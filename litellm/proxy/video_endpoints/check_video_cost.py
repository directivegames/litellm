"""
Bills video jobs once the provider finishes them.

No video adapter on these routes bills at create. Each marks the created VideoObject with
VIDEO_COST_POLL_ID_KEY, and the create endpoint calls record_pending_video_job, which stores
the job and the identity that created it in LiteLLM_ManagedObjectTable with file_purpose
'video'. CheckVideoCost runs on the proxy scheduler. Each cycle it checks the jobs it checked
longest ago, and bills each finished one exactly once, to the key that created it. A job that
ends without a reported cost closes unbilled.

The caller's own status checks and downloads never bill (NON_INFERENCE_CALL_TYPES), so spend
does not depend on the caller polling, or on how often. Every path that leaves a job unbilled
logs an error and sends a failed-tracking alert.

Callers: the /videos create endpoint (record_pending_video_job) and
ProxyStartupEvent.initialize_scheduled_background_jobs (CheckVideoCost).
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import (
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
    MANAGED_OBJECT_STALENESS_CUTOFF_DAYS,
    MAX_OBJECTS_PER_POLL_CYCLE,
)
from litellm.types.utils import BACKGROUND_VIDEO_COST_POLL_CALL_ORIGIN
from litellm.types.videos.utils import VIDEO_COST_POLL_ID_KEY, decode_video_id_with_provider

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router
    from litellm.types.videos.main import VideoObject

VIDEO_FILE_PURPOSE: Final = "video"
CHECK_VIDEO_COST_USER_AGENT: Final = "LiteLLM Proxy/CheckVideoCost"
TERMINAL_VIDEO_JOB_STATUSES: Final = ("completed", "failed", "stale_expired")
# A finished job bills whatever its provider reported, so a provider that charges
# for a failed or cancelled job is still billed.
PROVIDER_FINISHED_VIDEO_STATUSES: Final = frozenset({"completed", "failed", "cancelled", "expired"})


class VideoCostAlerts(Protocol):
    """ProxyLogging's alert for spend the proxy could not record."""

    async def failed_tracking_alert(self, error_message: str, failing_model: str) -> None: ...


class _VideoJobRow(Protocol):
    """The LiteLLM_ManagedObjectTable columns this poller reads."""

    @property
    def id(self) -> str: ...
    @property
    def unified_object_id(self) -> str: ...
    @property
    def status(self) -> str | None: ...
    @property
    def file_object(self) -> object: ...


class _VideoJobTable(Protocol):
    async def create(self, data: Mapping[str, object]) -> object: ...

    async def find_many(
        self, where: Mapping[str, object], take: int, order: Mapping[str, str]
    ) -> Sequence[_VideoJobRow]: ...

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> object: ...

    async def update_many(self, where: Mapping[str, object], data: Mapping[str, object]) -> int: ...


def _video_jobs(prisma_client: "PrismaClient") -> _VideoJobTable:
    return cast(_VideoJobTable, prisma_client.db.litellm_managedobjecttable)  # cast-ok: generated Prisma client


class PollRecord(BaseModel):
    """What record_pending_video_job stores in file_object for the poller."""

    model_config = ConfigDict(frozen=True)

    poll_video_id: str
    custom_llm_provider: str
    model_id: str
    # The creator's spend metadata, captured at create so a later rename or
    # key rotation cannot move the charge.
    metadata: Mapping[str, object] = Field(default_factory=lambda: MappingProxyType({}))


async def _alert(alerts: VideoCostAlerts, message: str, model: str = "") -> None:
    verbose_proxy_logger.error(message)
    try:
        await alerts.failed_tracking_alert(error_message=message, failing_model=model)
    except Exception as e:
        verbose_proxy_logger.error("CheckVideoCost: could not send the alert: %s", e)


def _unbilled(video_id: str, reason: str) -> str:
    return f"CheckVideoCost: video {video_id} will not be billed: {reason}"


def _creator_identity(user_api_key_dict: "UserAPIKeyAuth", request_tags: Sequence[str]) -> Mapping[str, object]:
    """The spend metadata the create request itself would have billed under."""
    fields: Final = (
        ("user_api_key", user_api_key_dict.api_key),
        ("user_api_key_hash", user_api_key_dict.api_key),
        ("user_api_key_alias", user_api_key_dict.key_alias),
        ("user_api_key_user_id", user_api_key_dict.user_id),
        ("user_api_key_user_email", user_api_key_dict.user_email),
        ("user_api_key_team_id", user_api_key_dict.team_id),
        ("user_api_key_team_alias", user_api_key_dict.team_alias),
        ("user_api_key_org_id", user_api_key_dict.org_id),
        ("user_api_key_org_alias", user_api_key_dict.organization_alias),
        ("user_api_key_project_id", user_api_key_dict.project_id),
        ("user_api_key_end_user_id", user_api_key_dict.end_user_id),
        ("tags", tuple(request_tags) or None),
    )
    return MappingProxyType({key: value for key, value in fields if value is not None})


def _request_tags(request_data: Mapping[str, object]) -> tuple[str, ...]:
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.batch_attribution import (
        request_tags_from_metadata,
    )

    metadata: Final = request_data.get("metadata")
    if not isinstance(metadata, Mapping):
        return ()
    return tuple(request_tags_from_metadata(cast("Mapping[str, object]", metadata)) or ())  # cast-ok: request JSON


def _poll_id(response: object) -> str | None:
    hidden_params: Final[object] = getattr(response, "_hidden_params", None)
    if not isinstance(hidden_params, Mapping):
        return None
    poll_id: Final[object] = cast("Mapping[str, object]", hidden_params).get(VIDEO_COST_POLL_ID_KEY)  # cast-ok: SDK
    return poll_id if isinstance(poll_id, str) and poll_id else None


def _pending_row(
    response: object,
    video_id: str,
    poll_id: str,
    user_api_key_dict: "UserAPIKeyAuth",
    request_data: Mapping[str, object],
) -> Mapping[str, object]:
    decoded: Final = decode_video_id_with_provider(video_id)
    provider: Final = decoded.get("custom_llm_provider")
    model_id: Final = decoded.get("model_id")
    if not provider or not model_id:
        raise ValueError("its id names no provider deployment")
    record: Final = PollRecord(
        poll_video_id=poll_id,
        custom_llm_provider=provider,
        model_id=model_id,
        metadata=_creator_identity(user_api_key_dict, _request_tags(request_data)),
    )
    status: Final[object] = getattr(response, "status", None)
    return {  # mutable-ok: prisma's query builder rejects a Mapping
        "unified_object_id": video_id,
        # Prefixed so a video task id can never match a batch id that
        # enforce_batch_object_access looks up by model_object_id.
        "model_object_id": f"{VIDEO_FILE_PURPOSE}:{provider}:{poll_id}",
        "file_object": record.model_dump_json(),
        "file_purpose": VIDEO_FILE_PURPOSE,
        "status": status if isinstance(status, str) and status else "queued",
        "created_by": user_api_key_dict.user_id,
        "updated_by": user_api_key_dict.user_id,
        "team_id": user_api_key_dict.team_id,
        "org_id": user_api_key_dict.org_id,
        "api_key": user_api_key_dict.api_key,
    }


async def record_pending_video_job(
    response: object,
    user_api_key_dict: "UserAPIKeyAuth",
    request_data: Mapping[str, object],
    prisma_client: "PrismaClient | None",
    alerts: VideoCostAlerts,
) -> None:
    """Store a created job for CheckVideoCost to bill.

    Never raises: the provider has already accepted the job, and failing the response
    now would still bill nobody and would lose the caller its video id.
    """
    poll_id: Final = _poll_id(response)
    if poll_id is None:
        return
    video_id: Final[object] = getattr(response, "id", None)
    if not isinstance(video_id, str) or not video_id:
        await _alert(alerts, _unbilled(str(video_id), "the created video has no id"))
        return
    if prisma_client is None:
        # With no database the proxy records no spend at all, so there is nothing to bill against.
        verbose_proxy_logger.warning("CheckVideoCost: no database, so video %s will not be billed", video_id)
        return
    try:
        row: Final = _pending_row(response, video_id, poll_id, user_api_key_dict, request_data)
        await _video_jobs(prisma_client).create(data=row)
    except Exception as e:
        await _alert(alerts, _unbilled(video_id, f"it could not be recorded: {e}"))


def _poll_record(job: _VideoJobRow) -> PollRecord | None:
    try:
        if isinstance(job.file_object, (str, bytes)):
            return PollRecord.model_validate_json(job.file_object)
        return PollRecord.model_validate(job.file_object)
    except ValidationError:
        return None


def _video_config(provider: str) -> "BaseVideoConfig | None":
    import litellm
    from litellm.utils import ProviderConfigManager

    try:
        llm_provider: Final = litellm.LlmProviders(provider)
    except ValueError:
        return None
    return ProviderConfigManager.get_provider_video_config(model=None, provider=llm_provider)


def _reported_cost(video: "VideoObject") -> float | None:
    usage: Final[object] = video.usage
    if not isinstance(usage, Mapping):
        return None
    cost: Final[object] = cast("Mapping[str, object]", usage).get("provider_reported_cost_usd")  # cast-ok: SDK usage
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost <= 0:
        return None
    return float(cost)


class CheckVideoCost:
    def __init__(
        self,
        prisma_client: "PrismaClient",
        llm_router: "Router",
        alerts: VideoCostAlerts,
    ) -> None:
        self.jobs: Final = _video_jobs(prisma_client)
        self.llm_router: Final = llm_router
        self.alerts: Final = alerts

    async def check_video_cost(self) -> None:
        try:
            await self._expire_stale_jobs()
        except Exception as e:
            verbose_proxy_logger.warning("CheckVideoCost: stale sweep failed (poll will continue): %s", e)

        # Every check rewrites the row, so ordering by updated_at sends a job that cannot be
        # billed yet to the back, instead of letting it hold a slot in every cycle.
        jobs: Final = await self.jobs.find_many(
            where={  # mutable-ok: prisma's query builder rejects a Mapping
                "file_purpose": VIDEO_FILE_PURPOSE,
                "batch_processed": False,
                "status": {"not_in": [*TERMINAL_VIDEO_JOB_STATUSES]},  # mutable-ok: prisma needs a dict
            },
            take=MAX_OBJECTS_PER_POLL_CYCLE,
            order={"updated_at": "asc"},  # mutable-ok: prisma's query builder rejects a Mapping
        )
        for job in jobs:
            try:
                await self._check_job(job)
            except Exception as e:
                verbose_proxy_logger.error(
                    "CheckVideoCost: could not check video %s; retrying later: %s", job.unified_object_id, e
                )
                await self._touch(job)

    async def _expire_stale_jobs(self) -> None:
        """A job the provider has not finished within the staleness window never will; stop polling it."""
        cutoff: Final = datetime.now(timezone.utc) - timedelta(days=MANAGED_OBJECT_STALENESS_CUTOFF_DAYS)
        expired: Final = await self.jobs.update_many(
            where={  # mutable-ok: prisma's query builder rejects a Mapping
                "file_purpose": VIDEO_FILE_PURPOSE,
                "batch_processed": False,
                "created_at": {"lt": cutoff},  # mutable-ok: prisma needs a dict
            },
            data={"status": "stale_expired", "batch_processed": True},  # mutable-ok: prisma needs a dict
        )
        if expired > 0:
            await _alert(
                self.alerts,
                f"CheckVideoCost: gave up on {expired} video jobs that did not finish within "
                f"{MANAGED_OBJECT_STALENESS_CUTOFF_DAYS} days; they will not be billed",
            )

    async def _check_job(self, job: _VideoJobRow) -> None:
        record: Final = _poll_record(job)
        if record is None:
            await self._close_unbilled(job, "failed", "its poll record cannot be read")
            return
        config: Final = _video_config(record.custom_llm_provider)
        if config is None:
            reason: Final = f"{record.custom_llm_provider} has no video adapter"
            await self._close_unbilled(job, "failed", reason, record.model_id)
            return
        video: Final = await self._fetch_status(record, config)
        if video is None:
            await self._touch(job)
            return
        if video.status not in PROVIDER_FINISHED_VIDEO_STATUSES:
            await self._set_status(job, video.status)
            return
        closed_status: Final = "completed" if video.status == "completed" else "failed"
        cost: Final = _reported_cost(video)
        if cost is None and closed_status == "completed":
            await self._close_unbilled(job, closed_status, "it finished but reported no cost", record.model_id)
            return
        if cost is None:
            await self._set_status(job, closed_status, processed=True)
            return
        if not await self._claim(job):
            return
        try:
            await self._bill(job, record, video, cost)
        except Exception:
            await self._release(job)
            raise
        await self._set_status(job, closed_status)

    async def _set_status(self, job: _VideoJobRow, status: str, processed: bool = False) -> None:
        data: Final = (
            {"status": status, "batch_processed": True}  # mutable-ok: prisma needs a dict
            if processed
            else {"status": status}  # mutable-ok: prisma needs a dict
        )
        await self.jobs.update(where={"id": job.id}, data=data)  # mutable-ok: prisma's query builder rejects a Mapping

    async def _touch(self, job: _VideoJobRow) -> None:
        try:
            await self._set_status(job, job.status or "queued")
        except Exception as e:
            verbose_proxy_logger.error("CheckVideoCost: could not requeue video %s: %s", job.unified_object_id, e)

    async def _close_unbilled(self, job: _VideoJobRow, status: str, reason: str, model: str = "") -> None:
        await self._set_status(job, status, processed=True)
        await _alert(self.alerts, _unbilled(job.unified_object_id, reason), model)

    async def _claim(self, job: _VideoJobRow) -> bool:
        """Every pod runs this poller; only the one whose flip from unbilled succeeds may bill the job."""
        claimed: Final = await self.jobs.update_many(
            where={"id": job.id, "batch_processed": False},  # mutable-ok: prisma's query builder rejects a Mapping
            data={"batch_processed": True},  # mutable-ok: prisma's query builder rejects a Mapping
        )
        return claimed > 0

    async def _release(self, job: _VideoJobRow) -> None:
        try:
            await self.jobs.update_many(
                where={"id": job.id, "batch_processed": True},  # mutable-ok: prisma's query builder rejects a Mapping
                data={"batch_processed": False},  # mutable-ok: prisma's query builder rejects a Mapping
            )
        except Exception as e:
            reason: Final = f"its claim could not be released after a failed charge: {e}"
            await _alert(self.alerts, _unbilled(job.unified_object_id, reason))

    async def _fetch_status(self, record: PollRecord, config: "BaseVideoConfig") -> "VideoObject | None":
        """The provider's view of the job, or nothing while its deployment is gone or paused.

        The caller-facing status route logs through every proxy callback, and its call type
        bills nothing, so the poller calls the status handler with a logging object nobody reports.
        """
        from litellm.main import base_llm_http_handler
        from litellm.types.router import GenericLiteLLMParams

        deployment: Final = self.llm_router.get_deployment(model_id=record.model_id)
        credentials: Final = self.llm_router.get_deployment_credentials_with_provider(record.model_id)
        if deployment is None or credentials is None:
            verbose_proxy_logger.warning(
                "CheckVideoCost: deployment %s is gone or paused, so video %s waits for it",
                record.model_id,
                record.poll_video_id,
            )
            return None
        litellm_params: Final = GenericLiteLLMParams.model_validate(
            MappingProxyType({**credentials, "model_info": deployment.model_info.model_dump(exclude_none=True)})
        )
        api_key: Final = credentials.get("api_key")
        return await base_llm_http_handler.async_video_status_handler(  # pyright: ignore[reportUnknownMemberType]  # its litellm_params and logging_obj are untyped
            video_id=record.poll_video_id,
            video_status_provider_config=config,
            custom_llm_provider=record.custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=_logging_obj("", None, {}, record.custom_llm_provider),  # mutable-ok: Logging writes into it
            api_key=api_key if isinstance(api_key, str) else None,
        )

    async def _bill(self, job: _VideoJobRow, record: PollRecord, video: "VideoObject", cost: float) -> None:
        deployment: Final = self.llm_router.get_deployment(model_id=record.model_id)
        # The spend row names the video the caller holds, not the provider's task id.
        billed: Final = video.model_copy(update=MappingProxyType({"id": job.unified_object_id}))
        billed._hidden_params = {  # pyright: ignore[reportPrivateUsage]  # logging reads the cost here  # mutable-ok: logging writes into it
            **video._hidden_params,  # pyright: ignore[reportPrivateUsage]  # the adapter's facts carry over
            "response_cost": cost,
            "model_id": record.model_id,
        }
        metadata: Final = {  # mutable-ok: the success handler writes hidden_params into this metadata
            **record.metadata,
            "model_info": {"id": record.model_id},  # mutable-ok: spend logs read the deployment id here
            "model_group": deployment.model_name if deployment is not None else None,
            INTERNAL_CALL_ORIGIN_METADATA_KEY: BACKGROUND_VIDEO_COST_POLL_CALL_ORIGIN,
        }
        request: Final = {"headers": {"user-agent": CHECK_VIDEO_COST_USER_AGENT}}  # mutable-ok: logging reads a dict
        params: Final[dict[str, object]] = {  # mutable-ok: Logging writes into it
            "proxy_server_request": request,
            "metadata": metadata,
        }
        model: Final = video.model or (deployment.litellm_params.model if deployment is not None else "")
        logging_obj: Final = _logging_obj(model, "<video_cost_poll>", params, record.custom_llm_provider)
        await logging_obj.async_success_handler(result=billed)


def _logging_obj(
    model: str,
    prompt: str | None,
    litellm_params: dict[str, object],  # mutable-ok: Logging writes into its litellm_params
    custom_llm_provider: str,
) -> "LiteLLMLogging":
    """A logging object for one poller call. Only the object passed to async_success_handler reports anywhere."""
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

    logging_obj: Final = LiteLLMLogging(
        model=model,
        messages=[{"role": "user", "content": prompt}] if prompt else [],  # mutable-ok: Logging's messages is a list
        stream=False,
        call_type="avideo_retrieve",
        start_time=datetime.now(),
        litellm_call_id=str(uuid.uuid4()),
        function_id=str(uuid.uuid4()),
    )
    logging_obj.update_environment_variables(  # pyright: ignore[reportUnknownMemberType]  # its dict parameters are untyped
        litellm_params=litellm_params,
        optional_params={},  # mutable-ok: Logging writes into its optional_params
        custom_llm_provider=custom_llm_provider,
    )
    return logging_obj
