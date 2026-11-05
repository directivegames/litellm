"""BytePlus ModelArk Seedance video jobs on the OpenAI /videos routes.

Callers: ProviderConfigManager.get_provider_video_config.
BytePlusVideoConfig, RATE_KEYS, seedance_rate, seedance_completed_cost,
seedance_input_kind, attach_rate_class, split_rate_class.

The finished task reports tokens, not whether the create body included video.
The per-token rate depends on that class and on the output resolution, and
comes from RATE_KEYS on the deployment. Operators set rates on litellm_params;
proxy update strips them from persisted model_info. deployment_pricing_model_info
merges both. Create refuses a job whose rate is not set.

Create does not bill. The class is suffixed onto the task id handed to the
proxy's video cost poller (VIDEO_COST_POLL_ID_KEY), and stripped on the way
back to BytePlus. The caller's video id carries the same class, so a status
response prices at the rate the job is billed at. An id with no suffix, from
before this change, prices at the plain rate. Spend is billed from the
server-stored poll id, not from what the caller polls.
"""

import asyncio
import math
from collections.abc import Mapping
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NoReturn
from urllib.parse import urlparse

import httpx
from httpx._types import RequestFiles
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.deployment_model_info import deployment_pricing_model_info
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # house cached-client factory has no public alias and its stub leaves params untyped
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    PROVIDER_RESPONSE_KEY,
    PROVIDER_TASK_ID_KEY,
    VIDEO_COST_POLL_ID_KEY,
    encode_video_id_with_provider,
    extract_original_video_id,
    stamp_provider_task_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

# International ModelArk. Volcengine's China host is a different provider.
_DEFAULT_API_BASE: Final = "https://ark.ap-southeast.bytepluses.com"
_TASKS_PATH: Final = "/api/v3/contents/generations/tasks"
_OPTIONAL_PARAMS: Final = (
    "duration",
    "seconds",
    "resolution",
    "ratio",
    "aspect_ratio",
    "generate_audio",
    "size",
    "content",
)
# Dollars per token on the deployment's model_info. BytePlus bills the whole
# task at one rate, chosen by whether the create input had a video and by the
# output resolution. Keys end in the resolution in lower case (480p, 720p, 1080p).
RATE_KEYS: Final = MappingProxyType(
    {
        "plain": "output_cost_per_video_token_without_video_input_{}",
        "video": "output_cost_per_video_token_with_video_input_{}",
    }
)
_STATUS: Final = MappingProxyType(
    {
        "queued": "queued",
        "pending": "queued",
        "submitted": "queued",
        "preparing": "queued",
        "running": "in_progress",
        "processing": "in_progress",
        "in_progress": "in_progress",
        "succeeded": "completed",
        "success": "completed",
        "completed": "completed",
        "failed": "failed",
        "error": "failed",
        "cancelled": "failed",
        "canceled": "failed",
        "expired": "failed",
    }
)
_FINISHED: Final = ("completed", "failed")
_VIDEO_ITEM_TYPES: Final = frozenset({"video", "video_url"})

_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_FIELDS_ADAPTER: Final = TypeAdapter(dict[str, object])
_ITEMS_ADAPTER: Final = TypeAdapter(tuple[object, ...])


class BytePlusVideoError(BaseLLMException):
    pass


def _fields(value: object) -> Mapping[str, object]:
    """A JSON object's fields, or none when the value is not an object."""
    try:
        return MappingProxyType(_FIELDS_ADAPTER.validate_python(value))
    except ValidationError:
        return _EMPTY


def _json_object(raw_response: httpx.Response) -> Mapping[str, object] | None:
    try:
        return MappingProxyType(_FIELDS_ADAPTER.validate_python(raw_response.json()))
    except ValidationError:
        return None


def _items(value: object) -> tuple[object, ...]:
    try:
        return _ITEMS_ADAPTER.validate_python(value)
    except ValidationError:
        return ()


def attach_rate_class(task_id: str, kind: str) -> str:
    """Add the input class to the id the cost poller checks. BytePlus never sees this suffix."""
    if kind not in RATE_KEYS:
        return task_id
    return f"{task_id}~{kind}"


def split_rate_class(stored_id: str) -> tuple[str, str | None]:
    """Task id for the provider, and plain or video when the suffix is one of those."""
    task_id, separator, kind = stored_id.rpartition("~")
    if separator == "" or kind not in RATE_KEYS or not task_id:
        return stored_id, None
    return task_id, kind


def _is_text_item(item: object) -> bool:
    kind: Final = _fields(item).get("type")
    return isinstance(kind, str) and kind.lower() == "text"


def _caller_media(content: object) -> tuple[object, ...]:
    """
    ``prompt`` is what guardrails scanned, so it is the only text BytePlus takes; a caller's ``content``
    array only contributes media, never text that would bypass that check.
    """
    items: Final = _items(content)
    if any(_is_text_item(item) for item in items):
        raise UnsupportedParamsError(
            message="content must hold only BytePlus image, video or audio items; pass the text of the request as prompt.",
            llm_provider="byteplus",
        )
    return items


def _is_video_item(item: object) -> bool:
    fields: Final = _fields(item)
    kind: Final = fields.get("type")
    return (isinstance(kind, str) and kind.lower() in _VIDEO_ITEM_TYPES) or "video_url" in fields


def seedance_input_kind(body: object) -> str | None:
    """plain or video, from the create body. Nothing if the body has no content list."""
    content: Final = _fields(body).get("content")
    items: Final = _items(content)
    if not isinstance(content, list):
        return None
    return "video" if any(_is_video_item(item) for item in items) else "plain"


def seedance_rate(model_info: Mapping[str, object], kind: str, resolution: object) -> Decimal:
    """Dollars per token for this input class and resolution. Raises when the deployment does not set it."""
    if not isinstance(resolution, str) or not resolution.strip():
        raise ValueError("BytePlus video needs a resolution to price the job.")
    key: Final = RATE_KEYS[kind].format(resolution.strip().lower())
    value: Final = model_info.get(key)
    # bool is an int subclass; true/false is not a price.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"BytePlus video needs model_info.{key} (dollars per token) on this deployment.")
    return Decimal(str(value))


def _whole_number(value: object) -> int | None:
    # bool is an int subclass; a true/false count is not usage. A JSON count may arrive as 5.0,
    # and refusing it would leave a finished job unbilled. is_integer() is False for nan and inf.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def _map_status(status: object) -> str:
    if not isinstance(status, str):
        return "in_progress"
    return _STATUS.get(status.lower(), "in_progress")


def seedance_completed_cost(body: object, kind: str | None, model_info: Mapping[str, object]) -> float | None:
    """Dollars for the tokens a Seedance task reports, whatever its status, at the deployment's rate for its input class and resolution."""
    fields: Final = _fields(body)
    if kind is None or kind not in RATE_KEYS or _map_status(fields.get("status")) not in _FINISHED:
        return None
    tokens: Final = _whole_number(_fields(fields.get("usage")).get("total_tokens"))
    if tokens is None or tokens == 0:
        return None
    # A finished job with no rate raises rather than being served unbilled.
    cost: Final = Decimal(tokens) * seedance_rate(model_info, kind, fields.get("resolution"))
    if cost <= 0:
        return None
    return float(cost)


def _seedance_billed_usage(
    body: Mapping[str, object], kind: str, model_info: Mapping[str, object], cost: float
) -> dict[str, object]:  # mutable-ok: VideoObject.usage is a dict field
    """The price, the token count, and the per-token rate that produced it."""
    resolution: Final = body.get("resolution")
    resolution_name: Final = resolution.strip().lower() if isinstance(resolution, str) else ""
    rate: Final = (
        {RATE_KEYS[kind].format(resolution_name): float(seedance_rate(model_info, kind, resolution))}
        if resolution_name
        else _EMPTY
    )
    return {
        "provider_reported_cost_usd": cost,
        "total_tokens": _whole_number(_fields(body.get("usage")).get("total_tokens")),
        "video_resolution": resolution_name or None,
        **rate,
    }


def _failure(status: str, error: object) -> dict[str, object] | None:  # mutable-ok: VideoObject.error is a dict field
    fields: Final = _fields(error)
    if status != "failed" or not fields:
        return None
    return dict(fields)


def _api_root(api_base: str | None) -> str:
    return (api_base or _DEFAULT_API_BASE).rstrip("/")


def _https_url(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parsed: Final = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return None
    return value


def _requested_video_id(logging_obj: object) -> str:
    details: Final = _fields(getattr(logging_obj, "model_call_details", None))
    requested: Final = _fields(details.get("additional_args")).get("video_id")
    return requested if isinstance(requested, str) else ""


def _status_request_video_id(logging_obj: object, status_video_id: str) -> str:
    """The video id from the status HTTP call, preferring what the handler logged."""
    return _requested_video_id(logging_obj) or status_video_id


def _pricing_input_kind(logging_obj: object, status_video_id: str) -> str:
    """plain or video for token pricing. Callers without a class suffix use plain."""
    _task_id, kind = split_rate_class(extract_original_video_id(_status_request_video_id(logging_obj, status_video_id)))
    return kind if kind is not None and kind in RATE_KEYS else "plain"


def _download_url(fields: Mapping[str, object] | None) -> str | None:
    if fields is None or _map_status(fields.get("status")) != "completed":
        return None
    return _https_url(_fields(fields.get("content")).get("video_url"))


def _download_signed_video_bytes(video_url: str) -> bytes:
    """Fetch a provider-signed CDN URL with the sync httpx client.

    TOS/CDN responses can carry non-ASCII header values (for example
    ``x-tos-expiration``). LiteLLM's default async aiohttp transport forwards
    those into ``httpx.Response`` and httpx rejects them; the sync client does not.
    """
    client: Final[HTTPHandler] = _get_httpx_client()
    downloaded: Final = client.get(video_url)  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.get leaves its params untyped
    downloaded.raise_for_status()
    return downloaded.content


def _not_ready() -> ValueError:
    return ValueError("BytePlus video file is not ready.")


class BytePlusVideoConfig(BaseVideoConfig):
    def __init__(self) -> None:
        super().__init__()
        # The status response hook gets no litellm_params, so the status request
        # keeps the deployment's model_info for pricing. One instance per call.
        self._model_info: Mapping[str, object] = _EMPTY
        self._status_video_id: str = ""

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig contract returns list
        return ["model", "prompt", *_OPTIONAL_PARAMS]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseVideoConfig contract
        sent: Final = _fields(video_create_optional_params)
        return {key: sent[key] for key in _OPTIONAL_PARAMS if sent.get(key) is not None}

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig contract
        key: Final = (
            api_key
            or (litellm_params.api_key if litellm_params is not None else None)
            or get_secret_str("ARK_API_KEY")
            or litellm.api_key
        )
        if key is None:
            raise ValueError("BytePlus API key is required. Set ARK_API_KEY or pass api_key.")
        return {**headers, "Authorization": f"Bearer {key}"}

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],  # mutable-ok: BaseVideoConfig contract
    ) -> str:
        return _api_root(api_base)

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict[str, object],  # mutable-ok: BaseVideoConfig contract
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[dict[str, object], RequestFiles, str]:  # mutable-ok: BaseVideoConfig contract
        params: Final = video_create_optional_request_params
        media: Final = _caller_media(params.get("content"))
        duration: Final = params.get("duration", params.get("seconds"))
        resolution: Final = params.get("resolution", params.get("size"))
        ratio: Final = params.get("ratio", params.get("aspect_ratio"))
        optional: Final = (
            ("duration", duration),
            ("resolution", resolution if isinstance(resolution, str) else None),
            ("ratio", ratio if isinstance(ratio, str) else None),
        )
        audio: Final = MappingProxyType(
            {"generate_audio": params["generate_audio"]} if "generate_audio" in params else {}
        )
        body: Final[dict[str, object]] = {  # mutable-ok: BaseVideoConfig contract returns the body as a dict
            "model": model,
            "content": [*(({"type": "text", "text": prompt},) if prompt else ()), *media],
            **{key: value for key, value in optional if value is not None},
            **audio,
        }
        # Refuse before BytePlus is called: a job with no rate would finish unbilled.
        seedance_rate(
            deployment_pricing_model_info(litellm_params), seedance_input_kind(body) or "plain", body.get("resolution")
        )
        return body, [], f"{_api_root(api_base)}{_TASKS_PATH}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: BaseVideoConfig contract
    ) -> VideoObject:
        payload: Final = _json_object(raw_response) or _EMPTY
        job_id: Final = payload.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise BytePlusVideoError(
                status_code=raw_response.status_code,
                message="BytePlus video response did not include a task id",
                headers=raw_response.headers,
            )
        # Matches the class create priced the gate with.
        classed_id: Final = attach_rate_class(job_id, seedance_input_kind(request_data) or "plain")
        video: Final = VideoObject(
            id=encode_video_id_with_provider(classed_id, custom_llm_provider, model) if custom_llm_provider else job_id,
            object="video",
            status=_map_status(payload.get("status", "queued")),
            created_at=0,
            model=model,
        )
        video._hidden_params[VIDEO_COST_POLL_ID_KEY] = classed_id  # pyright: ignore[reportPrivateUsage]  # the proxy reads adapter facts here
        stamp_provider_task_id(logging_obj, job_id)
        return video

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        self._model_info = deployment_pricing_model_info(litellm_params)
        self._status_video_id = video_id
        task_id, _kind = split_rate_class(extract_original_video_id(video_id))
        encoded: Final = encode_url_path_segment(task_id, field_name="video_id")
        return f"{_api_root(api_base)}{_TASKS_PATH}/{encoded}", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        client: HTTPHandler | None = None,
    ) -> VideoObject:
        payload: Final = _json_object(raw_response)
        if payload is None:
            raise BytePlusVideoError(
                status_code=raw_response.status_code,
                message="BytePlus video status was not an object",
                headers=raw_response.headers,
            )
        job_id: Final = payload.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise BytePlusVideoError(
                status_code=raw_response.status_code,
                message="BytePlus video status did not include a task id",
                headers=raw_response.headers,
            )
        request_video_id: Final = _status_request_video_id(logging_obj, self._status_video_id)
        _task_id, kind = split_rate_class(extract_original_video_id(request_video_id))
        pricing_kind: Final = _pricing_input_kind(logging_obj, self._status_video_id)
        stored_id: Final = attach_rate_class(job_id, kind) if kind is not None else job_id
        reported_model: Final = payload.get("model")
        model: Final = reported_model if isinstance(reported_model, str) else None
        status: Final = _map_status(payload.get("status"))
        cost: Final = seedance_completed_cost(payload, pricing_kind, self._model_info)
        video: Final = VideoObject(
            id=encode_video_id_with_provider(stored_id, custom_llm_provider, model)
            if custom_llm_provider
            else stored_id,
            object="video",
            status=status,
            created_at=0,
            model=model,
            completed_at=_whole_number(payload.get("completed_at")) or None,
            error=_failure(status, payload.get("error")),
            usage=_seedance_billed_usage(payload, pricing_kind, self._model_info, cost) if cost is not None else None,
        )
        # The poller logs the whole provider answer on the finish row; the caller never sees it.
        video._hidden_params[PROVIDER_RESPONSE_KEY] = dict(payload)  # pyright: ignore[reportPrivateUsage]  # the poller reads adapter facts here
        video._hidden_params[PROVIDER_TASK_ID_KEY] = job_id  # pyright: ignore[reportPrivateUsage]  # the poller reads adapter facts here
        return video

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
        variant: str | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        return self.transform_video_status_retrieve_request(video_id, api_base, litellm_params, headers)

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        client: HTTPHandler | None = None,
    ) -> bytes:
        video_url: Final = _download_url(_json_object(raw_response))
        if video_url is None:
            raise _not_ready()
        return _download_signed_video_bytes(video_url)

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        client: AsyncHTTPHandler | None = None,
    ) -> bytes:
        video_url: Final = _download_url(_json_object(raw_response))
        if video_url is None:
            raise _not_ready()
        return await asyncio.to_thread(_download_signed_video_bytes, video_url)

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
    ) -> NoReturn:
        raise NotImplementedError("Video remix is not supported for BytePlus")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
    ) -> NoReturn:
        raise NotImplementedError("Video remix is not supported for BytePlus")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, object],
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, object] | None = None,
    ) -> NoReturn:
        raise NotImplementedError("Video listing is not supported for BytePlus")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
    ) -> NoReturn:
        raise NotImplementedError("Video listing is not supported for BytePlus")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, object],
    ) -> NoReturn:
        raise NotImplementedError("Video delete is not supported for BytePlus")

    def transform_video_delete_response(self, raw_response: httpx.Response, logging_obj: "Logging") -> NoReturn:
        raise NotImplementedError("Video delete is not supported for BytePlus")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,  # mutable-ok: BaseVideoConfig contract
    ) -> BaseLLMException:
        return BytePlusVideoError(status_code=status_code, message=error_message, headers=headers)
