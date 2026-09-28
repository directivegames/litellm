"""MiniMax Hailuo video jobs on the OpenAI /videos routes.

Callers: ProviderConfigManager.get_provider_video_config.
MinimaxVideoConfig, MinimaxRates, minimax_rates, minimax_completed_cost.

Rates come from the deployment's model_info, keyed by output resolution in
lower case (480p, 768p, 2k). Create refuses a job whose rates are not set.
Only generation tasks are priced: this adapter creates no other task type.

Create does not bill. It hands the task id to the proxy's video cost poller
(VIDEO_COST_POLL_ID_KEY), which bills the job once it succeeds, from the
seconds and image count MiniMax reports. A job that fails is not charged.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final, cast
from urllib.parse import urlparse

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    VIDEO_COST_POLL_ID_KEY,
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

# International video API. The chat base ends in /v1; these routes are under /v2.
_DEFAULT_API_BASE: Final = "https://api.minimax.io"
_CREATE_PATH: Final = "/v2/video_generation"
_QUERY_PATH: Final = "/v2/query/video_generation"
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
# model_info keys. The two per-second keys end in the output resolution.
# MiniMax prices input video by the output resolution, not the input's. Audio is free.
OUTPUT_PER_SECOND_KEY: Final = "output_cost_per_second_{}"
INPUT_VIDEO_PER_SECOND_KEY: Final = "input_cost_per_video_per_second_{}"
PER_IMAGE_KEY: Final = "input_cost_per_image"
FREE_IMAGES_KEY: Final = "free_input_image_count"
_STATUS: Final = {
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


class MinimaxVideoError(BaseLLMException):
    pass


@dataclass(frozen=True)
class MinimaxRates:
    """One output resolution's rates, in dollars."""

    output_per_second: Decimal
    input_video_per_second: Decimal
    per_image: Decimal
    free_images: int


def _deployment_model_info(litellm_params: GenericLiteLLMParams) -> Mapping[str, object]:
    """The deployment's model_info, or empty when the call has none."""
    model_info: Final = cast(  # cast-ok: declared as a bare dict
        "Mapping[str, object] | None", litellm_params.model_info
    )
    return model_info if model_info is not None else {}


def minimax_rates(model_info: Mapping[str, object], resolution: object) -> MinimaxRates:
    """The deployment's rates for one output resolution. Raises when any is not set."""
    if not isinstance(resolution, str) or not resolution.strip():
        raise ValueError("MiniMax video needs a resolution to price the job.")
    suffix: Final = resolution.strip().lower()
    free: Final = model_info.get(FREE_IMAGES_KEY)
    if isinstance(free, bool) or not isinstance(free, int) or free < 0:
        raise ValueError(f"MiniMax video needs model_info.{FREE_IMAGES_KEY} (a whole number) on this deployment.")
    return MinimaxRates(
        output_per_second=_rate(model_info, OUTPUT_PER_SECOND_KEY.format(suffix)),
        input_video_per_second=_rate(model_info, INPUT_VIDEO_PER_SECOND_KEY.format(suffix)),
        per_image=_rate(model_info, PER_IMAGE_KEY),
        free_images=free,
    )


def minimax_completed_cost(body: object, model_info: Mapping[str, object]) -> float | None:
    """Dollars for a succeeded generation task at the deployment's rates, or nothing if it is not billable."""
    if not isinstance(body, dict):
        return None
    task = body.get("task")
    if not isinstance(task, dict) or task.get("status") != "succeeded":
        return None
    # Regeneration and Context-IR are separate MiniMax APIs this adapter never calls.
    if task.get("task_type", "generation") != "generation":
        return None
    usage = task.get("usage")
    if not isinstance(usage, dict):
        return None
    output_seconds = _whole_number(usage.get("output_seconds"))
    input_seconds = _whole_number(usage.get("input_seconds"))
    images = _whole_number(usage.get("input_image_count"))
    if output_seconds is None or input_seconds is None or images is None:
        return None
    # A finished job with no rate raises rather than being served unbilled.
    rates: Final = minimax_rates(model_info, task.get("resolution"))
    return _dollars(
        output_seconds * rates.output_per_second
        + input_seconds * rates.input_video_per_second
        + max(0, images - rates.free_images) * rates.per_image
    )


def _rate(model_info: Mapping[str, object], key: str) -> Decimal:
    value: Final = model_info.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"MiniMax video needs model_info.{key} (dollars) on this deployment.")
    return Decimal(str(value))


def _whole_number(value: object) -> int | None:
    # bool is an int subclass; a true/false count is not usage. A JSON count may arrive as 5.0,
    # and refusing it would leave a finished job unbilled. is_integer() is False for nan and inf.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def _dollars(cost: Decimal) -> float | None:
    if cost <= 0:
        return None
    return float(cost)


def _api_root(api_base: str | None) -> str:
    root = (api_base or _DEFAULT_API_BASE).rstrip("/")
    if root.endswith("/v1"):
        return root[: -len("/v1")]
    return root


def _map_status(status: object) -> str:
    if not isinstance(status, str):
        return "in_progress"
    return _STATUS.get(status.lower(), "in_progress")


def _https_url(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return None
    return value


class MinimaxVideoConfig(BaseVideoConfig):
    def __init__(self) -> None:
        super().__init__()
        # The status response hook gets no litellm_params, so the status request
        # keeps the deployment's model_info for pricing. One instance per call.
        self._model_info: Mapping[str, object] = {}

    def get_supported_openai_params(self, model: str) -> list:
        return ["model", "prompt", *_OPTIONAL_PARAMS]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped: dict[str, object] = {}
        for key in _OPTIONAL_PARAMS:
            value = video_create_optional_params.get(key)
            if value is not None:
                mapped[key] = value
        return mapped

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key
        api_key = api_key or get_secret_str("MINIMAX_API_KEY") or litellm.api_key
        if api_key is None:
            raise ValueError("MiniMax API key is required. Set MINIMAX_API_KEY or pass api_key.")
        headers.update({"Authorization": f"Bearer {api_key}"})
        return headers

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: dict) -> str:
        return _api_root(api_base)

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles, str]:
        params: Final = video_create_optional_request_params
        content = params.get("content")
        body: dict[str, object] = {
            "model": model,
            "content": content if isinstance(content, list) else [{"type": "text", "text": prompt}],
        }
        duration = params.get("duration", params.get("seconds"))
        if duration is not None:
            body["duration"] = duration
        resolution = params.get("resolution", params.get("size"))
        if isinstance(resolution, str):
            body["resolution"] = resolution
        ratio = params.get("ratio", params.get("aspect_ratio"))
        if isinstance(ratio, str):
            body["ratio"] = ratio
        if "generate_audio" in params:
            body["generate_audio"] = params["generate_audio"]
        # Refuse before MiniMax is called: a job with no rate would finish unbilled.
        minimax_rates(_deployment_model_info(litellm_params), body.get("resolution"))
        return body, [], f"{_api_root(api_base)}{_CREATE_PATH}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        payload: Final = raw_response.json()
        if not isinstance(payload, dict):
            raise MinimaxVideoError(
                status_code=raw_response.status_code,
                message="MiniMax video response was not an object",
                headers=raw_response.headers,
            )
        job_id = payload.get("task_id") or payload.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise MinimaxVideoError(
                status_code=raw_response.status_code,
                message="MiniMax video response did not include a task id",
                headers=raw_response.headers,
            )
        video = _video_object(job_id, _map_status(payload.get("status", "queued")), model)
        if custom_llm_provider:
            video.id = encode_video_id_with_provider(job_id, custom_llm_provider, model)
        video._hidden_params[VIDEO_COST_POLL_ID_KEY] = job_id  # pyright: ignore[reportPrivateUsage]  # the proxy reads adapter facts here
        return video

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        self._model_info = _deployment_model_info(litellm_params)
        task_id: Final = encode_url_path_segment(extract_original_video_id(video_id), field_name="video_id")
        return f"{_api_root(api_base)}{_QUERY_PATH}/{task_id}", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        payload: Final = raw_response.json()
        task = payload.get("task") if isinstance(payload, dict) else None
        if not isinstance(task, dict):
            raise MinimaxVideoError(
                status_code=raw_response.status_code,
                message="MiniMax video status did not include a task",
                headers=raw_response.headers,
            )
        job_id = task.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise MinimaxVideoError(
                status_code=raw_response.status_code,
                message="MiniMax video status did not include a task id",
                headers=raw_response.headers,
            )
        model = task.get("model") if isinstance(task.get("model"), str) else None
        video = _video_object(job_id, _map_status(task.get("status")), model)
        cost = minimax_completed_cost(payload, self._model_info)
        if cost is not None:
            video.usage = {"provider_reported_cost_usd": cost}
        if custom_llm_provider:
            video.id = encode_video_id_with_provider(job_id, custom_llm_provider, model)
        return video

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: str | None = None,
    ) -> tuple[str, dict]:
        # The file is a CDN link on the finished task, not a separate content route.
        return self.transform_video_status_retrieve_request(video_id, api_base, litellm_params, headers)

    def transform_video_content_response(self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj) -> bytes:
        video_url = _download_url(raw_response.json())
        if video_url is None:
            raise ValueError("MiniMax video file is not ready.")
        downloaded = _get_httpx_client().get(video_url)
        downloaded.raise_for_status()
        return downloaded.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        video_url = _download_url(raw_response.json())
        if video_url is None:
            raise ValueError("MiniMax video file is not ready.")
        client: Final[AsyncHTTPHandler] = get_async_httpx_client(llm_provider=litellm.LlmProviders.MINIMAX)
        downloaded = await client.get(video_url)
        downloaded.raise_for_status()
        return downloaded.content

    def transform_video_remix_request(self, video_id, prompt, api_base, litellm_params, headers, extra_body=None):
        raise NotImplementedError("Video remix is not supported for MiniMax")

    def transform_video_remix_response(self, raw_response, logging_obj, custom_llm_provider=None):
        raise NotImplementedError("Video remix is not supported for MiniMax")

    def transform_video_list_request(
        self, api_base, litellm_params, headers, after=None, limit=None, order=None, extra_query=None
    ):
        raise NotImplementedError("Video listing is not supported for MiniMax")

    def transform_video_list_response(self, raw_response, logging_obj, custom_llm_provider=None):
        raise NotImplementedError("Video listing is not supported for MiniMax")

    def transform_video_delete_request(self, video_id, api_base, litellm_params, headers):
        raise NotImplementedError("Video delete is not supported for MiniMax")

    def transform_video_delete_response(self, raw_response, logging_obj):
        raise NotImplementedError("Video delete is not supported for MiniMax")

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return MinimaxVideoError(status_code=status_code, message=error_message, headers=headers)


def _video_object(job_id: str, status: str, model: str | None) -> VideoObject:
    video = VideoObject(id=job_id, object="video", status=status, created_at=0)
    if model is not None:
        video.model = model
    return video


def _download_url(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    task = payload.get("task")
    if not isinstance(task, dict) or _map_status(task.get("status")) != "completed":
        return None
    content = task.get("content")
    if not isinstance(content, dict):
        return None
    return _https_url(content.get("url"))
