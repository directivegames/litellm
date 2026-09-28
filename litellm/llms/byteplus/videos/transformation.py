"""BytePlus ModelArk Seedance video jobs on the OpenAI /videos routes.

Callers: ProviderConfigManager.get_provider_video_config.
BytePlusVideoConfig, RATE_KEYS, seedance_rate, seedance_completed_cost,
attach_rate_class, split_rate_class.

The finished task reports tokens, not whether the create body included video.
That class is suffixed onto the task id before it is encoded, then stripped
on the way back to BytePlus. The per-token rate depends on that class and on
the output resolution, and comes from the deployment's model_info (RATE_KEYS).
Create refuses a job whose rate is not set.
"""

import math
from collections.abc import Mapping
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
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

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
RATE_KEYS: Final = {
    "plain": "output_cost_per_video_token_without_video_input_{}",
    "video": "output_cost_per_video_token_with_video_input_{}",
}
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


class BytePlusVideoError(BaseLLMException):
    pass


def attach_rate_class(task_id: str, kind: str) -> str:
    """Hide the input class in the id the caller polls. BytePlus never sees this suffix."""
    if kind not in RATE_KEYS:
        return task_id
    return f"{task_id}~{kind}"


def split_rate_class(stored_id: str) -> tuple[str, str | None]:
    """Task id for the provider, and plain or video when the suffix is one of those."""
    task_id, separator, kind = stored_id.rpartition("~")
    if separator == "" or kind not in RATE_KEYS or not task_id:
        return stored_id, None
    return task_id, kind


def seedance_input_kind(body: object) -> str | None:
    """plain or video, from the create body. Nothing if the body has no content list."""
    if not isinstance(body, dict) or not isinstance(body.get("content"), list):
        return None
    for item in body["content"]:
        if isinstance(item, dict) and _is_video_item(item):
            return "video"
    return "plain"


def _deployment_model_info(litellm_params: GenericLiteLLMParams) -> Mapping[str, object]:
    """The deployment's model_info, or empty when the call has none."""
    model_info: Final = cast("Mapping[str, object] | None", litellm_params.model_info)  # cast-ok: declared as a bare dict
    return model_info if model_info is not None else {}


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


def seedance_completed_cost(body: object, kind: str | None, model_info: Mapping[str, object]) -> float | None:
    """Dollars for a succeeded Seedance task at the deployment's rate for its input class and resolution."""
    if kind not in RATE_KEYS or not isinstance(body, dict) or body.get("status") != "succeeded":
        return None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    tokens = usage.get("total_tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        return None
    # A finished job with no rate raises rather than being served unbilled.
    cost = Decimal(tokens) * seedance_rate(model_info, kind, body.get("resolution"))
    if cost <= 0:
        return None
    return float(cost)


def _is_video_item(item: dict[str, Any]) -> bool:
    kind = item.get("type")
    if isinstance(kind, str) and kind.lower() in {"video", "video_url"}:
        return True
    return "video_url" in item


def _api_root(api_base: str | None) -> str:
    return (api_base or _DEFAULT_API_BASE).rstrip("/")


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


def _requested_video_id(logging_obj: object) -> str:
    details = getattr(logging_obj, "model_call_details", None)
    if not isinstance(details, dict):
        return ""
    additional = details.get("additional_args")
    if isinstance(additional, dict) and isinstance(additional.get("video_id"), str):
        return additional["video_id"]
    return ""


class BytePlusVideoConfig(BaseVideoConfig):
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
        api_key = api_key or get_secret_str("ARK_API_KEY") or litellm.api_key
        if api_key is None:
            raise ValueError("BytePlus API key is required. Set ARK_API_KEY or pass api_key.")
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
        # Refuse before BytePlus is called: a job with no rate would finish unbilled.
        seedance_rate(_deployment_model_info(litellm_params), seedance_input_kind(body) or "plain", body.get("resolution"))
        return body, [], f"{_api_root(api_base)}{_TASKS_PATH}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        payload: Final = raw_response.json()
        job_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(job_id, str) or not job_id:
            raise BytePlusVideoError(
                status_code=raw_response.status_code,
                message="BytePlus video response did not include a task id",
                headers=raw_response.headers,
            )
        kind = seedance_input_kind(request_data)
        stored_id = attach_rate_class(job_id, kind) if kind is not None else job_id
        video = VideoObject(id=stored_id, object="video", status=_map_status(payload.get("status", "queued")), created_at=0)
        video.model = model
        if custom_llm_provider:
            video.id = encode_video_id_with_provider(stored_id, custom_llm_provider, model)
        return video

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        self._model_info = _deployment_model_info(litellm_params)
        task_id, _kind = split_rate_class(extract_original_video_id(video_id))
        encoded: Final = encode_url_path_segment(task_id, field_name="video_id")
        return f"{_api_root(api_base)}{_TASKS_PATH}/{encoded}", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        payload: Final = raw_response.json()
        if not isinstance(payload, dict):
            raise BytePlusVideoError(
                status_code=raw_response.status_code,
                message="BytePlus video status was not an object",
                headers=raw_response.headers,
            )
        job_id = payload.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise BytePlusVideoError(
                status_code=raw_response.status_code,
                message="BytePlus video status did not include a task id",
                headers=raw_response.headers,
            )
        _task_id, kind = split_rate_class(extract_original_video_id(_requested_video_id(logging_obj)))
        stored_id = attach_rate_class(job_id, kind) if kind is not None else job_id
        model = payload.get("model") if isinstance(payload.get("model"), str) else None
        video = VideoObject(id=stored_id, object="video", status=_map_status(payload.get("status")), created_at=0)
        if model is not None:
            video.model = model
        cost = seedance_completed_cost(payload, kind, self._model_info)
        if cost is not None:
            video.usage = {"provider_reported_cost_usd": cost}
        if custom_llm_provider:
            video.id = encode_video_id_with_provider(stored_id, custom_llm_provider, model)
        return video

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: str | None = None,
    ) -> tuple[str, dict]:
        return self.transform_video_status_retrieve_request(video_id, api_base, litellm_params, headers)

    def transform_video_content_response(self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj) -> bytes:
        video_url = _download_url(raw_response.json())
        if video_url is None:
            raise ValueError("BytePlus video file is not ready.")
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
            raise ValueError("BytePlus video file is not ready.")
        client: Final[AsyncHTTPHandler] = get_async_httpx_client(llm_provider=litellm.LlmProviders.BYTEPLUS)
        downloaded = await client.get(video_url)
        downloaded.raise_for_status()
        return downloaded.content

    def transform_video_remix_request(self, video_id, prompt, api_base, litellm_params, headers, extra_body=None):
        raise NotImplementedError("Video remix is not supported for BytePlus")

    def transform_video_remix_response(self, raw_response, logging_obj, custom_llm_provider=None):
        raise NotImplementedError("Video remix is not supported for BytePlus")

    def transform_video_list_request(self, api_base, litellm_params, headers, after=None, limit=None, order=None, extra_query=None):
        raise NotImplementedError("Video listing is not supported for BytePlus")

    def transform_video_list_response(self, raw_response, logging_obj, custom_llm_provider=None):
        raise NotImplementedError("Video listing is not supported for BytePlus")

    def transform_video_delete_request(self, video_id, api_base, litellm_params, headers):
        raise NotImplementedError("Video delete is not supported for BytePlus")

    def transform_video_delete_response(self, raw_response, logging_obj):
        raise NotImplementedError("Video delete is not supported for BytePlus")

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return BytePlusVideoError(status_code=status_code, message=error_message, headers=headers)


def _download_url(payload: object) -> str | None:
    if not isinstance(payload, dict) or _map_status(payload.get("status")) != "completed":
        return None
    content = payload.get("content")
    if not isinstance(content, dict):
        return None
    return _https_url(content.get("video_url"))
