"""OpenRouter video jobs on the OpenAI /videos routes.

Callers: ProviderConfigManager.get_provider_video_config.
OpenRouterVideoConfig, openrouter_completed_cost.

The upstream model stays the OpenRouter catalog name, such as minimax/hailuo-3.
Create sends image inputs as frame_images and input_references. A video item is not sent.
Cost is the completed job's usage.cost. The finished usage keeps the provider's other
usage fields and a top-level duration. Create does not bill; it hands the job id to
the proxy's video cost poller (VIDEO_COST_POLL_ID_KEY), which bills the finished job.
Remix, edit, and extension are refused: only create records a job for the poller, so
a job started any other way would never be billed.
"""

import math
from collections.abc import Iterator, Mapping
from typing import Any, Final, NoReturn

import httpx
from httpx._types import FileContent

import litellm
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import VIDEO_COST_POLL_ID_KEY, extract_original_video_id

_DEFAULT_API_BASE: Final = "https://openrouter.ai/api/v1"
_FORWARDED: Final = (
    "duration",
    "seconds",
    "resolution",
    "ratio",
    "aspect_ratio",
    "generate_audio",
    "size",
)
# OpenRouter names a first or last frame separately from a reference image.
_FRAME_ROLES: Final = frozenset({"first_frame", "last_frame"})


def _length_seconds(value: object) -> float | None:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def _openrouter_billed_usage(body: object, cost: float) -> dict[str, object]:
    """The provider price, plus any tokens or length the status body already carried."""
    usage: dict[str, object] = {"provider_reported_cost_usd": cost}
    if not isinstance(body, dict):
        return usage
    raw = body.get("usage")
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key not in {"cost", "provider_reported_cost_usd"}:
                usage[key] = value
    if "duration_seconds" not in usage:
        length = _length_seconds(body.get("duration", body.get("seconds")))
        if length is not None:
            usage["duration_seconds"] = length
    return usage


def _image_url(item: Mapping[str, object]) -> str | None:
    image = item.get("image_url")
    if not isinstance(image, dict):
        return None
    url = image.get("url")
    if isinstance(url, str) and url:
        return url
    return None


def _frame_role(item: Mapping[str, object]) -> str | None:
    named = item.get("frame_type")
    role = named if isinstance(named, str) else item.get("role")
    if isinstance(role, str) and role.lower() in _FRAME_ROLES:
        return role.lower()
    return None


def _openrouter_image(url: str, frame: str | None) -> dict[str, object]:
    if frame is None:
        return {"type": "image_url", "image_url": {"url": url}}
    return {"type": "image_url", "image_url": {"url": url}, "frame_type": frame}


def _matching_image(item: object, *, frames: bool) -> dict[str, object] | None:
    if not isinstance(item, dict):
        return None
    url = _image_url(item)
    role = _frame_role(item)
    if url is None or (role is not None) != frames:
        return None
    return _openrouter_image(url, role if frames else None)


def _matching_images(items: object, *, frames: bool) -> Iterator[dict[str, object]]:
    if not isinstance(items, list):
        return
    for item in items:
        image = _matching_image(item, frames=frames)
        if image is not None:
            yield image


def _image_field(items: object, *, frames: bool) -> list[dict[str, object]]:
    return list(_matching_images(items, frames=frames))


def _openrouter_image_fields(params: Mapping[str, object]) -> dict[str, list[dict[str, object]]]:
    """Images for an OpenRouter video create.

    A first_frame or last_frame image is a frame. Any other image is a reference.
    A video item is left out. frame_images or input_references already on the
    request are sent as given, and content is not read in that case.
    """
    explicit_frames = params.get("frame_images")
    explicit_refs = params.get("input_references")
    using_content = not isinstance(explicit_frames, list) and not isinstance(explicit_refs, list)
    frame_source = params.get("content") if using_content else explicit_frames
    reference_source = params.get("content") if using_content else explicit_refs
    fields: dict[str, list[dict[str, object]]] = {}
    frames = _image_field(frame_source, frames=True)
    references = _image_field(reference_source, frames=False)
    if frames:
        fields["frame_images"] = frames
    if references:
        fields["input_references"] = references
    return fields


_FINISHED: Final = frozenset({"completed", "failed", "cancelled", "expired"})


def openrouter_completed_cost(body: object) -> float | None:
    """Dollars OpenRouter reports for a video job, whatever its status."""
    if not isinstance(body, dict) or body.get("status") not in _FINISHED:
        return None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    # bool is an int subclass; a true/false cost is not a price.
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    if not math.isfinite(cost) or cost <= 0:
        return None
    return float(cost)


class OpenRouterVideoConfig(OpenAIVideoConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return ["model", "prompt", "input_reference", "seconds", "size", "user", *_FORWARDED]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped = super().map_openai_params(video_create_optional_params, model, drop_params)
        for key in _FORWARDED:
            value = video_create_optional_params.get(key)
            if value is not None:
                mapped[key] = value
        return mapped

    def use_multipart_form_data(self) -> bool:
        # OpenRouter accepts the same JSON body the pass-through used to forward.
        return False

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key
        api_key = api_key or get_secret_str("OPENROUTER_API_KEY") or litellm.api_key
        if api_key is None:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY or pass api_key.")
        headers.update({"Authorization": f"Bearer {api_key}"})
        return headers

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: dict) -> str:
        base = (api_base or _DEFAULT_API_BASE).rstrip("/")
        if base.endswith("/videos"):
            return base
        return f"{base}/videos"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, list, str]:
        body: dict[str, object] = {"model": model, "prompt": prompt}
        duration = video_create_optional_request_params.get(
            "duration", video_create_optional_request_params.get("seconds")
        )
        if duration is not None:
            body["duration"] = duration
        for key in ("resolution", "ratio", "aspect_ratio", "generate_audio", "size"):
            value = video_create_optional_request_params.get(key)
            if value is not None:
                body[key] = value
        body.update(_openrouter_image_fields(video_create_optional_request_params))
        return body, [], api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        video = super().transform_video_create_response(
            model=model,
            raw_response=_normalized(raw_response),
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            request_data=request_data,
        )
        # Create has no finished price. Leave usage empty so this call is not billed.
        video.usage = {}
        video._hidden_params[VIDEO_COST_POLL_ID_KEY] = extract_original_video_id(video.id)  # pyright: ignore[reportPrivateUsage]  # the proxy reads adapter facts here
        return video

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Any,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        payload = raw_response.json()
        video = super().transform_video_status_retrieve_response(
            raw_response=_normalized(raw_response),
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
        )
        cost = openrouter_completed_cost(payload)
        if cost is not None:
            video.usage = _openrouter_billed_usage(payload, cost)
        return video

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
    ) -> NoReturn:
        raise NotImplementedError("Video remix is not supported for OpenRouter")

    def transform_video_edit_request(
        self,
        prompt: str,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, object],
        video_file: FileContent | None = None,
        extra_body: Mapping[str, object] | None = None,
        prefetched_source_data: Mapping[str, object] | None = None,
    ) -> NoReturn:
        raise NotImplementedError("Video edit is not supported for OpenRouter")

    def transform_video_extension_request(
        self,
        prompt: str,
        video_id: str,
        seconds: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
    ) -> NoReturn:
        raise NotImplementedError("Video extension is not supported for OpenRouter")


def _normalized(raw_response: httpx.Response) -> httpx.Response:
    """Fill the fields VideoObject requires when OpenRouter omits them."""
    payload = raw_response.json()
    if not isinstance(payload, dict):
        return raw_response
    filled = dict(payload)
    filled.setdefault("object", "video")
    filled.setdefault("created_at", 0)
    if "status" not in filled:
        filled["status"] = "queued"
    if filled is payload or filled == payload:
        return raw_response
    # httpx decodes while building the response. The new body is plain JSON, so a
    # copied Content-Encoding makes that decode fail with "incorrect header check".
    headers = {
        key: value
        for key, value in raw_response.headers.items()
        if key.lower() not in {"content-encoding", "content-length", "transfer-encoding"}
    }
    return httpx.Response(status_code=raw_response.status_code, json=filled, headers=headers)
