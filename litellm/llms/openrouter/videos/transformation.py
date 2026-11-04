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
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NoReturn

import httpx
from httpx._types import FileContent, RequestFiles
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    PROVIDER_RESPONSE_KEY,
    PROVIDER_TASK_ID_KEY,
    VIDEO_COST_POLL_ID_KEY,
    extract_original_video_id,
    stamp_provider_task_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

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
_BODY_KEYS: Final = ("resolution", "ratio", "aspect_ratio", "generate_audio", "size")
# OpenRouter names a first or last frame separately from a reference image.
_FRAME_ROLES: Final = frozenset({"first_frame", "last_frame"})
_COST_KEYS: Final = frozenset({"cost", "provider_reported_cost_usd"})
_FINISHED: Final = frozenset({"completed", "failed", "cancelled", "expired"})
# httpx decodes while building a response. A rebuilt body is plain JSON, so a copied
# Content-Encoding makes that decode fail with "incorrect header check".
_ENCODING_HEADERS: Final = frozenset({"content-encoding", "content-length", "transfer-encoding"})
# VideoObject requires these; OpenRouter may leave them out.
_VIDEO_DEFAULTS: Final = MappingProxyType({"object": "video", "created_at": 0, "status": "queued"})

_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_FIELDS_ADAPTER: Final = TypeAdapter(dict[str, object])
_ITEMS_ADAPTER: Final = TypeAdapter(tuple[object, ...])


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


def _length_seconds(value: object) -> float | None:
    if isinstance(value, str):
        try:
            return _length_seconds(float(value))
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def _openrouter_billed_usage(
    body: Mapping[str, object], cost: float
) -> dict[str, object]:  # mutable-ok: VideoObject.usage is a dict field
    """The provider price, plus any tokens or length the status body already carried."""
    reported: Final = {key: value for key, value in _fields(body.get("usage")).items() if key not in _COST_KEYS}
    length: Final = _length_seconds(body.get("duration", body.get("seconds")))
    duration: Final = (
        MappingProxyType({"duration_seconds": length})
        if "duration_seconds" not in reported and length is not None
        else _EMPTY
    )
    return {"provider_reported_cost_usd": cost, **reported, **duration}


def _image_url(item: Mapping[str, object]) -> str | None:
    url: Final = _fields(item.get("image_url")).get("url")
    return url if isinstance(url, str) and url else None


def _frame_role(item: Mapping[str, object]) -> str | None:
    named: Final = item.get("frame_type")
    role: Final = named if isinstance(named, str) else item.get("role")
    if isinstance(role, str) and role.lower() in _FRAME_ROLES:
        return role.lower()
    return None


def _openrouter_image(url: str, frame: str | None) -> dict[str, object]:  # mutable-ok: JSON request body fragment
    return {"type": "image_url", "image_url": {"url": url}, **({"frame_type": frame} if frame is not None else _EMPTY)}


def _matching_image(
    value: object, *, frames: bool
) -> dict[str, object] | None:  # mutable-ok: JSON request body fragment
    item: Final = _fields(value)
    url: Final = _image_url(item)
    role: Final = _frame_role(item)
    if url is None or (role is not None) != frames:
        return None
    return _openrouter_image(url, role if frames else None)


def _is_list(value: object) -> bool:
    return isinstance(value, list)


def _image_field(source: object, *, frames: bool) -> list[dict[str, object]]:  # mutable-ok: JSON request body field
    if not _is_list(source):
        return []
    images: Final = (_matching_image(item, frames=frames) for item in _items(source))
    return [image for image in images if image is not None]


def _openrouter_image_fields(params: Mapping[str, object]) -> Mapping[str, object]:
    """Images for an OpenRouter video create.

    A first_frame or last_frame image is a frame. Any other image is a reference.
    A video item is left out. frame_images or input_references already on the
    request are sent as given, and content is not read in that case.
    """
    explicit_frames: Final = params.get("frame_images")
    explicit_refs: Final = params.get("input_references")
    using_content: Final = not _is_list(explicit_frames) and not _is_list(explicit_refs)
    frames: Final = _image_field(params.get("content") if using_content else explicit_frames, frames=True)
    references: Final = _image_field(params.get("content") if using_content else explicit_refs, frames=False)
    fields: Final = (("frame_images", frames), ("input_references", references))
    return MappingProxyType({key: images for key, images in fields if images})


def openrouter_completed_cost(body: object) -> float | None:
    """Dollars OpenRouter reports for a video job, whatever its status."""
    fields: Final = _fields(body)
    if fields.get("status") not in _FINISHED:
        return None
    cost: Final = _fields(fields.get("usage")).get("cost")
    # bool is an int subclass; a true/false cost is not a price.
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    if not math.isfinite(cost) or cost <= 0:
        return None
    return float(cost)


def _normalized(raw_response: httpx.Response) -> httpx.Response:
    """Fill the fields VideoObject requires when OpenRouter omits them."""
    payload: Final = _json_object(raw_response)
    if payload is None or all(key in payload for key in _VIDEO_DEFAULTS):
        return raw_response
    headers: Final = {key: value for key, value in raw_response.headers.items() if key.lower() not in _ENCODING_HEADERS}
    return httpx.Response(status_code=raw_response.status_code, json={**_VIDEO_DEFAULTS, **payload}, headers=headers)


class OpenRouterVideoConfig(OpenAIVideoConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig contract returns list
        return ["model", "prompt", "input_reference", "seconds", "size", "user", *_FORWARDED]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseVideoConfig contract
        mapped: Final = _fields(
            super().map_openai_params(video_create_optional_params, model, drop_params)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # OpenAIVideoConfig returns an untyped dict
        )
        sent: Final = _fields(video_create_optional_params)
        return {**mapped, **{key: sent[key] for key in _FORWARDED if sent.get(key) is not None}}

    def use_multipart_form_data(self) -> bool:
        # OpenRouter accepts the same JSON body the pass-through used to forward.
        return False

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
            or get_secret_str("OPENROUTER_API_KEY")
            or litellm.api_key
        )
        if key is None:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY or pass api_key.")
        return {**headers, "Authorization": f"Bearer {key}"}

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],  # mutable-ok: BaseVideoConfig contract
    ) -> str:
        base: Final = (api_base or _DEFAULT_API_BASE).rstrip("/")
        return base if base.endswith("/videos") else f"{base}/videos"

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
        duration: Final = params.get("duration", params.get("seconds"))
        body: Final[dict[str, object]] = {  # mutable-ok: BaseVideoConfig contract returns the body as a dict
            "model": model,
            "prompt": prompt,
            **({"duration": duration} if duration is not None else _EMPTY),
            **{key: params[key] for key in _BODY_KEYS if params.get(key) is not None},
            **_openrouter_image_fields(params),
        }
        return body, [], api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: BaseVideoConfig contract
    ) -> VideoObject:
        video: Final = super().transform_video_create_response(  # pyright: ignore[reportUnknownMemberType]  # OpenAIVideoConfig leaves request_data untyped
            model=model,
            raw_response=_normalized(raw_response),
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            request_data=request_data,
        )
        # Create has no finished price. Leave usage empty so this call is not billed.
        video.usage = {}
        task_id: Final = extract_original_video_id(video.id)
        video._hidden_params[VIDEO_COST_POLL_ID_KEY] = task_id  # pyright: ignore[reportPrivateUsage]  # the proxy reads adapter facts here
        stamp_provider_task_id(logging_obj, task_id)
        return video

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        client: HTTPHandler | None = None,
    ) -> VideoObject:
        payload: Final = _json_object(raw_response) or _EMPTY
        video: Final = super().transform_video_status_retrieve_response(
            raw_response=_normalized(raw_response),
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            client=client,
        )
        cost: Final = openrouter_completed_cost(payload)
        if cost is not None:
            video.usage = _openrouter_billed_usage(payload, cost)
        task_id: Final = payload.get("id")
        # The poller logs the whole provider answer on the finish row; the caller never sees it.
        video._hidden_params[PROVIDER_RESPONSE_KEY] = dict(payload)  # pyright: ignore[reportPrivateUsage]  # the poller reads adapter facts here
        if isinstance(task_id, str):
            video._hidden_params[PROVIDER_TASK_ID_KEY] = task_id  # pyright: ignore[reportPrivateUsage]  # the poller reads adapter facts here
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
