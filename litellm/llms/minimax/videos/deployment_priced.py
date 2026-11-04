"""MiniMax video priced from the deployment's own rates, with reference video billed when the job ends.

Callers: ProviderConfigManager.get_provider_video_config, video_budget_hold.py.
DeploymentPricedMinimaxVideoConfig, reference_video_items, reference_video_rate, REFERENCE_VIDEO_LONGEST_SECONDS.

Upstream's MinimaxVideoConfig bills output seconds and input images at create. This keeps that, and adds:

- A deployment that sets per-second output rates must set the one for the requested resolution, or the plain one,
  or create is refused before MiniMax is called. Upstream would bill a missing tier at another tier's rate, or at
  nothing.
- free_input_image_count on the deployment replaces the price list's free image allowance.
- Reference video. MiniMax bills its seconds, which it reports only on the finished task (usage.input_seconds), so
  the job is marked for the proxy's video cost poller (VIDEO_COST_POLL_ID_KEY). The poller bills those seconds alone,
  at input_cost_per_video_per_second_<resolution>, on a second spend row.
- MiniMax's task id on the create spend row, and on the status answer the poller logs.
"""

import math
from collections.abc import Mapping
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.base_llm.videos.deployment_model_info import deployment_pricing_model_info
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.minimax.videos.transformation import MinimaxVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import (
    PROVIDER_RESPONSE_KEY,
    PROVIDER_TASK_ID_KEY,
    VIDEO_COST_POLL_ID_KEY,
    extract_original_video_id,
    stamp_provider_task_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

# MiniMax takes at most 15 seconds of reference video per request, summed over every clip
# (https://platform.minimax.io/docs/api-reference/video-generation-v2-create.md, read 2026-11-04).
REFERENCE_VIDEO_LONGEST_SECONDS: Final = Decimal(15)
REFERENCE_VIDEO_RATE_KEY: Final = "input_cost_per_video_per_second_{}"
OUTPUT_RATE_KEY: Final = "output_cost_per_second_{}"
PLAIN_OUTPUT_RATE_KEY: Final = "output_cost_per_second"
FREE_INPUT_IMAGE_COUNT_KEY: Final = "free_input_image_count"
BILLABLE_IMAGES_KEY: Final = "billable_input_image_count"
REPORTED_COST_KEY: Final = "provider_reported_cost_usd"

_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_PAYLOAD_ADAPTER: Final = TypeAdapter(dict[str, object])
_ITEMS_ADAPTER: Final = TypeAdapter(tuple[object, ...])


class _ReportedTask(BaseModel, frozen=True):
    id: str
    resolution: str | None = None


class _StatusAnswer(BaseModel, frozen=True):
    task: _ReportedTask


def _items(content: object) -> tuple[object, ...]:
    if isinstance(content, (str, bytes)):
        return ()
    try:
        return _ITEMS_ADAPTER.validate_python(content)
    except ValidationError:
        return ()


def _is_reference_video(item: object) -> bool:
    match item:
        case {"type": "video_url"}:
            return True
        case _:
            return False


def reference_video_items(content: object) -> tuple[object, ...]:
    """The reference videos in a create's content list."""
    return tuple(item for item in _items(content) if _is_reference_video(item))


def _rate(pricing: Mapping[str, object], key: str) -> Decimal | None:
    value: Final = pricing.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return Decimal(str(value))


def reference_video_rate(pricing: Mapping[str, object], resolution: str) -> Decimal | None:
    """Dollars per reference-video second at this output resolution, or None when the deployment sets none."""
    return _rate(pricing, REFERENCE_VIDEO_RATE_KEY.format(resolution.strip().lower()))


def _free_input_images(pricing: Mapping[str, object]) -> int | None:
    value: Final = pricing.get(FREE_INPUT_IMAGE_COUNT_KEY)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _prices_output_seconds(pricing: Mapping[str, object]) -> bool:
    return any(key == PLAIN_OUTPUT_RATE_KEY or key.startswith(f"{PLAIN_OUTPUT_RATE_KEY}_") for key in pricing)


def _missing_rates(body: Mapping[str, object], pricing: Mapping[str, object], has_reference: bool) -> tuple[str, ...]:
    resolution: Final = str(body.get("resolution", "")).strip().lower()
    priced: Final = _prices_output_seconds(pricing)
    output_key: Final = OUTPUT_RATE_KEY.format(resolution)
    output_missing: Final = (
        priced and _rate(pricing, output_key) is None and _rate(pricing, PLAIN_OUTPUT_RATE_KEY) is None
    )
    reference_key: Final = REFERENCE_VIDEO_RATE_KEY.format(resolution)
    reference_missing: Final = has_reference and _rate(pricing, reference_key) is None
    return tuple(key for key, missing in ((output_key, output_missing), (reference_key, reference_missing)) if missing)


def _with_billable_images(
    usage: Mapping[str, object] | None, free_images: int
) -> dict[str, object]:  # mutable-ok: VideoObject.usage is a dict
    counts: Final = usage or _EMPTY
    sent: Final = counts.get("input_image_count")
    billable: Final = max(0, (sent if isinstance(sent, int) else 0) - free_images)
    kept: Final = {key: value for key, value in counts.items() if key != BILLABLE_IMAGES_KEY}
    return {**kept, **({BILLABLE_IMAGES_KEY: billable} if billable else {})}


def _reference_video_cost(
    usage: Mapping[str, object] | None, resolution: str | None, pricing: Mapping[str, object]
) -> tuple[Decimal, str, Decimal] | None:
    """Dollars for the reference-video seconds MiniMax reports, with the rate key and rate that produced them."""
    seconds: Final = (usage or _EMPTY).get("input_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0 or not resolution:
        return None
    key: Final = REFERENCE_VIDEO_RATE_KEY.format(resolution.strip().lower())
    rate: Final = _rate(pricing, key)
    if rate is None or rate == 0:
        return None
    return Decimal(seconds) * rate, key, rate


class DeploymentPricedMinimaxVideoConfig(MinimaxVideoConfig):
    def __init__(self) -> None:
        super().__init__()
        # The response hooks get no litellm_params, so the request hooks keep the deployment's rates. One instance
        # serves one call.
        self._pricing: Mapping[str, object] = _EMPTY

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
        references: Final = reference_video_items(params.get("content"))
        without_references: Final = [item for item in _items(params.get("content")) if not _is_reference_video(item)]
        upstream_params: Final = {**params, "content": without_references} if references else params
        body, files, url = super().transform_video_create_request(
            model, prompt, api_base, upstream_params, litellm_params, headers
        )
        # MiniMax takes the ratio from the reference video when none is sent; upstream's text-only default would
        # override it.
        request_body: Final = (
            {
                **{key: value for key, value in body.items() if key != "ratio" or "ratio" in params},
                "content": self._content_param(params, prompt),
            }
            if references
            else body
        )
        self._pricing = deployment_pricing_model_info(litellm_params)
        missing: Final = _missing_rates(request_body, self._pricing, bool(references))
        if missing:
            raise UnsupportedParamsError(
                message=(
                    f"MiniMax {model} cannot be billed for this request on this deployment, which sets no "
                    f"{', '.join(missing)}. Ask for a resolution the deployment prices, or have an admin add the rate."
                ),
                model=model,
                llm_provider="minimax",
            )
        return request_body, files, url

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: BaseVideoConfig contract
    ) -> VideoObject:
        video: Final = super().transform_video_create_response(
            model, raw_response, logging_obj, custom_llm_provider, request_data
        )
        task_id: Final = extract_original_video_id(video.id)
        stamp_provider_task_id(logging_obj, task_id)
        free_images: Final = _free_input_images(self._pricing)
        if free_images is not None:
            video.usage = _with_billable_images(video.usage, free_images)
        if reference_video_items((request_data or _EMPTY).get("content")):
            video._hidden_params[VIDEO_COST_POLL_ID_KEY] = task_id  # pyright: ignore[reportPrivateUsage]  # the proxy reads adapter facts here
        return video

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        self._pricing = deployment_pricing_model_info(litellm_params)
        return super().transform_video_status_retrieve_request(video_id, api_base, litellm_params, headers)

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        client: HTTPHandler | None = None,
    ) -> VideoObject:
        video: Final = super().transform_video_status_retrieve_response(
            raw_response, logging_obj, custom_llm_provider, client
        )
        payload: Final = _PAYLOAD_ADAPTER.validate_python(raw_response.json())
        task: Final = _StatusAnswer.model_validate(payload).task
        video._hidden_params[PROVIDER_RESPONSE_KEY] = payload  # pyright: ignore[reportPrivateUsage]  # the poller logs it on the finish row
        video._hidden_params[PROVIDER_TASK_ID_KEY] = task.id  # pyright: ignore[reportPrivateUsage]  # the poller logs it on the finish row
        priced: Final = _reference_video_cost(video.usage, task.resolution, self._pricing)
        if priced is not None:
            cost, rate_key, rate = priced
            video.usage = {**(video.usage or _EMPTY), REPORTED_COST_KEY: float(cost), rate_key: float(rate)}
        return video
