"""OpenRouter video jobs on the OpenAI /videos routes.

Callers: ProviderConfigManager.get_provider_video_config.
OpenRouterVideoConfig, openrouter_completed_cost.

The upstream model stays the OpenRouter catalog name, such as minimax/hailuo-3.
Cost is the completed job's usage.cost. Create does not bill.
"""

import math
from typing import Any, Final

import httpx

import litellm
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject

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


def openrouter_completed_cost(body: object) -> float | None:
    """Dollars from a finished OpenRouter video job, or nothing if it is not billable."""
    if not isinstance(body, dict) or body.get("status") != "completed":
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
        duration = video_create_optional_request_params.get("duration", video_create_optional_request_params.get("seconds"))
        if duration is not None:
            body["duration"] = duration
        for key in ("resolution", "ratio", "aspect_ratio", "generate_audio", "size"):
            value = video_create_optional_request_params.get(key)
            if value is not None:
                body[key] = value
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
            video.usage = {"provider_reported_cost_usd": cost}
        return video


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
    return httpx.Response(status_code=raw_response.status_code, json=filled, headers=raw_response.headers)
