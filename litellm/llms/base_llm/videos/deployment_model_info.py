"""Video pricing keys on a deployment for adapter rate checks.

Proxy model update strips custom rates from persisted model_info, so operators
author them on litellm_params. Router spreads those params into each call; this
helper merges nested model_info with the pricing fields on litellm_params.
"""

from collections.abc import Mapping
from typing import Final, cast

from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import CustomPricingLiteLLMParams

_PRICING_PREFIXES: Final = (
    "output_cost_per_second_",
    "input_cost_per_video_per_second_",
    "output_cost_per_video_token_",
)

_PRICING_EXACT: Final = frozenset(
    {
        "input_cost_per_image",
        "free_input_image_count",
    }
)


def deployment_pricing_model_info(litellm_params: GenericLiteLLMParams) -> Mapping[str, object]:
    """Rates for video create and cost polling, from model_info and litellm_params."""
    nested: Final = cast("Mapping[str, object] | None", litellm_params.model_info)
    merged: dict[str, object] = dict(nested) if nested is not None else {}
    for field in CustomPricingLiteLLMParams.model_fields:
        value: Final = litellm_params.get(field)
        if value is not None:
            merged[field] = value
    for key, value in litellm_params.model_dump(exclude_none=True).items():
        if key in ("model", "api_key", "model_info"):
            continue
        if key in _PRICING_EXACT or any(key.startswith(prefix) for prefix in _PRICING_PREFIXES):
            merged[key] = value
    return merged
