"""Video pricing keys on a deployment for adapter rate checks.

Proxy model update strips custom rates from persisted model_info, so operators
author them on litellm_params. Router spreads those params into each call; this
helper merges nested model_info with the pricing fields on litellm_params.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

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
_NOT_PRICING: Final = frozenset({"model", "api_key", "model_info"})
_CUSTOM_PRICING_FIELDS: Final = frozenset(CustomPricingLiteLLMParams.model_fields)

_FIELDS_ADAPTER: Final = TypeAdapter(dict[str, object])


def _fields(value: object) -> Mapping[str, object]:
    try:
        return MappingProxyType(_FIELDS_ADAPTER.validate_python(value))
    except ValidationError:
        return MappingProxyType({})


def _is_pricing_key(key: str) -> bool:
    if key in _CUSTOM_PRICING_FIELDS:
        return True
    return key not in _NOT_PRICING and (key in _PRICING_EXACT or key.startswith(_PRICING_PREFIXES))


def deployment_pricing_model_info(litellm_params: GenericLiteLLMParams) -> Mapping[str, object]:
    """Rates for video create and cost polling, from model_info and litellm_params."""
    set_params: Final = _fields(litellm_params.model_dump(exclude_none=True))
    return MappingProxyType(
        {
            **_fields(getattr(litellm_params, "model_info", None)),
            **{key: value for key, value in set_params.items() if _is_pricing_key(key)},
        }
    )
