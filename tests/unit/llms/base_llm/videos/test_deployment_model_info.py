"""deployment_pricing_model_info merges litellm_params rates for video adapters."""

from litellm.llms.base_llm.videos.deployment_model_info import deployment_pricing_model_info
from litellm.types.router import GenericLiteLLMParams


def test_merges_top_level_litellm_params_over_partial_nested_model_info() -> None:
    litellm_params = GenericLiteLLMParams(
        model="minimax/MiniMax-H3",
        output_cost_per_second_768p=0.08,
        input_cost_per_video_per_second_768p=0.08,
        input_cost_per_image=0.04,
        free_input_image_count=5,
        model_info={"free_input_image_count": 5},
    )
    merged = deployment_pricing_model_info(litellm_params)
    assert merged["output_cost_per_second_768p"] == 0.08
    assert merged["free_input_image_count"] == 5
