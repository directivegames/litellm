"""MiniMax video priced from the deployment's own rates, with reference video billed when the job ends."""

from types import SimpleNamespace
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.minimax.videos.deployment_priced import DeploymentPricedMinimaxVideoConfig
from litellm.llms.minimax.videos.transformation import MinimaxVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import PROVIDER_TASK_ID_KEY, VIDEO_COST_POLL_ID_KEY

API_BASE: Final = "https://api.minimax.io"
H3_RATES: Final = {
    "output_cost_per_second_768p": 0.08,
    "output_cost_per_second_2k": 0.13,
    "input_cost_per_video_per_second_768p": 0.08,
    "input_cost_per_video_per_second_2k": 0.13,
    "input_cost_per_image": 0.04,
}
REFERENCE_VIDEO: Final = {
    "type": "video_url",
    "video_url": {"url": "https://cdn.example/ref.mp4"},
    "role": "reference_video",
}
REFERENCE_IMAGE: Final = {
    "type": "image_url",
    "image_url": {"url": "https://cdn.example/ref.png"},
    "role": "reference_image",
}


def _response(payload: dict) -> Mock:
    response: Final = Mock(spec=httpx.Response)
    response.json.return_value = payload
    return response


def _create_body(
    params: dict, rates: dict = H3_RATES, config: DeploymentPricedMinimaxVideoConfig | None = None
) -> dict:
    body, _files, _url = (config or DeploymentPricedMinimaxVideoConfig()).transform_video_create_request(
        model="MiniMax-H3",
        prompt="a red boat",
        api_base=API_BASE,
        video_create_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(**rates),
        headers={},
    )
    return body


def _created(params: dict, rates: dict = H3_RATES) -> tuple[SimpleNamespace, VideoObject]:
    config: Final = DeploymentPricedMinimaxVideoConfig()
    body: Final = _create_body(params, rates, config)
    logging_obj: Final = SimpleNamespace(model_call_details={"litellm_params": {"metadata": {}}})
    video: Final = config.transform_video_create_response(
        model="MiniMax-H3",
        raw_response=_response({"task_id": "mm-task"}),
        logging_obj=logging_obj,
        custom_llm_provider="minimax",
        request_data=body,
    )
    return logging_obj, video


def _status(task: dict, rates: dict = H3_RATES):
    config: Final = DeploymentPricedMinimaxVideoConfig()
    config.transform_video_status_retrieve_request("mm-task", API_BASE, GenericLiteLLMParams(**rates), {})
    return config.transform_video_status_retrieve_response(
        raw_response=_response({"task": task}), logging_obj=SimpleNamespace(litellm_params={})
    )


class TestReferenceVideo:
    def test_a_reference_video_reaches_minimax_beside_the_callers_other_media_in_order(self) -> None:
        body: Final = _create_body({"content": [REFERENCE_IMAGE, REFERENCE_VIDEO]})

        assert body["content"] == [{"type": "text", "text": "a red boat"}, REFERENCE_IMAGE, REFERENCE_VIDEO]

    def test_minimax_picks_the_ratio_from_the_reference_unless_the_caller_sends_one(self) -> None:
        assert "ratio" not in _create_body({"content": [REFERENCE_VIDEO]})
        assert _create_body({"content": [REFERENCE_VIDEO], "ratio": "9:16"})["ratio"] == "9:16"

    def test_a_reference_video_is_refused_when_the_deployment_has_no_rate_for_its_resolution(self) -> None:
        rates: Final = {key: value for key, value in H3_RATES.items() if key != "input_cost_per_video_per_second_2k"}

        with pytest.raises(UnsupportedParamsError, match="input_cost_per_video_per_second_2k") as raised:
            _create_body({"content": [REFERENCE_VIDEO], "resolution": "2K"}, rates)

        assert raised.value.status_code == 400
        assert _create_body({"content": [REFERENCE_VIDEO], "resolution": "768P"}, rates)["resolution"] == "768P"


class TestDeploymentRates:
    def test_a_resolution_the_deployment_does_not_price_is_refused(self) -> None:
        rates: Final = {"output_cost_per_second_768p": 0.08}

        with pytest.raises(UnsupportedParamsError, match="output_cost_per_second_2k"):
            _create_body({"resolution": "2K"}, rates)
        assert _create_body({"resolution": "768P"}, rates)["resolution"] == "768P"

    def test_a_plain_per_second_rate_prices_every_resolution(self) -> None:
        assert _create_body({"resolution": "2K"}, {"output_cost_per_second": 0.1})["resolution"] == "2K"

    def test_a_deployment_without_rates_is_left_to_upstreams_price_list(self) -> None:
        assert _create_body({"resolution": "2K"}, {})["resolution"] == "2K"


class TestCreateResponse:
    @pytest.mark.usefixtures("local_model_cost_map")
    def test_the_deployments_free_image_allowance_sets_the_billable_image_count(self) -> None:
        images: Final = [REFERENCE_IMAGE] * 3
        _logging, own_allowance = _created({"content": images}, {**H3_RATES, "free_input_image_count": 1})
        _logging, price_list = _created({"content": images}, H3_RATES)
        upstream: Final = MinimaxVideoConfig().transform_video_create_response(
            model="MiniMax-H3",
            raw_response=_response({"task_id": "mm-task"}),
            logging_obj=SimpleNamespace(),
            request_data=_create_body({"content": images}),
        )

        assert own_allowance.usage["billable_input_image_count"] == 2
        assert price_list.usage == upstream.usage

    def test_only_a_job_with_a_reference_video_waits_for_the_poller(self) -> None:
        _logging, plain = _created({})
        _logging, referenced = _created({"content": [REFERENCE_VIDEO]})

        assert VIDEO_COST_POLL_ID_KEY not in plain._hidden_params
        assert referenced._hidden_params[VIDEO_COST_POLL_ID_KEY] == "mm-task"

    def test_the_create_log_row_carries_minimaxs_task_id(self) -> None:
        logging_obj, _video = _created({})

        metadata: Final = logging_obj.model_call_details["litellm_params"]["metadata"]
        assert metadata["spend_logs_metadata"][PROVIDER_TASK_ID_KEY] == "mm-task"


class TestStatus:
    def test_a_finished_job_is_priced_at_its_reported_reference_seconds_and_resolution(self) -> None:
        video: Final = _status(
            {
                "id": "mm-task",
                "status": "succeeded",
                "resolution": "2K",
                "usage": {"input_seconds": 7, "output_seconds": 5},
            }
        )

        assert video.usage["provider_reported_cost_usd"] == pytest.approx(
            7 * H3_RATES["input_cost_per_video_per_second_2k"]
        )
        assert video._hidden_params[PROVIDER_TASK_ID_KEY] == "mm-task"

    def test_a_job_without_reference_seconds_has_no_price(self) -> None:
        video: Final = _status(
            {"id": "mm-task", "status": "succeeded", "resolution": "768P", "usage": {"output_seconds": 5}}
        )

        assert "provider_reported_cost_usd" not in video.usage

    def test_a_job_whose_resolution_the_deployment_does_not_price_has_no_price(self) -> None:
        video: Final = _status(
            {"id": "mm-task", "status": "succeeded", "resolution": "1080P", "usage": {"input_seconds": 7}}
        )

        assert "provider_reported_cost_usd" not in video.usage
