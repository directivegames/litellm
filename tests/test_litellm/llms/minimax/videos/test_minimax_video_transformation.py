"""MiniMax video adapter. Recorded payloads, no network."""

from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.minimax.videos.transformation import MinimaxVideoConfig, minimax_completed_cost
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import VIDEO_COST_POLL_ID_KEY, decode_video_id_with_provider

# MiniMax-H3 pay-as-you-go list prices, as a deployment would set them.
RATES: dict[str, object] = {
    "output_cost_per_second_768p": 0.08,
    "output_cost_per_second_2k": 0.13,
    "input_cost_per_video_per_second_768p": 0.08,
    "input_cost_per_video_per_second_2k": 0.13,
    "input_cost_per_image": 0.04,
    "free_input_image_count": 5,
}


def _response(payload: dict) -> Mock:
    raw = Mock(spec=httpx.Response)
    raw.json.return_value = payload
    raw.status_code = 200
    raw.headers = {}
    return raw


def _task(**overrides: object) -> dict:
    task: dict[str, object] = {
        "id": "424010985738629",
        "model": "MiniMax-H3",
        "status": "succeeded",
        "resolution": "768P",
        "task_type": "generation",
        "usage": {"output_seconds": 5, "input_seconds": 0, "input_image_count": 0},
        "content": {"url": "https://cdn.example/clip.mp4"},
    }
    task.update(overrides)
    return {"task": task}


def _create(config: MinimaxVideoConfig, model_info: dict | None, params: dict | None = None):
    return config.transform_video_create_request(
        model="MiniMax-H3",
        prompt="a red boat",
        api_base="https://api.minimax.io",
        video_create_optional_request_params=params or {"duration": 5, "resolution": "768P", "ratio": "16:9"},
        litellm_params=GenericLiteLLMParams(model_info=model_info),
        headers={},
    )


def _status(config: MinimaxVideoConfig, model_info: dict | None, payload: dict):
    config.transform_video_status_retrieve_request(
        video_id="424010985738629",
        api_base="https://api.minimax.io",
        litellm_params=GenericLiteLLMParams(model_info=model_info),
        headers={},
    )
    return config.transform_video_status_retrieve_response(
        raw_response=_response(payload),
        logging_obj=Mock(),
        custom_llm_provider="minimax",
    )


class TestMinimaxVideoTransformation:
    def setup_method(self) -> None:
        self.config = MinimaxVideoConfig()

    def test_create_request_uses_the_v2_video_route(self) -> None:
        data, files, url = _create(self.config, RATES)

        assert url == "https://api.minimax.io/v2/video_generation"
        assert files == []
        assert data["model"] == "MiniMax-H3"
        assert data["content"] == [{"type": "text", "text": "a red boat"}]
        assert data["duration"] == 5
        assert data["resolution"] == "768P"
        assert data["ratio"] == "16:9"

    def test_create_fails_without_rates(self) -> None:
        with pytest.raises(ValueError, match="free_input_image_count"):
            _create(self.config, None)

    def test_create_fails_without_a_rate_for_the_requested_resolution(self) -> None:
        with pytest.raises(ValueError, match="output_cost_per_second_480p"):
            _create(self.config, RATES, {"duration": 5, "resolution": "480P"})

    def test_create_fails_without_an_input_video_rate(self) -> None:
        rates = {k: v for k, v in RATES.items() if k != "input_cost_per_video_per_second_2k"}
        with pytest.raises(ValueError, match="input_cost_per_video_per_second_2k"):
            _create(self.config, rates, {"duration": 5, "resolution": "2K"})

    def test_create_fails_without_a_resolution(self) -> None:
        with pytest.raises(ValueError, match="resolution"):
            _create(self.config, RATES, {"duration": 5})

    def test_create_rejects_a_rate_that_is_not_a_number(self) -> None:
        with pytest.raises(ValueError, match="input_cost_per_image"):
            _create(self.config, {**RATES, "input_cost_per_image": "0.04"})

    def test_create_response_encodes_the_task_id_and_does_not_bill(self) -> None:
        video = self.config.transform_video_create_response(
            model="MiniMax-H3",
            raw_response=_response({"task_id": "4240", "status": "queued"}),
            logging_obj=Mock(),
            custom_llm_provider="minimax",
        )

        decoded = decode_video_id_with_provider(video.id)
        assert decoded.get("custom_llm_provider") == "minimax"
        assert decoded.get("video_id") == "4240"
        assert video.status == "queued"
        assert video.usage in (None, {})

    def _create_and_respond(self, params: dict):
        data, _files, _url = _create(self.config, RATES, params)
        return self.config.transform_video_create_response(
            model="MiniMax-H3",
            raw_response=_response({"task_id": "4240", "status": "queued"}),
            logging_obj=Mock(),
            custom_llm_provider="minimax",
            request_data=data,
        )

    @pytest.mark.parametrize(
        "content",
        [
            None,
            [{"type": "image_url", "image_url": {"url": f"https://cdn.example/{n}.png"}} for n in range(7)],
            [{"type": "video_url", "video_url": {"url": "https://cdn.example/in.mp4"}}],
        ],
        ids=["text", "images", "input_video"],
    )
    def test_every_job_bills_when_it_finishes(self, content: list | None) -> None:
        params: dict[str, object] = {"duration": 5, "resolution": "768P"}
        video = self._create_and_respond(params if content is None else {**params, "content": content})

        assert video.usage in (None, {})
        assert video._hidden_params[VIDEO_COST_POLL_ID_KEY] == "4240"

    def test_succeeded_status_reports_the_configured_cost(self) -> None:
        video = _status(self.config, RATES, _task())

        assert video.status == "completed"
        assert video.seconds == "5"
        assert video.usage == {
            "provider_reported_cost_usd": 0.4,
            "duration_seconds": 5,
            "input_seconds": 0,
            "input_image_count": 0,
            "video_resolution": "768p",
            "output_cost_per_second": 0.08,
            "input_cost_per_video_per_second": 0.08,
            "input_cost_per_image": 0.04,
            "free_input_image_count": 5,
        }

    def test_succeeded_status_without_rates_raises(self) -> None:
        with pytest.raises(ValueError, match="free_input_image_count"):
            _status(self.config, None, _task())

    def test_a_running_status_has_no_cost_and_needs_no_rates(self) -> None:
        video = _status(self.config, None, _task(status="running"))

        assert video.status == "in_progress"
        assert video.usage is None

    def test_content_request_polls_the_query_route(self) -> None:
        created = self.config.transform_video_create_response(
            model="MiniMax-H3",
            raw_response=_response({"task_id": "4240"}),
            logging_obj=Mock(),
            custom_llm_provider="minimax",
        )
        url, params = self.config.transform_video_content_request(
            video_id=created.id,
            api_base="https://api.minimax.io",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://api.minimax.io/v2/query/video_generation/4240"
        assert params == {}

    def test_remix_is_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            self.config.transform_video_remix_request(
                "id", "prompt", "https://api.minimax.io", GenericLiteLLMParams(), {}
            )


class TestMinimaxCompletedCost:
    def test_bills_output_seconds_at_the_768p_rate(self) -> None:
        assert minimax_completed_cost(_task(), RATES) == 0.4

    def test_bills_2k_input_video_and_an_image_past_the_free_count(self) -> None:
        body = _task(resolution="2K", usage={"output_seconds": 5, "input_seconds": 2, "input_image_count": 6})
        assert minimax_completed_cost(body, RATES) == 0.95

    def test_bills_input_video_at_its_own_rate(self) -> None:
        # MiniMax-H3-Max prices input video above its output rate.
        rates = {
            "output_cost_per_second_768p": 0.08,
            "input_cost_per_video_per_second_768p": 0.143,
            "input_cost_per_image": 0.074,
            "free_input_image_count": 2,
        }
        body = _task(model="MiniMax-H3-Max", usage={"output_seconds": 5, "input_seconds": 2, "input_image_count": 3})
        assert minimax_completed_cost(body, rates) == pytest.approx(0.4 + 0.286 + 0.074)

    def test_does_not_price_other_task_types(self) -> None:
        assert minimax_completed_cost(_task(task_type="regeneration", resolution="2K"), RATES) is None
        assert minimax_completed_cost(_task(task_type="h3_context_ir"), RATES) is None

    def test_ignores_a_job_that_is_still_running(self) -> None:
        assert minimax_completed_cost(_task(status="running"), RATES) is None

    def test_raises_for_a_resolution_without_a_configured_rate(self) -> None:
        with pytest.raises(ValueError, match="output_cost_per_second_1080p"):
            minimax_completed_cost(_task(resolution="1080P"), RATES)

    def test_a_zero_rate_is_free_not_missing(self) -> None:
        assert minimax_completed_cost(_task(), {**RATES, "output_cost_per_second_768p": 0}) is None

    def test_bills_whole_counts_sent_as_floats(self) -> None:
        body = _task(resolution="2K", usage={"output_seconds": 5.0, "input_seconds": 2.0, "input_image_count": 6.0})
        assert minimax_completed_cost(body, RATES) == 0.95

    @pytest.mark.parametrize(
        "count",
        [5.5, -1, True, float("nan"), float("inf"), "5", None],
        ids=["fraction", "negative", "bool", "nan", "inf", "string", "missing"],
    )
    def test_does_not_bill_a_count_that_is_not_a_whole_number(self, count: object) -> None:
        body = _task(usage={"output_seconds": count, "input_seconds": 0, "input_image_count": 0})
        assert minimax_completed_cost(body, RATES) is None
