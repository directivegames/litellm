"""MiniMax video adapter. Recorded payloads, no network."""

from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.minimax.videos.transformation import MinimaxVideoConfig, minimax_completed_cost
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import decode_video_id_with_provider


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


class TestMinimaxVideoTransformation:
    def setup_method(self) -> None:
        self.config = MinimaxVideoConfig()

    def test_create_request_uses_the_v2_video_route(self) -> None:
        data, files, url = self.config.transform_video_create_request(
            model="MiniMax-H3",
            prompt="a red boat",
            api_base="https://api.minimax.io",
            video_create_optional_request_params={"duration": 5, "resolution": "768P", "ratio": "16:9"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://api.minimax.io/v2/video_generation"
        assert files == []
        assert data["model"] == "MiniMax-H3"
        assert data["content"] == [{"type": "text", "text": "a red boat"}]
        assert data["duration"] == 5
        assert data["resolution"] == "768P"
        assert data["ratio"] == "16:9"

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

    def test_succeeded_status_reports_the_published_cost(self) -> None:
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response(_task()),
            logging_obj=Mock(),
            custom_llm_provider="minimax",
        )

        assert video.status == "completed"
        assert video.usage == {"provider_reported_cost_usd": 0.4}

    def test_a_running_status_has_no_cost(self) -> None:
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response(_task(status="running")),
            logging_obj=Mock(),
            custom_llm_provider="minimax",
        )

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
            self.config.transform_video_remix_request("id", "prompt", "https://api.minimax.io", GenericLiteLLMParams(), {})


class TestMinimaxCompletedCost:
    def test_bills_output_seconds_at_the_768p_rate(self) -> None:
        assert minimax_completed_cost(_task()) == 0.4

    def test_bills_2k_and_an_image_past_the_free_five(self) -> None:
        body = _task(resolution="2K", usage={"output_seconds": 5, "input_seconds": 2, "input_image_count": 6})
        assert minimax_completed_cost(body) == 0.95

    def test_bills_regeneration_separately_from_generation(self) -> None:
        body = _task(
            task_type="regeneration",
            resolution="2K",
            usage={"output_seconds": 5, "input_seconds": 0, "input_image_count": 6},
        )
        assert minimax_completed_cost(body) == 0.275

    def test_ignores_a_job_that_is_still_running(self) -> None:
        assert minimax_completed_cost(_task(status="running")) is None

    def test_ignores_a_resolution_without_a_published_rate(self) -> None:
        assert minimax_completed_cost(_task(resolution="1080P")) is None
