"""OpenRouter video adapter. Recorded payloads, no network."""

from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.openrouter.videos.transformation import OpenRouterVideoConfig, openrouter_completed_cost
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import VIDEO_COST_POLL_ID_KEY, decode_video_id_with_provider


def _response(payload: dict) -> httpx.Response:
    return httpx.Response(status_code=200, json=payload)


class TestOpenRouterVideoTransformation:
    def setup_method(self) -> None:
        self.config = OpenRouterVideoConfig()

    def test_create_request_keeps_the_openrouter_catalog_model(self) -> None:
        data, files, url = self.config.transform_video_create_request(
            model="minimax/hailuo-3",
            prompt="a red boat",
            api_base="https://openrouter.ai/api/v1/videos",
            video_create_optional_request_params={"duration": 5, "aspect_ratio": "16:9", "generate_audio": False},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://openrouter.ai/api/v1/videos"
        assert files == []
        assert data == {
            "model": "minimax/hailuo-3",
            "prompt": "a red boat",
            "duration": 5,
            "aspect_ratio": "16:9",
            "generate_audio": False,
        }
        assert self.config.use_multipart_form_data() is False

    def test_create_response_encodes_the_id_and_does_not_bill(self) -> None:
        video = self.config.transform_video_create_response(
            model="minimax/hailuo-3",
            raw_response=_response({"id": "job-abc123", "status": "pending"}),
            logging_obj=Mock(),
            custom_llm_provider="openrouter",
        )

        decoded = decode_video_id_with_provider(video.id)
        assert decoded.get("custom_llm_provider") == "openrouter"
        assert decoded.get("video_id") == "job-abc123"
        assert video.usage == {}
        assert video._hidden_params[VIDEO_COST_POLL_ID_KEY] == "job-abc123"

    def test_completed_status_reports_usage_cost(self) -> None:
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response({"id": "job-abc123", "object": "video", "status": "completed", "usage": {"cost": 0.5}}),
            logging_obj=Mock(),
            custom_llm_provider="openrouter",
        )

        assert video.status == "completed"
        assert video.usage == {"provider_reported_cost_usd": 0.5}

    def test_content_url_uses_the_openrouter_videos_route(self) -> None:
        created = self.config.transform_video_create_response(
            model="bytedance/seedance-2.5",
            raw_response=_response({"id": "job-1", "status": "completed"}),
            logging_obj=Mock(),
            custom_llm_provider="openrouter",
        )
        url, params = self.config.transform_video_content_request(
            video_id=created.id,
            api_base=self.config.get_complete_url("bytedance/seedance-2.5", None, {}),
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://openrouter.ai/api/v1/videos/job-1/content"
        assert params == {}

    def test_jobs_the_cost_poller_never_sees_are_refused(self) -> None:
        common = {"video_id": "job-1", "api_base": "https://openrouter.ai/api/v1/videos", "headers": {}}
        params = GenericLiteLLMParams()

        with pytest.raises(NotImplementedError, match="remix"):
            self.config.transform_video_remix_request(prompt="again", litellm_params=params, **common)
        with pytest.raises(NotImplementedError, match="edit"):
            self.config.transform_video_edit_request(prompt="again", litellm_params=params, **common)
        with pytest.raises(NotImplementedError, match="extension"):
            self.config.transform_video_extension_request(prompt="again", seconds="4", litellm_params=params, **common)


class TestOpenRouterCompletedCost:
    def test_reads_a_positive_cost(self) -> None:
        assert openrouter_completed_cost({"id": "job-abc123", "status": "completed", "usage": {"cost": 0.5}}) == 0.5

    def test_ignores_a_job_that_is_still_running(self) -> None:
        assert openrouter_completed_cost({"status": "pending", "usage": {"cost": 0.5}}) is None

    def test_ignores_zero_negative_and_non_numeric_costs(self) -> None:
        body = {"status": "completed", "usage": {"cost": 0}}
        assert openrouter_completed_cost(body) is None
        body["usage"] = {"cost": -1}
        assert openrouter_completed_cost(body) is None
        body["usage"] = {"cost": True}
        assert openrouter_completed_cost(body) is None
        body["usage"] = {"cost": "0.5"}
        assert openrouter_completed_cost(body) is None
