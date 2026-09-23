"""BytePlus Seedance video adapter. Recorded payloads, no network."""

from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.byteplus.videos.transformation import (
    BytePlusVideoConfig,
    seedance_completed_cost,
    split_rate_class,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider


def _response(payload: dict) -> Mock:
    raw = Mock(spec=httpx.Response)
    raw.json.return_value = payload
    raw.status_code = 200
    raw.headers = {}
    return raw


def _job(**overrides: object) -> dict:
    body: dict[str, object] = {
        "id": "cgt-2026-abc",
        "model": "dreamina-seedance-2-5-260628",
        "status": "succeeded",
        "usage": {"total_tokens": 1_000_000},
        "content": {"video_url": "https://cdn.example/clip.mp4"},
    }
    body.update(overrides)
    return body


class TestBytePlusVideoTransformation:
    def setup_method(self) -> None:
        self.config = BytePlusVideoConfig()

    def test_create_request_uses_the_modelark_tasks_route(self) -> None:
        data, files, url = self.config.transform_video_create_request(
            model="dreamina-seedance-2-5-260628",
            prompt="a red boat",
            api_base="https://ark.ap-southeast.bytepluses.com",
            video_create_optional_request_params={
                "duration": 4,
                "ratio": "16:9",
                "resolution": "480p",
                "generate_audio": False,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
        assert files == []
        assert data["content"] == [{"type": "text", "text": "a red boat"}]
        assert data["resolution"] == "480p"
        assert data["generate_audio"] is False

    def test_create_response_suffixes_the_input_class(self) -> None:
        video = self.config.transform_video_create_response(
            model="dreamina-seedance-2-5-260628",
            raw_response=_response({"id": "cgt-1"}),
            logging_obj=Mock(),
            custom_llm_provider="byteplus",
            request_data={"content": [{"type": "text", "text": "a boat"}, {"type": "video_url", "video_url": {"url": "https://cdn.example/in.mp4"}}]},
        )

        decoded = decode_video_id_with_provider(video.id)
        assert decoded.get("custom_llm_provider") == "byteplus"
        assert split_rate_class(decoded.get("video_id") or "") == ("cgt-1", "video")
        assert video.usage in (None, {})

    def test_status_strips_the_suffix_and_bills_the_video_rate(self) -> None:
        encoded = encode_video_id_with_provider("cgt-2026-abc~video", "byteplus", "dreamina-seedance-2-5-260628")
        url, _params = self.config.transform_video_status_retrieve_request(
            video_id=encoded,
            api_base="https://ark.ap-southeast.bytepluses.com",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        logging_obj = Mock()
        logging_obj.model_call_details = {"additional_args": {"video_id": encoded}}

        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response(_job()),
            logging_obj=logging_obj,
            custom_llm_provider="byteplus",
        )

        assert url.endswith("/api/v3/contents/generations/tasks/cgt-2026-abc")
        assert "~" not in url
        assert video.status == "completed"
        assert video.usage == {"provider_reported_cost_usd": 6.4}
        decoded = decode_video_id_with_provider(video.id)
        assert decoded.get("video_id") == "cgt-2026-abc~video"

    def test_a_missing_input_class_is_not_billed(self) -> None:
        logging_obj = Mock()
        logging_obj.model_call_details = {"additional_args": {"video_id": "cgt-2026-abc"}}

        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response(_job()),
            logging_obj=logging_obj,
            custom_llm_provider="byteplus",
        )

        assert video.usage is None

    def test_remix_is_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            self.config.transform_video_remix_request("id", "prompt", "https://ark.example", GenericLiteLLMParams(), {})


class TestSeedanceCompletedCost:
    def test_uses_the_published_token_rate_for_the_input_class(self) -> None:
        assert seedance_completed_cost(_job(), "plain") == 10.7
        assert seedance_completed_cost(_job(), "video") == 6.4

    def test_ignores_a_job_that_is_still_running(self) -> None:
        assert seedance_completed_cost(_job(status="running"), "plain") is None
