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

# Seedance 2.5 list prices per million tokens, as a deployment would set them:
# $10.70 and $6.40 at 480p and 720p, $11.70 and $7.00 at 1080p.
RATES: dict[str, object] = {
    "output_cost_per_video_token_without_video_input_480p": 0.0000107,
    "output_cost_per_video_token_with_video_input_480p": 0.0000064,
    "output_cost_per_video_token_without_video_input_720p": 0.0000107,
    "output_cost_per_video_token_with_video_input_720p": 0.0000064,
    "output_cost_per_video_token_without_video_input_1080p": 0.0000117,
    "output_cost_per_video_token_with_video_input_1080p": 0.000007,
}
_VIDEO_CONTENT: list[dict[str, object]] = [
    {"type": "text", "text": "a boat"},
    {"type": "video_url", "video_url": {"url": "https://cdn.example/in.mp4"}},
]


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
        "resolution": "720p",
        "usage": {"total_tokens": 1_000_000},
        "content": {"video_url": "https://cdn.example/clip.mp4"},
    }
    body.update(overrides)
    return body


def _create(config: BytePlusVideoConfig, model_info: dict | None, params: dict | None = None):
    return config.transform_video_create_request(
        model="dreamina-seedance-2-5-260628",
        prompt="a red boat",
        api_base="https://ark.ap-southeast.bytepluses.com",
        video_create_optional_request_params=params
        or {"duration": 4, "ratio": "16:9", "resolution": "480p", "generate_audio": False},
        litellm_params=GenericLiteLLMParams(model_info=model_info),
        headers={},
    )


def _status(config: BytePlusVideoConfig, model_info: dict | None, stored_id: str, payload: dict):
    encoded = encode_video_id_with_provider(stored_id, "byteplus", "dreamina-seedance-2-5-260628")
    url, _params = config.transform_video_status_retrieve_request(
        video_id=encoded,
        api_base="https://ark.ap-southeast.bytepluses.com",
        litellm_params=GenericLiteLLMParams(model_info=model_info),
        headers={},
    )
    logging_obj = Mock()
    logging_obj.model_call_details = {"additional_args": {"video_id": encoded}}
    video = config.transform_video_status_retrieve_response(
        raw_response=_response(payload),
        logging_obj=logging_obj,
        custom_llm_provider="byteplus",
    )
    return url, video


class TestBytePlusVideoTransformation:
    def setup_method(self) -> None:
        self.config = BytePlusVideoConfig()

    def test_create_request_uses_the_modelark_tasks_route(self) -> None:
        data, files, url = _create(self.config, RATES)

        assert url == "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
        assert files == []
        assert data["content"] == [{"type": "text", "text": "a red boat"}]
        assert data["resolution"] == "480p"
        assert data["generate_audio"] is False

    def test_create_fails_without_rates(self) -> None:
        with pytest.raises(ValueError, match="output_cost_per_video_token_without_video_input_480p"):
            _create(self.config, None)

    def test_create_with_video_input_needs_the_video_input_rate(self) -> None:
        rates = {"output_cost_per_video_token_without_video_input_720p": 0.0000107}
        with pytest.raises(ValueError, match="output_cost_per_video_token_with_video_input_720p"):
            _create(self.config, rates, {"content": _VIDEO_CONTENT, "resolution": "720p"})

    def test_create_without_video_input_needs_only_its_own_rate(self) -> None:
        data, _files, _url = _create(self.config, {"output_cost_per_video_token_without_video_input_480p": 0.0000107})
        assert data["content"] == [{"type": "text", "text": "a red boat"}]

    def test_create_fails_without_a_rate_for_the_requested_resolution(self) -> None:
        with pytest.raises(ValueError, match="output_cost_per_video_token_without_video_input_4k"):
            _create(self.config, RATES, {"resolution": "4K"})

    def test_create_fails_without_a_resolution(self) -> None:
        with pytest.raises(ValueError, match="resolution"):
            _create(self.config, RATES, {"duration": 4})

    def test_create_rejects_a_rate_that_is_not_a_number(self) -> None:
        with pytest.raises(ValueError, match="output_cost_per_video_token_without_video_input_480p"):
            _create(self.config, {**RATES, "output_cost_per_video_token_without_video_input_480p": True})

    def test_create_response_suffixes_the_input_class(self) -> None:
        video = self.config.transform_video_create_response(
            model="dreamina-seedance-2-5-260628",
            raw_response=_response({"id": "cgt-1"}),
            logging_obj=Mock(),
            custom_llm_provider="byteplus",
            request_data={"content": _VIDEO_CONTENT},
        )

        decoded = decode_video_id_with_provider(video.id)
        assert decoded.get("custom_llm_provider") == "byteplus"
        assert split_rate_class(decoded.get("video_id") or "") == ("cgt-1", "video")
        assert video.usage in (None, {})

    def test_status_strips_the_suffix_and_bills_the_video_rate(self) -> None:
        url, video = _status(self.config, RATES, "cgt-2026-abc~video", _job())

        assert url.endswith("/api/v3/contents/generations/tasks/cgt-2026-abc")
        assert "~" not in url
        assert video.status == "completed"
        assert video.usage == {"provider_reported_cost_usd": 6.4}
        decoded = decode_video_id_with_provider(video.id)
        assert decoded.get("video_id") == "cgt-2026-abc~video"

    def test_succeeded_status_without_rates_raises(self) -> None:
        with pytest.raises(ValueError, match="output_cost_per_video_token_with_video_input_720p"):
            _status(self.config, None, "cgt-2026-abc~video", _job())

    def test_a_running_status_needs_no_rates(self) -> None:
        _url, video = _status(self.config, None, "cgt-2026-abc~plain", _job(status="running"))
        assert video.usage is None

    def test_a_missing_input_class_is_not_billed(self) -> None:
        _url, video = _status(self.config, RATES, "cgt-2026-abc", _job())
        assert video.usage is None

    def test_remix_is_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            self.config.transform_video_remix_request("id", "prompt", "https://ark.example", GenericLiteLLMParams(), {})


class TestSeedanceCompletedCost:
    def test_uses_the_configured_token_rate_for_the_input_class(self) -> None:
        assert seedance_completed_cost(_job(), "plain", RATES) == 10.7
        assert seedance_completed_cost(_job(), "video", RATES) == 6.4

    def test_uses_the_rate_for_the_reported_resolution(self) -> None:
        assert seedance_completed_cost(_job(resolution="1080p"), "plain", RATES) == 11.7
        assert seedance_completed_cost(_job(resolution="1080p"), "video", RATES) == 7.0

    def test_raises_when_the_finished_job_reports_no_resolution(self) -> None:
        body = _job()
        del body["resolution"]
        with pytest.raises(ValueError, match="resolution"):
            seedance_completed_cost(body, "plain", RATES)

    def test_prices_any_model_the_deployment_sets_rates_for(self) -> None:
        assert seedance_completed_cost(_job(model="dreamina-seedance-2-0-260128"), "plain", RATES) == 10.7

    def test_ignores_a_job_that_is_still_running(self) -> None:
        assert seedance_completed_cost(_job(status="running"), "plain", RATES) is None
