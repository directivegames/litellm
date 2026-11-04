from types import MappingProxyType
from typing import Any, Final

import orjson
from pydantic import BaseModel, TypeAdapter

from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_character_id_with_provider,
    encode_video_id_with_provider,
)

_RESPONSE_FIELDS: Final = TypeAdapter(dict[str, object])


def extract_model_from_target_model_names(target_model_names: Any) -> str | None:
    if isinstance(target_model_names, str):
        model_names: Final = [m.strip() for m in target_model_names.split(",") if m.strip()]
        return model_names[0] if model_names else None
    if not isinstance(target_model_names, list):
        return None
    return target_model_names[0] if target_model_names else None


def video_reference_to_id(video_ref: object) -> str:
    if isinstance(video_ref, dict):
        return video_ref.get("id", "")
    if not isinstance(video_ref, str):
        return ""
    try:
        parsed_ref: Final = orjson.loads(video_ref)
    except orjson.JSONDecodeError:
        return video_ref
    return parsed_ref.get("id", "") if isinstance(parsed_ref, dict) else video_ref


def get_custom_provider_from_data(data: dict[str, Any]) -> str | None:
    custom_llm_provider: Final = data.get("custom_llm_provider")
    if custom_llm_provider:
        return custom_llm_provider

    extra_body = data.get("extra_body")
    if isinstance(extra_body, str):
        try:
            parsed_extra_body: Final = orjson.loads(extra_body)
            if isinstance(parsed_extra_body, dict):
                extra_body = parsed_extra_body
        except Exception:
            extra_body = None

    if isinstance(extra_body, dict):
        extra_body_custom_llm_provider: Final = extra_body.get("custom_llm_provider")
        if isinstance(extra_body_custom_llm_provider, str):
            return extra_body_custom_llm_provider

    return None


def _hidden_model_id(response: object) -> str | None:
    match getattr(response, "_hidden_params", None):
        case {"model_id": str(model_id)} if model_id:
            return model_id
        case _:
            return None


def _response_id(response: object) -> str | None:
    match response:
        case {"id": str(video_id)} if video_id:
            return video_id
        case _:
            attribute: Final[object] = getattr(response, "id", None)
            return attribute if isinstance(attribute, str) and attribute else None


def encode_video_id_in_response(response: object, fallback_model_id: str | None = None) -> object:
    """
    A copy of the response whose video id records the router deployment id.

    Provider adapters encode the provider's model name, which can match several deployments
    (a direct row and an OpenRouter row can share minimax/hailuo-3). A deployment id resolves to
    exactly one. Ids the adapter left unencoded are returned as the provider sent them.
    """
    model_id: Final = _hidden_model_id(response) or fallback_model_id
    video_id: Final = _response_id(response)
    if not model_id or video_id is None:
        return response

    decoded: Final = decode_video_id_with_provider(video_id)
    provider: Final = decoded.get("custom_llm_provider")
    if not provider:
        return response

    encoded: Final = encode_video_id_with_provider(decoded.get("video_id") or video_id, provider, model_id)
    if isinstance(response, BaseModel):
        # model_copy keeps _hidden_params, which the job record reads for the poll id.
        return response.model_copy(update=MappingProxyType({"id": encoded}))
    return {**_RESPONSE_FIELDS.validate_python(response), "id": encoded}


def encode_character_id_in_response(response: Any, custom_llm_provider: str, model_id: str | None) -> object:
    if isinstance(response, dict) and response.get("id"):
        response["id"] = encode_character_id_with_provider(
            character_id=response["id"],
            provider=custom_llm_provider,
            model_id=model_id,
        )
        return response

    character_id: Final = getattr(response, "id", None)
    if isinstance(character_id, str) and character_id:
        response.id = encode_character_id_with_provider(
            character_id=character_id,
            provider=custom_llm_provider,
            model_id=model_id,
        )
    return response
