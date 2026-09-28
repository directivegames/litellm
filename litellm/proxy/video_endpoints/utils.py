from typing import Any, Final

import orjson

from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_character_id_with_provider,
    encode_video_id_with_provider,
)


def extract_model_from_target_model_names(target_model_names: Any) -> str | None:
    if isinstance(target_model_names, str):
        target_model_names = [m.strip() for m in target_model_names.split(",") if m.strip()]
    elif not isinstance(target_model_names, list):
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


def encode_video_id_in_response(response: Any, fallback_model_id: str | None = None) -> Any:
    """
    Record the router deployment id in a returned video id.

    Provider adapters encode the provider's model name, which can match several deployments
    (a direct row and an OpenRouter row can share minimax/hailuo-3). A deployment id resolves to
    exactly one. Ids the adapter left unencoded are returned as the provider sent them.
    """
    hidden_params: Final = getattr(response, "_hidden_params", None) or {}
    model_id: Final = hidden_params.get("model_id") or fallback_model_id
    if not model_id:
        return response

    video_id: Final = response.get("id") if isinstance(response, dict) else getattr(response, "id", None)
    if not isinstance(video_id, str) or not video_id:
        return response

    decoded: Final = decode_video_id_with_provider(video_id)
    provider: Final = decoded.get("custom_llm_provider")
    if not provider:
        return response

    encoded: Final = encode_video_id_with_provider(decoded.get("video_id") or video_id, provider, model_id)
    if isinstance(response, dict):
        response["id"] = encoded
    else:
        response.id = encoded
    return response


def encode_character_id_in_response(response: Any, custom_llm_provider: str, model_id: str | None) -> Any:
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
