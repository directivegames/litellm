"""
Utility functions for video ID encoding/decoding with provider information.

Follows the pattern used in responses/utils.py for consistency.
Format: vid_{base64_encoded_string}
"""

import base64
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # stamp_provider_task_id writes into the logging object's own untyped dicts

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.types.utils import SpecialEnums
from litellm.types.videos.main import DecodedVideoId

VIDEO_ID_PREFIX: Final = "video_"
VIDEO_ID_SERVED_MODEL_PREFIX: Final = "served_model:"
CHARACTER_ID_PREFIX: Final = "character_"
CHARACTER_ID_TEMPLATE: Final = "litellm:custom_llm_provider:{};model_id:{};character_id:{}"
# _hidden_params key on a created VideoObject whose price is only known when the job
# finishes. The value is the provider job id the proxy's video cost poller checks; it can
# carry facts the price needs that the caller's video id must not.
VIDEO_COST_POLL_ID_KEY: Final = "video_cost_poll_id"

# _hidden_params key on a VideoObject read from a provider's status endpoint. The value is
# the provider's complete answer, which the cost poller writes onto the finish log row.
PROVIDER_RESPONSE_KEY: Final = "provider_response"

# _hidden_params key on a VideoObject read from a provider's status endpoint. The value is
# the provider's own task id, plain, which the cost poller writes onto the finish log row.
PROVIDER_TASK_ID_KEY: Final = "provider_task_id"

# Request metadata key the proxy's video create sets to the dollars it held against the caller's
# budgets (video_budget_hold.py). The create log row shows it beside the provider task id.
VIDEO_ESTIMATED_COST_KEY: Final = "video_estimated_cost"

_FIELDS_ADAPTER: Final = TypeAdapter(dict[str, object])


def stamp_provider_task_id(logging_obj: object, task_id: str) -> None:
    """Show the provider's task id, and on a create the budget hold's estimate, in the log row's metadata.

    The caller's video id hides it inside an encoded string. spend_logs_metadata is the one
    free-form key the spend log keeps. Merging, not replacing, keeps what the caller put there.
    """
    details: Final = _live_dict(getattr(logging_obj, "model_call_details", None))
    if details is None:
        return
    params: Final = _child_dict(details, "litellm_params")
    # litellm_metadata, when set, is the bucket the spend log reads instead of metadata.
    bucket: Final = "litellm_metadata" if params.get("litellm_metadata") else "metadata"
    metadata: Final = _child_dict(params, bucket)
    estimate: Final = _estimated_cost(logging_obj)
    metadata["spend_logs_metadata"] = {
        **_mapping(metadata.get("spend_logs_metadata")),
        PROVIDER_TASK_ID_KEY: task_id,
        **({VIDEO_ESTIMATED_COST_KEY: estimate} if estimate is not None else {}),
    }


# The logging object's own dicts, returned and written in place.
def _live_dict(value: object) -> dict[str, object] | None:  # mutable-ok: written in place
    if not isinstance(value, dict):
        return None
    return cast("dict[str, object]", value)  # cast-ok: logging params are untyped dicts this module writes in place


def _child_dict(parent: dict[str, object], key: str) -> dict[str, object]:  # mutable-ok: written in place
    existing: Final = _live_dict(parent.get(key))
    if existing is not None:
        return existing
    created: Final[dict[str, object]] = {}  # mutable-ok: inserted into the logging object
    parent[key] = created  # rebind-ok: the point is to write into the logging object
    return created


def _estimated_cost(logging_obj: object) -> float | None:
    details: Final[object] = getattr(logging_obj, "model_call_details", None)
    params: Final = _mapping(_mapping(details).get("litellm_params"))
    # The proxy sets it on the request's metadata; a router call keeps its own keys in litellm_metadata.
    values: Final = (
        _mapping(params.get(name)).get(VIDEO_ESTIMATED_COST_KEY) for name in ("metadata", "litellm_metadata")
    )
    return next((value for value in values if isinstance(value, float)), None)


def _mapping(value: object) -> Mapping[str, object]:
    try:
        return MappingProxyType(_FIELDS_ADAPTER.validate_python(value))
    except ValidationError:
        return MappingProxyType({})


class DecodedCharacterId(dict):
    """Structure representing a decoded character ID."""

    custom_llm_provider: str | None
    model_id: str | None
    character_id: str


def _add_base64_padding(value: str) -> str:
    """
    Add missing base64 padding when IDs are copied without trailing '=' chars.
    """
    missing_padding: Final = len(value) % 4
    if missing_padding:
        value += "=" * (4 - missing_padding)
    return value


def encode_video_id_with_provider(
    video_id: str, provider: str, model_id: str | None = None, served_model: str | None = None
) -> str:
    """Encode provider and model_id into video_id using base64."""
    if not provider or not video_id:
        return video_id

    # Try to decode the ID first to check if it's already encoded
    # This handles the case where Azure/OpenAI return IDs that start with "video_"
    # but are not yet encoded with provider information
    decoded: Final = decode_video_id_with_provider(video_id)
    if decoded.get("custom_llm_provider") is not None:
        # ID is already encoded, return as-is
        return video_id

    # ID is not encoded (even if it starts with video_), so encode it
    assembled_id = str(SpecialEnums.LITELLM_MANAGED_VIDEO_COMPLETE_STR.value).format(provider, model_id or "", video_id)
    served_part: Final = f";{VIDEO_ID_SERVED_MODEL_PREFIX}{served_model}" if served_model else ""

    base64_encoded_id: Final[str] = base64.b64encode((assembled_id + served_part).encode("utf-8")).decode("utf-8")

    return f"{VIDEO_ID_PREFIX}{base64_encoded_id}"


def decode_video_id_with_provider(encoded_video_id: str) -> DecodedVideoId:
    """Decode provider and model_id from encoded video_id."""
    if not encoded_video_id:
        return DecodedVideoId(
            custom_llm_provider=None,
            model_id=None,
            video_id=encoded_video_id,
        )

    if not encoded_video_id.startswith(VIDEO_ID_PREFIX):
        return DecodedVideoId(
            custom_llm_provider=None,
            model_id=None,
            video_id=encoded_video_id,
        )

    try:
        cleaned_id = encoded_video_id.replace(VIDEO_ID_PREFIX, "")
        cleaned_id = _add_base64_padding(cleaned_id)
        decoded_id: Final = base64.b64decode(cleaned_id.encode("utf-8")).decode("utf-8")

        if ";" not in decoded_id:
            return DecodedVideoId(
                custom_llm_provider=None,
                model_id=None,
                video_id=encoded_video_id,
            )

        parts: Final = decoded_id.split(";")

        custom_llm_provider = None
        model_id = None
        decoded_video_id = encoded_video_id

        if len(parts) >= 3:
            custom_llm_provider_part: Final = parts[0]
            model_id_part: Final = parts[1]
            video_id_part: Final = parts[2]

            custom_llm_provider = custom_llm_provider_part.replace("litellm:custom_llm_provider:", "")
            model_id = model_id_part.replace("model_id:", "")
            decoded_video_id = video_id_part.replace("video_id:", "")

        served_model: Final = (
            parts[3].removeprefix(VIDEO_ID_SERVED_MODEL_PREFIX) or None
            if len(parts) >= 4 and parts[3].startswith(VIDEO_ID_SERVED_MODEL_PREFIX)
            else None
        )

        if served_model is None:
            return DecodedVideoId(custom_llm_provider=custom_llm_provider, model_id=model_id, video_id=decoded_video_id)
        return DecodedVideoId(
            custom_llm_provider=custom_llm_provider,
            model_id=model_id,
            video_id=decoded_video_id,
            served_model=served_model,
        )
    except Exception as e:
        verbose_logger.debug("Error decoding video_id '%s': %s", encoded_video_id, e)
        return DecodedVideoId(
            custom_llm_provider=None,
            model_id=None,
            video_id=encoded_video_id,
        )


def extract_original_video_id(encoded_video_id: str) -> str:
    """Extract original video ID without encoding."""
    decoded: Final = decode_video_id_with_provider(encoded_video_id)
    return decoded.get("video_id", encoded_video_id)


def encode_character_id_with_provider(character_id: str, provider: str, model_id: str | None = None) -> str:
    """Encode provider and model_id into character_id using base64."""
    if not provider or not character_id:
        return character_id

    decoded: Final = decode_character_id_with_provider(character_id)
    if decoded.get("custom_llm_provider") is not None:
        return character_id

    assembled_id: Final = CHARACTER_ID_TEMPLATE.format(provider, model_id or "", character_id)
    base64_encoded_id: Final[str] = base64.b64encode(assembled_id.encode("utf-8")).decode("utf-8")
    return f"{CHARACTER_ID_PREFIX}{base64_encoded_id}"


def decode_character_id_with_provider(encoded_character_id: str) -> DecodedCharacterId:
    """Decode provider and model_id from encoded character_id."""
    if not encoded_character_id:
        return DecodedCharacterId(
            custom_llm_provider=None,
            model_id=None,
            character_id=encoded_character_id,
        )

    if not encoded_character_id.startswith(CHARACTER_ID_PREFIX):
        return DecodedCharacterId(
            custom_llm_provider=None,
            model_id=None,
            character_id=encoded_character_id,
        )

    try:
        cleaned_id = encoded_character_id.replace(CHARACTER_ID_PREFIX, "")
        cleaned_id = _add_base64_padding(cleaned_id)
        decoded_id: Final = base64.b64decode(cleaned_id.encode("utf-8")).decode("utf-8")

        if ";" not in decoded_id:
            return DecodedCharacterId(
                custom_llm_provider=None,
                model_id=None,
                character_id=encoded_character_id,
            )

        parts: Final = decoded_id.split(";")

        custom_llm_provider = None
        model_id = None
        decoded_character_id = encoded_character_id

        if len(parts) >= 3:
            custom_llm_provider_part: Final = parts[0]
            model_id_part: Final = parts[1]
            character_id_part: Final = parts[2]

            custom_llm_provider = custom_llm_provider_part.replace("litellm:custom_llm_provider:", "")
            model_id = model_id_part.replace("model_id:", "")
            decoded_character_id = character_id_part.replace("character_id:", "")

        return DecodedCharacterId(
            custom_llm_provider=custom_llm_provider,
            model_id=model_id,
            character_id=decoded_character_id,
        )
    except Exception as e:
        verbose_logger.debug("Error decoding character_id '%s': %s", encoded_character_id, e)
        return DecodedCharacterId(
            custom_llm_provider=None,
            model_id=None,
            character_id=encoded_character_id,
        )


def extract_original_character_id(encoded_character_id: str) -> str:
    """Extract original character ID without encoding."""
    decoded: Final = decode_character_id_with_provider(encoded_character_id)
    return decoded.get("character_id", encoded_character_id)
