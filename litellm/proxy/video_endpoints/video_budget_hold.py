"""Holds each video job's estimated price against its creator's key and Team budgets until the job is billed.

A create bills nothing; check_video_cost.py bills the job once the provider finishes it. Until then the
usual budget check sees no spend for it, so without a hold every create would pass. hold_video_budget
prices the request at the most expensive deployment its model name can route to, adds the estimates of
the caller's jobs not yet billed, and refuses when that would pass a max budget. The estimate is stored
on the pending job (PollRecord.estimated_cost) and, through metadata_with_estimate, on the create log row.

Per deployment, the estimate is model_info.max_cost_per_video_job when set. Otherwise MiniMax is the output
rate times the requested seconds plus images past the free count, and BytePlus Seedance is its published
token count, (input video seconds + output seconds) x width x height x 24 fps / 1024, at the widest frame
for the resolution, counting an input video as long as the output. Other providers have no estimate.

video_budget_locks serializes creates per Team and per key within one process. Separate processes can
still each pass one job on the same total.

Callers: the /videos create endpoint.
video_budget_locks, hold_video_budget, raise_video_budget_refusal, metadata_with_estimate, estimate_video_job_cost,
VideoBudgetHold, NoVideoEstimate, VideoOverBudget, MAX_COST_PER_VIDEO_JOB_KEY.
"""

import asyncio
import math
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NoReturn, TypeAlias, assert_never, cast

from pydantic import ValidationError

import litellm
from litellm.llms.base_llm.videos.deployment_model_info import deployment_pricing_model_info
from litellm.llms.byteplus.videos.transformation import RATE_KEYS, seedance_input_kind, seedance_rate
from litellm.llms.minimax.videos.transformation import OUTPUT_PER_SECOND_KEY, minimax_rates
from litellm.proxy._types import ProxyException
from litellm.proxy.video_endpoints.check_video_cost import unbilled_video_estimate
from litellm.types.router import Deployment, GenericLiteLLMParams
from litellm.types.videos.utils import VIDEO_ESTIMATED_COST_KEY

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router

MAX_COST_PER_VIDEO_JOB_KEY: Final = "max_cost_per_video_job"
SEEDANCE_FRAMES_PER_SECOND: Final = Decimal(24)
SEEDANCE_PIXELS_PER_TOKEN: Final = Decimal(1024)
SEEDANCE_WIDEST_RATIO: Final = Decimal(21) / Decimal(9)
SEEDANCE_SHORT_SIDE: Final = MappingProxyType({"480p": 480, "720p": 720, "1080p": 1080, "2k": 1440, "4k": 2160})
SEEDANCE_LONGEST_SECONDS: Final = Decimal(15)
MINIMAX_LONGEST_SECONDS: Final = Decimal(10)
MINIMAX_IMAGE_ITEM_TYPES: Final = frozenset({"image", "image_url", "first_frame_image", "last_frame_image"})

_LOCKS: Final[weakref.WeakValueDictionary[str, asyncio.Lock]] = weakref.WeakValueDictionary()

SpendReader: TypeAlias = Callable[[str, float, float], Awaitable[float]]
UnbilledReader: TypeAlias = Callable[["PrismaClient", Mapping[str, str]], Awaitable[float]]


@dataclass(frozen=True, slots=True)
class VideoBudgetHold:
    estimated_cost: float | None


@dataclass(frozen=True, slots=True)
class NoVideoEstimate:
    model: str


@dataclass(frozen=True, slots=True)
class VideoOverBudget:
    entity_type: str
    entity_id: str
    spend: float
    unbilled: float
    estimate: float
    max_budget: float


@dataclass(frozen=True, slots=True)
class _Budget:
    entity_type: str
    entity_id: str
    counter_key: str
    fallback_spend: float
    max_budget: float
    unbilled_owner: Mapping[str, str]


def video_budget_locks(user_api_key_dict: "UserAPIKeyAuth") -> AsyncExitStack:
    """Locks for the caller's Team and key, entered in a fixed order so two creates never wait on each other."""
    return _LockStack(
        tuple(sorted({f"{budget.entity_type}:{budget.entity_id}" for budget in _budgets(user_api_key_dict)}))
    )


async def hold_video_budget(
    user_api_key_dict: "UserAPIKeyAuth",
    request_data: Mapping[str, object],
    llm_router: "Router | None",
    prisma_client: "PrismaClient | None",
    read_spend: SpendReader | None = None,
    read_unbilled: UnbilledReader = unbilled_video_estimate,
) -> VideoBudgetHold | NoVideoEstimate | VideoOverBudget:
    """Call inside video_budget_locks, and record the job with the hold's estimate before leaving them."""
    budgets: Final = _budgets(user_api_key_dict)
    if not budgets or llm_router is None or prisma_client is None:
        return VideoBudgetHold(estimated_cost=None)
    model: Final = request_data.get("model")
    if not isinstance(model, str):
        return NoVideoEstimate(model=str(model))
    estimate: Final = estimate_video_job_cost(request_data, _deployments(llm_router, model, user_api_key_dict.team_id))
    if estimate is None:
        return NoVideoEstimate(model=model)
    reader: Final = read_spend or _proxy_spend
    for budget in budgets:
        spend = await reader(budget.counter_key, budget.fallback_spend, budget.max_budget)
        unbilled = await read_unbilled(prisma_client, budget.unbilled_owner)
        if spend + unbilled + float(estimate) > budget.max_budget:
            return VideoOverBudget(
                entity_type=budget.entity_type,
                entity_id=budget.entity_id,
                spend=spend,
                unbilled=unbilled,
                estimate=float(estimate),
                max_budget=budget.max_budget,
            )
    return VideoBudgetHold(estimated_cost=float(estimate))


def raise_video_budget_refusal(refusal: NoVideoEstimate | VideoOverBudget) -> NoReturn:
    match refusal:
        case NoVideoEstimate(model=model):
            raise ProxyException(
                message=(
                    f"Video model {model} has no price estimate, so it cannot be held against your budget. "
                    f"An admin sets model_info.{MAX_COST_PER_VIDEO_JOB_KEY} on its deployments."
                ),
                type="budget_exceeded",
                param="model",
                code=400,
            )
        case VideoOverBudget():
            raise litellm.BudgetExceededError(
                current_cost=refusal.spend + refusal.unbilled,
                max_budget=refusal.max_budget,
                message=(
                    f"Budget has been exceeded for {refusal.entity_type} {refusal.entity_id}: spend {refusal.spend:.4f}, "
                    f"videos not yet billed {refusal.unbilled:.4f}, and this video up to {refusal.estimate:.4f} "
                    f"would pass the max budget {refusal.max_budget}"
                ),
                entity_type=refusal.entity_type,
                entity_id=refusal.entity_id,
            )
        case _:
            assert_never(refusal)


def metadata_with_estimate(metadata: object, hold: VideoBudgetHold) -> Mapping[str, object] | None:
    """Request metadata carrying the hold's estimate for the create log row, with any value the caller sent
    under that key dropped. None means leave the request's metadata as it is."""
    if metadata is not None and not isinstance(metadata, Mapping):
        return None
    sent: Final = cast("Mapping[str, object]", metadata or {})  # cast-ok: request JSON
    if hold.estimated_cost is None and VIDEO_ESTIMATED_COST_KEY not in sent:
        return None
    kept: Final = MappingProxyType({key: value for key, value in sent.items() if key != VIDEO_ESTIMATED_COST_KEY})
    if hold.estimated_cost is None:
        return kept
    return MappingProxyType({**kept, VIDEO_ESTIMATED_COST_KEY: hold.estimated_cost})


def estimate_video_job_cost(request_data: Mapping[str, object], deployments: Sequence[Deployment]) -> Decimal | None:
    """Dollars at the most expensive deployment; None when there is none or any of them has no estimate."""
    estimates: Final = tuple(_deployment_estimate(request_data, deployment) for deployment in deployments)
    if not estimates or any(estimate is None for estimate in estimates):
        return None
    return max(estimate for estimate in estimates if estimate is not None)


class _LockStack(AsyncExitStack):
    def __init__(self, scopes: tuple[str, ...]) -> None:
        super().__init__()
        self._scopes: Final = scopes

    async def __aenter__(self) -> "_LockStack":
        await super().__aenter__()
        for scope in self._scopes:
            await self.enter_async_context(_lock(scope))
        return self


def _lock(scope: str) -> asyncio.Lock:
    existing: Final = _LOCKS.get(scope)
    if existing is not None:
        return existing
    created: Final = asyncio.Lock()
    _LOCKS[scope] = created
    return created


async def _proxy_spend(counter_key: str, fallback_spend: float, max_budget: float) -> float:
    from litellm.proxy.proxy_server import get_current_spend

    return await get_current_spend(counter_key=counter_key, fallback_spend=fallback_spend, max_budget=max_budget)


def _budgets(user_api_key_dict: "UserAPIKeyAuth") -> tuple[_Budget, ...]:
    key_hash: Final = user_api_key_dict.token or user_api_key_dict.api_key
    key_max: Final = user_api_key_dict.max_budget
    key_budget: Final = (
        _Budget(
            entity_type="key",
            entity_id=key_hash,
            counter_key=f"spend:key:{key_hash}",
            fallback_spend=user_api_key_dict.spend or 0.0,
            max_budget=key_max,
            unbilled_owner=MappingProxyType({"api_key": user_api_key_dict.api_key or key_hash}),
        )
        if key_hash and key_max is not None and math.isfinite(key_max)
        else None
    )
    team_id: Final = user_api_key_dict.team_id
    team_max: Final = user_api_key_dict.team_max_budget
    team_budget: Final = (
        _Budget(
            entity_type="team",
            entity_id=team_id,
            counter_key=f"spend:team:{team_id}",
            fallback_spend=user_api_key_dict.team_spend or 0.0,
            max_budget=team_max,
            unbilled_owner=MappingProxyType({"team_id": team_id}),
        )
        if team_id and team_max is not None and math.isfinite(team_max)
        else None
    )
    return tuple(budget for budget in (key_budget, team_budget) if budget is not None)


def _deployments(llm_router: "Router", model: str, team_id: str | None) -> tuple[Deployment, ...]:
    listed: Final = llm_router.get_model_list(model_name=model, team_id=team_id) or []
    try:
        return tuple(Deployment.model_validate(entry) for entry in listed)
    except ValidationError:
        # A deployment that cannot be read cannot be priced; no deployments refuses the create.
        return ()


def _deployment_estimate(request_data: Mapping[str, object], deployment: Deployment) -> Decimal | None:
    pricing: Final = deployment_pricing_model_info(
        GenericLiteLLMParams.model_validate(
            MappingProxyType(
                {
                    **deployment.litellm_params.model_dump(exclude_none=True),
                    "model_info": deployment.model_info.model_dump(exclude_none=True),
                }
            )
        )
    )
    ceiling: Final = _dollars(pricing.get(MAX_COST_PER_VIDEO_JOB_KEY))
    if ceiling is not None:
        return ceiling
    provider: Final = deployment.litellm_params.custom_llm_provider or deployment.litellm_params.model.split("/", 1)[0]
    if provider == "minimax":
        return _minimax_estimate(request_data, pricing)
    if provider == "byteplus":
        return _seedance_estimate(request_data, pricing)
    return None


def _minimax_estimate(request_data: Mapping[str, object], pricing: Mapping[str, object]) -> Decimal | None:
    seconds: Final = _seconds(request_data, MINIMAX_LONGEST_SECONDS)
    images: Final = sum(1 for item in _content_items(request_data) if _is_image_item(item)) + (
        1 if request_data.get("input_reference") is not None else 0
    )
    costs: Final = tuple(
        _minimax_cost(pricing, resolution, seconds, images)
        for resolution in _resolutions(request_data, pricing, OUTPUT_PER_SECOND_KEY)
    )
    return max((cost for cost in costs if cost is not None), default=None)


def _minimax_cost(pricing: Mapping[str, object], resolution: str, seconds: Decimal, images: int) -> Decimal | None:
    try:
        rates: Final = minimax_rates(pricing, resolution)
    except ValueError:
        return None
    return rates.output_per_second * seconds + rates.per_image * max(images - rates.free_images, 0)


def _seedance_estimate(request_data: Mapping[str, object], pricing: Mapping[str, object]) -> Decimal | None:
    kind: Final = seedance_input_kind(dict(request_data)) or "plain"
    seconds: Final = _seconds(request_data, SEEDANCE_LONGEST_SECONDS)
    billed_seconds: Final = seconds * 2 if kind == "video" else seconds
    costs: Final = tuple(
        _seedance_cost(pricing, kind, resolution, billed_seconds)
        for resolution in _resolutions(request_data, pricing, RATE_KEYS[kind])
    )
    return max((cost for cost in costs if cost is not None), default=None)


def _seedance_cost(pricing: Mapping[str, object], kind: str, resolution: str, seconds: Decimal) -> Decimal | None:
    short_side: Final = SEEDANCE_SHORT_SIDE.get(resolution)
    if short_side is None:
        return None
    try:
        rate: Final = seedance_rate(pricing, kind, resolution)
    except ValueError:
        return None
    pixels: Final = Decimal(short_side) * Decimal(short_side) * SEEDANCE_WIDEST_RATIO
    return seconds * pixels * SEEDANCE_FRAMES_PER_SECOND / SEEDANCE_PIXELS_PER_TOKEN * rate


def _resolutions(
    request_data: Mapping[str, object], pricing: Mapping[str, object], key_pattern: str
) -> tuple[str, ...]:
    """The request's resolution, or every resolution the deployment prices when it names none."""
    requested: Final = request_data.get("resolution", request_data.get("size"))
    if isinstance(requested, str) and requested.strip():
        return (requested.strip().lower(),)
    prefix: Final = key_pattern.format("")
    return tuple(sorted({key[len(prefix) :].lower() for key in pricing if key.startswith(prefix) and key != prefix}))


def _seconds(request_data: Mapping[str, object], longest: Decimal) -> Decimal:
    value: Final = request_data.get("duration", request_data.get("seconds"))
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return longest
    try:
        seconds: Final = Decimal(str(value).strip())
    except InvalidOperation:
        return longest
    return seconds if seconds.is_finite() and seconds > 0 else longest


def _content_items(request_data: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    content: Final = request_data.get("content")
    if not isinstance(content, (list, tuple)):
        return ()
    items: Final = cast("Sequence[object]", content)  # cast-ok: request JSON
    return tuple(mapping for mapping in map(_mapping, items) if mapping is not None)


def _mapping(item: object) -> Mapping[str, object] | None:
    if not isinstance(item, Mapping):
        return None
    return cast("Mapping[str, object]", item)  # cast-ok: request JSON


def _is_image_item(item: Mapping[str, object]) -> bool:
    kind: Final = item.get("type")
    return isinstance(kind, str) and kind.lower() in MINIMAX_IMAGE_ITEM_TYPES


def _dollars(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    return Decimal(str(value))
