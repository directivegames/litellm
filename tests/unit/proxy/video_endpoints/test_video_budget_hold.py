"""A video create is priced up front and held against its creator's key and Team budgets until the job is billed.

The Router and the deployments' rates are real. The managed-object table is an in-memory stand-in, and spend
comes from an injected reader in place of the Redis counter.
"""

import asyncio
from collections.abc import Mapping
from decimal import Decimal
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.video_endpoints.check_video_cost import record_pending_video_job, unbilled_video_estimate
from litellm.proxy.video_endpoints.video_budget_hold import (
    MAX_COST_PER_VIDEO_JOB_KEY,
    MINIMAX_LONGEST_SECONDS,
    NoVideoEstimate,
    VideoBudgetHold,
    VideoOverBudget,
    estimate_video_job_cost,
    hold_video_budget,
    metadata_with_estimate,
    video_budget_locks,
)
from litellm.router import Router
from litellm.types.router import Deployment
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import VIDEO_COST_POLL_ID_KEY, VIDEO_ESTIMATED_COST_KEY, encode_video_id_with_provider

MINIMAX_RATE: Final = 0.08
MINIMAX_IMAGE_RATE: Final = 0.04
SEEDANCE_PLAIN_RATE: Final = 0.0000107
SEEDANCE_VIDEO_RATE: Final = 0.0000064
MINIMAX: Final = {
    "model_name": "minimax/hailuo-3",
    "litellm_params": {"model": "minimax/MiniMax-H3", "api_key": "mm-key"},
    "model_info": {
        "id": "mm-dep",
        "mode": "video_generation",
        "output_cost_per_second_768p": MINIMAX_RATE,
        "input_cost_per_video_per_second_768p": MINIMAX_RATE,
        "input_cost_per_image": MINIMAX_IMAGE_RATE,
        "free_input_image_count": 1,
    },
}
SEEDANCE: Final = {
    "model_name": "bytedance/seedance-2.5",
    "litellm_params": {"model": "byteplus/dreamina-seedance-2-5-260628", "api_key": "bp-key"},
    "model_info": {
        "id": "bp-dep",
        "mode": "video_generation",
        "output_cost_per_video_token_without_video_input_720p": SEEDANCE_PLAIN_RATE,
        "output_cost_per_video_token_with_video_input_720p": SEEDANCE_VIDEO_RATE,
        "output_cost_per_video_token_without_video_input_1080p": SEEDANCE_PLAIN_RATE * 2,
    },
}
OPENROUTER: Final = {
    "model_name": "openrouter/minimax/hailuo-3",
    "litellm_params": {"model": "openrouter/minimax/hailuo-3", "api_key": "or-key"},
    "model_info": {"id": "or-dep", "mode": "video_generation"},
}
INPUT_VIDEO: Final = {"type": "video_url", "video_url": {"url": "https://cdn.example/in.mp4"}}
IMAGE: Final = {"type": "image_url", "image_url": {"url": "https://cdn.example/in.png"}}


class FakeJobTable:
    """LiteLLM_ManagedObjectTable, with the equality filters unbilled_video_estimate uses."""

    def __init__(self) -> None:
        self.rows: list[SimpleNamespace] = []  # mutable-ok: an in-memory table stand-in

    async def create(self, data: Mapping[str, object]) -> SimpleNamespace:
        row: Final = SimpleNamespace(batch_processed=False, **data)
        self.rows.append(row)
        return row

    async def find_many(
        self, where: Mapping[str, object], take: int, order: Mapping[str, str]
    ) -> list[SimpleNamespace]:  # mutable-ok: prisma returns a list
        return [row for row in self.rows if all(getattr(row, key) == value for key, value in where.items())][:take]


def _deployment(entry: Mapping[str, object]) -> Deployment:
    return Deployment.model_validate(entry)


def _caller(
    team_max_budget: float | None = 10.0, max_budget: float | None = None, key: str = "hashed-key"
) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key=key,
        token=key,
        user_id="user-1",
        team_id="team-1",
        team_max_budget=team_max_budget,
        max_budget=max_budget,
    )


def _spend(by_counter: Mapping[str, float]):
    async def read(counter_key: str, fallback_spend: float, max_budget: float) -> float:
        return by_counter.get(counter_key, 0.0)

    return read


def _prisma(table: FakeJobTable) -> SimpleNamespace:
    return SimpleNamespace(db=SimpleNamespace(litellm_managedobjecttable=table))


async def _record(table: FakeJobTable, caller: UserAPIKeyAuth, task_id: str, estimate: float | None) -> None:
    video: Final = VideoObject(
        id=encode_video_id_with_provider(task_id, "minimax", "mm-dep"), object="video", status="queued", created_at=0
    )
    video._hidden_params[VIDEO_COST_POLL_ID_KEY] = task_id
    await record_pending_video_job(video, caller, {}, _prisma(table), SimpleNamespace(), estimated_cost=estimate)


class TestEstimate:
    def test_minimax_is_the_output_rate_times_the_seconds_plus_images_past_the_free_ones(self) -> None:
        request: Final = {
            "model": "minimax/hailuo-3",
            "resolution": "768P",
            "duration": 6,
            "content": [IMAGE, IMAGE, IMAGE],
        }

        estimate: Final = estimate_video_job_cost(request, (_deployment(MINIMAX),))

        assert estimate == Decimal(str(MINIMAX_RATE)) * 6 + Decimal(str(MINIMAX_IMAGE_RATE)) * 2

    def test_a_request_with_no_duration_is_priced_at_the_longest_job(self) -> None:
        estimate: Final = estimate_video_job_cost({"resolution": "768p"}, (_deployment(MINIMAX),))

        assert estimate == Decimal(str(MINIMAX_RATE)) * MINIMAX_LONGEST_SECONDS

    def test_seedance_grows_with_the_seconds_and_counts_an_input_video_at_its_own_rate(self) -> None:
        deployments: Final = (_deployment(SEEDANCE),)
        plain_5s: Final = estimate_video_job_cost({"resolution": "720p", "duration": 5}, deployments)
        plain_10s: Final = estimate_video_job_cost({"resolution": "720p", "duration": 10}, deployments)
        with_video: Final = estimate_video_job_cost(
            {"resolution": "720p", "duration": 5, "content": [INPUT_VIDEO]}, deployments
        )

        assert plain_5s is not None and plain_10s is not None and with_video is not None
        assert plain_10s == plain_5s * 2
        # The input video counts as long as the output, at the cheaper with-video rate.
        assert with_video == plain_5s * 2 * Decimal(str(SEEDANCE_VIDEO_RATE)) / Decimal(str(SEEDANCE_PLAIN_RATE))

    def test_no_resolution_is_priced_at_the_dearest_resolution_with_a_rate(self) -> None:
        deployments: Final = (_deployment(SEEDANCE),)

        unnamed: Final = estimate_video_job_cost({"duration": 5}, deployments)
        dearest: Final = estimate_video_job_cost({"duration": 5, "resolution": "1080p"}, deployments)

        assert unnamed == dearest

    def test_the_dearest_deployment_of_a_model_name_sets_the_estimate(self) -> None:
        cheap: Final = {
            **MINIMAX,
            "model_info": {**MINIMAX["model_info"], "id": "cheap", "output_cost_per_second_768p": 0.01},
        }
        request: Final = {"resolution": "768p", "duration": 6}

        both: Final = estimate_video_job_cost(request, (_deployment(cheap), _deployment(MINIMAX)))

        assert both == estimate_video_job_cost(request, (_deployment(MINIMAX),))

    def test_a_provider_with_no_rate_needs_a_ceiling_and_then_uses_it(self) -> None:
        with_ceiling: Final = {
            **OPENROUTER,
            "model_info": {**OPENROUTER["model_info"], MAX_COST_PER_VIDEO_JOB_KEY: 2.5},
        }

        assert estimate_video_job_cost({"duration": 5}, (_deployment(OPENROUTER),)) is None
        assert estimate_video_job_cost({"duration": 5}, (_deployment(MINIMAX), _deployment(OPENROUTER))) is None
        assert estimate_video_job_cost({"duration": 5}, (_deployment(with_ceiling),)) == Decimal("2.5")


class TestHold:
    @pytest.mark.asyncio
    async def test_a_create_that_fits_the_budget_is_held_at_its_estimate(self) -> None:
        router: Final = Router(model_list=[MINIMAX])
        request: Final = {"model": "minimax/hailuo-3", "resolution": "768p", "duration": 6}

        hold: Final = await hold_video_budget(_caller(), request, router, _prisma(FakeJobTable()), _spend({}))

        assert hold == VideoBudgetHold(
            estimated_cost=float(estimate_video_job_cost(request, (_deployment(MINIMAX),)) or 0)
        )

    @pytest.mark.asyncio
    async def test_jobs_not_yet_billed_count_against_the_team_and_billed_ones_do_not(self) -> None:
        router: Final = Router(model_list=[MINIMAX])
        table: Final = FakeJobTable()
        request: Final = {"model": "minimax/hailuo-3", "resolution": "768p", "duration": 10}
        estimate: Final = float(estimate_video_job_cost(request, (_deployment(MINIMAX),)) or 0)
        caller: Final = _caller(team_max_budget=estimate * 2.5)
        await _record(table, caller, "running-1", estimate)
        await _record(table, _caller(key="other-key"), "running-2", estimate)
        await _record(table, caller, "billed", estimate)
        table.rows[-1].batch_processed = True

        assert await unbilled_video_estimate(_prisma(table), {"team_id": "team-1"}) == pytest.approx(estimate * 2)
        refused: Final = await hold_video_budget(caller, request, router, _prisma(table), _spend({}))

        assert isinstance(refused, VideoOverBudget)
        assert (refused.entity_type, refused.entity_id) == ("team", "team-1")
        assert refused.unbilled == pytest.approx(estimate * 2)

    @pytest.mark.asyncio
    async def test_spend_already_counted_refuses_a_create_that_would_pass_the_key_budget(self) -> None:
        router: Final = Router(model_list=[MINIMAX])
        request: Final = {"model": "minimax/hailuo-3", "resolution": "768p", "duration": 6}
        estimate: Final = float(estimate_video_job_cost(request, (_deployment(MINIMAX),)) or 0)
        caller: Final = _caller(team_max_budget=None, max_budget=1.0)

        fits: Final = await hold_video_budget(
            caller, request, router, _prisma(FakeJobTable()), _spend({"spend:key:hashed-key": 1.0 - estimate})
        )
        refused: Final = await hold_video_budget(
            caller, request, router, _prisma(FakeJobTable()), _spend({"spend:key:hashed-key": 1.0 - estimate + 0.01})
        )

        assert isinstance(fits, VideoBudgetHold)
        assert isinstance(refused, VideoOverBudget) and refused.entity_type == "key"

    @pytest.mark.asyncio
    async def test_a_model_with_no_estimate_is_refused_only_while_a_budget_applies(self) -> None:
        router: Final = Router(model_list=[OPENROUTER])
        request: Final = {"model": "openrouter/minimax/hailuo-3", "duration": 5}

        budgeted: Final = await hold_video_budget(_caller(), request, router, _prisma(FakeJobTable()), _spend({}))
        unbudgeted: Final = await hold_video_budget(
            _caller(team_max_budget=None), request, router, _prisma(FakeJobTable()), _spend({})
        )

        assert budgeted == NoVideoEstimate(model="openrouter/minimax/hailuo-3")
        assert unbudgeted == VideoBudgetHold(estimated_cost=None)

    @pytest.mark.asyncio
    async def test_concurrent_creates_for_one_team_cannot_all_pass_on_the_same_total(self) -> None:
        router: Final = Router(model_list=[MINIMAX])
        table: Final = FakeJobTable()
        request: Final = {"model": "minimax/hailuo-3", "resolution": "768p", "duration": 10}
        estimate: Final = float(estimate_video_job_cost(request, (_deployment(MINIMAX),)) or 0)
        caller: Final = _caller(team_max_budget=estimate * 3.5)

        async def create(task_id: str) -> bool:
            async with video_budget_locks(caller):
                hold: Final = await hold_video_budget(caller, request, router, _prisma(table), _spend({}))
                if not isinstance(hold, VideoBudgetHold):
                    return False
                # The provider call: the other creates run while this one waits on it.
                await asyncio.sleep(0)
                await _record(table, caller, task_id, hold.estimated_cost)
                return True

        results: Final = await asyncio.gather(*(create(f"task-{index}") for index in range(10)))

        assert sum(results) == 3
        assert len(table.rows) == 3


class TestMetadataWithEstimate:
    def test_the_hold_replaces_any_estimate_the_caller_sent_and_keeps_the_rest(self) -> None:
        sent: Final = {"tags": ["prod"], VIDEO_ESTIMATED_COST_KEY: 0.01}

        held: Final = metadata_with_estimate(sent, VideoBudgetHold(estimated_cost=4.5))
        unheld: Final = metadata_with_estimate(sent, VideoBudgetHold(estimated_cost=None))

        assert held == {"tags": ["prod"], VIDEO_ESTIMATED_COST_KEY: 4.5}
        assert unheld == {"tags": ["prod"]}
        assert metadata_with_estimate("not a mapping", VideoBudgetHold(estimated_cost=4.5)) is None
