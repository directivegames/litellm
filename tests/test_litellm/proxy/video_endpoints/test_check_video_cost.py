"""Video jobs are recorded at create and billed once, by CheckVideoCost, when they finish.

Every test runs the real Router and provider adapters. Only the provider's HTTP is faked,
through respx, and the managed-object table is an in-memory stand-in.
"""

import asyncio
import itertools
import json
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.video_endpoints import check_video_cost
from litellm.proxy.video_endpoints.check_video_cost import CheckVideoCost, record_pending_video_job
from litellm.proxy.video_endpoints.utils import encode_video_id_in_response
from litellm.router import Router
from litellm.types.utils import BACKGROUND_VIDEO_COST_POLL_CALL_ORIGIN
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import VIDEO_COST_POLL_ID_KEY, encode_video_id_with_provider

MINIMAX_CREATE = "https://api.minimax.io/v2/video_generation"
MINIMAX_STATUS = "https://api.minimax.io/v2/query/video_generation/mm-task"
BYTEPLUS_CREATE = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
BYTEPLUS_STATUS = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks/cgt-1"
OPENROUTER_CREATE = "https://openrouter.ai/api/v1/videos"
OPENROUTER_STATUS = "https://openrouter.ai/api/v1/videos/or-job"

DEPLOYMENTS = [
    {
        "model_name": "minimax/hailuo-3",
        "litellm_params": {"model": "minimax/MiniMax-H3", "api_key": "mm-key"},
        "model_info": {
            "id": "mm-dep",
            "mode": "video_generation",
            "output_cost_per_second_768p": 0.08,
            "input_cost_per_video_per_second_768p": 0.08,
            "input_cost_per_image": 0.04,
            "free_input_image_count": 5,
        },
    },
    {
        "model_name": "bytedance/seedance-2.5",
        "litellm_params": {"model": "byteplus/dreamina-seedance-2-5-260628", "api_key": "bp-key"},
        "model_info": {
            "id": "bp-dep",
            "mode": "video_generation",
            "output_cost_per_video_token_without_video_input_720p": 0.0000107,
            "output_cost_per_video_token_with_video_input_720p": 0.0000064,
        },
    },
    {
        "model_name": "openrouter/minimax/hailuo-3",
        "litellm_params": {"model": "openrouter/minimax/hailuo-3", "api_key": "or-key"},
        "model_info": {"id": "or-dep", "mode": "video_generation"},
    },
]
CALLER = UserAPIKeyAuth(api_key="hashed-key", user_id="user-1", team_id="team-1", org_id="org-1", key_alias="ci")
INPUT_VIDEO = {"type": "video_url", "video_url": {"url": "https://cdn.example/in.mp4"}}


def _minimax_task(status: str = "succeeded", input_seconds: int = 0) -> dict:
    usage = {"output_seconds": 5, "input_seconds": input_seconds, "input_image_count": 0}
    return {"task": {"id": "mm-task", "status": status, "resolution": "768P", "usage": usage}}


def _byteplus_task(status: str = "succeeded") -> dict:
    return {"id": "cgt-1", "status": status, "resolution": "720p", "usage": {"total_tokens": 100_000}}


class FakeProviders:
    """Answers the provider routes a test registers. Any other request fails the test."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], tuple[int, object]] = {}
        self.seen: list[tuple[str, str]] = []

    def reply(self, method: str, url: str, body: object, status: int = 200) -> None:
        self.routes[(method, url)] = (status, body)

    def handle(self, request: httpx.Request) -> httpx.Response:
        key = (request.method, str(request.url))
        self.seen.append(key)
        if key not in self.routes:
            raise AssertionError(f"unexpected provider request {key}")
        status, body = self.routes[key]
        return httpx.Response(status, json=body)


class FakeJobTable:
    """LiteLLM_ManagedObjectTable with the filters CheckVideoCost uses. Every write bumps updated_at."""

    def __init__(self) -> None:
        self.rows: list[SimpleNamespace] = []
        self._ids = itertools.count(1)
        self._clock = itertools.count(1)
        self.fail_create = False
        # When set, each find_many holds its snapshot until every reader has one: pods that read together.
        self.readers: asyncio.Barrier | None = None

    async def create(self, data: Mapping[str, object]) -> SimpleNamespace:
        if self.fail_create:
            raise RuntimeError("db down")
        now = datetime.now(timezone.utc)
        row = SimpleNamespace(
            id=f"row-{next(self._ids)}",
            batch_processed=False,
            created_at=now,
            updated_at=next(self._clock),
            **data,
        )
        self.rows.append(row)
        return row

    async def find_many(
        self, where: Mapping[str, object], take: int, order: Mapping[str, str]
    ) -> list[SimpleNamespace]:
        ((column, direction),) = order.items()
        found = sorted(
            (row for row in self.rows if _matches(row, where)),
            key=lambda row: getattr(row, column),
            reverse=direction == "desc",
        )
        if self.readers is not None:
            await self.readers.wait()
        return found[:take]

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> None:
        for row in self.rows:
            if _matches(row, where):
                self._write(row, data)

    async def update_many(self, where: Mapping[str, object], data: Mapping[str, object]) -> int:
        matched = [row for row in self.rows if _matches(row, where)]
        for row in matched:
            self._write(row, data)
        return len(matched)

    def _write(self, row: SimpleNamespace, data: Mapping[str, object]) -> None:
        for key, value in data.items():
            setattr(row, key, value)
        row.updated_at = next(self._clock)


def _matches(row: SimpleNamespace, where: Mapping[str, object]) -> bool:
    for key, condition in where.items():
        value = getattr(row, key)
        if isinstance(condition, Mapping):
            if "not_in" in condition and value in condition["not_in"]:
                return False
            if "lt" in condition and not value < condition["lt"]:
                return False
        elif value != condition:
            return False
    return True


class Alerts:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def failed_tracking_alert(self, error_message: str, failing_model: str) -> None:
        self.messages.append(error_message)


class SpendCapture(CustomLogger):
    """Sees exactly what the proxy's spend writer sees for every logged call."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[dict, object]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.events.append((kwargs, response_obj))

    def create_charges(self) -> list[tuple[dict, object]]:
        return [event for event in self.events if event[0].get("call_type") == "avideo_generation"]

    def poll_charges(self) -> list[tuple[dict, object]]:
        return [event for event in self.events if event[0].get("call_type") == "avideo_retrieve"]

    async def wait_for_creates(self, count: int) -> None:
        """A create logs its spend in a background task, after the response returns."""
        async with asyncio.timeout(5):
            while len(self.create_charges()) < count:
                await asyncio.sleep(0.01)


class Stack:
    def __init__(self, spend: SpendCapture) -> None:
        self.providers = FakeProviders()
        self.table = FakeJobTable()
        self.alerts = Alerts()
        self.spend = spend
        self.router = Router(model_list=DEPLOYMENTS)
        self.prisma = SimpleNamespace(db=SimpleNamespace(litellm_managedobjecttable=self.table))

    async def create(self, model: str, **params: object) -> VideoObject:
        created_before: Final = len(self.spend.create_charges())
        video = await self.router.avideo_generation(model=model, prompt="a red boat", **params)
        await self.spend.wait_for_creates(created_before + 1)
        encoded = encode_video_id_in_response(video)
        await record_pending_video_job(encoded, CALLER, {"metadata": {"tags": ["prod"]}}, self.prisma, self.alerts)
        return encoded

    def poller(self) -> CheckVideoCost:
        return CheckVideoCost(prisma_client=self.prisma, llm_router=self.router, alerts=self.alerts)

    async def poll(self, cycles: int = 1) -> None:
        for _ in range(cycles):
            await self.poller().check_video_cost()
            await asyncio.sleep(0)


@pytest.fixture
def stack(monkeypatch: pytest.MonkeyPatch) -> Iterator[Stack]:
    # respx intercepts httpx only, so LiteLLM's default aiohttp transport is switched off
    # and any cached client built with it is dropped.
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    spend = SpendCapture()
    monkeypatch.setattr(litellm, "callbacks", [spend])
    monkeypatch.setattr(litellm, "_async_success_callback", [spend])
    stack = Stack(spend)
    with respx.mock(assert_all_called=False) as provider_http:
        provider_http.route().mock(side_effect=stack.providers.handle)
        yield stack
    litellm.in_memory_llm_clients_cache.flush_cache()


PROVIDER_CASES = [
    pytest.param(
        "minimax/hailuo-3",
        {"duration": 5, "resolution": "768P"},
        (MINIMAX_CREATE, {"task_id": "mm-task", "status": "queued"}),
        (MINIMAX_STATUS, _minimax_task()),
        0.4,
        id="minimax",
    ),
    pytest.param(
        "minimax/hailuo-3",
        {"duration": 5, "resolution": "768P", "content": [INPUT_VIDEO]},
        (MINIMAX_CREATE, {"task_id": "mm-task", "status": "queued"}),
        (MINIMAX_STATUS, _minimax_task(input_seconds=3)),
        0.64,
        id="minimax_input_video",
    ),
    pytest.param(
        "bytedance/seedance-2.5",
        {"duration": 5, "resolution": "720p"},
        (BYTEPLUS_CREATE, {"id": "cgt-1"}),
        (BYTEPLUS_STATUS, _byteplus_task()),
        1.07,
        id="byteplus",
    ),
    pytest.param(
        "bytedance/seedance-2.5",
        {"duration": 5, "resolution": "720p", "content": [INPUT_VIDEO]},
        (BYTEPLUS_CREATE, {"id": "cgt-1"}),
        (BYTEPLUS_STATUS, _byteplus_task()),
        0.64,
        id="byteplus_input_video",
    ),
    pytest.param(
        "openrouter/minimax/hailuo-3",
        {"duration": 5, "resolution": "720p"},
        (OPENROUTER_CREATE, {"id": "or-job", "status": "pending"}),
        (OPENROUTER_STATUS, {"id": "or-job", "status": "completed", "usage": {"cost": 0.5}}),
        0.5,
        id="openrouter",
    ),
]


class TestBilledWhenTheJobFinishes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("model", "params", "create", "status", "cost"), PROVIDER_CASES)
    async def test_the_create_is_free_and_the_finished_job_bills_its_creator_once(
        self, stack: Stack, model: str, params: dict, create: tuple, status: tuple, cost: float
    ) -> None:
        stack.providers.reply("POST", *create)
        stack.providers.reply("GET", *status)

        video = await stack.create(model, **params)
        started = datetime.now(timezone.utc) - timedelta(minutes=2)
        stack.table.rows[0].created_at = started
        await stack.poll(cycles=2)

        assert [kwargs["response_cost"] for kwargs, _ in stack.spend.create_charges()] == [0.0]
        charges = stack.spend.poll_charges()
        assert len(charges) == 1
        kwargs, billed = charges[0]
        assert kwargs["response_cost"] == pytest.approx(cost)
        assert kwargs["standard_logging_object"]["response_cost"] == pytest.approx(cost)
        logged = kwargs["standard_logging_object"]
        assert logged["endTime"] - logged["startTime"] == pytest.approx(120, abs=5)
        assert billed.usage["provider_reported_cost_usd"] == pytest.approx(cost)
        if model.startswith("minimax/"):
            assert logged["metadata"]["usage_object"]["duration_seconds"] == 5
            assert billed.usage["duration_seconds"] == 5
            assert billed.usage["input_seconds"] == (3 if "content" in params else 0)
            assert billed.usage["output_cost_per_second"] == pytest.approx(0.08)
            assert billed.seconds == "5"
        elif model.startswith("bytedance/"):
            assert billed.usage["total_tokens"] == 100_000
            assert logged["total_tokens"] == 100_000
            assert billed.usage["video_resolution"] == "720p"
            rate_key = (
                "output_cost_per_video_token_with_video_input_720p"
                if "content" in params
                else "output_cost_per_video_token_without_video_input_720p"
            )
            assert billed.usage[rate_key] == pytest.approx(0.0000107 if "content" not in params else 0.0000064)
        else:
            assert set(billed.usage) == {"provider_reported_cost_usd"}
        assert billed.id == video.id
        metadata = kwargs["litellm_params"]["metadata"]
        assert metadata["user_api_key"] == "hashed-key"
        assert metadata["user_api_key_team_id"] == "team-1"
        assert metadata["tags"] == ["prod"]
        assert metadata[INTERNAL_CALL_ORIGIN_METADATA_KEY] == BACKGROUND_VIDEO_COST_POLL_CALL_ORIGIN
        assert ("GET", status[0]) in stack.providers.seen
        (row,) = stack.table.rows
        assert (row.status, row.batch_processed) == ("completed", True)
        assert stack.alerts.messages == []

    @pytest.mark.asyncio
    async def test_the_finish_log_ends_when_the_provider_says_the_job_completed(self, stack: Stack) -> None:
        completed = int(datetime.now(timezone.utc).timestamp()) - 30
        started = datetime.now(timezone.utc) - timedelta(minutes=3)
        body = _minimax_task()
        body["task"]["completed_at"] = completed
        stack.providers.reply("POST", MINIMAX_CREATE, {"task_id": "mm-task", "status": "queued"})
        stack.providers.reply("GET", MINIMAX_STATUS, body)

        await stack.create("minimax/hailuo-3", duration=5, resolution="768P")
        stack.table.rows[0].created_at = started
        await stack.poll(cycles=2)

        kwargs, _billed = stack.spend.poll_charges()[0]
        logged = kwargs["standard_logging_object"]
        assert logged["startTime"] == pytest.approx(started.timestamp(), abs=1)
        assert logged["endTime"] == pytest.approx(completed, abs=1)

    @pytest.mark.asyncio
    async def test_a_running_job_is_not_billed(self, stack: Stack) -> None:
        stack.providers.reply("POST", MINIMAX_CREATE, {"task_id": "mm-task", "status": "queued"})
        stack.providers.reply("GET", MINIMAX_STATUS, _minimax_task(status="processing"))

        await stack.create("minimax/hailuo-3", duration=5, resolution="768P")
        await stack.poll()

        assert stack.spend.poll_charges() == []
        (row,) = stack.table.rows
        assert (row.status, row.batch_processed) == ("in_progress", False)

    @pytest.mark.asyncio
    async def test_a_failed_job_closes_without_a_charge_or_an_alert(self, stack: Stack) -> None:
        stack.providers.reply("POST", MINIMAX_CREATE, {"task_id": "mm-task", "status": "queued"})
        stack.providers.reply("GET", MINIMAX_STATUS, _minimax_task(status="failed"))

        await stack.create("minimax/hailuo-3", duration=5, resolution="768P")
        await stack.poll(cycles=2)

        assert stack.spend.poll_charges() == []
        (row,) = stack.table.rows
        assert (row.status, row.batch_processed) == ("failed", True)
        assert stack.alerts.messages == []

    @pytest.mark.asyncio
    async def test_a_finished_job_without_a_cost_closes_with_an_alert(self, stack: Stack) -> None:
        stack.providers.reply("POST", OPENROUTER_CREATE, {"id": "or-job", "status": "pending"})
        stack.providers.reply("GET", OPENROUTER_STATUS, {"id": "or-job", "status": "completed"})

        video = await stack.create("openrouter/minimax/hailuo-3", duration=5)
        await stack.poll(cycles=2)

        assert stack.spend.poll_charges() == []
        (row,) = stack.table.rows
        assert (row.status, row.batch_processed) == ("completed", True)
        assert stack.alerts.messages == [
            f"CheckVideoCost: video {video.id} will not be billed: it finished but reported no cost"
        ]

    @pytest.mark.asyncio
    async def test_two_pods_polling_together_bill_the_job_once(self, stack: Stack) -> None:
        stack.providers.reply("POST", BYTEPLUS_CREATE, {"id": "cgt-1"})
        stack.providers.reply("GET", BYTEPLUS_STATUS, _byteplus_task())

        await stack.create("bytedance/seedance-2.5", duration=5, resolution="720p")
        stack.table.readers = asyncio.Barrier(2)
        await asyncio.gather(stack.poller().check_video_cost(), stack.poller().check_video_cost())

        assert len(stack.spend.poll_charges()) == 1


class TestJobsThatCannotBeBilledYet:
    @pytest.mark.asyncio
    async def test_a_job_that_keeps_waiting_does_not_starve_newer_jobs(
        self, stack: Stack, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(check_video_cost, "MAX_OBJECTS_PER_POLL_CYCLE", 1)
        stack.providers.reply("POST", MINIMAX_CREATE, {"task_id": "mm-task", "status": "queued"})
        stack.providers.reply("GET", MINIMAX_STATUS, _minimax_task(status="processing"))
        stack.providers.reply("POST", OPENROUTER_CREATE, {"id": "or-job", "status": "pending"})
        stack.providers.reply("GET", OPENROUTER_STATUS, {"id": "or-job", "status": "completed", "usage": {"cost": 0.5}})

        await stack.create("minimax/hailuo-3", duration=5, resolution="768P")
        await stack.create("openrouter/minimax/hailuo-3", duration=5)
        await stack.poll(cycles=2)

        assert [kwargs["response_cost"] for kwargs, _ in stack.spend.poll_charges()] == [0.5]

    @pytest.mark.asyncio
    async def test_a_removed_deployment_leaves_the_job_waiting(self, stack: Stack) -> None:
        await stack.table.create(
            data={
                "unified_object_id": encode_video_id_with_provider("mm-task", "minimax", "gone"),
                "file_object": json.dumps(
                    {"poll_video_id": "mm-task", "custom_llm_provider": "minimax", "model_id": "gone"}
                ),
                "file_purpose": "video",
                "status": "queued",
            }
        )

        await stack.poll()

        assert stack.providers.seen == []
        (row,) = stack.table.rows
        assert (row.status, row.batch_processed) == ("queued", False)
        assert stack.alerts.messages == []

    @pytest.mark.asyncio
    async def test_one_provider_error_does_not_stop_the_cycle(self, stack: Stack) -> None:
        stack.providers.reply("POST", MINIMAX_CREATE, {"task_id": "mm-task", "status": "queued"})
        stack.providers.reply("GET", MINIMAX_STATUS, {"error": "upstream"}, status=500)
        stack.providers.reply("POST", OPENROUTER_CREATE, {"id": "or-job", "status": "pending"})
        stack.providers.reply("GET", OPENROUTER_STATUS, {"id": "or-job", "status": "completed", "usage": {"cost": 0.5}})

        await stack.create("minimax/hailuo-3", duration=5, resolution="768P")
        await stack.create("openrouter/minimax/hailuo-3", duration=5)
        await stack.poll()

        assert [kwargs["response_cost"] for kwargs, _ in stack.spend.poll_charges()] == [0.5]
        waiting = next(row for row in stack.table.rows if row.batch_processed is False)
        assert waiting.status == "queued"


class TestJobsThatWillNeverBeBilled:
    @pytest.mark.asyncio
    async def test_an_unreadable_record_closes_the_job_with_an_alert(self, stack: Stack) -> None:
        await stack.table.create(
            data={
                "unified_object_id": "video_x",
                "file_object": "not json",
                "file_purpose": "video",
                "status": "queued",
            }
        )

        await stack.poll()

        (row,) = stack.table.rows
        assert (row.status, row.batch_processed) == ("failed", True)
        assert stack.alerts.messages == [
            "CheckVideoCost: video video_x will not be billed: its poll record cannot be read"
        ]

    @pytest.mark.asyncio
    async def test_a_job_older_than_the_staleness_window_is_dropped_with_an_alert(self, stack: Stack) -> None:
        stack.providers.reply("POST", MINIMAX_CREATE, {"task_id": "mm-task", "status": "queued"})
        await stack.create("minimax/hailuo-3", duration=5, resolution="768P")
        (row,) = stack.table.rows
        row.created_at = datetime.now(timezone.utc) - timedelta(days=30)

        await stack.poll()

        assert (row.status, row.batch_processed) == ("stale_expired", True)
        assert len(stack.alerts.messages) == 1
        assert "gave up on 1 video jobs" in stack.alerts.messages[0]
        assert ("GET", MINIMAX_STATUS) not in stack.providers.seen


def _created_video(poll_id: str | None, video_id: str) -> VideoObject:
    video = VideoObject(id=video_id, object="video", status="queued", created_at=0)
    if poll_id is not None:
        video._hidden_params[VIDEO_COST_POLL_ID_KEY] = poll_id
    return video


class TestRecordPendingVideoJob:
    CALLER_VIDEO_ID = encode_video_id_with_provider("cgt-1", "byteplus", "bp-dep")

    async def _record(
        self, video: VideoObject, table: FakeJobTable | None = None, alerts: Alerts | None = None
    ) -> tuple[FakeJobTable, Alerts]:
        table = table or FakeJobTable()
        alerts = alerts or Alerts()
        prisma = SimpleNamespace(db=SimpleNamespace(litellm_managedobjecttable=table))
        await record_pending_video_job(video, CALLER, {"metadata": {"tags": ["prod"]}}, prisma, alerts)
        return table, alerts

    @pytest.mark.asyncio
    async def test_records_the_job_and_who_created_it(self) -> None:
        table, alerts = await self._record(_created_video("cgt-1~video", self.CALLER_VIDEO_ID))

        (row,) = table.rows
        assert row.unified_object_id == self.CALLER_VIDEO_ID
        assert row.model_object_id == "video:byteplus:cgt-1~video"
        assert (row.file_purpose, row.status) == ("video", "queued")
        assert (row.created_by, row.team_id, row.org_id, row.api_key) == ("user-1", "team-1", "org-1", "hashed-key")
        record = json.loads(row.file_object)
        assert (record["poll_video_id"], record["model_id"]) == ("cgt-1~video", "bp-dep")
        assert record["metadata"]["user_api_key_alias"] == "ci"
        assert record["metadata"]["tags"] == ["prod"]
        assert alerts.messages == []

    @pytest.mark.asyncio
    async def test_a_video_without_a_poll_id_is_not_recorded(self) -> None:
        table, alerts = await self._record(_created_video(None, self.CALLER_VIDEO_ID))

        assert table.rows == []
        assert alerts.messages == []

    @pytest.mark.asyncio
    async def test_a_failed_write_alerts_instead_of_failing_the_create(self) -> None:
        table = FakeJobTable()
        table.fail_create = True

        _table, alerts = await self._record(_created_video("cgt-1~video", self.CALLER_VIDEO_ID), table=table)

        assert alerts.messages == [
            f"CheckVideoCost: video {self.CALLER_VIDEO_ID} will not be billed: it could not be recorded: db down"
        ]

    @pytest.mark.asyncio
    async def test_an_id_without_a_deployment_alerts(self) -> None:
        table, alerts = await self._record(_created_video("raw-task", "raw-task"))

        assert table.rows == []
        assert alerts.messages == [
            "CheckVideoCost: video raw-task will not be billed: it could not be recorded: its id names no provider deployment"
        ]

    @pytest.mark.asyncio
    async def test_without_a_database_nothing_is_recorded_or_alerted(self) -> None:
        alerts = Alerts()

        await record_pending_video_job(_created_video("cgt-1~video", self.CALLER_VIDEO_ID), CALLER, {}, None, alerts)

        assert alerts.messages == []
