"""The smoke-test command must exit non-zero whenever the run did not fully succeed."""

import pytest

from msu_llm_lab.classification import smoke
from msu_llm_lab.classification.records import Task
from msu_llm_lab.core.budget import BudgetService
from msu_llm_lab.core.live_guard import LiveAuthorization
from test_runner_adapters import http_err, jev_body, jev_choice, noul_body
from test_smoke import cli_env, fake_net, run_cli  # noqa: F401  (fixtures and helper)


@pytest.fixture
def approved(monkeypatch):
    import msu_llm_lab.core.live_guard as lg
    monkeypatch.setattr(lg, "confirm_live_run", lambda models, summary_lines=(), **k: LiveAuthorization(models=tuple(models)))


def live(*extra):
    return run_cli("--model", "jev-1.14", "--task", "INTENT", "--live", *extra)


def good():
    return [jev_body(jev_choice(l)) for l in ("GREETING", "QUERY", "OUT_OF_SCOPE", "UPLOAD_STATUS_CHECK", "HUMAN_HANDOFF_REQUEST")]


def test_a_complete_valid_run_exits_zero(cli_env, fake_net, approved):
    fake_net.script = good()
    r = live()
    assert r.exit_code == 0 and "Smoke test completed" in r.output


def test_a_partial_run_that_stops_after_one_record_exits_non_zero(cli_env, fake_net, approved):
    fake_net.script = [good()[0]] + [http_err(400) for _ in range(3)]
    r = live()
    assert r.exit_code == 1 and "SMOKE TEST FAILED" in r.output and "planned requests produced a record" in r.output and "errors" in r.output


def test_invalid_outputs_exit_non_zero_even_though_every_request_returned(cli_env, fake_net, approved):
    fake_net.script = [jev_body(jev_choice(choice="greeting")) for _ in range(5)]
    r = live()
    assert r.exit_code == 1 and "invalid outputs or errors" in r.output and "5 record(s)" in r.output


def test_api_errors_exit_non_zero(cli_env, fake_net, approved):
    fake_net.script = [http_err(401)]
    r = live()
    assert r.exit_code == 1 and "SMOKE TEST FAILED" in r.output


def test_an_unexpected_exception_during_execution_exits_non_zero_after_writing_what_exists(cli_env, fake_net, approved, monkeypatch):
    fake_net.script = good()
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(smoke, "execute_plan", boom)
    r = live()
    assert r.exit_code == 1 and "Smoke test stopped" in r.output and "the run stopped before finishing" in r.output


def test_missing_cost_stops_the_run_leaves_unresolved_spend_and_fails_the_command(cli_env, fake_net, approved):
    body = jev_body(jev_choice())
    del body["usage"]
    fake_net.script = [body, jev_body(jev_choice())]
    r = live()
    assert r.exit_code == 1 and "unresolved spend" in r.output.lower()
    assert len(BudgetService().unresolved()) == 1


def test_unresolved_spend_blocks_the_next_plan_and_live_run(cli_env, fake_net, approved):
    BudgetService().record_unresolved_spend("tx-old", "typesafe/jev-1.14", "openrouter", "earlier request with unknown cost")
    plan_only = run_cli("--model", "jev-1.14")
    assert plan_only.exit_code == 2 and "unresolved-spend" in plan_only.output
    r = live()
    assert r.exit_code == 1 and "plan is blocked" in r.output and fake_net.requests == []
