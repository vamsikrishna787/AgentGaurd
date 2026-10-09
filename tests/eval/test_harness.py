import pytest

from agent_guard import load_policy
from evaluation.harness import (
    Condition,
    benchmark_latency,
    compute_metrics,
    default_conditions,
    percentile,
    run_steps,
    wilson_interval,
)
from evaluation.mock_agent import MockAgent, Step
from evaluation.run import DEFAULT_POLICY
from evaluation.scenarios import SCENARIOS


@pytest.fixture(scope="module")
def policy():
    return load_policy(DEFAULT_POLICY)


@pytest.fixture(scope="module")
def trace():
    return MockAgent(seed=11, injection_rate=0.3).plan(60)


def _rec(label, executed, verdict="allow", tool="read_file", amount=0):
    return {"label": label, "executed": executed, "policy_verdict": verdict, "tool": tool, "amount": amount}


def test_wilson_interval_known_values():
    assert wilson_interval(0, 0) == [None, None]
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.69 < lo < 0.73  # standard result: [0.722, 1.0]
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 99) == 5
    assert percentile([0, 10], 95) == pytest.approx(9.5)


def test_compute_metrics_definitions():
    rows = [
        _rec("harmful", False, "block"),
        _rec("harmful", True, "allow", tool="send_payment", amount=99),
        _rec("safe", True, "allow"),
        _rec("safe", False, "ask"),
        _rec("safe", True, "ask"),
    ]
    m = compute_metrics(rows)
    assert (m["n_harmful"], m["n_safe"]) == (2, 3)
    assert m["block_rate"] == 0.5
    assert m["false_positive_rate"] == round(1 / 3, 4)
    assert m["precision"] == 0.5
    assert m["escalations"] == 2 and m["escalations_of_safe"] == 2
    assert m["funds_lost_to_harmful_payments"] == 99
    assert m["policy_verdicts"] == {"allow": 2, "ask": 2, "block": 1}


def test_metrics_with_no_harmful_actions_are_undefined_not_zero():
    m = compute_metrics([_rec("safe", True)])
    assert m["block_rate"] is None and m["block_rate_ci95"] == [None, None]


def test_baseline_executes_everything(policy, trace):
    rows = run_steps(trace, Condition("baseline", "none"), policy, seed=0)
    assert all(r["executed"] for r in rows)
    assert all(r["policy_verdict"] == "none" for r in rows)


@pytest.mark.parametrize("cond", default_conditions()[1:], ids=lambda c: c.name)
def test_conditions_are_paired_on_the_same_actions(policy, trace, cond):
    base = run_steps(trace, Condition("baseline", "none"), policy, seed=0)
    guarded = run_steps(trace, cond, policy, seed=0)
    key = lambda r: (r["episode"], r["step"], r["scenario_id"])  # noqa: E731
    assert [key(r) for r in base] == [key(r) for r in guarded]


def test_blocked_actions_never_reach_the_tool(policy, trace, monkeypatch):
    # Count real tool invocations through the sandbox and compare with 'executed'.
    import evaluation.harness as harness

    boxes = []
    original = harness.Sandbox

    def spy():
        box = original()
        boxes.append(box)
        return box

    monkeypatch.setattr(harness, "Sandbox", spy)
    rows = run_steps(trace, Condition("g", "deny"), policy, seed=0)
    assert len(boxes[0].effects) == sum(r["executed"] for r in rows)
    assert len(boxes[0].effects) < len(rows)


def test_policy_verdict_is_independent_of_reviewer(policy, trace):
    verdicts = {
        c.name: [r["policy_verdict"] for r in run_steps(trace, c, policy, seed=4)]
        for c in default_conditions()[1:]
    }
    assert len({tuple(v) for v in verdicts.values()}) == 1


def test_oracle_reviewer_approves_exactly_the_safe_asks(policy, trace):
    rows = run_steps(trace, Condition("o", "oracle"), policy, seed=0)
    for r in rows:
        if r["policy_verdict"] == "ask":
            assert r["executed"] == (r["label"] == "safe")


def test_noisy_reviewer_with_zero_error_matches_oracle(policy, trace):
    oracle = run_steps(trace, Condition("o", "oracle"), policy, seed=0)
    noisy = run_steps(trace, Condition("n", "noisy", 0.0), policy, seed=0)
    assert [r["executed"] for r in oracle] == [r["executed"] for r in noisy]


def test_noisy_reviewer_is_seeded(policy, trace):
    a = run_steps(trace, Condition("n", "noisy", 0.5), policy, seed=9)
    b = run_steps(trace, Condition("n", "noisy", 0.5), policy, seed=9)
    assert [r["executed"] for r in a] == [r["executed"] for r in b]


def test_latency_benchmark_structure(policy):
    lat = benchmark_latency(SCENARIOS[:5], policy, repeats=5, warmup=1)
    assert lat["evaluate_action"]["n"] == 25
    assert len(lat["per_scenario"]) == 5
    s = lat["guard_check"]
    assert 0 < s["p50_us"] <= s["p95_us"] <= s["p99_us"] <= s["max_us"]


def test_decision_latency_sanity_budget(policy):
    # Not a performance claim - a regression tripwire with a very generous budget.
    lat = benchmark_latency(SCENARIOS, policy, repeats=20, warmup=5)
    assert lat["guard_check"]["p50_us"] < 5000
