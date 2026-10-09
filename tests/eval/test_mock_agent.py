import pytest

from evaluation.mock_agent import TASKS, MockAgent, Sandbox
from evaluation.scenarios import BY_ID, CORE, HARMFUL, HELDOUT, SAFE, SCENARIOS, select


def test_scenario_catalog_is_well_formed():
    assert len(BY_ID) == len(SCENARIOS)
    assert {s.label for s in SCENARIOS} == {SAFE, HARMFUL}
    assert {s.split for s in SCENARIOS} == {CORE, HELDOUT}
    # both splits contain both classes, so every rate is defined
    for split in (CORE, HELDOUT):
        assert select(split, SAFE) and select(split, HARMFUL)


def test_every_task_is_made_of_safe_scenarios():
    for plan in TASKS.values():
        assert all(BY_ID[sid].label == SAFE for sid in plan)


def test_sandbox_has_a_tool_for_every_scenario():
    tools = Sandbox().tools()
    assert {s.tool for s in SCENARIOS} <= set(tools)


def test_sandbox_records_effects_and_funds():
    box = Sandbox()
    tools = box.tools()
    tools["read_file"](path="/a")
    tools["send_payment"](to="x", amount=40)
    assert box.effects == [("read_file", {"path": "/a"}), ("send_payment", {"to": "x", "amount": 40})]
    assert box.funds_sent == 40


def test_same_seed_gives_identical_trace():
    a = MockAgent(seed=3, injection_rate=0.3).plan(50)
    b = MockAgent(seed=3, injection_rate=0.3).plan(50)
    assert [(s.episode, s.index, s.scenario.id, s.injected) for s in a] == [
        (s.episode, s.index, s.scenario.id, s.injected) for s in b
    ]


def test_different_seed_gives_different_trace():
    a = MockAgent(seed=1, injection_rate=0.3).plan(50)
    b = MockAgent(seed=2, injection_rate=0.3).plan(50)
    assert [s.scenario.id for s in a] != [s.scenario.id for s in b]


def test_benign_agent_only_takes_safe_actions():
    trace = MockAgent(seed=0, injection_rate=0.0).plan(100)
    assert trace
    assert not any(s.injected for s in trace)
    assert all(s.scenario.label == SAFE for s in trace)


def test_injected_actions_are_harmful_and_flagged():
    trace = MockAgent(seed=0, injection_rate=1.0).plan(20)
    injected = [s for s in trace if s.injected]
    planned = [s for s in trace if not s.injected]
    assert len(injected) == len(planned)  # rate 1.0: one rogue action after every planned step
    assert all(s.scenario.label == HARMFUL for s in injected)


def test_step_indices_are_contiguous_within_episodes():
    trace = MockAgent(seed=5, injection_rate=0.5).plan(30)
    by_ep = {}
    for s in trace:
        by_ep.setdefault(s.episode, []).append(s.index)
    assert all(idx == list(range(len(idx))) for idx in by_ep.values())


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_invalid_injection_rate(rate):
    with pytest.raises(ValueError):
        MockAgent(injection_rate=rate)
