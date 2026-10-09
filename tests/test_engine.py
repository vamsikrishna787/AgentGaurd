import pytest

from agent_guard import Action, Verdict, evaluate_action, parse_policy


@pytest.fixture
def policy():
    return parse_policy(
        {
            "default": "block",
            "allowlist": ["read_file", "search_*"],
            "denylist": ["delete_*"],
            "rules": [
                {
                    "id": "no-secrets",
                    "tools": ["read_file"],
                    "action": "block",
                    "when": [{"arg": "path", "op": "matches", "value": r"\.env$"}],
                    "reason": "secret file",
                },
                {
                    "id": "big-payment",
                    "tools": ["send_payment"],
                    "action": "ask",
                    "when": [{"arg": "amount", "op": "gt", "value": 100}],
                    "reason": "large payment",
                },
                {
                    "id": "small-payment",
                    "tools": ["send_payment"],
                    "action": "allow",
                    "when": [{"arg": "amount", "op": "lte", "value": 100}],
                },
            ],
        }
    )


def test_allowlisted_tool_is_allowed(policy):
    d = evaluate_action({"tool": "read_file", "args": {"path": "README.md"}}, policy)
    assert d.verdict is Verdict.ALLOW
    assert d.allowed
    assert d.matched_rules == ["allowlist"]


def test_allowlist_supports_globs(policy):
    assert evaluate_action({"tool": "search_web"}, policy).allowed


def test_denylisted_tool_is_blocked(policy):
    d = evaluate_action(Action("delete_user", {"id": 1}), policy)
    assert d.blocked
    assert "denylist" in d.reasons[0]


def test_denylist_beats_allowlist():
    p = parse_policy({"allowlist": ["*"], "denylist": ["rm"]})
    assert evaluate_action({"tool": "rm"}, p).blocked
    assert evaluate_action({"tool": "ls"}, p).allowed


def test_block_rule_overrides_allowlist(policy):
    d = evaluate_action({"tool": "read_file", "args": {"path": "app/.env"}}, policy)
    assert d.blocked
    assert d.reasons == ["secret file"]
    assert d.matched_rules == ["no-secrets"]


def test_ask_rule_requires_approval(policy):
    d = evaluate_action({"tool": "send_payment", "args": {"amount": 500}}, policy)
    assert d.verdict is Verdict.ASK
    assert d.needs_approval
    assert d.reasons == ["large payment"]


def test_conditional_allow(policy):
    assert evaluate_action({"tool": "send_payment", "args": {"amount": 20}}, policy).allowed


def test_default_applies_when_nothing_matches(policy):
    d = evaluate_action({"tool": "unknown_tool"}, policy)
    assert d.blocked
    assert "default" in d.reasons[0]


def test_default_can_be_ask():
    p = parse_policy({"default": "ask"})
    assert evaluate_action({"tool": "anything"}, p).needs_approval


def test_most_restrictive_verdict_wins():
    p = parse_policy(
        {
            "rules": [
                {"id": "a", "tools": ["t"], "action": "allow"},
                {"id": "b", "tools": ["t"], "action": "ask", "reason": "check"},
            ]
        }
    )
    d = evaluate_action({"tool": "t"}, p)
    assert d.verdict is Verdict.ASK
    assert d.matched_rules == ["b"]


def test_all_block_reasons_are_reported():
    p = parse_policy(
        {
            "rules": [
                {"id": "a", "tools": ["t"], "action": "block", "reason": "r1"},
                {"id": "b", "tools": ["t"], "action": "block", "reason": "r2"},
            ]
        }
    )
    assert evaluate_action({"tool": "t"}, p).reasons == ["r1", "r2"]


def test_missing_argument_does_not_match_condition(policy):
    # amount missing -> neither payment rule matches -> default block
    d = evaluate_action({"tool": "send_payment", "args": {}}, policy)
    assert d.blocked
    assert d.matched_rules == []


def test_incomparable_types_do_not_match(policy):
    d = evaluate_action({"tool": "send_payment", "args": {"amount": "lots"}}, policy)
    assert d.matched_rules == []


def test_nested_argument_paths():
    p = parse_policy(
        {
            "default": "allow",
            "rules": [
                {
                    "id": "no-prod",
                    "tools": ["http.*"],
                    "action": "block",
                    "when": [{"arg": "target.env", "op": "equals", "value": "prod"}],
                }
            ],
        }
    )
    assert evaluate_action({"tool": "http.post", "args": {"target": {"env": "prod"}}}, p).blocked
    assert evaluate_action({"tool": "http.post", "args": {"target": {"env": "dev"}}}, p).allowed


@pytest.mark.parametrize(
    "op,value,arg,expected",
    [
        ("equals", 5, 5, True),
        ("not_equals", 5, 6, True),
        ("contains", "rm", "sudo rm -rf", True),
        ("contains", "a", ["a", "b"], True),
        ("not_contains", "x", "abc", True),
        ("startswith", "/tmp", "/tmp/x", True),
        ("endswith", ".py", "x.py", True),
        ("matches", r"^\d+$", "123", True),
        ("in", ["a", "b"], "a", True),
        ("not_in", ["a", "b"], "c", True),
        ("gt", 10, 11, True),
        ("gte", 10, 10, True),
        ("lt", 10, 9, True),
        ("lte", 10, 11, False),
        ("exists", True, "anything", True),
    ],
)
def test_operators(op, value, arg, expected):
    p = parse_policy(
        {
            "default": "allow",
            "rules": [{"id": "r", "tools": ["t"], "action": "block", "when": [{"arg": "x", "op": op, "value": value}]}],
        }
    )
    assert evaluate_action({"tool": "t", "args": {"x": arg}}, p).blocked is expected


def test_exists_false_matches_missing_argument():
    p = parse_policy(
        {
            "default": "allow",
            "rules": [
                {"id": "r", "tools": ["t"], "action": "block", "when": [{"arg": "token", "op": "exists", "value": False}]}
            ],
        }
    )
    assert evaluate_action({"tool": "t"}, p).blocked
    assert evaluate_action({"tool": "t", "args": {"token": "x"}}, p).allowed


def test_decision_to_dict(policy):
    d = evaluate_action({"tool": "delete_all"}, policy)
    assert d.to_dict() == {"verdict": "block", "reasons": d.reasons, "matched_rules": ["denylist"]}
