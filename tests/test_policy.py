import json
from pathlib import Path

import pytest

from agent_guard import PolicyError, Verdict, load_policy, parse_policy

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_load_example_yaml():
    p = load_policy(EXAMPLES / "policy.yaml")
    assert p.name == "example-agent-policy"
    assert p.default is Verdict.BLOCK
    assert "read_file" in p.allowlist
    assert "drop_database" in p.denylist
    assert any(r.id == "large-payment-needs-approval" and r.action is Verdict.ASK for r in p.rules)


def test_load_example_json():
    p = load_policy(EXAMPLES / "policy.json")
    assert p.default is Verdict.ASK
    assert len(p.rules) == 2


def test_load_json_from_tmp(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"default": "allow", "denylist": "rm"}))
    p = load_policy(f)
    assert p.default is Verdict.ALLOW
    assert p.denylist == ["rm"]
    assert p.name == "p"


def test_missing_file():
    with pytest.raises(PolicyError, match="not found"):
        load_policy("does/not/exist.yaml")


def test_unsupported_extension(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("{}")
    with pytest.raises(PolicyError, match="unsupported"):
        load_policy(f)


def test_invalid_json(tmp_path):
    f = tmp_path / "p.json"
    f.write_text("{not json")
    with pytest.raises(PolicyError, match="invalid JSON"):
        load_policy(f)


@pytest.mark.parametrize(
    "data,message",
    [
        ({"default": "maybe"}, "default"),
        ({"bogus": 1}, "unknown top-level"),
        ({"rules": {}}, "must be a list"),
        ({"rules": [{"id": "r", "action": "block"}]}, "'tools' is required"),
        ({"rules": [{"id": "r", "tools": ["t"]}]}, "'action' is required"),
        ({"rules": [{"id": "r", "tools": ["t"], "action": "nope"}]}, "invalid verdict"),
        ({"rules": [{"id": "r", "tools": ["t"], "action": "block", "when": [{"arg": "x", "op": "~="}]}]}, "unknown operator"),
        ({"rules": [{"id": "r", "tools": ["t"], "action": "block", "when": [{"arg": "x", "op": "in", "value": 1}]}]}, "list value"),
        ({"rules": [{"id": "r", "tools": ["t"], "action": "block", "when": [{"op": "equals"}]}]}, "'arg' and 'op'"),
        ({"rules": [{"id": "r", "tools": ["t"], "action": "allow"}, {"id": "r", "tools": ["u"], "action": "allow"}]}, "duplicate"),
        ({"allowlist": [1, 2]}, "allowlist"),
    ],
)
def test_validation_errors(data, message):
    with pytest.raises(PolicyError, match=message):
        parse_policy(data)


def test_rule_defaults():
    p = parse_policy({"rules": [{"tool": "t", "action": "ASK", "when": {"arg": "a", "op": "exists"}}]})
    rule = p.rules[0]
    assert rule.id == "rule-0"
    assert rule.tools == ["t"]
    assert rule.action is Verdict.ASK
    assert len(rule.when) == 1
