"""Loading and validating policies from YAML, JSON or plain dicts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Union

from .exceptions import PolicyError
from .models import Condition, Policy, Rule, Verdict

OPERATORS = {
    "equals",
    "not_equals",
    "contains",
    "not_contains",
    "startswith",
    "endswith",
    "matches",
    "in",
    "not_in",
    "gt",
    "gte",
    "lt",
    "lte",
    "exists",
}


def load_policy(source: Union[str, Path, Mapping[str, Any]]) -> Policy:
    """Build a :class:`Policy` from a ``.yaml``/``.yml``/``.json`` file path or a dict."""
    if isinstance(source, Mapping):
        return parse_policy(source)

    path = Path(source)
    if not path.is_file():
        raise PolicyError(f"policy file not found: {path}")
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PolicyError(f"invalid JSON in {path}: {exc}") from exc
    elif path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise PolicyError("PyYAML is required for YAML policies: pip install pyyaml") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PolicyError(f"invalid YAML in {path}: {exc}") from exc
    else:
        raise PolicyError(f"unsupported policy file type: {path.suffix!r} (use .yaml, .yml or .json)")

    if not isinstance(data, Mapping):
        raise PolicyError(f"policy in {path} must be a mapping at the top level")
    return parse_policy(data, name=path.stem)


def parse_policy(data: Mapping[str, Any], name: str = "policy") -> Policy:
    unknown = set(data) - {"version", "name", "default", "allowlist", "denylist", "rules"}
    if unknown:
        raise PolicyError(f"unknown top-level policy keys: {sorted(unknown)}")

    try:
        default = Verdict.parse(data.get("default", "block"))
    except ValueError as exc:
        raise PolicyError(f"default: {exc}") from exc

    rules_data = data.get("rules")
    if rules_data is None:
        rules_data = []
    if not isinstance(rules_data, list):
        raise PolicyError("'rules' must be a list")

    rules = [_parse_rule(r, i) for i, r in enumerate(rules_data)]
    ids = [r.id for r in rules]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise PolicyError(f"duplicate rule ids: {sorted(duplicates)}")

    return Policy(
        default=default,
        allowlist=_str_list(data.get("allowlist"), "allowlist"),
        denylist=_str_list(data.get("denylist"), "denylist"),
        rules=rules,
        name=str(data.get("name", name)),
    )


def _parse_rule(data: Any, index: int) -> Rule:
    where = f"rules[{index}]"
    if not isinstance(data, Mapping):
        raise PolicyError(f"{where} must be a mapping")

    rule_id = str(data.get("id") or f"rule-{index}")
    where = f"rule {rule_id!r}"

    tools = data.get("tools", data.get("tool"))
    if tools is None:
        raise PolicyError(f"{where}: 'tools' is required")
    tools = _str_list(tools, f"{where}.tools")
    if not tools:
        raise PolicyError(f"{where}: 'tools' must not be empty")

    if "action" not in data:
        raise PolicyError(f"{where}: 'action' is required (allow, block or ask)")
    try:
        verdict = Verdict.parse(data["action"])
    except ValueError as exc:
        raise PolicyError(f"{where}: {exc}") from exc

    when_data = data.get("when") or []
    if isinstance(when_data, Mapping):
        when_data = [when_data]
    if not isinstance(when_data, list):
        raise PolicyError(f"{where}: 'when' must be a list of conditions")

    conditions = []
    for c in when_data:
        if not isinstance(c, Mapping) or "arg" not in c or "op" not in c:
            raise PolicyError(f"{where}: each condition needs 'arg' and 'op'")
        op = str(c["op"])
        if op not in OPERATORS:
            raise PolicyError(f"{where}: unknown operator {op!r}; valid: {sorted(OPERATORS)}")
        if op in ("in", "not_in") and not isinstance(c.get("value"), list):
            raise PolicyError(f"{where}: operator {op!r} requires a list value")
        conditions.append(Condition(arg=str(c["arg"]), op=op, value=c.get("value")))

    return Rule(
        id=rule_id,
        tools=tools,
        action=verdict,
        when=conditions,
        reason=str(data.get("reason", "")),
    )


def _str_list(value: Any, field_name: str) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise PolicyError(f"'{field_name}' must be a string or list of strings")
