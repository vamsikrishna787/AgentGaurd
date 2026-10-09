"""Policy evaluation: decides whether an action is allowed, blocked or needs approval."""

from __future__ import annotations

import fnmatch
import re
from typing import Any, List, Mapping, Tuple, Union

from .models import SEVERITY, Action, Condition, Decision, Policy, Rule, Verdict

_MISSING = object()


def evaluate_action(action: Union[Action, Mapping[str, Any]], policy: Policy) -> Decision:
    """Evaluate a tool call against a policy.

    Order of evaluation:

    1. **denylist** - a matching tool is blocked immediately; nothing can override it.
    2. **rules** and **allowlist** - every matching rule (and the allowlist, which acts
       like an unconditional ``allow`` rule) contributes a verdict. The most restrictive
       one wins: ``block`` > ``ask`` > ``allow``.
    3. **default** - used when nothing matched.
    """
    action = Action.coerce(action)

    for pattern in policy.denylist:
        if _tool_matches(action.tool, pattern):
            return Decision(
                verdict=Verdict.BLOCK,
                reasons=[f"tool {action.tool!r} is on the denylist ({pattern!r})"],
                matched_rules=["denylist"],
                action=action,
            )

    hits: List[Tuple[Verdict, str, str]] = []  # (verdict, rule id, reason)

    for rule in policy.rules:
        if _rule_matches(rule, action):
            reason = rule.reason or f"matched rule {rule.id!r}"
            hits.append((rule.action, rule.id, reason))

    for pattern in policy.allowlist:
        if _tool_matches(action.tool, pattern):
            hits.append((Verdict.ALLOW, "allowlist", f"tool {action.tool!r} is on the allowlist ({pattern!r})"))
            break

    if not hits:
        return Decision(
            verdict=policy.default,
            reasons=[f"no rule matched; default is {policy.default.value!r}"],
            matched_rules=[],
            action=action,
        )

    verdict = max((h[0] for h in hits), key=SEVERITY.__getitem__)
    winning = [h for h in hits if h[0] is verdict]
    return Decision(
        verdict=verdict,
        reasons=[h[2] for h in winning],
        matched_rules=[h[1] for h in winning],
        action=action,
    )


def _tool_matches(tool: str, pattern: str) -> bool:
    """Tool names match exactly or by shell-style glob (``fs.*``, ``delete_*``)."""
    return fnmatch.fnmatchcase(tool, pattern)


def _rule_matches(rule: Rule, action: Action) -> bool:
    if not any(_tool_matches(action.tool, p) for p in rule.tools):
        return False
    return all(_condition_matches(c, action.args) for c in rule.when)


def _lookup(args: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted path like ``headers.host`` inside nested dicts."""
    current: Any = args
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def _condition_matches(cond: Condition, args: Mapping[str, Any]) -> bool:
    actual = _lookup(args, cond.arg)
    expected = cond.value

    if cond.op == "exists":
        return (actual is not _MISSING) == (expected is None or bool(expected))
    if actual is _MISSING:
        return False

    try:
        if cond.op == "equals":
            return actual == expected
        if cond.op == "not_equals":
            return actual != expected
        if cond.op == "contains":
            return expected in actual
        if cond.op == "not_contains":
            return expected not in actual
        if cond.op == "startswith":
            return str(actual).startswith(str(expected))
        if cond.op == "endswith":
            return str(actual).endswith(str(expected))
        if cond.op == "matches":
            return re.search(str(expected), str(actual)) is not None
        if cond.op == "in":
            return actual in expected
        if cond.op == "not_in":
            return actual not in expected
        if cond.op == "gt":
            return actual > expected
        if cond.op == "gte":
            return actual >= expected
        if cond.op == "lt":
            return actual < expected
        if cond.op == "lte":
            return actual <= expected
    except TypeError:
        # Incomparable types (e.g. "abc" > 5) simply don't match.
        return False
    raise ValueError(f"unknown operator {cond.op!r}")  # pragma: no cover - validated at load
