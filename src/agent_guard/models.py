"""Core data types: actions, rules, policies and decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class Verdict(str, Enum):
    """The outcome of evaluating an action against a policy."""

    ALLOW = "allow"
    BLOCK = "block"
    ASK = "ask"  # pause and require human approval before executing

    @classmethod
    def parse(cls, value: Any) -> "Verdict":
        try:
            return cls(str(value).lower())
        except ValueError:
            valid = ", ".join(v.value for v in cls)
            raise ValueError(f"invalid verdict {value!r}; expected one of: {valid}") from None


# When several rules match, the most restrictive verdict wins.
SEVERITY = {Verdict.ALLOW: 0, Verdict.ASK: 1, Verdict.BLOCK: 2}


@dataclass(frozen=True)
class Action:
    """A tool call an agent wants to perform."""

    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, value: Any) -> "Action":
        if isinstance(value, Action):
            return value
        if isinstance(value, Mapping):
            if "tool" not in value:
                raise ValueError("action dict must contain a 'tool' key")
            return cls(tool=str(value["tool"]), args=dict(value.get("args") or {}))
        raise TypeError(f"cannot convert {type(value).__name__} to Action")


@dataclass(frozen=True)
class Condition:
    """A single test against an argument of the action, e.g. ``amount > 100``."""

    arg: str
    op: str
    value: Any = None


@dataclass(frozen=True)
class Rule:
    id: str
    tools: List[str]
    action: Verdict
    when: List[Condition] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class Policy:
    default: Verdict = Verdict.BLOCK
    allowlist: List[str] = field(default_factory=list)
    denylist: List[str] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    name: str = "policy"


@dataclass(frozen=True)
class Decision:
    """Result of :func:`agent_guard.evaluate_action`."""

    verdict: Verdict
    reasons: List[str]
    matched_rules: List[str] = field(default_factory=list)
    action: Optional[Action] = None

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    @property
    def blocked(self) -> bool:
        return self.verdict is Verdict.BLOCK

    @property
    def needs_approval(self) -> bool:
        return self.verdict is Verdict.ASK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "matched_rules": list(self.matched_rules),
        }
