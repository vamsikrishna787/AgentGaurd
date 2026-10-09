"""Exceptions raised by Agent Guard."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Decision


class AgentGuardError(Exception):
    """Base class for all Agent Guard errors."""


class PolicyError(AgentGuardError):
    """The policy file or dict is malformed."""


class ActionBlocked(AgentGuardError):
    """Raised by the interceptor when an action is blocked or approval is denied."""

    def __init__(self, decision: "Decision"):
        self.decision = decision
        tool = decision.action.tool if decision.action else "<unknown>"
        super().__init__(f"action {tool!r} blocked: " + "; ".join(decision.reasons))
