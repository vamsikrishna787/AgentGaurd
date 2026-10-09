"""Agent Guard - a policy interceptor for AI agent tool calls.

Every tool call is evaluated before it runs and gets one of three verdicts:
``allow``, ``block``, or ``ask`` (pause for human approval).
"""

from .engine import evaluate_action
from .exceptions import ActionBlocked, AgentGuardError, PolicyError
from .guard import Guard, console_approver, deny_all_approver
from .models import Action, Condition, Decision, Policy, Rule, Verdict
from .policy import load_policy, parse_policy

__version__ = "0.1.0"

__all__ = [
    "Action",
    "ActionBlocked",
    "AgentGuardError",
    "Condition",
    "Decision",
    "Guard",
    "Policy",
    "PolicyError",
    "Rule",
    "Verdict",
    "console_approver",
    "deny_all_approver",
    "evaluate_action",
    "load_policy",
    "parse_policy",
]
