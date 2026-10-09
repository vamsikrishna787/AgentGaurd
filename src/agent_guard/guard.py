"""The interceptor: sits between the agent and its tools and enforces the policy."""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, TypeVar, Union

from .engine import evaluate_action
from .exceptions import ActionBlocked
from .models import Action, Decision, Policy, Verdict
from .policy import load_policy

Approver = Callable[[Decision], bool]
F = TypeVar("F", bound=Callable[..., Any])


def console_approver(decision: Decision) -> bool:
    """Ask a human on the terminal whether a flagged action may run."""
    action = decision.action
    print("\n[agent-guard] Human approval required")
    print(f"  tool:    {action.tool if action else '?'}")
    print(f"  args:    {dict(action.args) if action else {}}")
    print(f"  reasons: {'; '.join(decision.reasons)}")
    answer = input("  Allow this action? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def deny_all_approver(decision: Decision) -> bool:
    """Approver for unattended runs: anything that needs approval is refused."""
    return False


class Guard:
    """Intercepts tool calls, evaluates them, and only lets allowed ones execute.

    ``ask`` decisions are routed to ``approver`` (a callable returning ``True`` to
    approve). Without an approver, ``ask`` is treated as ``block`` - fail closed.
    """

    def __init__(
        self,
        policy: Union[Policy, str, Path, Mapping[str, Any]],
        approver: Optional[Approver] = None,
        on_decision: Optional[Callable[[Decision], None]] = None,
    ):
        self.policy = policy if isinstance(policy, Policy) else load_policy(policy)
        self.approver = approver
        self.on_decision = on_decision
        self.audit_log: List[Decision] = []

    def check(self, action: Union[Action, Mapping[str, Any]]) -> Decision:
        """Evaluate an action and resolve any ``ask`` into a final allow/block."""
        decision = evaluate_action(action, self.policy)
        if decision.verdict is Verdict.ASK:
            decision = self._resolve_approval(decision)
        self.audit_log.append(decision)
        if self.on_decision:
            self.on_decision(decision)
        return decision

    def execute(self, tool: str, fn: Callable[..., Any], args: Optional[Dict[str, Any]] = None) -> Any:
        """Check ``tool(args)`` and run ``fn(**args)`` only if allowed; raise otherwise."""
        args = dict(args or {})
        decision = self.check(Action(tool=tool, args=args))
        if not decision.allowed:
            raise ActionBlocked(decision)
        return fn(**args)

    def wrap(self, fn: F, name: Optional[str] = None) -> F:
        """Return a guarded version of ``fn``; its arguments are checked by name."""
        tool_name = name or fn.__name__
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            decision = self.check(Action(tool=tool_name, args=dict(bound.arguments)))
            if not decision.allowed:
                raise ActionBlocked(decision)
            return fn(*bound.args, **bound.kwargs)

        guarded.__agent_guard__ = self  # type: ignore[attr-defined]
        return guarded  # type: ignore[return-value]

    def tool(self, name: Optional[str] = None) -> Callable[[F], F]:
        """Decorator form of :meth:`wrap`: ``@guard.tool()`` or ``@guard.tool("fs.read")``."""

        def decorator(fn: F) -> F:
            return self.wrap(fn, name=name)

        return decorator

    def _resolve_approval(self, decision: Decision) -> Decision:
        if self.approver is None:
            return Decision(
                verdict=Verdict.BLOCK,
                reasons=decision.reasons + ["human approval required but no approver is configured"],
                matched_rules=decision.matched_rules,
                action=decision.action,
            )
        try:
            approved = bool(self.approver(decision))
        except Exception as exc:  # an approver failure must never let an action through
            approved = False
            note = f"approver raised {type(exc).__name__}: {exc}"
        else:
            note = "approved by human" if approved else "denied by human"
        return Decision(
            verdict=Verdict.ALLOW if approved else Verdict.BLOCK,
            reasons=decision.reasons + [note],
            matched_rules=decision.matched_rules,
            action=decision.action,
        )
