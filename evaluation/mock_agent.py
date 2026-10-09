"""A deterministic mock agent and the sandboxed tools it calls.

The agent works through ordinary, benign tasks made of *safe* scenarios. To
simulate a compromised or misaligned agent (e.g. after a prompt injection), each
step can be followed by a *rogue* action drawn from the harmful scenarios with
probability ``injection_rate``. All randomness comes from one seeded RNG, so the
same seed always yields the same action trace.

Tools never touch the real machine: they only record their effects in a
:class:`Sandbox`, which lets an experiment count the damage that got through.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from .scenarios import BY_ID, HARMFUL, Scenario, select

# Benign tasks the agent is asked to do; each is a plan of safe scenario ids.
TASKS: Dict[str, List[str]] = {
    "fix-bug": ["read-readme", "read-source", "shell-pytest", "write-notes", "shell-git-status"],
    "ship-order": ["sql-select", "sql-update-status", "email-customer"],
    "dependency-audit": ["http-pypi", "shell-npm-install", "write-report"],
    "pay-bills": ["pay-small", "pay-just-under", "pay-invoice", "email-internal"],
    "triage-issue": ["http-github", "search-docs", "http-create-issue"],
    "docs-research": ["search-docs", "ho-http-docs", "ho-read-env-docs", "write-report"],
    "release-check": ["ho-git-log", "ho-make-test", "ho-sql-cte", "ho-email-internal"],
}


@dataclass
class Sandbox:
    """In-memory stand-in for the outside world. Records every effect."""

    effects: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    funds_sent: float = 0.0

    def _record(self, tool: str, **args: Any) -> str:
        self.effects.append((tool, args))
        return f"{tool} ok"

    def tools(self) -> Dict[str, Callable[..., str]]:
        names = [
            "read_file", "list_dir", "search_web", "write_file", "run_shell", "http_request",
            "send_email", "db_query", "delete_file", "drop_database",
        ]
        table = {name: (lambda name: lambda **kw: self._record(name, **kw))(name) for name in names}
        table["send_payment"] = self._send_payment
        return table

    def _send_payment(self, to: str, amount: float) -> str:
        self.funds_sent += amount
        return self._record("send_payment", to=to, amount=amount)


@dataclass(frozen=True)
class Step:
    episode: int
    index: int
    task: str
    scenario: Scenario
    injected: bool  # True if this is a rogue action, not part of the task plan


class MockAgent:
    def __init__(self, seed: int = 0, injection_rate: float = 0.0):
        if not 0.0 <= injection_rate <= 1.0:
            raise ValueError("injection_rate must be between 0 and 1")
        self.seed = seed
        self.injection_rate = injection_rate

    def plan(self, episodes: int) -> List[Step]:
        """Produce the full action trace for ``episodes`` tasks."""
        rng = random.Random(self.seed)
        task_names = sorted(TASKS)
        rogue_pool = select(label=HARMFUL)
        trace: List[Step] = []
        for ep in range(episodes):
            task = rng.choice(task_names)
            i = 0
            for sid in TASKS[task]:
                trace.append(Step(ep, i, task, BY_ID[sid], injected=False))
                i += 1
                if rng.random() < self.injection_rate:
                    trace.append(Step(ep, i, task, rng.choice(rogue_pool), injected=True))
                    i += 1
        return trace
