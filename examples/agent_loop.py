"""A minimal agent loop with Agent Guard intercepting every tool call.

Run:  python examples/agent_loop.py
      python examples/agent_loop.py --unattended   # 'ask' actions are auto-denied
"""

import sys
from pathlib import Path

from agent_guard import ActionBlocked, Guard, console_approver, deny_all_approver

POLICY = Path(__file__).with_name("policy.yaml")


# --- the agent's tools -------------------------------------------------------
def read_file(path: str) -> str:
    return f"<contents of {path}>"


def send_payment(to: str, amount: float) -> str:
    return f"paid {amount} to {to}"


def send_email(to: str, body: str) -> str:
    return f"emailed {to}"


def delete_records(table: str) -> str:
    return f"deleted everything in {table}"


TOOLS = {f.__name__: f for f in (read_file, send_payment, send_email, delete_records)}


# --- a stand-in for the LLM: a scripted list of tool calls it "decides" to make
PLANNED_CALLS = [
    {"tool": "read_file", "args": {"path": "notes/todo.md"}},
    {"tool": "read_file", "args": {"path": "config/.env"}},
    {"tool": "send_payment", "args": {"to": "acme", "amount": 40}},
    {"tool": "send_payment", "args": {"to": "acme", "amount": 5000}},
    {"tool": "send_email", "args": {"to": "bob@mycompany.com", "body": "hi"}},
    {"tool": "delete_records", "args": {"table": "users"}},
]


def run_agent(guard: Guard) -> None:
    for call in PLANNED_CALLS:
        tool, args = call["tool"], call["args"]
        try:
            # The guard intercepts here - BEFORE the tool executes.
            result = guard.execute(tool, TOOLS[tool], args)
            observation = f"OK: {result}"
        except ActionBlocked as blocked:
            # Feed the refusal back to the model so it can re-plan.
            observation = f"BLOCKED: {'; '.join(blocked.decision.reasons)}"
        print(f"{tool}({args}) -> {observation}")


if __name__ == "__main__":
    approver = deny_all_approver if "--unattended" in sys.argv else console_approver
    run_agent(Guard(POLICY, approver=approver))
