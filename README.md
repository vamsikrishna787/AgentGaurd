# Agent Guard

**Agent Guard** is a small Python library that intercepts an AI agent's tool calls *before they execute* and decides, from a declarative policy, whether each one is:

| Verdict | Meaning |
|---------|---------|
| `allow` | The tool runs. |
| `block` | The tool never runs; the agent gets the reasons back. |
| `ask`   | Execution pauses until a human approves or denies it. |

Policies are YAML or JSON and support an **allowlist**, a **denylist**, and **conditional rules** on tool arguments.

> See [`docs/architecture.html`](docs/architecture.html) for an animated architecture diagram (open it in a browser).

---

## Install

```bash
pip install -e .            # from a clone
pip install -e ".[dev]"     # with pytest
```

Requires Python 3.8+ and PyYAML (JSON policies work with the standard library alone).

## Policy format

```yaml
version: 1
name: my-agent-policy
default: block              # verdict when nothing matches: allow | block | ask

allowlist:                  # always allowed, unless a rule blocks/asks
  - read_file
  - search_*                # shell-style globs

denylist:                   # always blocked - nothing overrides this
  - delete_*
  - shell.exec

rules:
  - id: no-secret-files
    tools: [read_file]
    action: block
    when:
      - { arg: path, op: matches, value: '(\.env|id_rsa)$' }
    reason: Reading secret files is not allowed

  - id: large-payment-needs-approval
    tools: [send_payment]
    action: ask
    when:
      - { arg: amount, op: gt, value: 100 }
    reason: Payments over 100 require human approval
```

The same structure works as JSON - see [`examples/policy.json`](examples/policy.json).

### How a decision is made

1. **Denylist** - if the tool matches, it is blocked immediately.
2. **Rules + allowlist** - every rule whose `tools` match *and* whose `when` conditions all hold contributes its verdict; an allowlist match contributes `allow`. **The most restrictive verdict wins**: `block` > `ask` > `allow`. All reasons at that level are returned.
3. **Default** - used when nothing matched.

### Condition operators

`equals`, `not_equals`, `contains`, `not_contains`, `startswith`, `endswith`, `matches` (regex search), `in`, `not_in`, `gt`, `gte`, `lt`, `lte`, `exists`.

- `arg` supports dotted paths into nested arguments: `arg: target.env`.
- A condition on a missing argument, or with incomparable types (`"abc" > 5`), **does not match**. If you rely on a conditional `allow`, keep `default: block` so malformed calls fall through to a block.

## Quick evaluation

```python
from agent_guard import load_policy, evaluate_action

policy = load_policy("examples/policy.yaml")

decision = evaluate_action({"tool": "send_payment", "args": {"amount": 5000}}, policy)
decision.verdict        # Verdict.ASK
decision.reasons        # ['Payments over 100 require human approval']
decision.matched_rules  # ['large-payment-needs-approval']
decision.allowed, decision.blocked, decision.needs_approval
```

`evaluate_action` is pure: it never runs anything or prompts anyone.

## Integrating into an agent loop (the interceptor)

`Guard` sits between your agent and its tools. It evaluates each call, routes `ask` verdicts to a human approver, and only executes the tool if the final verdict is `allow`. Otherwise it raises `ActionBlocked` - **the tool function is never called.**

```python
from agent_guard import Guard, ActionBlocked, console_approver

guard = Guard("examples/policy.yaml", approver=console_approver)

TOOLS = {"read_file": read_file, "send_payment": send_payment}

while not done:
    tool_call = llm.next_tool_call(messages)           # e.g. {"tool": ..., "args": {...}}
    name, args = tool_call["tool"], tool_call["args"]

    try:
        # Intercept BEFORE execution
        result = guard.execute(name, TOOLS[name], args)
        observation = str(result)
    except ActionBlocked as blocked:
        # Tell the model why, so it can re-plan
        observation = "Action blocked: " + "; ".join(blocked.decision.reasons)

    messages.append({"role": "tool", "content": observation})
```

### Decorator style

```python
guard = Guard("policy.yaml", approver=console_approver)

@guard.tool()                    # tool name defaults to the function name
def send_payment(to: str, amount: float): ...

@guard.tool("fs.delete")         # or give it an explicit name
def remove(path: str): ...

send_payment("acme", 50)         # allowed -> runs
send_payment("acme", 5000)       # ask -> prompts the human
remove("/etc")                   # raises ActionBlocked if denied
```

Positional and keyword arguments are both bound to parameter names before evaluation, so policies always see `{"to": ..., "amount": ...}`.

### Human approval

An approver is any callable `(Decision) -> bool`:

```python
def slack_approver(decision):
    return ask_on_slack(f"Allow {decision.action.tool}? {decision.reasons}")  # your code

guard = Guard(policy, approver=slack_approver)
```

Built-ins: `console_approver` (terminal y/N prompt) and `deny_all_approver` (unattended runs).

Agent Guard **fails closed**: if no approver is configured, the approver says no, or the approver raises, the action is blocked.

### Auditing

```python
guard = Guard(policy, on_decision=lambda d: logger.info(d.to_dict()))
guard.audit_log   # list of every final Decision
```

## Example

```bash
python examples/agent_loop.py                # prompts you for 'ask' actions
python examples/agent_loop.py --unattended   # auto-denies them
```

```
read_file({'path': 'notes/todo.md'}) -> OK: <contents of notes/todo.md>
read_file({'path': 'config/.env'}) -> BLOCKED: Reading secret or credential files is not allowed
send_payment({'to': 'acme', 'amount': 40}) -> OK: paid 40 to acme
send_payment({'to': 'acme', 'amount': 5000}) -> BLOCKED: Payments over 100 require human approval; denied by human
send_email({'to': 'bob@mycompany.com', 'body': 'hi'}) -> OK: emailed bob@mycompany.com
delete_records({'table': 'users'}) -> BLOCKED: tool 'delete_records' is on the denylist ('delete_*')
```

## Tests

```bash
pytest
```

## Project layout

```
src/agent_guard/
  models.py      Action, Rule, Policy, Decision, Verdict
  policy.py      load_policy / parse_policy (YAML, JSON, dict) + validation
  engine.py      evaluate_action - the pure decision function
  guard.py       Guard interceptor, approvers, decorator
  exceptions.py  PolicyError, ActionBlocked
examples/        policy.yaml, policy.json, agent_loop.py
evaluation/      reproducible evaluation framework (scenarios, mock agent, harness, report)
results/         reference/ - committed results of the reference evaluation run
tests/           pytest suite (tests/eval/ verifies the evaluation)
docs/            architecture.html (animated diagram)
```

## Evaluation

A reproducible evaluation compares the same mock agent **with and without** the interceptor
on 61 labeled actions and a 1,383-step simulated workload, measuring block rate, false
positives, human-escalation rate and latency. Headline: 100% of anticipated harmful actions
blocked, but only 40% of held-out evasions; ~0.07 ms median overhead per action.
See the [methods and results report](evaluation/REPORT.md).

```bash
python -m evaluation.run --out results/latest   # writes CSV + JSON results
pytest tests/eval                                # re-verifies the published numbers
```
