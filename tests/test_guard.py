import pytest

from agent_guard import ActionBlocked, Guard, Verdict, deny_all_approver

POLICY = {
    "default": "block",
    "allowlist": ["read_file"],
    "denylist": ["delete_*"],
    "rules": [
        {
            "id": "big-payment",
            "tools": ["send_payment"],
            "action": "ask",
            "when": [{"arg": "amount", "op": "gt", "value": 100}],
            "reason": "large payment",
        },
        {
            "id": "small-payment",
            "tools": ["send_payment"],
            "action": "allow",
            "when": [{"arg": "amount", "op": "lte", "value": 100}],
        },
    ],
}


class Spy:
    """A fake tool that records whether it actually ran."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return "done"


def test_execute_runs_allowed_action():
    tool = Spy()
    assert Guard(POLICY).execute("read_file", tool, {"path": "a.txt"}) == "done"
    assert tool.calls == [{"path": "a.txt"}]


def test_execute_never_runs_blocked_action():
    tool = Spy()
    with pytest.raises(ActionBlocked) as exc:
        Guard(POLICY).execute("delete_db", tool, {})
    assert tool.calls == []
    assert exc.value.decision.blocked
    assert "delete_db" in str(exc.value)


def test_ask_without_approver_fails_closed():
    tool = Spy()
    with pytest.raises(ActionBlocked) as exc:
        Guard(POLICY).execute("send_payment", tool, {"amount": 500})
    assert tool.calls == []
    assert "no approver" in exc.value.decision.reasons[-1]


def test_ask_approved_by_human_runs():
    seen = []

    def approver(decision):
        seen.append(decision)
        return True

    tool = Spy()
    Guard(POLICY, approver=approver).execute("send_payment", tool, {"amount": 500})
    assert tool.calls == [{"amount": 500}]
    assert seen[0].verdict is Verdict.ASK
    assert seen[0].reasons == ["large payment"]


def test_ask_denied_by_human_blocks():
    tool = Spy()
    guard = Guard(POLICY, approver=deny_all_approver)
    with pytest.raises(ActionBlocked) as exc:
        guard.execute("send_payment", tool, {"amount": 500})
    assert tool.calls == []
    assert exc.value.decision.reasons[-1] == "denied by human"


def test_approver_exception_fails_closed():
    def broken(_):
        raise RuntimeError("UI crashed")

    with pytest.raises(ActionBlocked) as exc:
        Guard(POLICY, approver=broken).execute("send_payment", Spy(), {"amount": 500})
    assert "RuntimeError" in exc.value.decision.reasons[-1]


def test_approver_not_called_for_allow_or_block():
    calls = []
    guard = Guard(POLICY, approver=lambda d: calls.append(d) or True)
    guard.check({"tool": "read_file"})
    guard.check({"tool": "delete_x"})
    assert calls == []


def test_decorator_intercepts_positional_and_keyword_args():
    guard = Guard(POLICY)
    ran = []

    @guard.tool()
    def send_payment(to, amount=0):
        ran.append((to, amount))
        return "paid"

    assert send_payment("acme", 50) == "paid"
    assert send_payment(to="acme", amount=10) == "paid"
    with pytest.raises(ActionBlocked):
        send_payment("acme", amount=1000)
    assert ran == [("acme", 50), ("acme", 10)]


def test_decorator_with_custom_name():
    guard = Guard(POLICY)

    @guard.tool("delete_everything")
    def harmless():
        return "ok"

    with pytest.raises(ActionBlocked):
        harmless()


def test_audit_log_and_callback():
    events = []
    guard = Guard(POLICY, on_decision=events.append)
    guard.check({"tool": "read_file"})
    guard.check({"tool": "delete_x"})
    assert [d.verdict for d in guard.audit_log] == [Verdict.ALLOW, Verdict.BLOCK]
    assert events == guard.audit_log


def test_guard_accepts_policy_file(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('{"default": "allow"}')
    assert Guard(f).check({"tool": "x"}).allowed
