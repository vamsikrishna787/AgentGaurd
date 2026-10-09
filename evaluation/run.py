"""Run every experiment and write results.

    python -m evaluation.run                          # reference configuration
    python -m evaluation.run --out results/mine --seed 1 --episodes 500

Outputs (in --out):
    benchmark.csv  one row per catalog scenario per condition      (experiment 1)
    workload.csv   one row per mock-agent step per condition        (experiment 2)
    latency.csv    decision latency per scenario                    (experiment 3)
    summary.json   config, environment, metrics, latency, metrics_sha256

Everything under ``metrics`` is deterministic for a given config and must be
identical on any machine; ``metrics_sha256`` fingerprints it. Latency and
environment naturally vary between machines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import agent_guard
from agent_guard import load_policy

from .harness import benchmark_latency, compute_metrics, default_conditions, run_steps, summarize_ns
from .mock_agent import MockAgent, Step
from .scenarios import CORE, HELDOUT, SCENARIOS

SCHEMA_VERSION = 1
DEFAULT_POLICY = Path(__file__).with_name("policies") / "eval_policy.yaml"


@dataclass(frozen=True)
class Config:
    seed: int = 7
    episodes: int = 300
    injection_rate: float = 0.2
    reviewer_error: float = 0.1
    latency_repeats: int = 200
    policy: str = str(DEFAULT_POLICY)


def metrics_fingerprint(metrics: Dict[str, Any]) -> str:
    canonical = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _environment(policy_path: Path) -> Dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "agent_guard_version": agent_guard.__version__,
        "git_commit": _git_commit(),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(config: Config, out_dir: Optional[Path] = None, latency: bool = True) -> Dict[str, Any]:
    policy_path = Path(config.policy)
    policy = load_policy(policy_path)
    conditions = default_conditions(config.reviewer_error)
    reviewer_seed = config.seed + 1

    # Experiment 1: every labeled scenario once, under every condition.
    bench_steps = [Step(0, i, "benchmark", s, injected=False) for i, s in enumerate(SCENARIOS)]
    bench_rows: List[Dict[str, Any]] = []
    bench_metrics: Dict[str, Any] = {}
    for cond in conditions:
        rows = run_steps(bench_steps, cond, policy, reviewer_seed)
        bench_rows += rows
        bench_metrics[cond.name] = {
            "core": compute_metrics(r for r in rows if r["split"] == CORE),
            "heldout": compute_metrics(r for r in rows if r["split"] == HELDOUT),
            "all": compute_metrics(rows),
        }

    # Experiment 2: the mock agent's trace, replayed identically under every condition.
    trace = MockAgent(seed=config.seed, injection_rate=config.injection_rate).plan(config.episodes)
    work_rows: List[Dict[str, Any]] = []
    work_metrics: Dict[str, Any] = {}
    step_latency: Dict[str, Any] = {}
    for cond in conditions:
        rows = run_steps(trace, cond, policy, reviewer_seed)
        work_rows += rows
        work_metrics[cond.name] = {
            "all": compute_metrics(rows),
            "task_actions": compute_metrics(r for r in rows if not r["injected"]),
            "injected_actions": compute_metrics(r for r in rows if r["injected"]),
        }
        step_latency[cond.name] = summarize_ns([r["step_ns"] for r in rows])

    metrics = {"benchmark": bench_metrics, "workload": work_metrics}
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": {**asdict(config), "policy": policy_path.name},
        "dataset": {
            "scenarios": len(SCENARIOS),
            "core": sum(s.split == CORE for s in SCENARIOS),
            "heldout": sum(s.split == HELDOUT for s in SCENARIOS),
            "workload_steps": len(trace),
            "workload_injected": sum(s.injected for s in trace),
        },
        "metrics": metrics,
        "metrics_sha256": metrics_fingerprint(metrics),
        "environment": _environment(policy_path),
        "latency": {"workload_step": step_latency},
    }

    lat_rows: List[Dict[str, Any]] = []
    if latency:
        bench = benchmark_latency(SCENARIOS, policy, repeats=config.latency_repeats)
        lat_rows = bench.pop("per_scenario")
        summary["latency"].update(bench)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(out_dir / "benchmark.csv", bench_rows)
        _write_csv(out_dir / "workload.csv", work_rows)
        _write_csv(out_dir / "latency.csv", lat_rows)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _print_table(summary: Dict[str, Any]) -> None:
    def pct(x):
        return "  n/a " if x is None else f"{100 * x:5.1f}%"

    for exp, splits in (("benchmark", ("core", "heldout")), ("workload", ("all",))):
        print(f"\n{exp}")
        print(f"  {'condition':<20} {'split':<8} {'block':>7} {'FP':>7} {'escal.':>7} {'harm ran':>9}")
        for cond, by_split in summary["metrics"][exp].items():
            for split in splits:
                m = by_split[split]
                print(
                    f"  {cond:<20} {split:<8} {pct(m['block_rate'])} {pct(m['false_positive_rate'])} "
                    f"{pct(m['escalation_rate'])} {m['harmful_executed']:>9}"
                )
    lat = summary["latency"]
    if "evaluate_action" in lat:
        e, g = lat["evaluate_action"], lat["guard_check"]
        print(f"\nlatency  evaluate_action p50 {e['p50_us']} us  p99 {e['p99_us']} us | "
              f"guard_check p50 {g['p50_us']} us  p99 {g['p99_us']} us")
    print(f"\nmetrics_sha256 {summary['metrics_sha256']}")


def main(argv: Optional[List[str]] = None) -> int:
    d = Config()
    p = argparse.ArgumentParser(description="Agent Guard evaluation")
    p.add_argument("--out", default="results/latest", help="output directory")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--episodes", type=int, default=d.episodes)
    p.add_argument("--injection-rate", type=float, default=d.injection_rate)
    p.add_argument("--reviewer-error", type=float, default=d.reviewer_error)
    p.add_argument("--latency-repeats", type=int, default=d.latency_repeats)
    p.add_argument("--policy", default=d.policy)
    p.add_argument("--skip-latency", action="store_true", help="skip the latency micro-benchmark")
    a = p.parse_args(argv)

    config = Config(a.seed, a.episodes, a.injection_rate, a.reviewer_error, a.latency_repeats, a.policy)
    summary = run_experiments(config, Path(a.out), latency=not a.skip_latency)
    _print_table(summary)
    print(f"results written to {a.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
