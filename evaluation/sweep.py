"""Repeat the workload experiment over several seeds to show run-to-run variation.

    python -m evaluation.sweep --seeds 1 2 3 4 5 6 7 8 9 10 --out results/reference/seed_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .run import Config, run_experiments

FIELDS = ("block_rate", "false_positive_rate", "escalation_rate", "harmful_executed")


def sweep(seeds: Sequence[int], base: Config = Config()) -> List[Dict[str, object]]:
    rows = []
    for seed in seeds:
        summary = run_experiments(replace(base, seed=seed), latency=False)
        for cond, by_split in summary["metrics"]["workload"].items():
            m = by_split["all"]
            rows.append({"seed": seed, "condition": cond, "n_actions": m["n_actions"], **{f: m[f] for f in FIELDS}})
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Seed sweep for the workload experiment")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--out", default="results/latest/seed_sweep.csv")
    a = p.parse_args(argv)

    rows = sweep(a.seeds)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'condition':<20} {'block mean±sd':>16} {'FP mean±sd':>16}")
    for cond in dict.fromkeys(r["condition"] for r in rows):
        sub = [r for r in rows if r["condition"] == cond]
        b = [r["block_rate"] for r in sub]
        fp = [r["false_positive_rate"] for r in sub]
        sd = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0  # noqa: E731
        print(f"{cond:<20} {statistics.mean(b):8.3f}±{sd(b):.3f}   {statistics.mean(fp):8.3f}±{sd(fp):.3f}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
