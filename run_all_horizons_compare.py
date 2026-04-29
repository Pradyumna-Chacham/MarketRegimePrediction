"""
Run baselines, fair flattened baseline, and GCN for horizons 1, 3, 5.
Writes per-horizon CSV files and final comparison CSV files.

Usage:
    python run_all_horizons_compare.py

Optional:
    python run_all_horizons_compare.py --horizons 1 3 5 --results_dir results_horizon_runs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--results_dir", type=str, default="results_horizon_runs")
    parser.add_argument("--skip_existing", action="store_true", help="Skip a script if its expected CSV already exists")
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


JOBS = [
    ("baselines", "baselinetp_cli.py", "baselines_h{h}.csv"),
    ("flattened", "flattened_node_fair_cli.py", "flattened_fair_h{h}.csv"),
    ("gcn", "gcn_attention_cli.py", "gcn_attention_h{h}.csv"),
]


def run_command(cmd: List[str], log_path: Path) -> None:
    print("\n" + "=" * 90)
    print("RUN:", " ".join(cmd))
    print("LOG:", log_path)
    print("=" * 90)
    with log_path.open("w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    logs_dir = results_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    for h in args.horizons:
        for job_name, script, csv_template in JOBS:
            csv_path = results_dir / csv_template.format(h=h)
            if args.skip_existing and csv_path.exists():
                print(f"Skipping existing: {csv_path}")
                continue
            cmd = [args.python, script, "--horizon", str(h), "--results_dir", str(results_dir)]
            log_path = logs_dir / f"{job_name}_h{h}.log"
            run_command(cmd, log_path)

    csv_files = []
    for h in args.horizons:
        for _, _, csv_template in JOBS:
            p = results_dir / csv_template.format(h=h)
            if p.exists():
                csv_files.append(p)
            else:
                print(f"WARNING: missing expected CSV: {p}")

    if not csv_files:
        raise RuntimeError("No result CSV files found.")

    all_results = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
    all_path = results_dir / "final_horizon_comparison.csv"
    all_results.to_csv(all_path, index=False)

    test_only = all_results[all_results["split"].str.lower() == "test"].copy()
    test_only = test_only.sort_values(["horizon", "macro_f1"], ascending=[True, False])
    test_path = results_dir / "final_test_only_ranking.csv"
    test_only.to_csv(test_path, index=False)

    pivot = test_only.pivot_table(index="model", columns="horizon", values="macro_f1", aggfunc="max")
    pivot_path = results_dir / "final_test_macro_f1_pivot.csv"
    pivot.to_csv(pivot_path)

    print("\nDONE")
    print(f"Saved all results: {all_path}")
    print(f"Saved test ranking: {test_path}")
    print(f"Saved macro-F1 pivot: {pivot_path}")
    print("\nTest-only ranking:")
    print(test_only[["horizon", "model", "accuracy", "macro_f1", "weighted_f1", "f1_calm", "f1_normal", "f1_turbulent"]].to_string(index=False))


if __name__ == "__main__":
    main()
