#!/usr/bin/env python3
"""
Run a small grid of Merlin configs sequentially and summarize their
final TEST metrics in a simple table.

Usage (from repo root):

    source .venv/bin/activate
    python3 run_merlin_grid.py

This script assumes:
  - Merlin Parquet files exist under tmp/merlin/{train,valid,test}.parquet
  - You have already run:  python3 preprocess_merlin_data.py
  - The configs in configs/merlin/*.gin are present.
"""

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ConfigResult:
    name: str
    config_path: Path
    ndcg5: Optional[float] = None
    ndcg10: Optional[float] = None
    ndcg50: Optional[float] = None
    hr5: Optional[float] = None
    hr10: Optional[float] = None
    hr50: Optional[float] = None
    mrr: Optional[float] = None
    eval_loss: Optional[float] = None
    elapsed_sec: Optional[float] = None
    failed: bool = False
    error: Optional[str] = None


TEST_LINE_RE = re.compile(
    r"NDCG@5\s+([0-9.]+),\s+NDCG@10\s+([0-9.]+),\s+NDCG@50\s+([0-9.]+),\s+"
    r"HR@5\s+([0-9.]+),\s+HR@10\s+([0-9.]+),\s+HR@50\s+([0-9.]+),\s+"
    r"MRR\s+([0-9.]+),\s+eval_loss\s+([0-9.]+)"
)


def run_config(config: ConfigResult) -> ConfigResult:
    cmd = [
        sys.executable,
        "main.py",
        "--gin_config_file",
        str(config.config_path),
        "--master_port",
        "12345",
    ]
    print(f"\n=== Running config: {config.name} ===")
    print("Command:", " ".join(cmd))

    start_time = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        config.failed = True
        config.error = f"Exception while starting: {e}"
        print(config.error)
        return config

    assert proc.stdout is not None
    lines: List[str] = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    config.elapsed_sec = time.time() - start_time

    output = "".join(lines)

    if proc.returncode != 0:
        config.failed = True
        config.error = f"Non-zero exit code: {proc.returncode}"
        return config

    # Parse the final TEST eval line.
    match = None
    for line in output.splitlines():
        if "TEST eval in" in line and "NDCG@10" in line:
            match = TEST_LINE_RE.search(line)

    if not match:
        config.failed = True
        config.error = "Could not find TEST eval line in output."
        return config

    # group(1) = ndcg@5, (2) = ndcg@10, (3) = ndcg@50,
    # group(4) = hr@5, (5) = hr@10, (6) = hr@50,
    # group(7) = mrr, (8) = eval_loss.
    config.ndcg5 = float(match.group(1))
    config.ndcg10 = float(match.group(2))
    config.ndcg50 = float(match.group(3))
    config.hr5 = float(match.group(4))
    config.hr10 = float(match.group(5))
    config.hr50 = float(match.group(6))
    config.mrr = float(match.group(7))
    config.eval_loss = float(match.group(8))
    return config


def format_float(x: Optional[float]) -> str:
    return f"{x:.4f}" if x is not None else "NA"


def format_time_minutes(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x / 60.0:.1f}"


def print_summary(results: List[ConfigResult]) -> None:
    print("\n=== Merlin config comparison (TEST metrics) ===")
    headers = [
        "Config",
        "HR@5",
        "HR@10",
        "NDCG@5",
        "NDCG@10",
        "MRR",
        "Time (min)",
        "Eval loss",
        "Status",
    ]
    # Ensure the first column is wide enough for long config names.
    col_widths = [max(len(headers[0]), 22), 8, 8, 8, 8, 10, 10, 10, 8]

    def fmt_row(cols: List[str]) -> str:
        return (
            cols[0].ljust(col_widths[0])
            + "  "
            + "  ".join(c.rjust(w) for c, w in zip(cols[1:], col_widths[1:]))
        )

    print(fmt_row(headers))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))

    # Choose best config by highest HR@5 (only among non-failed).
    best_idx: Optional[int] = None
    best_hr5: float = -1.0
    for i, r in enumerate(results):
        if r.failed or r.hr5 is None:
            continue
        if r.hr5 > best_hr5:
            best_hr5 = r.hr5
            best_idx = i

    # Sort rows by HR@5 (descending), keeping failed runs at the bottom.
    def sort_key(r: ConfigResult):
        # Failed runs or missing HR@5 get lowest priority.
        if r.failed or r.hr5 is None:
            return (1, 0.0)
        return (0, -r.hr5)

    sorted_results = sorted(results, key=sort_key)

    for i, r in enumerate(sorted_results):
        status = "OK" if not r.failed else "FAIL"
        # Mark the best config (by HR@5) with '*'.
        mark = "*" if best_idx is not None and r is results[best_idx] and not r.failed else " "
        row = [
            f"{mark} {r.name}",
            format_float(r.hr5),
            format_float(r.hr10),
            format_float(r.ndcg5),
            format_float(r.ndcg10),
            format_float(r.mrr),
            format_time_minutes(r.elapsed_sec),
            format_float(r.eval_loss),
            status,
        ]
        print(fmt_row(row))

    if best_idx is not None:
        best = results[best_idx]
        print(
            f"\nBest config by HR@5: {best.name} "
            f"(HR@5={format_float(best.hr5)}, MRR={format_float(best.mrr)})"
        )
    else:
        print("\nNo successful runs to select a best config from.")

    # Print any recorded errors for easier debugging.
    any_errors = False
    for r in results:
        if r.failed and r.error:
            if not any_errors:
                print("\nErrors:")
                any_errors = True
            print(f"- {r.name}: {r.error}")


def main() -> None:
    root = Path(__file__).resolve().parent
    merlin_configs: Dict[str, str] = {
        # SASRec variants
        # "sasrec_full_softmax_rated": "configs/merlin/sasrec-gpt-like-full-softmax-mini.gin",
        # "sasrec_full_softmax_no_ratings": "configs/merlin/sasrec-gpt-like-full-softmax-no-ratings-mini.gin",
        # "sasrec_sampled_softmax_rated": "configs/merlin/sasrec-gpt-like-sampled-softmax-mini.gin",
        "sasrec_full_softmax_rated_mol": "configs/merlin/sasrec-gpt-like-full-softmax-mol-mini.gin",
        # HSTU variants
        # "hstu_full_softmax_rated": "configs/merlin/hstu-gpt-like-full-softmax-mini.gin",
        # "hstu_full_softmax_no_ratings": "configs/merlin/hstu-gpt-like-full-softmax-no-ratings-mini.gin",
        # "hstu_sampled_softmax_rated": "configs/merlin/hstu-gpt-like-sampled-softmax-mini.gin",
        "hstu_full_softmax_rated_mol": "configs/merlin/hstu-gpt-like-full-softmax-mol-mini.gin",
    }

    results: List[ConfigResult] = []
    for name, rel_path in merlin_configs.items():
        config_path = root / rel_path
        if not config_path.exists():
            print(f"Skipping {name}: config not found at {config_path}")
            results.append(
                ConfigResult(
                    name=name,
                    config_path=config_path,
                    failed=True,
                    error="Config file missing",
                )
            )
            continue
        result = run_config(ConfigResult(name=name, config_path=config_path))
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
