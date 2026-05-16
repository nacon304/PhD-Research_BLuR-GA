#!/usr/bin/env python3
from __future__ import annotations

"""Prepare synthetic linkage-correctness datasets for BLuR-GA.

Examples
--------
Debug/smoke set, one seed per family::

    python scripts/prepare_linkage_benchmarks.py \
        --output-root ../Dataset/prepared_blurga \
        --preset debug --seeds 0

Main scalable set::

    python scripts/prepare_linkage_benchmarks.py \
        --output-root ../Dataset/prepared_blurga \
        --preset main --seeds 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any

import pandas as pd

from blur_ga.benchmarks import build_problem, write_problem


# FAMILIES = ["pairwise", "ising", "planted_xor", "nk", "maxsat", "gametes"]
FAMILIES = ["pairwise", "ising", "nk", "maxsat"]


DEBUG_PRESET: dict[str, list[dict[str, Any]]] = {
    "pairwise": [{"n_features": 30, "n_edges": 30, "signed": True, "noise_std": 0.0}],
    "ising": [{"grid_rows": 5, "grid_cols": 5, "signed": True}],
    # "planted_xor": [{"n_features": 30, "n_samples": 800, "group_sizes": (3, 3), "label_noise": 0.03}],
    "nk": [{"n_features": 30, "k": 3, "neighborhood": "random"}],
    "maxsat": [{"n_features": 35, "n_clauses": 90, "clause_size": 3, "weighted": True}],
    # "gametes": [{"n_features": 30, "n_samples": 800, "n_loci": 2, "maf": 0.25, "label_noise": 0.02}],
}


MAIN_PRESET: dict[str, list[dict[str, Any]]] = {
    "pairwise": [
        {"n_features": 50, "n_edges": 75, "signed": True, "noise_std": 0.0},
        {"n_features": 100, "n_edges": 150, "signed": True, "noise_std": 0.0},
        {"n_features": 50, "n_edges": 75, "signed": True, "noise_std": 0.05},
    ],
    "ising": [
        {"grid_rows": 6, "grid_cols": 6, "signed": True},
        {"grid_rows": 10, "grid_cols": 10, "signed": True},
    ],
    # "planted_xor": [
    #     {"n_features": 50, "n_samples": 2000, "group_sizes": (3, 3), "label_noise": 0.03},
    #     {"n_features": 100, "n_samples": 3000, "group_sizes": (4, 4), "label_noise": 0.05},
    # ],
    "nk": [
        {"n_features": 50, "k": 3, "neighborhood": "random"},
        {"n_features": 80, "k": 5, "neighborhood": "random"},
    ],
    "maxsat": [
        {"n_features": 50, "n_clauses": 150, "clause_size": 3, "weighted": True},
        {"n_features": 100, "n_clauses": 300, "clause_size": 3, "weighted": True},
    ],
    # "gametes": [
    #     {"n_features": 50, "n_samples": 2000, "n_loci": 2, "maf": 0.25, "label_noise": 0.02},
    #     {"n_features": 100, "n_samples": 3000, "n_loci": 3, "maf": 0.25, "label_noise": 0.03},
    # ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BLuR-GA synthetic linkage benchmark datasets.")
    parser.add_argument("--output-root", required=True, help="Root folder to write prepared_blurga-style data.")
    parser.add_argument("--preset", choices=["debug", "main"], default="debug")
    parser.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES + ["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--manifest-name", default="manifest.csv")
    parser.add_argument("--overwrite-manifest", action="store_true", help="Replace an existing manifest instead of appending/deduplicating.")
    parser.add_argument("--summary-json", default=None, help="Optional path to write a small JSON run summary.")
    return parser.parse_args()


def selected_families(raw: list[str]) -> list[str]:
    if "all" in raw:
        return FAMILIES
    return raw


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    preset = DEBUG_PRESET if args.preset == "debug" else MAIN_PRESET
    rows: list[dict[str, Any]] = []

    for family in selected_families(args.families):
        for params in preset[family]:
            for seed in args.seeds:
                problem = build_problem(family, seed=int(seed), params=params)
                row = write_problem(problem, output_root, group_folder="linkage")
                rows.append(row)
                print(
                    f"prepared {row['dataset_key']} | family={row['problem_family']} | "
                    f"features={row['n_features']} | true_edges={row['n_true_edges']}"
                )

    manifest_path = output_root / args.manifest_name
    new_df = pd.DataFrame(rows)
    if manifest_path.exists() and not args.overwrite_manifest:
        old_df = pd.read_csv(manifest_path)
        all_df = pd.concat([old_df, new_df], ignore_index=True)
        all_df = all_df.drop_duplicates(subset=["dataset_key"], keep="last")
    else:
        all_df = new_df
    all_df = all_df.sort_values(["eval_group", "problem_family", "dataset_key"]).reset_index(drop=True)
    all_df.to_csv(manifest_path, index=False)
    print(f"wrote manifest: {manifest_path} ({len(all_df)} datasets total)")

    summary = {
        "output_root": str(output_root),
        "preset": args.preset,
        "families": selected_families(args.families),
        "seeds": args.seeds,
        "n_new_datasets": len(rows),
        "n_manifest_datasets": int(len(all_df)),
        "by_family": new_df.groupby("problem_family")["dataset_key"].count().to_dict() if not new_df.empty else {},
    }
    summary_path = Path(args.summary_json) if args.summary_json else output_root / "prepare_linkage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
