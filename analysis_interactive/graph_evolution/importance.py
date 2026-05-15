from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RUN_KEY_COLUMNS = ["repeat_id", "outer_fold", "run_id"]
SCORE_CANDIDATES = [
    "outer_score",
    "best_inner_fitness",
    "best_so_far_fitness",
    "best_population_fitness",
    "fitness",
    "score",
]


def run_key_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in RUN_KEY_COLUMNS if c in df.columns]


def normalise_pair(i: int, j: int) -> tuple[int, int]:
    i, j = int(i), int(j)
    return (i, j) if i <= j else (j, i)


def auto_related_paths(snapshots_path: str | Path) -> dict[str, object]:
    """Discover common files next to graph_snapshots_<dataset>_cX_aY_rZ.csv."""
    p = Path(snapshots_path)
    name = p.name
    m = re.match(r"graph_snapshots_(.+)_r\d+\.csv$", name)
    if not m:
        return {"selected_features": None, "run_summary": None, "importance_snapshots": [p]}
    base = m.group(1)
    selected = p.with_name(f"selected_features_{base}.csv")
    summary = p.with_name(f"nested_summary_{base}.csv")
    snaps = sorted(p.parent.glob(f"graph_snapshots_{base}_r*.csv"))
    return {
        "selected_features": selected if selected.exists() else None,
        "run_summary": summary if summary.exists() else None,
        "importance_snapshots": snaps if snaps else [p],
    }


def load_run_summary(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Run-summary file not found: {p}")
    return pd.read_csv(p)


def choose_score_column(run_summary: pd.DataFrame | None, requested: str | None = None) -> str | None:
    if run_summary is None or run_summary.empty:
        return None
    if requested:
        if requested not in run_summary.columns:
            raise ValueError(f"Requested score column {requested!r} is not in run summary columns.")
        return requested
    for col in SCORE_CANDIDATES:
        if col in run_summary.columns:
            return col
    return None


def _make_run_id(df: pd.DataFrame) -> pd.Series:
    keys = run_key_columns(df)
    if keys:
        return df[keys].astype(str).agg("|".join, axis=1)
    return pd.Series(["run0"] * len(df), index=df.index)


def _score_table(run_summary: pd.DataFrame | None, score_column: str | None) -> pd.DataFrame:
    if run_summary is None or run_summary.empty or score_column is None:
        return pd.DataFrame(columns=["run_uid", "score"])
    df = run_summary.copy()
    df["run_uid"] = _make_run_id(df)
    df["score"] = pd.to_numeric(df[score_column], errors="coerce")
    df = df.dropna(subset=["score"])
    if df.empty:
        return pd.DataFrame(columns=["run_uid", "score"])
    return df[["run_uid", "score"]].drop_duplicates("run_uid", keep="last")


def compute_feature_importance(
    selected_features: pd.DataFrame | None,
    *,
    n_nodes: int,
    labels: dict[int, str] | None = None,
    run_summary: pd.DataFrame | None = None,
    score_column: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """
    Fitness-weighted feature-selection importance.

    Formula:
        FI_i = sum_r 1(i in S_r) * q_r / sum_r q_r

    q_r is the selected score column for run r. If no score is available, q_r=1.
    """
    labels = labels or {i: str(i) for i in range(n_nodes)}
    if selected_features is None or selected_features.empty or "feature_index" not in selected_features.columns:
        meta = {
            "available": False,
            "formula": "FI_i = sum_r 1(i in S_r) * score_r / sum_r score_r",
            "scoreColumn": score_column,
            "nRuns": 0,
        }
        return [], meta

    sel = selected_features.copy()
    sel["feature_index"] = sel["feature_index"].astype(int)
    sel["run_uid"] = _make_run_id(sel)

    scores = _score_table(run_summary, score_column)
    run_ids = sorted(set(sel["run_uid"].unique()) | set(scores["run_uid"].unique() if not scores.empty else []))
    if not run_ids:
        return [], {"available": False, "formula": "FI_i = sum_r 1(i in S_r) * score_r / sum_r score_r", "scoreColumn": score_column, "nRuns": 0}

    score_map = {r: 1.0 for r in run_ids}
    if not scores.empty:
        raw = dict(zip(scores["run_uid"], scores["score"]))
        # Use non-negative scores; if scores are all non-positive, fall back to uniform weights.
        vals = [float(raw.get(r, np.nan)) for r in run_ids if pd.notna(raw.get(r, np.nan))]
        if vals and max(vals) > 0:
            for r in run_ids:
                v = raw.get(r, np.nan)
                score_map[r] = max(float(v), 0.0) if pd.notna(v) else 1.0
    total_score = sum(score_map.values()) or float(len(run_ids))

    selected_by_run = {r: set(sel.loc[sel["run_uid"] == r, "feature_index"].astype(int)) for r in run_ids}
    rows: list[dict[str, object]] = []
    for i in range(n_nodes):
        selected_runs = [r for r in run_ids if i in selected_by_run.get(r, set())]
        weighted_sum = sum(score_map[r] for r in selected_runs)
        scores_when_selected = [score_map[r] for r in selected_runs]
        rows.append(
            {
                "feature": int(i),
                "label": labels.get(i, str(i)),
                "importance": float(weighted_sum / total_score) if total_score else 0.0,
                "selected_count": int(len(selected_runs)),
                "selection_rate": float(len(selected_runs) / len(run_ids)) if run_ids else 0.0,
                "weighted_selected_score": float(weighted_sum),
                "mean_score_when_selected": float(np.mean(scores_when_selected)) if scores_when_selected else 0.0,
            }
        )
    rows.sort(key=lambda r: (float(r["importance"]), int(r["selected_count"])), reverse=True)
    meta = {
        "available": True,
        "formula": "FI_i = sum_r 1(i in S_r) * score_r / sum_r score_r",
        "scoreColumn": score_column or "uniform",
        "nRuns": len(run_ids),
        "totalScore": float(total_score),
    }
    return rows, meta


def _final_edges_by_run(snapshot_paths: Iterable[str | Path]) -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    synthetic_idx = 0
    for path in snapshot_paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Snapshot file not found: {p}")
        df = pd.read_csv(p)
        if df.empty:
            continue
        for c in ["feature_i", "feature_j", "generation"]:
            if c not in df.columns:
                raise ValueError(f"{p} is missing required column {c!r}")
        if "weight" not in df.columns:
            raise ValueError(f"{p} is missing required column 'weight'")
        keys = run_key_columns(df)
        if not keys:
            df = df.copy()
            df["run_uid"] = f"file{synthetic_idx}"
            synthetic_idx += 1
            keys = ["run_uid"]
        for _, sub in df.groupby(keys, dropna=False, sort=False):
            final_gen = int(sub["generation"].max())
            final = sub[sub["generation"] == final_gen].copy()
            final["run_uid"] = _make_run_id(final) if "run_uid" not in final.columns else final["run_uid"].astype(str)
            final["source_file"] = p.name
            frames.append(final)
    if not frames:
        return pd.DataFrame(), 0
    all_edges = pd.concat(frames, ignore_index=True)
    n_runs = int(all_edges["run_uid"].nunique())
    return all_edges, n_runs


def compute_edge_importance(
    snapshot_paths: Iterable[str | Path],
    *,
    labels: dict[int, str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """
    Multi-run edge importance from final edge weights.

    Formula:
        EI_ij = (1/R) * sum_r w_ij,r / max_e(w_e,r)

    Missing edge in a run contributes 0. The table also reports raw occurrence
    rate and mean raw weight when the edge exists.
    """
    labels = labels or {}
    all_edges, n_runs = _final_edges_by_run(snapshot_paths)
    if all_edges.empty or n_runs == 0:
        meta = {
            "available": False,
            "formula": "EI_ij = (1/R) * sum_r w_ij,r / max_e(w_e,r)",
            "nRuns": 0,
        }
        return [], meta

    all_edges["feature_i"] = all_edges["feature_i"].astype(int)
    all_edges["feature_j"] = all_edges["feature_j"].astype(int)
    all_edges["weight"] = pd.to_numeric(all_edges["weight"], errors="coerce").fillna(0.0)
    pairs = all_edges.apply(lambda r: normalise_pair(r["feature_i"], r["feature_j"]), axis=1)
    all_edges["edge_key"] = [f"{i}-{j}" for i, j in pairs]
    all_edges["a"] = [i for i, _ in pairs]
    all_edges["b"] = [j for _, j in pairs]
    max_by_run = all_edges.groupby("run_uid")["weight"].max().replace(0, np.nan)
    all_edges["norm_weight"] = all_edges.apply(lambda r: float(r["weight"]) / float(max_by_run.loc[r["run_uid"]]) if pd.notna(max_by_run.loc[r["run_uid"]]) else 0.0, axis=1)

    rows: list[dict[str, object]] = []
    for edge_key, grp in all_edges.groupby("edge_key", sort=False):
        a = int(grp["a"].iloc[0]); b = int(grp["b"].iloc[0])
        occurrence = int(grp["run_uid"].nunique())
        rows.append(
            {
                "edge": edge_key,
                "i": a,
                "j": b,
                "label": f"{labels.get(a, str(a))} – {labels.get(b, str(b))}",
                "importance": float(grp["norm_weight"].sum() / n_runs),
                "occurrence_count": occurrence,
                "occurrence_rate": float(occurrence / n_runs),
                "mean_weight_present": float(grp["weight"].mean()),
                "max_weight": float(grp["weight"].max()),
                "sum_weight": float(grp["weight"].sum()),
                "positive_count_sum": int(grp["positive_count"].sum()) if "positive_count" in grp.columns else 0,
                "tested_count_sum": int(grp["tested_count"].sum()) if "tested_count" in grp.columns else 0,
            }
        )
    rows.sort(key=lambda r: (float(r["importance"]), float(r["occurrence_rate"]), float(r["mean_weight_present"])), reverse=True)
    meta = {
        "available": True,
        "formula": "EI_ij = (1/R) * sum_r w_ij,r / max_e(w_e,r); absent edge contributes 0",
        "nRuns": n_runs,
    }
    return rows, meta
