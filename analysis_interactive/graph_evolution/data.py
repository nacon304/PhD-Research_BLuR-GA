from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SNAPSHOT_COLUMNS = {
    "generation",
    "feature_i",
    "feature_j",
    "weight",
    "positive_count",
    "tested_count",
    "first_seen_generation",
    "last_updated_generation",
}


@dataclass(frozen=True)
class GraphEvolutionData:
    snapshots: pd.DataFrame
    trace: pd.DataFrame | None
    n_nodes: int
    labels: dict[int, str]


def load_snapshots(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    df = pd.read_csv(path)
    missing = SNAPSHOT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Snapshot file is missing required columns: {sorted(missing)}")
    int_cols = [
        "generation",
        "feature_i",
        "feature_j",
        "positive_count",
        "tested_count",
        "first_seen_generation",
        "last_updated_generation",
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)
    df["weight"] = df["weight"].astype(float)
    if "weight_sum" in df.columns:
        df["weight_sum"] = df["weight_sum"].astype(float)
    return df.sort_values(["generation", "weight"], ascending=[True, False]).reset_index(drop=True)


def load_trace(path: str | Path | None, *, run_id: int | None = None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")
    df = pd.read_csv(path)
    if run_id is not None and "run_id" in df.columns:
        df = df[df["run_id"] == run_id].copy()
    if "generation" in df.columns:
        df["generation"] = df["generation"].astype(int)
    return df.reset_index(drop=True)


def infer_n_nodes(snapshots: pd.DataFrame, n_nodes: int | None = None) -> int:
    if n_nodes is not None:
        return int(n_nodes)
    if snapshots.empty:
        raise ValueError("Cannot infer n_nodes from an empty snapshot file. Pass --n-nodes.")
    return int(max(snapshots["feature_i"].max(), snapshots["feature_j"].max()) + 1)


def load_labels(path: str | Path | None, n_nodes: int) -> dict[int, str]:
    if path is None:
        return {i: str(i) for i in range(n_nodes)}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Node label file not found: {path}")
    df = pd.read_csv(path)
    if {"node_id", "label"}.issubset(df.columns):
        return {int(r.node_id): str(r.label) for r in df.itertuples(index=False)}
    if {"feature_index", "feature_name"}.issubset(df.columns):
        return {int(r.feature_index): str(r.feature_name) for r in df.itertuples(index=False)}
    raise ValueError("Node label CSV must contain either node_id,label or feature_index,feature_name columns.")


def load_graph_evolution(
    snapshots_path: str | Path,
    *,
    trace_path: str | Path | None = None,
    n_nodes: int | None = None,
    labels_path: str | Path | None = None,
    run_id: int | None = None,
) -> GraphEvolutionData:
    snapshots = load_snapshots(snapshots_path)
    trace = load_trace(trace_path, run_id=run_id)
    inferred = infer_n_nodes(snapshots, n_nodes=n_nodes)
    labels = load_labels(labels_path, inferred)
    return GraphEvolutionData(snapshots=snapshots, trace=trace, n_nodes=inferred, labels=labels)
