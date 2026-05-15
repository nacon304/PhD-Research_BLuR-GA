from __future__ import annotations

import pandas as pd


def edges_for_generation(snapshots: pd.DataFrame, generation: int) -> pd.DataFrame:
    """Return the latest available cumulative snapshot at or before generation."""
    available = snapshots.loc[snapshots["generation"] <= generation, "generation"]
    if available.empty:
        return snapshots.iloc[0:0].copy()
    chosen = int(available.max())
    return snapshots[snapshots["generation"] == chosen].copy()


def apply_view_mode(edges: pd.DataFrame, generation: int, *, view_mode: str, window: int = 20) -> pd.DataFrame:
    if edges.empty:
        return edges
    if view_mode == "cumulative":
        return edges
    if view_mode in {"new", "new_only", "delta"}:
        return edges[edges["first_seen_generation"] == generation].copy()
    if view_mode in {"window", "sliding_window"}:
        lo = int(generation) - int(window) + 1
        return edges[edges["last_updated_generation"] >= lo].copy()
    raise ValueError(f"Unknown view_mode={view_mode!r}")


def filter_edges(
    edges: pd.DataFrame,
    *,
    mode: str = "topk",
    top_k: int = 40,
    threshold: float = 0.0,
    percentile: float = 95.0,
) -> pd.DataFrame:
    if edges.empty:
        return edges
    mode = mode.lower()
    if mode in {"topk", "top-k", "top_k"}:
        return edges.sort_values("weight", ascending=False).head(int(top_k)).copy()
    if mode in {"topk_per_node", "top-k-per-node", "top_k_per_node", "per_node_topk"}:
        k = max(1, int(top_k))
        keep: set[int] = set()
        nodes = sorted(set(edges["feature_i"].astype(int)).union(set(edges["feature_j"].astype(int))))
        for node in nodes:
            incident = edges[(edges["feature_i"] == node) | (edges["feature_j"] == node)]
            keep.update(incident.sort_values("weight", ascending=False).head(k).index.tolist())
        if not keep:
            return edges.iloc[0:0].copy()
        return edges.loc[sorted(keep)].sort_values("weight", ascending=False).copy()
    if mode in {"threshold", "weight"}:
        return edges[edges["weight"] >= float(threshold)].sort_values("weight", ascending=False).copy()
    if mode in {"percentile", "quantile"}:
        q = edges["weight"].quantile(float(percentile) / 100.0)
        return edges[edges["weight"] >= q].sort_values("weight", ascending=False).copy()
    if mode in {"all", "none"}:
        return edges.sort_values("weight", ascending=False).copy()
    raise ValueError(f"Unknown filter mode: {mode!r}")
