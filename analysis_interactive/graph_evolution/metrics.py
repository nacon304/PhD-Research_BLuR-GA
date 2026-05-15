from __future__ import annotations

import pandas as pd

from .filters import edges_for_generation


def snapshot_metrics(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(columns=["generation", "n_edges", "mean_weight", "max_weight", "total_positive_count"])
    rows = []
    for gen, grp in snapshots.groupby("generation", sort=True):
        rows.append(
            {
                "generation": int(gen),
                "n_edges": int(len(grp)),
                "mean_weight": float(grp["weight"].mean()) if len(grp) else 0.0,
                "max_weight": float(grp["weight"].max()) if len(grp) else 0.0,
                "total_positive_count": int(grp["positive_count"].sum()) if "positive_count" in grp.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def metrics_at_generation(snapshots: pd.DataFrame, generation: int) -> dict[str, float | int]:
    edges = edges_for_generation(snapshots, generation)
    if edges.empty:
        return {"generation": int(generation), "n_edges": 0, "mean_weight": 0.0, "max_weight": 0.0}
    return {
        "generation": int(generation),
        "n_edges": int(len(edges)),
        "mean_weight": float(edges["weight"].mean()),
        "max_weight": float(edges["weight"].max()),
    }
