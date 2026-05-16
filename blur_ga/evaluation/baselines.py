from __future__ import annotations

import numpy as np
import pandas as pd


def _edge_frame(rows: list[dict[str, object]], *, dataset: str, method: str) -> pd.DataFrame:
    cols = ["dataset", "method", "generation", "feature_i", "feature_j", "score", "coefficient", "sign", "stability", "selected_count", "tested_count", "first_seen_generation", "last_updated_generation"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values(["score", "feature_i", "feature_j"], ascending=[False, True, True]).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1, dtype=int))
    else:
        df.insert(0, "rank", [])
    return df


def cooccurrence_edges(chromosomes: np.ndarray, fitness: np.ndarray, *, dataset: str, top_fraction: float = 0.2) -> pd.DataFrame:
    X = np.asarray(chromosomes, dtype=np.int8)
    y = np.asarray(fitness, dtype=float)
    if X.ndim != 2 or len(X) == 0:
        return _edge_frame([], dataset=dataset, method="top_cooccurrence")
    n = len(X)
    n_top = max(1, int(np.ceil(float(top_fraction) * n)))
    top_idx = np.argsort(y)[-n_top:]
    X_top = X[top_idx]
    rows = []
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            top_freq = float(np.mean((X_top[:, i] == 1) & (X_top[:, j] == 1)))
            all_freq = float(np.mean((X[:, i] == 1) & (X[:, j] == 1)))
            score = max(0.0, top_freq - all_freq)
            if score > 0:
                rows.append({"dataset": dataset, "method": "top_cooccurrence", "generation": "", "feature_i": i, "feature_j": j, "score": score, "coefficient": score, "sign": 1, "stability": 0.0, "selected_count": int(np.sum((X_top[:, i] == 1) & (X_top[:, j] == 1))), "tested_count": n_top, "first_seen_generation": -1, "last_updated_generation": -1})
    return _edge_frame(rows, dataset=dataset, method="top_cooccurrence")


def random_edges(n_features: int, n_edges: int, *, dataset: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    pairs = [(i, j) for i in range(int(n_features)) for j in range(i + 1, int(n_features))]
    if not pairs:
        return _edge_frame([], dataset=dataset, method="random_graph")
    chosen = rng.choice(len(pairs), size=min(int(n_edges), len(pairs)), replace=False)
    rows = []
    for rank, idx in enumerate(chosen, start=1):
        i, j = pairs[int(idx)]
        score = 1.0 / rank
        rows.append({"dataset": dataset, "method": "random_graph", "generation": "", "feature_i": i, "feature_j": j, "score": score, "coefficient": score, "sign": 1, "stability": 0.0, "selected_count": 0, "tested_count": 0, "first_seen_generation": -1, "last_updated_generation": -1})
    return _edge_frame(rows, dataset=dataset, method="random_graph")
