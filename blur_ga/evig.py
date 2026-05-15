from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EdgeObservation:
    """Result of one linkage-learning edge test."""

    feature_i: int
    feature_j: int
    observed_weight: float
    is_positive: bool
    was_new_edge: bool
    tested_count_after: int
    positive_count_after: int
    edge_weight_after: float
    weight_sum_after: float
    first_seen_generation: int
    last_updated_generation: int


@dataclass
class EmpiricalVIG:
    """Empirical weighted Variable Interaction Graph.

    The graph stores three related quantities:

    - tested_count: how many times a pair was tested by linkage mutation.
    - count: how many positive observations passed the edge threshold.
    - weight: average positive observed interaction weight.

    Keeping tested_count separately is important because an absent edge can mean
    either "never tested" or "tested but consistently non-positive".
    """

    n_features: int
    weight_sum: np.ndarray = field(init=False)
    count: np.ndarray = field(init=False)
    tested_count: np.ndarray = field(init=False)
    weight: np.ndarray = field(init=False)
    first_seen_generation: np.ndarray = field(init=False)
    last_updated_generation: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        shape = (self.n_features, self.n_features)
        self.weight_sum = np.zeros(shape, dtype=float)
        self.count = np.zeros(shape, dtype=int)
        self.tested_count = np.zeros(shape, dtype=int)
        self.weight = np.zeros(shape, dtype=float)
        self.first_seen_generation = np.full(shape, -1, dtype=int)
        self.last_updated_generation = np.full(shape, -1, dtype=int)

    def add_observation(
        self,
        a: int,
        b: int,
        observed_weight: float,
        *,
        epsilon: float = 0.0,
        generation: int = -1,
    ) -> EdgeObservation:
        if a == b:
            raise ValueError("Self-linkage observations are not valid.")
        if a > b:
            a, b = b, a

        observed = float(observed_weight)
        self.tested_count[a, b] += 1
        self.tested_count[b, a] = self.tested_count[a, b]
        is_positive = observed > epsilon
        was_new = bool(is_positive and self.count[a, b] == 0)

        if is_positive:
            self.weight_sum[a, b] += observed
            self.count[a, b] += 1
            avg = self.weight_sum[a, b] / self.count[a, b]
            self.weight[a, b] = avg
            self.weight[b, a] = avg
            self.weight_sum[b, a] = self.weight_sum[a, b]
            self.count[b, a] = self.count[a, b]
            if was_new:
                self.first_seen_generation[a, b] = int(generation)
                self.first_seen_generation[b, a] = int(generation)
            self.last_updated_generation[a, b] = int(generation)
            self.last_updated_generation[b, a] = int(generation)

        return EdgeObservation(
            feature_i=int(a),
            feature_j=int(b),
            observed_weight=observed,
            is_positive=bool(is_positive),
            was_new_edge=was_new,
            tested_count_after=int(self.tested_count[a, b]),
            positive_count_after=int(self.count[a, b]),
            edge_weight_after=float(self.weight[a, b]),
            weight_sum_after=float(self.weight_sum[a, b]),
            first_seen_generation=int(self.first_seen_generation[a, b]),
            last_updated_generation=int(self.last_updated_generation[a, b]),
        )

    @property
    def n_edges(self) -> int:
        return int(np.count_nonzero(np.triu(self.count, k=1)))

    @property
    def n_tested_pairs(self) -> int:
        return int(np.count_nonzero(np.triu(self.tested_count, k=1)))

    def adjacency_matrix(self) -> np.ndarray:
        return self.weight.copy()

    def edge_table(
        self,
        *,
        min_weight: float = 0.0,
        top_k: int | None = None,
    ) -> list[tuple[int, int, float, int, int, float, int, int]]:
        """Return positive edges sorted by descending weight.

        Columns are:
        feature_i, feature_j, weight, positive_count, tested_count,
        weight_sum, first_seen_generation, last_updated_generation.
        """
        rows: list[tuple[int, int, float, int, int, float, int, int]] = []
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                if self.count[i, j] > 0 and self.weight[i, j] >= min_weight:
                    rows.append(
                        (
                            i,
                            j,
                            float(self.weight[i, j]),
                            int(self.count[i, j]),
                            int(self.tested_count[i, j]),
                            float(self.weight_sum[i, j]),
                            int(self.first_seen_generation[i, j]),
                            int(self.last_updated_generation[i, j]),
                        )
                    )
        rows.sort(key=lambda r: (-r[2], r[0], r[1]))
        if top_k is not None:
            rows = rows[: int(top_k)]
        return rows

    def tested_pair_table(self) -> list[tuple[int, int, int, int, float]]:
        """Return all tested pairs, including those without positive edges."""
        rows: list[tuple[int, int, int, int, float]] = []
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                if self.tested_count[i, j] > 0:
                    rows.append((i, j, int(self.tested_count[i, j]), int(self.count[i, j]), float(self.weight[i, j])))
        rows.sort(key=lambda r: (-r[2], r[0], r[1]))
        return rows

    def save_matrix(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, self.weight, delimiter=",", fmt="%.8f")

    def save_edges(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(
                "feature_i,feature_j,weight,positive_count,tested_count,weight_sum,"
                "first_seen_generation,last_updated_generation\n"
            )
            for i, j, weight, count, tested, weight_sum, first_seen, last_updated in self.edge_table():
                f.write(
                    f"{i},{j},{weight:.12f},{count},{tested},{weight_sum:.12f},"
                    f"{first_seen},{last_updated}\n"
                )

    def save_tested_pairs(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("feature_i,feature_j,tested_count,positive_count,weight\n")
            for i, j, tested, count, weight in self.tested_pair_table():
                f.write(f"{i},{j},{tested},{count},{weight:.12f}\n")
