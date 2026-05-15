from __future__ import annotations

"""Ground-truth linkage benchmark problems for BLuR-GA.

The classes in this module deliberately separate two concepts:

1. a *problem definition* stored in ``problem.json`` and used later by a
   synthetic-landscape evaluator; and
2. optional supervised synthetic data stored in ``dataset.npz`` for problems
   such as planted XOR and GAMETES-style SNP epistasis.

Every benchmark folder follows the same prepared-data style used by the current
TabArena workflow: ``metadata.json``, ``feature_names.json``, and a manifest row.
For linkage correctness evaluation, every folder additionally stores
``ground_truth_edges.csv`` and ``ground_truth_graph_snapshots.csv``.
"""

from dataclasses import asdict, dataclass
from itertools import combinations, product
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ArrayLikeInt = Iterable[int] | np.ndarray


def sanitize_key(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_.-")
    return text or "dataset"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _as_binary(x: ArrayLikeInt, n_features: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int8).reshape(-1)
    if arr.size != n_features:
        raise ValueError(f"Expected chromosome of length {n_features}, got {arr.size}.")
    return (arr > 0).astype(np.int8)


def _complete_pair_edges(nodes: list[int], weight: float = 1.0) -> dict[tuple[int, int], float]:
    edges: dict[tuple[int, int], float] = {}
    for i, j in combinations(sorted(set(nodes)), 2):
        edges[(int(i), int(j))] = float(weight)
    return edges


def _edge_records_from_map(edge_weights: Mapping[tuple[int, int], float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (i, j), w in sorted(edge_weights.items()):
        if i == j:
            continue
        ii, jj = sorted((int(i), int(j)))
        rows.append(
            {
                "feature_i": ii,
                "feature_j": jj,
                "true_weight": float(w),
                "true_sign": int(np.sign(w)) if float(w) != 0.0 else 0,
                "abs_weight": abs(float(w)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature_i", "feature_j", "true_weight", "true_sign", "abs_weight"])
    df = pd.DataFrame(rows).drop_duplicates(["feature_i", "feature_j"], keep="last")
    return df.sort_values(["feature_i", "feature_j"]).reset_index(drop=True)


def _matrix_from_edges(n_features: int, edges: pd.DataFrame) -> pd.DataFrame:
    mat = np.zeros((int(n_features), int(n_features)), dtype=float)
    for row in edges.itertuples(index=False):
        i = int(row.feature_i)
        j = int(row.feature_j)
        w = float(row.true_weight)
        mat[i, j] = w
        mat[j, i] = w
    return pd.DataFrame(mat, columns=[f"f{i}" for i in range(n_features)], index=[f"f{i}" for i in range(n_features)])


def _ground_truth_snapshots(edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in edges.itertuples(index=False):
        rows.append(
            {
                "generation": 0,
                "feature_i": int(row.feature_i),
                "feature_j": int(row.feature_j),
                "weight": float(row.abs_weight),
                "positive_count": int(float(row.true_weight) > 0),
                "tested_count": 1,
                "weight_sum": float(row.abs_weight),
                "first_seen_generation": 0,
                "last_updated_generation": 0,
                "true_weight": float(row.true_weight),
                "true_sign": int(row.true_sign),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "generation",
            "feature_i",
            "feature_j",
            "weight",
            "positive_count",
            "tested_count",
            "weight_sum",
            "first_seen_generation",
            "last_updated_generation",
            "true_weight",
            "true_sign",
        ],
    )


@dataclass(frozen=True)
class BenchmarkSpec:
    family: str
    dataset_key: str
    seed: int
    params: dict[str, Any]


@dataclass
class LinkageBenchmarkProblem:
    dataset_key: str
    problem_family: str
    n_features: int
    seed: int
    eval_group: str = "linkage"
    evaluator_type: str = "synthetic_landscape"
    fitness_direction: str = "maximize"
    description: str = ""

    def evaluate(self, chromosome: ArrayLikeInt) -> float:
        raise NotImplementedError

    def true_edges(self) -> pd.DataFrame:
        raise NotImplementedError

    def problem_payload(self) -> dict[str, Any]:
        payload = asdict(self).copy()
        payload["problem_type"] = self.problem_family
        return payload

    def metadata(self, *, relative_folder: str, data_path: str | None = None) -> dict[str, Any]:
        edges = self.true_edges()
        return {
            "dataset_key": self.dataset_key,
            "eval_group": self.eval_group,
            "problem_family": self.problem_family,
            "evaluator_type": self.evaluator_type,
            "fitness_direction": self.fitness_direction,
            "n_samples": None,
            "n_features": int(self.n_features),
            "n_classes": None,
            "n_true_edges": int(len(edges)),
            "has_true_edges": True,
            "has_true_blocks": False,
            "split_mode": "none",
            "n_outer_folds": 1,
            "seed": int(self.seed),
            "data_path": data_path,
            "problem_path": f"{relative_folder}/problem.json",
            "ground_truth_edges_path": f"{relative_folder}/ground_truth_edges.csv",
            "ground_truth_matrix_path": f"{relative_folder}/ground_truth_matrix.csv",
            "ground_truth_graph_snapshots_path": f"{relative_folder}/ground_truth_graph_snapshots.csv",
            "prepared_folder": relative_folder,
            "description": self.description,
        }

    def feature_metadata(self) -> list[dict[str, Any]]:
        return [{"feature_index": int(i), "feature_name": f"f{i}"} for i in range(int(self.n_features))]

    def optional_dataset(self) -> tuple[np.ndarray, np.ndarray] | None:
        return None


@dataclass
class PairwiseQUBOProblem(LinkageBenchmarkProblem):
    intercept: float = 0.0
    main_effects: list[float] | None = None
    pairwise_edges: list[dict[str, float | int]] | None = None
    noise_std: float = 0.0

    @classmethod
    def generate(
        cls,
        *,
        n_features: int = 50,
        n_edges: int = 75,
        seed: int = 0,
        signed: bool = True,
        main_effect_scale: float = 0.05,
        edge_weight_low: float = 0.4,
        edge_weight_high: float = 1.0,
        noise_std: float = 0.0,
    ) -> "PairwiseQUBOProblem":
        rng = _rng(seed)
        all_pairs = np.array(list(combinations(range(n_features), 2)), dtype=int)
        if n_edges > len(all_pairs):
            raise ValueError(f"n_edges={n_edges} exceeds total possible pairs {len(all_pairs)}.")
        chosen = all_pairs[rng.choice(len(all_pairs), size=int(n_edges), replace=False)]
        magnitudes = rng.uniform(edge_weight_low, edge_weight_high, size=int(n_edges))
        signs = rng.choice([-1.0, 1.0], size=int(n_edges)) if signed else np.ones(int(n_edges))
        edges = [
            {"i": int(i), "j": int(j), "weight": float(w * s)}
            for (i, j), w, s in zip(chosen, magnitudes, signs)
        ]
        main = rng.normal(0.0, main_effect_scale, size=int(n_features)).astype(float).tolist()
        name = sanitize_key(
            f"pairwise_qubo_d{n_features}_e{n_edges}_{'posneg' if signed else 'pos'}_noise{noise_std:g}_seed{seed}"
        )
        return cls(
            dataset_key=name,
            problem_family="pairwise_qubo",
            n_features=int(n_features),
            seed=int(seed),
            description="Sparse pairwise pseudo-Boolean/QUBO landscape with known quadratic coefficients.",
            intercept=0.0,
            main_effects=main,
            pairwise_edges=edges,
            noise_std=float(noise_std),
        )

    def evaluate(self, chromosome: ArrayLikeInt) -> float:
        x = _as_binary(chromosome, self.n_features).astype(float)
        main = np.asarray(self.main_effects or [0.0] * self.n_features, dtype=float)
        score = float(self.intercept + np.dot(main, x))
        for edge in self.pairwise_edges or []:
            score += float(edge["weight"]) * float(x[int(edge["i"])] * x[int(edge["j"])])
        return score

    def true_edges(self) -> pd.DataFrame:
        edge_map = {(int(e["i"]), int(e["j"])): float(e["weight"]) for e in self.pairwise_edges or []}
        return _edge_records_from_map(edge_map)


@dataclass
class IsingSpinGlassProblem(LinkageBenchmarkProblem):
    grid_rows: int = 5
    grid_cols: int = 5
    fields: list[float] | None = None
    couplings: list[dict[str, float | int]] | None = None

    @classmethod
    def generate(
        cls,
        *,
        grid_rows: int = 6,
        grid_cols: int = 6,
        seed: int = 0,
        signed: bool = True,
        coupling_low: float = 0.5,
        coupling_high: float = 1.0,
        field_scale: float = 0.02,
    ) -> "IsingSpinGlassProblem":
        rng = _rng(seed)
        n_features = int(grid_rows) * int(grid_cols)
        edges: list[dict[str, float | int]] = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                idx = r * grid_cols + c
                if c + 1 < grid_cols:
                    j = r * grid_cols + c + 1
                    mag = float(rng.uniform(coupling_low, coupling_high))
                    sign = float(rng.choice([-1, 1])) if signed else 1.0
                    edges.append({"i": int(idx), "j": int(j), "weight": mag * sign})
                if r + 1 < grid_rows:
                    j = (r + 1) * grid_cols + c
                    mag = float(rng.uniform(coupling_low, coupling_high))
                    sign = float(rng.choice([-1, 1])) if signed else 1.0
                    edges.append({"i": int(idx), "j": int(j), "weight": mag * sign})
        fields = rng.normal(0.0, field_scale, size=n_features).astype(float).tolist()
        name = sanitize_key(f"ising_spin_glass_grid{grid_rows}x{grid_cols}_{'posneg' if signed else 'pos'}_seed{seed}")
        return cls(
            dataset_key=name,
            problem_family="ising_spin_glass",
            n_features=n_features,
            seed=int(seed),
            description="Grid Ising/spin-glass landscape with known nearest-neighbor couplings.",
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
            fields=fields,
            couplings=edges,
        )

    def evaluate(self, chromosome: ArrayLikeInt) -> float:
        x = _as_binary(chromosome, self.n_features)
        spins = 2.0 * x.astype(float) - 1.0
        fields = np.asarray(self.fields or [0.0] * self.n_features, dtype=float)
        score = float(np.dot(fields, spins))
        for edge in self.couplings or []:
            score += float(edge["weight"]) * float(spins[int(edge["i"])] * spins[int(edge["j"])])
        return score

    def true_edges(self) -> pd.DataFrame:
        edge_map = {(int(e["i"]), int(e["j"])): float(e["weight"]) for e in self.couplings or []}
        return _edge_records_from_map(edge_map)

    def feature_metadata(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                idx = r * self.grid_cols + c
                rows.append({"feature_index": idx, "feature_name": f"s{r}_{c}", "grid_row": r, "grid_col": c})
        return rows


@dataclass
class PlantedXORProblem(LinkageBenchmarkProblem):
    n_samples: int = 1000
    interaction_groups: list[list[int]] | None = None
    label_noise: float = 0.0
    X: list[list[int]] | None = None
    y: list[int] | None = None

    @classmethod
    def generate(
        cls,
        *,
        n_features: int = 40,
        n_samples: int = 1000,
        group_sizes: tuple[int, ...] = (3, 3),
        seed: int = 0,
        label_noise: float = 0.03,
    ) -> "PlantedXORProblem":
        rng = _rng(seed)
        if sum(group_sizes) > n_features:
            raise ValueError("Total planted group size cannot exceed n_features.")
        perm = rng.permutation(n_features)
        groups: list[list[int]] = []
        cursor = 0
        for size in group_sizes:
            groups.append(sorted(int(v) for v in perm[cursor : cursor + int(size)]))
            cursor += int(size)
        X = rng.integers(0, 2, size=(int(n_samples), int(n_features)), dtype=np.int8)
        y = np.zeros(int(n_samples), dtype=np.int8)
        for group in groups:
            y ^= np.bitwise_xor.reduce(X[:, group], axis=1).astype(np.int8)
        if label_noise > 0:
            flip = rng.random(int(n_samples)) < float(label_noise)
            y[flip] = 1 - y[flip]
        name = sanitize_key(f"planted_xor_d{n_features}_n{n_samples}_g{'-'.join(map(str, group_sizes))}_noise{label_noise:g}_seed{seed}")
        return cls(
            dataset_key=name,
            problem_family="planted_xor_parity",
            n_features=int(n_features),
            seed=int(seed),
            evaluator_type="supervised_synthetic_classification",
            description="Synthetic classification data with planted XOR/parity interaction groups.",
            n_samples=int(n_samples),
            interaction_groups=groups,
            label_noise=float(label_noise),
            X=X.astype(int).tolist(),
            y=y.astype(int).tolist(),
        )

    def evaluate(self, chromosome: ArrayLikeInt) -> float:
        # A deterministic proxy fitness for synthetic-landscape smoke tests: count
        # how many complete planted groups are selected, penalize unrelated bits.
        x = _as_binary(chromosome, self.n_features)
        selected = set(np.flatnonzero(x).astype(int).tolist())
        score = 0.0
        planted = set()
        for group in self.interaction_groups or []:
            planted.update(group)
            group_set = set(group)
            score += 1.0 if group_set.issubset(selected) else 0.0
        score -= 0.01 * len(selected - planted)
        return float(score)

    def true_edges(self) -> pd.DataFrame:
        edge_map: dict[tuple[int, int], float] = {}
        for group in self.interaction_groups or []:
            for key, value in _complete_pair_edges(group, 1.0).items():
                edge_map[key] = edge_map.get(key, 0.0) + value
        return _edge_records_from_map(edge_map)

    def metadata(self, *, relative_folder: str, data_path: str | None = None) -> dict[str, Any]:
        meta = super().metadata(relative_folder=relative_folder, data_path=data_path)
        meta.update({"n_samples": int(self.n_samples), "n_classes": 2, "split_mode": "outer_cv", "n_outer_folds": 3})
        return meta

    def problem_payload(self) -> dict[str, Any]:
        payload = super().problem_payload()
        payload.pop("X", None)
        payload.pop("y", None)
        return payload

    def optional_dataset(self) -> tuple[np.ndarray, np.ndarray] | None:
        return np.asarray(self.X, dtype=np.float32), np.asarray(self.y, dtype=np.int64)


@dataclass
class NKLandscapeProblem(LinkageBenchmarkProblem):
    k: int = 3
    neighborhoods: list[list[int]] | None = None
    contribution_tables: list[list[float]] | None = None

    @classmethod
    def generate(
        cls,
        *,
        n_features: int = 40,
        k: int = 3,
        seed: int = 0,
        neighborhood: str = "random",
    ) -> "NKLandscapeProblem":
        if k < 1 or k >= n_features:
            raise ValueError("k must satisfy 1 <= k < n_features.")
        rng = _rng(seed)
        neighborhoods: list[list[int]] = []
        tables: list[list[float]] = []
        for i in range(n_features):
            if neighborhood == "adjacent":
                partners = [int((i + offset + 1) % n_features) for offset in range(k)]
            elif neighborhood == "random":
                candidates = [j for j in range(n_features) if j != i]
                partners = sorted(int(v) for v in rng.choice(candidates, size=k, replace=False))
            else:
                raise ValueError("neighborhood must be 'random' or 'adjacent'.")
            loci = [int(i)] + partners
            neighborhoods.append(loci)
            tables.append(rng.random(2 ** (k + 1)).astype(float).tolist())
        name = sanitize_key(f"nk_landscape_d{n_features}_k{k}_{neighborhood}_seed{seed}")
        return cls(
            dataset_key=name,
            problem_family="nk_landscape",
            n_features=int(n_features),
            seed=int(seed),
            description="NK landscape with known variable neighborhoods; true pair edges are induced by shared subfunctions.",
            k=int(k),
            neighborhoods=neighborhoods,
            contribution_tables=tables,
        )

    def evaluate(self, chromosome: ArrayLikeInt) -> float:
        x = _as_binary(chromosome, self.n_features)
        total = 0.0
        for loci, table in zip(self.neighborhoods or [], self.contribution_tables or []):
            idx = 0
            for bit in x[np.asarray(loci, dtype=int)]:
                idx = (idx << 1) | int(bit)
            total += float(table[int(idx)])
        return float(total / max(1, self.n_features))

    def true_edges(self) -> pd.DataFrame:
        edge_map: dict[tuple[int, int], float] = {}
        for loci in self.neighborhoods or []:
            for i, j in combinations(sorted(set(loci)), 2):
                key = (int(i), int(j))
                edge_map[key] = edge_map.get(key, 0.0) + 1.0
        return _edge_records_from_map(edge_map)


@dataclass
class MaxSATProblem(LinkageBenchmarkProblem):
    n_clauses: int = 120
    clause_size: int = 3
    clauses: list[dict[str, Any]] | None = None

    @classmethod
    def generate(
        cls,
        *,
        n_features: int = 50,
        n_clauses: int = 150,
        clause_size: int = 3,
        seed: int = 0,
        weighted: bool = True,
    ) -> "MaxSATProblem":
        if clause_size < 2 or clause_size > n_features:
            raise ValueError("clause_size must satisfy 2 <= clause_size <= n_features.")
        rng = _rng(seed)
        clauses: list[dict[str, Any]] = []
        for _ in range(int(n_clauses)):
            variables = sorted(int(v) for v in rng.choice(n_features, size=int(clause_size), replace=False))
            signs = rng.choice([-1, 1], size=int(clause_size)).astype(int).tolist()
            weight = float(rng.uniform(0.5, 1.5)) if weighted else 1.0
            clauses.append({"variables": variables, "signs": signs, "weight": weight})
        name = sanitize_key(f"maxsat_d{n_features}_m{n_clauses}_k{clause_size}_{'weighted' if weighted else 'unweighted'}_seed{seed}")
        return cls(
            dataset_key=name,
            problem_family="maxsat",
            n_features=int(n_features),
            seed=int(seed),
            description="Weighted random MAX-SAT instance; true pair edges connect variables co-occurring in clauses.",
            n_clauses=int(n_clauses),
            clause_size=int(clause_size),
            clauses=clauses,
        )

    def evaluate(self, chromosome: ArrayLikeInt) -> float:
        x = _as_binary(chromosome, self.n_features)
        score = 0.0
        max_score = 0.0
        for clause in self.clauses or []:
            variables = clause["variables"]
            signs = clause["signs"]
            weight = float(clause["weight"])
            max_score += weight
            satisfied = False
            for var, sign in zip(variables, signs):
                bit = int(x[int(var)])
                literal = bool(bit == 1) if int(sign) > 0 else bool(bit == 0)
                satisfied = satisfied or literal
            if satisfied:
                score += weight
        return float(score / max(max_score, 1e-12))

    def true_edges(self) -> pd.DataFrame:
        edge_map: dict[tuple[int, int], float] = {}
        for clause in self.clauses or []:
            weight = float(clause["weight"])
            for i, j in combinations(sorted(set(int(v) for v in clause["variables"])), 2):
                key = (int(i), int(j))
                edge_map[key] = edge_map.get(key, 0.0) + weight
        return _edge_records_from_map(edge_map)


@dataclass
class GametesStyleProblem(LinkageBenchmarkProblem):
    n_samples: int = 1000
    interaction_loci: list[int] | None = None
    maf: float = 0.25
    prevalence: float = 0.5
    effect_scale: float = 0.30
    label_noise: float = 0.0
    penetrance_table: list[float] | None = None
    penetrance_shape: list[int] | None = None
    X: list[list[int]] | None = None
    y: list[int] | None = None

    @staticmethod
    def _centered_penetrance_signal(rng: np.random.Generator, n_loci: int, maf: float, iterations: int = 80) -> np.ndarray:
        probs_1d = np.asarray([(1.0 - maf) ** 2, 2.0 * maf * (1.0 - maf), maf**2], dtype=float)
        shape = (3,) * int(n_loci)
        table = rng.normal(0.0, 1.0, size=shape)
        weights = np.ones(shape, dtype=float)
        for axis in range(n_loci):
            view_shape = [1] * n_loci
            view_shape[axis] = 3
            weights *= probs_1d.reshape(view_shape)
        table -= float(np.sum(table * weights) / np.sum(weights))
        # Remove one-locus marginal effects under the genotype distribution.  This
        # approximates the pure/strict-epistasis property needed for feature-
        # selection linkage tests without depending on the original Java GAMETES UI.
        for _ in range(int(iterations)):
            global_mean = float(np.sum(table * weights) / np.sum(weights))
            for axis in range(n_loci):
                for g in range(3):
                    sl = [slice(None)] * n_loci
                    sl[axis] = g
                    w_slice = weights[tuple(sl)]
                    marg = float(np.sum(table[tuple(sl)] * w_slice) / np.sum(w_slice))
                    table[tuple(sl)] -= marg - global_mean
            table -= float(np.sum(table * weights) / np.sum(weights))
        std = float(np.sqrt(np.sum((table**2) * weights) / np.sum(weights)))
        if std <= 1e-12:
            std = 1.0
        return table / std

    @classmethod
    def generate(
        cls,
        *,
        n_features: int = 40,
        n_samples: int = 1000,
        n_loci: int = 2,
        seed: int = 0,
        maf: float = 0.25,
        prevalence: float = 0.5,
        effect_scale: float = 0.30,
        label_noise: float = 0.02,
    ) -> "GametesStyleProblem":
        if n_loci < 2 or n_loci > n_features:
            raise ValueError("n_loci must satisfy 2 <= n_loci <= n_features.")
        rng = _rng(seed)
        loci = sorted(int(v) for v in rng.choice(n_features, size=int(n_loci), replace=False))
        probs = [(1.0 - maf) ** 2, 2.0 * maf * (1.0 - maf), maf**2]
        X = rng.choice(np.array([0, 1, 2], dtype=np.int8), size=(int(n_samples), int(n_features)), p=probs).astype(np.int8)
        signal = cls._centered_penetrance_signal(rng, int(n_loci), float(maf))
        penetrance = np.clip(float(prevalence) + float(effect_scale) * signal, 0.01, 0.99)
        lookup = penetrance[tuple(X[:, loci].T)]
        y = (rng.random(int(n_samples)) < lookup).astype(np.int8)
        if label_noise > 0:
            flip = rng.random(int(n_samples)) < float(label_noise)
            y[flip] = 1 - y[flip]
        name = sanitize_key(f"gametes_style_d{n_features}_n{n_samples}_loci{n_loci}_maf{maf:g}_seed{seed}")
        return cls(
            dataset_key=name,
            problem_family="gametes_style_epistasis",
            n_features=int(n_features),
            seed=int(seed),
            evaluator_type="supervised_synthetic_classification",
            description="GAMETES-style biallelic SNP epistasis dataset with a penetrance table and known interacting loci.",
            n_samples=int(n_samples),
            interaction_loci=loci,
            maf=float(maf),
            prevalence=float(prevalence),
            effect_scale=float(effect_scale),
            label_noise=float(label_noise),
            penetrance_table=penetrance.reshape(-1).astype(float).tolist(),
            penetrance_shape=list(penetrance.shape),
            X=X.astype(int).tolist(),
            y=y.astype(int).tolist(),
        )

    def evaluate(self, chromosome: ArrayLikeInt) -> float:
        x = _as_binary(chromosome, self.n_features)
        selected = set(np.flatnonzero(x).astype(int).tolist())
        loci = set(self.interaction_loci or [])
        return float((1.0 if loci.issubset(selected) else 0.0) - 0.01 * len(selected - loci))

    def true_edges(self) -> pd.DataFrame:
        return _edge_records_from_map(_complete_pair_edges(self.interaction_loci or [], 1.0))

    def metadata(self, *, relative_folder: str, data_path: str | None = None) -> dict[str, Any]:
        meta = super().metadata(relative_folder=relative_folder, data_path=data_path)
        meta.update({"n_samples": int(self.n_samples), "n_classes": 2, "split_mode": "outer_cv", "n_outer_folds": 3})
        return meta

    def problem_payload(self) -> dict[str, Any]:
        payload = super().problem_payload()
        payload.pop("X", None)
        payload.pop("y", None)
        return payload

    def optional_dataset(self) -> tuple[np.ndarray, np.ndarray] | None:
        return np.asarray(self.X, dtype=np.float32), np.asarray(self.y, dtype=np.int64)


def build_problem(family: str, *, seed: int, params: Mapping[str, Any] | None = None) -> LinkageBenchmarkProblem:
    params = dict(params or {})
    family = family.lower().replace("-", "_")
    if family in {"pairwise", "pairwise_qubo", "qubo", "pseudo_boolean"}:
        return PairwiseQUBOProblem.generate(seed=seed, **params)
    if family in {"ising", "spin_glass", "ising_spin_glass"}:
        return IsingSpinGlassProblem.generate(seed=seed, **params)
    if family in {"planted_xor", "xor", "parity", "planted_xor_parity"}:
        return PlantedXORProblem.generate(seed=seed, **params)
    if family in {"nk", "nk_landscape"}:
        return NKLandscapeProblem.generate(seed=seed, **params)
    if family in {"maxsat", "max_sat", "sat"}:
        return MaxSATProblem.generate(seed=seed, **params)
    if family in {"gametes", "gametes_style", "gametes_style_epistasis"}:
        return GametesStyleProblem.generate(seed=seed, **params)
    raise ValueError(f"Unknown linkage benchmark family: {family!r}")


def write_problem(problem: LinkageBenchmarkProblem, output_root: str | Path, *, group_folder: str = "linkage") -> dict[str, Any]:
    output_root = Path(output_root)
    relative_folder = f"{group_folder}/{problem.dataset_key}"
    out_dir = output_root / relative_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    edges = problem.true_edges()
    matrix = _matrix_from_edges(problem.n_features, edges)
    snapshots = _ground_truth_snapshots(edges)
    features = pd.DataFrame(problem.feature_metadata())

    dataset = problem.optional_dataset()
    data_path: str | None = None
    if dataset is not None:
        X, y = dataset
        np.savez_compressed(out_dir / "dataset.npz", X=X, y=y)
        data_path = f"{relative_folder}/dataset.npz"

    (out_dir / "problem.json").write_text(json.dumps(problem.problem_payload(), indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(problem.metadata(relative_folder=relative_folder, data_path=data_path), indent=2), encoding="utf-8"
    )
    features.to_json(out_dir / "feature_names.json", orient="records", indent=2)
    edges.to_csv(out_dir / "ground_truth_edges.csv", index=False)
    matrix.to_csv(out_dir / "ground_truth_matrix.csv", index=True, index_label="feature")
    snapshots.to_csv(out_dir / "ground_truth_graph_snapshots.csv", index=False)

    # A labels CSV compatible with analysis_interactive.graph_evolution.data.load_labels.
    labels = features.rename(columns={"feature_index": "node_id", "feature_name": "label"})
    labels[["node_id", "label"]].to_csv(out_dir / "node_labels.csv", index=False)

    meta = problem.metadata(relative_folder=relative_folder, data_path=data_path)
    return {
        "dataset_key": problem.dataset_key,
        "eval_group": problem.eval_group,
        "problem_family": problem.problem_family,
        "evaluator_type": problem.evaluator_type,
        "data_path": data_path or f"{relative_folder}/problem.json",
        "problem_path": f"{relative_folder}/problem.json",
        "ground_truth_edges_path": f"{relative_folder}/ground_truth_edges.csv",
        "ground_truth_graph_snapshots_path": f"{relative_folder}/ground_truth_graph_snapshots.csv",
        "n_samples": meta["n_samples"],
        "n_features": int(problem.n_features),
        "n_classes": meta["n_classes"],
        "n_true_edges": int(len(edges)),
        "split_mode": meta["split_mode"],
        "n_outer_folds": int(meta["n_outer_folds"]),
        "seed": int(problem.seed),
        "prepared_folder": relative_folder,
        "dtype": "float32" if data_path else "synthetic_problem_json",
    }


def load_problem(path: str | Path) -> LinkageBenchmarkProblem:
    """Load a benchmark problem from a saved ``problem.json`` file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    problem_type = str(payload.pop("problem_type", payload.get("problem_family", ""))).lower()
    payload.pop("X", None)
    payload.pop("y", None)
    # For supervised synthetic datasets, X/y live in dataset.npz; the problem JSON
    # keeps only the generative metadata and ground-truth structure.
    cls_map = {
        "pairwise_qubo": PairwiseQUBOProblem,
        "ising_spin_glass": IsingSpinGlassProblem,
        "planted_xor_parity": PlantedXORProblem,
        "nk_landscape": NKLandscapeProblem,
        "maxsat": MaxSATProblem,
        "gametes_style_epistasis": GametesStyleProblem,
    }
    if problem_type not in cls_map:
        raise ValueError(f"Unsupported problem_type in {path}: {problem_type!r}")
    return cls_map[problem_type](**payload)
