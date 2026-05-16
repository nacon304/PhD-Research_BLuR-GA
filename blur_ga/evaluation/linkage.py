from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EdgeScore:
    feature_i: int
    feature_j: int
    score: float
    coefficient: float = 0.0
    sign: int = 0
    source: str = "learned"


def canonical_pair(i: int, j: int) -> tuple[int, int]:
    ii, jj = int(i), int(j)
    return (ii, jj) if ii < jj else (jj, ii)


def load_true_edges(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "feature_i" not in df.columns or "feature_j" not in df.columns:
        raise ValueError(f"{path} must contain feature_i and feature_j columns.")
    if "true_weight" not in df.columns:
        df["true_weight"] = 1.0
    if "true_sign" not in df.columns:
        df["true_sign"] = np.sign(df["true_weight"].astype(float)).astype(int)
    if "abs_weight" not in df.columns:
        df["abs_weight"] = df["true_weight"].astype(float).abs()
    df["feature_i"] = df["feature_i"].astype(int)
    df["feature_j"] = df["feature_j"].astype(int)
    return df.sort_values(["feature_i", "feature_j"]).reset_index(drop=True)


def evig_to_edges(evig: object, *, method: str, dataset: str, generation: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if evig is None:
        return pd.DataFrame(columns=_predicted_edge_columns())
    n_features = int(getattr(evig, "n_features", 0))
    weight = np.asarray(getattr(evig, "weight"), dtype=float)
    coefficient = np.asarray(getattr(evig, "coefficient", np.zeros_like(weight)), dtype=float)
    stability = np.asarray(getattr(evig, "stability", np.zeros_like(weight)), dtype=float)
    selected_count = np.asarray(getattr(evig, "selected_count", getattr(evig, "count", np.zeros_like(weight))), dtype=int)
    tested_count = np.asarray(getattr(evig, "tested_count", np.zeros_like(weight)), dtype=int)
    first_seen = np.asarray(getattr(evig, "first_seen_generation", np.full_like(weight, -1, dtype=int)), dtype=int)
    last_updated = np.asarray(getattr(evig, "last_updated_generation", np.full_like(weight, -1, dtype=int)), dtype=int)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            score = float(weight[i, j])
            if score <= 0.0:
                continue
            coef = float(coefficient[i, j])
            sign = int(np.sign(coef)) if abs(coef) > 0 else 0
            rows.append({
                "dataset": dataset,
                "method": method,
                "generation": "" if generation is None else int(generation),
                "feature_i": i,
                "feature_j": j,
                "score": score,
                "coefficient": coef,
                "sign": sign,
                "stability": float(stability[i, j]),
                "selected_count": int(selected_count[i, j]),
                "tested_count": int(tested_count[i, j]),
                "first_seen_generation": int(first_seen[i, j]),
                "last_updated_generation": int(last_updated[i, j]),
            })
    df = pd.DataFrame(rows, columns=_predicted_edge_columns())
    if not df.empty:
        df = df.sort_values(["score", "feature_i", "feature_j"], ascending=[False, True, True]).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1, dtype=int))
    else:
        df.insert(0, "rank", [])
    return df


def _predicted_edge_columns() -> list[str]:
    return [
        "dataset", "method", "generation", "feature_i", "feature_j", "score", "coefficient", "sign",
        "stability", "selected_count", "tested_count", "first_seen_generation", "last_updated_generation",
    ]


def _edge_set(df: pd.DataFrame) -> set[tuple[int, int]]:
    if df.empty:
        return set()
    return {canonical_pair(r.feature_i, r.feature_j) for r in df.itertuples(index=False)}


def precision_recall_f1_at_k(pred: pd.DataFrame, true: pd.DataFrame, k: int) -> tuple[float, float, float, int]:
    true_set = _edge_set(true)
    if not true_set:
        return 0.0, 0.0, 0.0, 0
    top = pred.head(max(0, int(k))) if not pred.empty else pred
    pred_set = _edge_set(top)
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp


def average_precision(pred: pd.DataFrame, true: pd.DataFrame) -> float:
    true_set = _edge_set(true)
    if not true_set or pred.empty:
        return 0.0
    hits = 0
    precisions: list[float] = []
    seen: set[tuple[int, int]] = set()
    ranked = pred.sort_values(["score", "feature_i", "feature_j"], ascending=[False, True, True])
    for rank, row in enumerate(ranked.itertuples(index=False), start=1):
        pair = canonical_pair(int(row.feature_i), int(row.feature_j))
        if pair in seen:
            continue
        seen.add(pair)
        if pair in true_set:
            hits += 1
            precisions.append(hits / rank)
    return float(np.sum(precisions) / len(true_set)) if precisions else 0.0


def spearman_weight_correlation(pred: pd.DataFrame, true: pd.DataFrame, n_features: int) -> float:
    if n_features < 2:
        return 0.0
    pred_map = {canonical_pair(r.feature_i, r.feature_j): float(r.score) for r in pred.itertuples(index=False)} if not pred.empty else {}
    true_map = {canonical_pair(r.feature_i, r.feature_j): abs(float(r.true_weight)) for r in true.itertuples(index=False)} if not true.empty else {}
    y_pred: list[float] = []
    y_true: list[float] = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            p = (i, j)
            y_pred.append(pred_map.get(p, 0.0))
            y_true.append(true_map.get(p, 0.0))
    s_pred = pd.Series(y_pred).rank(method="average")
    s_true = pd.Series(y_true).rank(method="average")
    corr = s_pred.corr(s_true, method="pearson")
    return 0.0 if pd.isna(corr) else float(corr)


def sign_accuracy(pred: pd.DataFrame, true: pd.DataFrame) -> float:
    if pred.empty or true.empty or "sign" not in pred.columns:
        return 0.0
    pred_map = {canonical_pair(r.feature_i, r.feature_j): int(r.sign) for r in pred.itertuples(index=False)}
    true_map = {canonical_pair(r.feature_i, r.feature_j): int(r.true_sign) for r in true.itertuples(index=False)}
    common = [p for p in pred_map if p in true_map and pred_map[p] != 0 and true_map[p] != 0]
    if not common:
        return 0.0
    return float(np.mean([pred_map[p] == true_map[p] for p in common]))


def evaluate_predicted_edges(
    pred: pd.DataFrame,
    true: pd.DataFrame,
    *,
    dataset: str,
    method: str,
    repeat_id: int,
    outer_fold: int,
    run_id: int,
    seed: int,
    n_features: int,
    k_values: Iterable[int] | None = None,
) -> pd.DataFrame:
    n_true = int(len(true))
    default_ks = [n_true, 2 * n_true, 50, 100]
    ks = sorted({int(k) for k in (k_values or default_ks) if int(k) > 0})
    total_pairs = n_features * (n_features - 1) // 2
    ap = average_precision(pred, true)
    spear = spearman_weight_correlation(pred, true, n_features)
    sign_acc = sign_accuracy(pred, true)
    rows: list[dict[str, object]] = []
    for k in ks:
        kk = min(k, total_pairs)
        precision, recall, f1, tp = precision_recall_f1_at_k(pred, true, kk)
        rows.append({
            "dataset": dataset,
            "method": method,
            "repeat_id": int(repeat_id),
            "outer_fold": int(outer_fold),
            "run_id": int(run_id),
            "seed": int(seed),
            "k": int(kk),
            "true_positives_at_k": int(tp),
            "precision_at_k": precision,
            "recall_at_k": recall,
            "f1_at_k": f1,
            "average_precision": ap,
            "spearman_abs_weight": spear,
            "sign_accuracy": sign_acc,
            "n_true_edges": n_true,
            "n_pred_edges": int(len(pred)),
            "n_features": int(n_features),
        })
    return pd.DataFrame(rows)


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
