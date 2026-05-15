from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Dataset, SupervisedSplit


@dataclass(frozen=True)
class Fold:
    train_indices: np.ndarray
    test_indices: np.ndarray
    fold_id: int
    repeat_id: int = 0


class KFoldSplitter:
    """Deterministic K-fold splitter with stratification for classification."""

    def __init__(self, n_splits: int, *, shuffle: bool = True, seed: int = 1, stratified: bool = True) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed
        self.stratified = stratified

    def split(self, dataset: Dataset, *, repeat_id: int = 0, indices: np.ndarray | None = None) -> list[Fold]:
        base_indices = np.arange(dataset.n_samples, dtype=int) if indices is None else np.asarray(indices, dtype=int)
        if len(base_indices) < self.n_splits:
            raise ValueError("Number of samples must be at least n_splits.")
        rng = np.random.default_rng(self.seed + 1009 * repeat_id)

        if self.stratified and dataset.problem_type == "classification":
            fold_bins: list[list[int]] = [[] for _ in range(self.n_splits)]
            y = dataset.y[base_indices]
            for cls in np.unique(y):
                cls_indices = base_indices[y == cls].copy()
                if self.shuffle:
                    rng.shuffle(cls_indices)
                for rank, idx in enumerate(cls_indices):
                    fold_bins[rank % self.n_splits].append(int(idx))
        else:
            perm = base_indices.copy()
            if self.shuffle:
                rng.shuffle(perm)
            fold_bins = [list(map(int, part)) for part in np.array_split(perm, self.n_splits)]

        all_set = set(map(int, base_indices))
        folds: list[Fold] = []
        for fold_id, test_list in enumerate(fold_bins):
            test_indices = np.asarray(sorted(test_list), dtype=int)
            test_set = set(map(int, test_indices))
            train_indices = np.asarray(sorted(all_set - test_set), dtype=int)
            folds.append(Fold(train_indices=train_indices, test_indices=test_indices, fold_id=fold_id, repeat_id=repeat_id))
        return folds


def make_split(dataset: Dataset, train_indices: np.ndarray, eval_indices: np.ndarray) -> SupervisedSplit:
    return SupervisedSplit(
        X_train=dataset.X[train_indices],
        y_train=dataset.y[train_indices],
        X_eval=dataset.X[eval_indices],
        y_eval=dataset.y[eval_indices],
        problem_type=dataset.problem_type,
        n_classes=dataset.n_classes,
    )


def remap_local_to_global(local_indices: np.ndarray, global_pool: np.ndarray) -> np.ndarray:
    return np.asarray(global_pool[local_indices], dtype=int)
