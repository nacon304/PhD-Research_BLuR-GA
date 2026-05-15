from __future__ import annotations

import numpy as np

from .data import SupervisedSplit


class KNNScorer:
    """Dependency-light KNN scorer for wrapper feature selection.

    Computes squared Euclidean distances in 2-D batches using
    ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab^T, then selects only the k nearest
    rows with argpartition. This avoids the memory-heavy 3-D broadcast used in
    the earlier port and is safer for Windows/HPC runs.
    """

    def __init__(self, k: int, *, batch_size: int = 256) -> None:
        if k < 1:
            raise ValueError("k must be positive.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.k = int(k)
        self.batch_size = int(batch_size)

    def score(self, split: SupervisedSplit, selected_features: np.ndarray) -> float:
        if selected_features.size == 0:
            return 0.0
        k = min(self.k, split.n_train)
        X_train = np.ascontiguousarray(split.X_train[:, selected_features], dtype=np.float64)
        X_eval = np.ascontiguousarray(split.X_eval[:, selected_features], dtype=np.float64)
        if X_train.ndim != 2 or X_eval.ndim != 2:
            raise ValueError("KNN expects 2-D train/eval matrices after feature selection.")
        if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(X_eval)):
            raise ValueError("KNN received NaN/Inf values. Re-run preprocessing/prepare step.")
        if split.problem_type == "classification":
            pred = self._predict_classification(X_train, split.y_train.astype(int), X_eval, k, split.n_classes)
            return float(np.mean(pred == split.y_eval.astype(int)))
        pred_reg = self._predict_regression(X_train, split.y_train.astype(float), X_eval, k)
        mse = float(np.mean((pred_reg - split.y_eval.astype(float)) ** 2))
        return 1.0 - mse

    def _nearest_indices(self, X_train: np.ndarray, X_eval: np.ndarray, k: int) -> np.ndarray:
        n_eval = int(X_eval.shape[0])
        n_train = int(X_train.shape[0])
        if n_eval == 0 or n_train == 0:
            return np.empty((n_eval, 0), dtype=int)
        k = min(int(k), n_train)
        nearest_all = np.empty((n_eval, k), dtype=int)
        train_sq = np.einsum("ij,ij->i", X_train, X_train, optimize=True)
        for start in range(0, n_eval, self.batch_size):
            end = min(start + self.batch_size, n_eval)
            X_batch = X_eval[start:end]
            batch_sq = np.einsum("ij,ij->i", X_batch, X_batch, optimize=True)
            dists = batch_sq[:, None] + train_sq[None, :] - 2.0 * (X_batch @ X_train.T)
            np.maximum(dists, 0.0, out=dists)
            if k == n_train:
                nearest = np.argsort(dists, axis=1, kind="mergesort")[:, :k]
            else:
                part = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
                part_dists = np.take_along_axis(dists, part, axis=1)
                order = np.argsort(part_dists, axis=1, kind="mergesort")
                nearest = np.take_along_axis(part, order, axis=1)
            nearest_all[start:end] = nearest.astype(int, copy=False)
        return nearest_all

    def _predict_classification(
        self, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, k: int, n_classes: int
    ) -> np.ndarray:
        nearest = self._nearest_indices(X_train, X_eval, k)
        labels = y_train[nearest]
        out = np.empty(X_eval.shape[0], dtype=int)
        if labels.size == 0:
            out.fill(0)
            return out
        max_label = max(int(n_classes) - 1, int(np.max(y_train)))
        for i, row in enumerate(labels):
            counts = np.bincount(row.astype(int), minlength=max_label + 1)
            out[i] = int(np.argmax(counts))
        return out

    def _predict_regression(self, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, k: int) -> np.ndarray:
        nearest = self._nearest_indices(X_train, X_eval, k)
        if nearest.shape[1] == 0:
            return np.zeros(X_eval.shape[0], dtype=float)
        return np.mean(y_train[nearest], axis=1)
