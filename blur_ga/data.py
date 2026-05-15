from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

ProblemType = Literal["classification", "regression"]


@dataclass(frozen=True)
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    problem_type: ProblemType
    n_classes: int = 0

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def subset(self, indices: Iterable[int]) -> "Dataset":
        idx = np.asarray(list(indices), dtype=int)
        return Dataset(self.name, self.X[idx].copy(), self.y[idx].copy(), self.problem_type, self.n_classes)


@dataclass(frozen=True)
class SupervisedSplit:
    """A train/evaluation split used by the fitness evaluator or final test."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    problem_type: ProblemType
    n_classes: int = 0

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_eval(self) -> int:
        return int(self.X_eval.shape[0])


def _safe_dataset_name(path: Path, metadata_name: object | None = None) -> str:
    raw = str(metadata_name) if metadata_name not in (None, "") else path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_.-") or path.stem


def _read_metadata_name(path: Path) -> str | None:
    meta_path = path / "metadata.json" if path.is_dir() else path.with_suffix(".metadata.json")
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return str(meta.get("dataset_key") or meta.get("dataset_name") or meta.get("name") or "") or None
    except Exception:
        return None


def _coerce_classification_labels(y: np.ndarray) -> tuple[np.ndarray, int]:
    """Return integer labels and number of classes for already-encoded targets.

    The OpenML preprocessing stage writes labels as 0..K-1. This helper also
    handles any integer-coded target that is not perfectly contiguous by remapping
    it to 0..K-1, which keeps np.bincount-based KNN voting safe.
    """
    y = np.asarray(y).reshape(-1)
    if y.size == 0:
        raise ValueError("Target vector is empty.")
    if np.issubdtype(y.dtype, np.number):
        if not np.all(np.isfinite(y.astype(float))):
            raise ValueError("Target contains NaN/Inf values.")
        y_int = y.astype(int)
        if not np.allclose(y.astype(float), y_int.astype(float)):
            raise ValueError("Classification target must be integer encoded.")
    else:
        # Fallback for CSV targets read as objects.
        _, y_int = np.unique(y.astype(str), return_inverse=True)
        y_int = y_int.astype(int)
    classes = np.unique(y_int)
    if classes.size < 2:
        raise ValueError("Classification target has fewer than two classes.")
    expected = np.arange(classes.size)
    if not np.array_equal(classes, expected):
        mapping = {int(old): int(new) for new, old in enumerate(classes)}
        y_int = np.asarray([mapping[int(v)] for v in y_int], dtype=int)
    return y_int.astype(int), int(classes.size)


class ProcessedTabularParser:
    """Reader for prepared OpenML/TabArena data without downloading anything.

    Supported inputs:
    - a directory containing ``dataset.npz`` produced by scripts/prepare_existing_tabarena.py
    - a direct ``.npz`` file with arrays ``X`` and ``y``
    - a directory containing ``X_preprocessed.csv`` and ``y.csv`` from the OpenML
      preprocessing script

    The returned Dataset is classification-only and is already numeric/finite.
    """

    @classmethod
    def read(cls, path: str | Path) -> Dataset:
        p = Path(path)
        if p.is_dir() and (p / "dataset.npz").exists():
            return cls._read_npz(p / "dataset.npz", name=_safe_dataset_name(p, _read_metadata_name(p)))
        if p.suffix.lower() == ".npz":
            return cls._read_npz(p, name=_safe_dataset_name(p, _read_metadata_name(p.parent)))
        if p.is_dir() and (p / "X_preprocessed.csv").exists() and (p / "y.csv").exists():
            return cls._read_processed_csv(p)
        raise FileNotFoundError(
            f"Cannot read processed dataset from {p}. Expected dataset.npz, .npz, or X_preprocessed.csv + y.csv."
        )

    @staticmethod
    def _read_npz(path: Path, *, name: str) -> Dataset:
        with np.load(path, allow_pickle=False) as data:
            if "X" not in data or "y" not in data:
                raise ValueError(f"{path} must contain arrays named 'X' and 'y'.")
            X = np.asarray(data["X"], dtype=np.float64)
            y_raw = np.asarray(data["y"])
            n_classes = 0
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D matrix, got shape {X.shape}.")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"X in {path} contains NaN/Inf values; run prepare_existing_tabarena.py first.")
        y, inferred_classes = _coerce_classification_labels(y_raw)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y row mismatch in {path}: {X.shape[0]} vs {y.shape[0]}.")
        return Dataset(name=name, X=X, y=y, problem_type="classification", n_classes=max(n_classes, inferred_classes))

    @staticmethod
    def _read_processed_csv(folder: Path) -> Dataset:
        import pandas as pd

        X_df = pd.read_csv(folder / "X_preprocessed.csv")
        y_df = pd.read_csv(folder / "y.csv")
        if y_df.shape[1] != 1:
            raise ValueError(f"Expected y.csv with one column, got {y_df.shape[1]} columns in {folder}.")
        X = X_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(X)):
            raise ValueError(f"X_preprocessed.csv in {folder} contains NaN/Inf values; run prepare_existing_tabarena.py first.")
        y, n_classes = _coerce_classification_labels(y_df.iloc[:, 0].to_numpy())
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y row mismatch in {folder}: {X.shape[0]} vs {y.shape[0]}.")
        name = _safe_dataset_name(folder, _read_metadata_name(folder))
        return Dataset(name=name, X=X, y=y, problem_type="classification", n_classes=n_classes)


_TOKEN_RE = re.compile(r"[ :=\n\t\r\f\v]+")


class DatParser:
    """Parser for original .dat files and prepared processed TabArena folders.

    Unlike the old port, this parser does *not* create a 70/30 split. It returns
    the full dataset so that split policy is controlled by the experiment layer.
    For the OpenML workflow, pass either ``--data-dir prepared_tabarena`` plus a
    dataset key from ``manifest.csv``, or pass a direct path to a prepared folder
    or ``dataset.npz`` file. No OpenML download is performed here.
    """

    @staticmethod
    def _tokens(line: str) -> list[str]:
        return [tok for tok in _TOKEN_RE.split(line.strip()) if tok]

    @classmethod
    def read(cls, problem: str | Path, search_dir: str | Path = ".") -> Dataset:
        path_arg = Path(problem)
        search_root = Path(search_dir)

        # Direct path to prepared folder/.npz, or a dataset key under --data-dir.
        candidates: list[Path] = []
        if path_arg.exists():
            candidates.append(path_arg)
        candidates.extend([
            search_root / path_arg,
            search_root / f"{path_arg}.npz",
            search_root / str(path_arg),
        ])
        for candidate in candidates:
            if candidate.exists() and (candidate.is_dir() or candidate.suffix.lower() == ".npz"):
                try:
                    return ProcessedTabularParser.read(candidate)
                except FileNotFoundError:
                    pass

        # Original .dat path or dataset name.
        if path_arg.suffix == ".dat":
            dataset_path = path_arg if path_arg.exists() else search_root / path_arg
            name = path_arg.stem
        else:
            dataset_path = search_root / f"{problem}.dat"
            name = Path(problem).name

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"cannot open dataset file: {dataset_path}. For prepared OpenML data, run scripts/prepare_existing_tabarena.py "
                f"and pass a dataset key from the manifest with --data-dir pointing to the prepared folder."
            )

        lines = dataset_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        prob_type_raw: int | None = None
        n_features: int | None = None
        n_samples: int | None = None
        n_classes = 0
        dataset_start: int | None = None

        for idx, line in enumerate(lines):
            toks = cls._tokens(line)
            if not toks:
                continue
            key = toks[0]
            if key == "TYPE":
                prob_type_raw = int(toks[1])
            elif key == "N_ATTRIBUTES":
                n_features = int(toks[1])
            elif key == "N_EXAMPLES":
                n_samples = int(toks[1])
            elif key == "N_CLASSES":
                n_classes = int(toks[1])
            elif key == "DATASET":
                dataset_start = idx + 1
                break

        if prob_type_raw is None or n_features is None or n_samples is None or dataset_start is None:
            raise ValueError("Dataset metadata incomplete. Need TYPE, N_ATTRIBUTES, N_EXAMPLES, and DATASET.")

        problem_type: ProblemType = "classification" if prob_type_raw == 1 else "regression"
        numeric_tokens: list[str] = []
        for line in lines[dataset_start:]:
            numeric_tokens.extend(line.split())

        row_width = n_features + 1
        required = n_samples * row_width
        if len(numeric_tokens) < required:
            raise ValueError(f"DATASET block is incomplete: expected {required} values, got {len(numeric_tokens)}")

        values = np.asarray([float(v) for v in numeric_tokens[:required]], dtype=float).reshape(n_samples, row_width)
        X = values[:, :n_features].astype(float, copy=True)
        y_raw = values[:, n_features]
        if problem_type == "classification":
            y, inferred_classes = _coerce_classification_labels(y_raw)
            n_classes = max(n_classes, inferred_classes)
        else:
            y = y_raw.astype(float)
        return Dataset(name=name, X=X, y=y, problem_type=problem_type, n_classes=n_classes)
