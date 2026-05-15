#!/usr/bin/env python3
"""Prepare already-downloaded/preprocessed OpenML TabArena data for BLuR-GA.

This script deliberately does NOT contact OpenML. It consumes the folders already
created under openml_tabarena_processed/ and produces a compact, HPC-friendly
prepared data directory:

    prepared_tabarena/
      manifest.csv
      skipped_datasets.csv
      <dataset_key>/
        dataset.npz
        metadata.json
        feature_names.json
        class_mapping.json

Filtering defaults implement the current research requirement:
- keep only datasets with n_samples <= 10,000
- keep only datasets with at least 10 processed features
- keep only classification targets with at least 2 classes
- validate X is numeric and finite after robust sanitisation

The generated dataset.npz files are directly readable by run_blur_ga.py via:

    python run_blur_ga.py <dataset_key> 1 1 --data-dir prepared_tabarena ...
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PreparedDatasetMetadata:
    dataset_key: str
    source_folder: str
    source_dataset_name: str | None
    task_id: int | None
    dataset_id: int | None
    n_samples: int
    n_features: int
    n_classes: int
    class_distribution: dict[str, int]
    feature_count_basis: str
    max_samples: int
    min_features: int
    dropped_non_numeric_columns: list[str]
    dropped_constant_columns: list[str]
    imputed_missing_values: int
    dtype: str
    preprocessing_scope: str
    notes: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare existing TabArena processed folders for BLuR-GA/HPC without OpenML download.")
    p.add_argument("--source-dir", default="openml_tabarena_processed", help="Existing processed OpenML folder.")
    p.add_argument("--output-dir", default="prepared_tabarena", help="Output folder for compact prepared datasets.")
    p.add_argument("--max-samples", type=int, default=10_000, help="Keep datasets with n_samples <= this value.")
    p.add_argument("--min-features", type=int, default=10, help="Keep datasets with at least this many features.")
    p.add_argument(
        "--feature-count-basis",
        choices=["processed", "raw"],
        default="processed",
        help="Whether --min-features is applied to processed feature count or raw feature count from metadata.",
    )
    p.add_argument("--float-dtype", choices=["float32", "float64"], default="float32", help="Numeric dtype saved to dataset.npz.")
    p.add_argument("--drop-constant", type=str, default="true", choices=["true", "false"], help="Drop zero-variance processed columns.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output directory.")
    p.add_argument("--limit", type=int, default=None, help="Optional limit for smoke testing.")
    return p.parse_args()


def sanitize_key(text: Any, max_len: int = 96) -> str:
    text = str(text) if text not in (None, "") else "dataset"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.-")
    return (text[:max_len] or "dataset")


def str2bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_dataset_folders(source_dir: Path) -> list[Path]:
    summary_path = source_dir / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        folders: list[Path] = []
        for row in summary.itertuples(index=False):
            out_folder = getattr(row, "output_folder", None)
            candidate = Path(str(out_folder)) if out_folder is not None else None
            if candidate is not None:
                # Previous runs may have Windows-style paths in summary.csv.
                if not candidate.exists():
                    candidate = source_dir / Path(str(out_folder).replace("\\", "/")).name
                if candidate.exists():
                    folders.append(candidate)
        if folders:
            return sorted(dict.fromkeys(folders))
    return sorted(p for p in source_dir.iterdir() if p.is_dir() and (p / "X_preprocessed.csv").exists() and (p / "y.csv").exists())


def read_xy(folder: Path) -> tuple[pd.DataFrame, pd.Series]:
    x_path = folder / "X_preprocessed.csv"
    y_path = folder / "y.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("missing X_preprocessed.csv or y.csv")
    X = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    if y_df.shape[1] != 1:
        raise ValueError(f"y.csv must contain exactly one column, got shape={y_df.shape}")
    return X, y_df.iloc[:, 0]


def sanitize_X(X: pd.DataFrame, *, drop_constant: bool) -> tuple[np.ndarray, list[int], list[str], list[str], int]:
    """Coerce X to numeric finite matrix and return kept column indices.

    The source data should already be imputed/scaled/one-hot encoded. This stage
    is a defensive gate for HPC runs: it removes accidental non-numeric columns,
    handles NaN/Inf if any slipped through, and optionally removes constant
    features that cannot help KNN/SVM/RF or wrapper FS.
    """
    X_num = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    dropped_non_numeric = [str(c) for c in X_num.columns[X_num.isna().all()].tolist()]
    if dropped_non_numeric:
        X_num = X_num.drop(columns=dropped_non_numeric)
    if X_num.shape[1] == 0:
        raise ValueError("no numeric features remain after coercion")

    missing_before = int(X_num.isna().sum().sum())
    if missing_before:
        med = X_num.median(axis=0, skipna=True).fillna(0.0)
        X_num = X_num.fillna(med)

    kept_indices = [int(X.columns.get_loc(c)) for c in X_num.columns]
    dropped_constant: list[str] = []
    if drop_constant:
        variances = X_num.var(axis=0, ddof=0).to_numpy(dtype=float)
        keep_mask = np.isfinite(variances) & (variances > 0.0)
        dropped_constant = [str(c) for c, keep in zip(X_num.columns, keep_mask) if not keep]
        if not np.any(keep_mask):
            raise ValueError("all features are constant after sanitisation")
        kept_indices = [idx for idx, keep in zip(kept_indices, keep_mask) if keep]
        X_num = X_num.loc[:, keep_mask]

    X_arr = X_num.to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X still contains NaN/Inf after sanitisation")
    return X_arr, kept_indices, dropped_non_numeric, dropped_constant, missing_before


def encode_y(y: pd.Series) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    if y.isna().any():
        raise ValueError("target contains missing values")
    if pd.api.types.is_numeric_dtype(y):
        y_num = pd.to_numeric(y, errors="coerce")
        if y_num.isna().any():
            raise ValueError("numeric target contains non-numeric values")
        y_int = y_num.astype(int).to_numpy()
        if not np.allclose(y_num.to_numpy(dtype=float), y_int.astype(float)):
            raise ValueError("classification target is not integer encoded")
        classes = np.unique(y_int)
        mapping = {str(int(cls)): int(i) for i, cls in enumerate(classes)}
        y_enc = np.asarray([mapping[str(int(v))] for v in y_int], dtype=np.int64)
    else:
        labels, y_enc = np.unique(y.astype(str).to_numpy(), return_inverse=True)
        mapping = {str(label): int(i) for i, label in enumerate(labels)}
        y_enc = y_enc.astype(np.int64)
    if np.unique(y_enc).size < 2:
        raise ValueError("target has fewer than two classes")
    counts = pd.Series(y_enc).value_counts().sort_index().astype(int).to_dict()
    return y_enc, mapping, {str(k): int(v) for k, v in counts.items()}


def filtered_feature_names(folder: Path, kept_indices: list[int], n_features: int) -> list[dict[str, str]]:
    old_path = folder / "feature_names.json"
    old_records = []
    if old_path.exists():
        try:
            old_records = json.loads(old_path.read_text(encoding="utf-8"))
        except Exception:
            old_records = []
    transformed: list[str] = []
    for original_idx in kept_indices:
        name = None
        if 0 <= original_idx < len(old_records):
            rec = old_records[original_idx]
            name = rec.get("transformed_name") if isinstance(rec, dict) else None
        transformed.append(str(name or f"feature_{original_idx}"))
    # Safety if feature-name metadata was shorter than X.
    if len(transformed) < n_features:
        transformed.extend(f"feature_{i}" for i in range(len(transformed), n_features))
    return [
        {"feature_index": int(i), "feature_id": f"f{i:05d}", "transformed_name": str(transformed[i])}
        for i in range(n_features)
    ]


def process_folder(folder: Path, out_root: Path, args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    meta = load_json(folder / "metadata.json")
    source_name = meta.get("dataset_name") or folder.name
    task_id = meta.get("task_id")
    dataset_key = sanitize_key(f"{source_name}_task{task_id}" if task_id is not None else folder.name)

    try:
        X_df, y_series = read_xy(folder)
        n_samples = int(X_df.shape[0])
        n_raw_features = int(meta.get("n_features_raw", X_df.shape[1]))
        n_processed_features = int(X_df.shape[1])
        basis_count = n_processed_features if args.feature_count_basis == "processed" else n_raw_features

        if n_samples > args.max_samples:
            return None, {"dataset_key": dataset_key, "source_folder": str(folder), "reason": "too_many_samples", "detail": n_samples}
        if basis_count < args.min_features:
            return None, {
                "dataset_key": dataset_key,
                "source_folder": str(folder),
                "reason": f"too_few_{args.feature_count_basis}_features",
                "detail": basis_count,
            }

        y, class_mapping, class_distribution = encode_y(y_series)
        X, kept_indices, dropped_non_numeric, dropped_constant, n_imputed = sanitize_X(
            X_df, drop_constant=str2bool(args.drop_constant)
        )
        if X.shape[1] < args.min_features:
            return None, {
                "dataset_key": dataset_key,
                "source_folder": str(folder),
                "reason": "too_few_features_after_sanitisation",
                "detail": X.shape[1],
            }
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y row mismatch after loading: {X.shape[0]} vs {y.shape[0]}")

        out_dir = out_root / dataset_key
        out_dir.mkdir(parents=True, exist_ok=True)
        dtype = np.float32 if args.float_dtype == "float32" else np.float64
        np.savez_compressed(out_dir / "dataset.npz", X=X.astype(dtype, copy=False), y=y.astype(np.int64), n_classes=np.array([np.unique(y).size], dtype=np.int64))

        feature_records = filtered_feature_names(folder, kept_indices, X.shape[1])
        write_json(out_dir / "feature_names.json", feature_records)

        # Preserve original class mapping if available, otherwise write the validated mapping.
        original_class_mapping = load_json(folder / "class_mapping.json") or class_mapping
        write_json(out_dir / "class_mapping.json", original_class_mapping)

        prepared_meta = PreparedDatasetMetadata(
            dataset_key=dataset_key,
            source_folder=str(folder),
            source_dataset_name=str(source_name) if source_name is not None else None,
            task_id=int(task_id) if task_id is not None and not pd.isna(task_id) else None,
            dataset_id=int(meta.get("dataset_id")) if meta.get("dataset_id") is not None else None,
            n_samples=int(X.shape[0]),
            n_features=int(X.shape[1]),
            n_classes=int(np.unique(y).size),
            class_distribution=class_distribution,
            feature_count_basis=args.feature_count_basis,
            max_samples=int(args.max_samples),
            min_features=int(args.min_features),
            dropped_non_numeric_columns=dropped_non_numeric,
            dropped_constant_columns=dropped_constant,
            imputed_missing_values=int(n_imputed),
            dtype=args.float_dtype,
            preprocessing_scope="existing_full_dataset_processed_then_validated_for_blur_ga",
            notes=[
                "No OpenML download was performed.",
                "Source X_preprocessed.csv/y.csv were consumed from an existing local folder.",
                "The matrix is numeric, finite, classification-labelled, and saved as compressed npz for HPC use.",
            ],
        )
        write_json(out_dir / "metadata.json", asdict(prepared_meta))

        row = {
            "dataset_key": dataset_key,
            "data_path": str(out_dir / "dataset.npz"),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(np.unique(y).size),
            "source_dataset_name": source_name,
            "task_id": task_id,
            "dataset_id": meta.get("dataset_id"),
            "source_folder": str(folder),
            "prepared_folder": str(out_dir),
            "dtype": args.float_dtype,
        }
        return row, None
    except Exception as exc:
        return None, {"dataset_key": dataset_key, "source_folder": str(folder), "reason": "error", "detail": repr(exc)}


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    out_root = Path(args.output_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    folders = discover_dataset_folders(source_dir)
    if args.limit is not None:
        folders = folders[: args.limit]

    manifest_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    print(f"Source dir: {source_dir.resolve()}")
    print(f"Output dir: {out_root.resolve()}")
    print(f"Discovered processed dataset folders: {len(folders)}")
    print(f"Filter: n_samples <= {args.max_samples}, {args.feature_count_basis}_features >= {args.min_features}")

    for i, folder in enumerate(folders, start=1):
        row, skipped = process_folder(folder, out_root, args)
        if row is not None:
            manifest_rows.append(row)
            print(f"[{i:03d}/{len(folders):03d}] OK   {row['dataset_key']} | X={row['n_samples']}x{row['n_features']}")
        if skipped is not None:
            skipped_rows.append(skipped)
            print(f"[{i:03d}/{len(folders):03d}] SKIP {skipped['dataset_key']} | {skipped['reason']} | {skipped.get('detail', '')}")

    manifest = pd.DataFrame(manifest_rows).sort_values(["n_samples", "n_features", "dataset_key"]) if manifest_rows else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped_rows)
    manifest.to_csv(out_root / "manifest.csv", index=False)
    skipped_df.to_csv(out_root / "skipped_datasets.csv", index=False)
    print("\nDone.")
    print(f"Prepared datasets: {len(manifest_rows)}")
    print(f"Skipped datasets: {len(skipped_rows)}")
    print(f"Manifest: {out_root / 'manifest.csv'}")
    print(f"Skipped log: {out_root / 'skipped_datasets.csv'}")


if __name__ == "__main__":
    main()
