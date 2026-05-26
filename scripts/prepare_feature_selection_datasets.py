#!/usr/bin/env python3
"""Download and prepare feature-selection evaluation datasets for BLuR-GA.

This script prepares the dataset table used for the Feature Selection Performance
Evaluation.  It outputs the compact format already consumed by ``run_blur_ga.py``:

    prepared_feature_selection/
      manifest.csv
      dataset_registry.csv
      skipped_datasets.csv
      missing_datasets.csv
      dataset_groups/
        small_tabular.txt
        medium_tabular.txt
        medium_highdim.txt
        large_highdim.txt
        small.txt              # compatibility alias: small_tabular
        medium.txt             # compatibility alias: medium_tabular
        highdim.txt            # compatibility alias: medium_highdim + large_highdim
        all.txt
      <dataset_key>/
        dataset.npz            # arrays: X, y, n_classes
        metadata.json
        feature_names.json
        class_mapping.json

Preprocessing is intentionally conservative and unsupervised:

- classification targets are label-encoded to 0..K-1;
- numeric columns: median imputation + optional StandardScaler;
- categorical/string columns: missing-category imputation + OneHotEncoder;
- ASU/scikit-feature .mat datasets: numeric validation + median imputation + scaling;
- zero-variance transformed features are removed;
- all processed features are finite numeric arrays.

The script supports two modes:

1. ``--mode download``: download/load datasets from UCI, OpenML/TabArena, and
   ASU/scikit-feature, then preprocess them.
2. ``--mode existing``: convert/copy an already prepared folder. This is useful
   for recovery when raw data has already been processed locally.

Important evaluation note:
The output ``dataset.npz`` is a globally preprocessed matrix for compatibility
with the current BLuR-GA code base. This uses only unsupervised preprocessing,
but the strictest ML protocol would fit imputers/scalers/encoders inside each
outer-training fold. If the project later supports fold-wise preprocessing,
prefer that for final leakage-free evaluation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import string
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases"
ASU_GITHUB_RAW = "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data"
ASU_WEB = "https://jundongl.github.io/scikit-feature/files/datasets"

CANONICAL_GROUPS = ("small_tabular", "medium_tabular", "medium_highdim", "large_highdim")
GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "small": ("small_tabular",),
    "medium": ("medium_tabular",),
    "highdim": ("medium_highdim", "large_highdim"),
    "all": CANONICAL_GROUPS,
}
ALL_GROUP_CHOICES = tuple(list(CANONICAL_GROUPS) + list(GROUP_ALIASES))


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    group: str
    display_name: str
    source: str
    expected_n_samples: int
    expected_n_features_raw: int
    expected_n_classes: int
    loader: str
    openml_data_id: int | None = None
    openml_name: str | None = None
    openml_version: int | None = None
    urls: tuple[str, ...] = ()
    asu_mat_name: str | None = None
    notes: str = ""


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    # ------------------------------------------------------------------
    # Small tabular datasets.
    # ------------------------------------------------------------------
    DatasetSpec(
        key="lymphography_uci",
        group="small_tabular",
        display_name="Lymphography",
        source="UCI",
        expected_n_samples=148,
        expected_n_features_raw=18,
        expected_n_classes=4,
        loader="uci_lymphography",
        urls=(f"{UCI_BASE}/lymphography/lymphography.data",),
    ),
    DatasetSpec(
        key="sonar_uci",
        group="small_tabular",
        display_name="Sonar",
        source="UCI",
        expected_n_samples=208,
        expected_n_features_raw=60,
        expected_n_classes=2,
        loader="uci_sonar",
        urls=(f"{UCI_BASE}/undocumented/connectionist-bench/sonar/sonar.all-data",),
    ),
    DatasetSpec(
        key="ionosphere_uci",
        group="small_tabular",
        display_name="Ionosphere",
        source="UCI",
        expected_n_samples=351,
        expected_n_features_raw=34,
        expected_n_classes=2,
        loader="uci_ionosphere",
        urls=(f"{UCI_BASE}/ionosphere/ionosphere.data",),
    ),
    DatasetSpec(
        key="heart_disease_uci",
        group="small_tabular",
        display_name="Heart Disease",
        source="UCI",
        expected_n_samples=303,
        expected_n_features_raw=13,
        expected_n_classes=2,
        loader="uci_heart",
        urls=(f"{UCI_BASE}/heart-disease/processed.cleveland.data",),
        notes="Cleveland heart-disease target is binarised as disease/no-disease.",
    ),
    DatasetSpec(
        key="australian_credit_uci_openml",
        group="small_tabular",
        display_name="Australian Credit",
        source="UCI / OpenML",
        expected_n_samples=690,
        expected_n_features_raw=14,
        expected_n_classes=2,
        loader="uci_australian_credit",
        urls=(f"{UCI_BASE}/statlog/australian/australian.dat",),
    ),
    DatasetSpec(
        key="vehicle_uci",
        group="small_tabular",
        display_name="Vehicle",
        source="UCI",
        expected_n_samples=846,
        expected_n_features_raw=18,
        expected_n_classes=4,
        loader="uci_vehicle",
        urls=tuple(f"{UCI_BASE}/statlog/vehicle/x{suffix}.dat" for suffix in ("aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai")),
    ),
    DatasetSpec(
        key="hepatitis_uci",
        group="small_tabular",
        display_name="Hepatitis",
        source="UCI",
        expected_n_samples=155,
        expected_n_features_raw=19,
        expected_n_classes=2,
        loader="uci_hepatitis",
        urls=(f"{UCI_BASE}/hepatitis/hepatitis.data",),
    ),
    DatasetSpec(
        key="parkinsons_uci",
        group="small_tabular",
        display_name="Parkinsons",
        source="UCI",
        expected_n_samples=195,
        expected_n_features_raw=22,
        expected_n_classes=2,
        loader="uci_parkinsons",
        urls=(f"{UCI_BASE}/parkinsons/parkinsons.data",),
    ),
    # ------------------------------------------------------------------
    # Medium tabular datasets.
    # ------------------------------------------------------------------
    DatasetSpec(
        key="credit-g_task363611",
        group="medium_tabular",
        display_name="Credit-g",
        source="OpenML / TabArena",
        expected_n_samples=1000,
        expected_n_features_raw=21,
        expected_n_classes=2,
        loader="openml",
        openml_data_id=46918,
        openml_name="credit-g",
    ),
    DatasetSpec(
        key="qsar_biodeg_openml",
        group="medium_tabular",
        display_name="QSAR biodeg",
        source="UCI / OpenML",
        expected_n_samples=1054,
        expected_n_features_raw=42,
        expected_n_classes=2,
        loader="openml",
        openml_data_id=1494,
        openml_name="qsar-biodeg",
    ),
    DatasetSpec(
        key="website_phishing_task363707",
        group="medium_tabular",
        display_name="Website phishing",
        source="UCI / OpenML / TabArena",
        expected_n_samples=1353,
        expected_n_features_raw=10,
        expected_n_classes=3,
        loader="openml",
        openml_data_id=46963,
        openml_name="website_phishing",
    ),
    DatasetSpec(
        key="MIC_task363687",
        group="medium_tabular",
        display_name="MIC",
        source="TabArena",
        expected_n_samples=1699,
        expected_n_features_raw=112,
        expected_n_classes=8,
        loader="openml",
        openml_data_id=46943,
        openml_name="MIC",
    ),
    # Kept in medium_tabular because this is the grouping in the current table.
    # If you want grouping by dimensionality, move Isolet_asu to medium_highdim.
    DatasetSpec(
        key="Isolet_asu",
        group="medium_tabular",
        display_name="Isolet",
        source="ASU / scikit-feature / UCI",
        expected_n_samples=1560,
        expected_n_features_raw=617,
        expected_n_classes=26,
        loader="asu_mat",
        asu_mat_name="Isolet.mat",
    ),
    DatasetSpec(
        key="Is-this-a-good-customer_task363682",
        group="medium_tabular",
        display_name="Is-this-a-good-customer",
        source="TabArena",
        expected_n_samples=1723,
        expected_n_features_raw=14,
        expected_n_classes=2,
        loader="openml",
        openml_data_id=46938,
    ),
    DatasetSpec(
        key="Marketing_Campaign_task363684",
        group="medium_tabular",
        display_name="Marketing Campaign",
        source="TabArena",
        expected_n_samples=2240,
        expected_n_features_raw=26,
        expected_n_classes=2,
        loader="openml",
        openml_data_id=46940,
    ),
    DatasetSpec(
        key="hazelnut-spread-contaminant-detection_task363674",
        group="medium_tabular",
        display_name="Hazelnut contaminant detection",
        source="TabArena",
        expected_n_samples=2400,
        expected_n_features_raw=31,
        expected_n_classes=2,
        loader="openml",
        openml_data_id=46930,
    ),
    DatasetSpec(
        key="seismic-bumps_task363700",
        group="medium_tabular",
        display_name="Seismic-bumps",
        source="TabArena / OpenML",
        expected_n_samples=2584,
        expected_n_features_raw=16,
        expected_n_classes=2,
        loader="openml",
        openml_data_id=46956,
        openml_name="seismic-bumps",
    ),
    DatasetSpec(
        key="splice_uci_openml_tabarena",
        group="medium_tabular",
        display_name="Splice",
        source="UCI / OpenML / TabArena",
        expected_n_samples=3190,
        expected_n_features_raw=61,
        expected_n_classes=3,
        loader="uci_splice",
        urls=(f"{UCI_BASE}/molecular-biology/splice-junction-gene-sequences/splice.data",),
        notes="UCI file has class, sequence id, and DNA string. The sequence id is dropped; the DNA string is expanded to per-position categorical features.",
    ),
    DatasetSpec(
        key="segment_openml_cc18_uci",
        group="medium_tabular",
        display_name="Segment",
        source="OpenML-CC18 / UCI",
        expected_n_samples=2310,
        expected_n_features_raw=19,
        expected_n_classes=7,
        loader="openml",
        openml_data_id=40984,
        openml_name="segment",
    ),
    # ------------------------------------------------------------------
    # Medium high-dimensional datasets.
    # ------------------------------------------------------------------
    DatasetSpec(
        key="ORL_asu",
        group="medium_highdim",
        display_name="ORL",
        source="ASU / scikit-feature",
        expected_n_samples=400,
        expected_n_features_raw=1024,
        expected_n_classes=40,
        loader="asu_mat",
        asu_mat_name="ORL.mat",
    ),
    DatasetSpec(
        key="warpAR10P_asu",
        group="medium_highdim",
        display_name="warpAR10P",
        source="ASU / scikit-feature",
        expected_n_samples=130,
        expected_n_features_raw=2400,
        expected_n_classes=10,
        loader="asu_mat",
        asu_mat_name="warpAR10P.mat",
    ),
    DatasetSpec(
        key="warpPIE10P_asu",
        group="medium_highdim",
        display_name="warpPIE10P",
        source="ASU / scikit-feature",
        expected_n_samples=210,
        expected_n_features_raw=2420,
        expected_n_classes=10,
        loader="asu_mat",
        asu_mat_name="warpPIE10P.mat",
    ),
    DatasetSpec(
        key="Yale_asu",
        group="medium_highdim",
        display_name="Yale",
        source="ASU / scikit-feature",
        expected_n_samples=165,
        expected_n_features_raw=1024,
        expected_n_classes=15,
        loader="asu_mat",
        asu_mat_name="Yale.mat",
    ),
    DatasetSpec(
        key="lung_discrete_asu",
        group="medium_highdim",
        display_name="lung_discrete",
        source="ASU / scikit-feature",
        expected_n_samples=73,
        expected_n_features_raw=325,
        expected_n_classes=7,
        loader="asu_mat",
        asu_mat_name="lung_discrete.mat",
    ),
    DatasetSpec(
        key="colon_asu",
        group="medium_highdim",
        display_name="Colon",
        source="ASU / scikit-feature",
        expected_n_samples=62,
        expected_n_features_raw=2000,
        expected_n_classes=2,
        loader="asu_mat",
        asu_mat_name="colon.mat",
    ),
    # ------------------------------------------------------------------
    # Large high-dimensional datasets.
    # ------------------------------------------------------------------
    DatasetSpec(
        key="GLIOMA_asu",
        group="large_highdim",
        display_name="GLIOMA",
        source="ASU / scikit-feature",
        expected_n_samples=50,
        expected_n_features_raw=4434,
        expected_n_classes=4,
        loader="asu_mat",
        asu_mat_name="GLIOMA.mat",
    ),
    DatasetSpec(
        key="Lung_asu",
        group="large_highdim",
        display_name="Lung",
        source="ASU / scikit-feature",
        expected_n_samples=203,
        expected_n_features_raw=3312,
        expected_n_classes=5,
        loader="asu_mat",
        asu_mat_name="lung.mat",
    ),
    DatasetSpec(
        key="Lymphoma_asu",
        group="large_highdim",
        display_name="Lymphoma",
        source="ASU / scikit-feature",
        expected_n_samples=96,
        expected_n_features_raw=4026,
        expected_n_classes=9,
        loader="asu_mat",
        asu_mat_name="lymphoma.mat",
    ),
    DatasetSpec(
        key="Prostate_GE_asu",
        group="large_highdim",
        display_name="Prostate_GE",
        source="ASU / scikit-feature",
        expected_n_samples=102,
        expected_n_features_raw=5966,
        expected_n_classes=2,
        loader="asu_mat",
        asu_mat_name="Prostate-GE.mat",
    ),
    DatasetSpec(
        key="TOX_171_asu",
        group="large_highdim",
        display_name="TOX_171",
        source="ASU / scikit-feature",
        expected_n_samples=171,
        expected_n_features_raw=5748,
        expected_n_classes=4,
        loader="asu_mat",
        asu_mat_name="TOX-171.mat",
    ),
)


@dataclass
class PreparedDatasetMetadata:
    dataset_key: str
    group: str
    display_name: str
    source: str
    source_detail: str
    n_samples: int
    n_features: int
    n_features_raw_expected: int
    n_classes: int
    class_distribution: dict[str, int]
    expected_n_samples: int
    expected_n_features_raw: int
    expected_n_classes: int
    dtype: str
    preprocessing_version: str
    preprocessing_scope: str
    standardize_numeric: bool
    numeric_columns: list[str]
    categorical_columns: list[str]
    dropped_identifier_columns: list[str]
    dropped_all_missing_columns: list[str]
    dropped_constant_columns: list[str]
    imputed_missing_values: int
    categorical_missing_strategy: str
    max_categories: int | None
    shape_warnings: list[str]
    notes: list[str]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download/process BLuR-GA feature-selection evaluation datasets.")
    p.add_argument("--mode", choices=["download", "existing"], default="download")
    p.add_argument("--output-dir", default="prepared_feature_selection")
    p.add_argument("--raw-cache-dir", default="dataset_cache/raw")
    p.add_argument("--processed-dir", default="dataset_cache/processed")
    p.add_argument("--existing-prepared-dir", default=None)
    p.add_argument("--groups", nargs="+", choices=ALL_GROUP_CHOICES, default=["all"], help="Dataset groups or compatibility aliases to prepare.")
    p.add_argument("--datasets", nargs="*", default=None, help="Optional explicit dataset keys to prepare.")
    p.add_argument("--float-dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--standardize-numeric", type=str2bool, default=True, help="Apply StandardScaler to numeric/transformed numeric features.")
    p.add_argument("--categorical-missing-strategy", choices=["missing_category", "most_frequent"], default="missing_category")
    p.add_argument("--max-categories", type=int, default=0, help="OneHotEncoder max_categories; 0 disables the cap.")
    p.add_argument("--save-processed-csv", type=str2bool, default=False, help="Optionally save X_preprocessed.csv/y.csv audit copies. Can be large.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Limit selected specs for quick smoke tests.")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--list-datasets", action="store_true", help="Print the registry after group/dataset filtering and exit.")
    return p.parse_args(argv)


def resolve_groups(groups: Iterable[str]) -> set[str]:
    resolved: set[str] = set()
    for group in groups:
        if group in GROUP_ALIASES:
            resolved.update(GROUP_ALIASES[group])
        else:
            resolved.add(group)
    return resolved


def selected_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    group_set = resolve_groups(args.groups)
    specs = [s for s in DATASET_SPECS if s.group in group_set]
    if args.datasets:
        wanted = set(args.datasets)
        specs = [s for s in specs if s.key in wanted]
        known = {s.key for s in DATASET_SPECS}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"Unknown dataset key(s): {unknown}")
        filtered_out = sorted(wanted - {s.key for s in specs})
        if filtered_out:
            raise ValueError(f"Dataset key(s) not in selected groups: {filtered_out}")
    if args.limit is not None:
        specs = specs[: int(args.limit)]
    if not specs:
        raise ValueError("No datasets selected.")
    return specs


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def download_file(url: str, dest: Path, *, timeout: int = 180) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"    downloading {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(dest)
    return dest


def _make_feature_names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i:03d}" for i in range(n)]


def load_uci_lymphography(spec: DatasetSpec, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    path = download_file(spec.urls[0], raw_dir / spec.key / "lymphography.data")
    df = pd.read_csv(path, header=None, na_values="?")
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:].copy()
    X.columns = _make_feature_names("lymph", X.shape[1])
    return X, y, str(path), []


def load_uci_sonar(spec: DatasetSpec, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    path = download_file(spec.urls[0], raw_dir / spec.key / "sonar.all-data")
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1].copy()
    X.columns = _make_feature_names("sonar", X.shape[1])
    return X, y, str(path), []


def load_uci_ionosphere(spec: DatasetSpec, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    path = download_file(spec.urls[0], raw_dir / spec.key / "ionosphere.data")
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1].copy()
    X.columns = _make_feature_names("ionosphere", X.shape[1])
    return X, y, str(path), []


def load_uci_heart(spec: DatasetSpec, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    path = download_file(spec.urls[0], raw_dir / spec.key / "processed.cleveland.data")
    cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv(path, header=None, names=cols, na_values="?")
    target = pd.to_numeric(df.pop("target"), errors="coerce")
    if target.isna().any():
        raise ValueError("Heart Disease target contains missing/non-numeric values")
    y = (target > 0).astype(int)
    return df, y, str(path), []


def load_uci_australian_credit(spec: DatasetSpec, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    path = download_file(spec.urls[0], raw_dir / spec.key / "australian.dat")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1].copy()
    X.columns = _make_feature_names("aus", X.shape[1])
    return X, y, str(path), []


def load_uci_vehicle(spec: DatasetSpec, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    frames: list[pd.DataFrame] = []
    paths: list[str] = []
    for url in spec.urls:
        path = download_file(url, raw_dir / spec.key / Path(url).name)
        paths.append(str(path))
        frames.append(pd.read_csv(path, sep=r"\s+", header=None))
    df = pd.concat(frames, ignore_index=True)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1].copy()
    X.columns = _make_feature_names("vehicle", X.shape[1])
    return X, y, ";".join(paths), []


def load_uci_hepatitis(spec: DatasetSpec, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    path = download_file(spec.urls[0], raw_dir / spec.key / "hepatitis.data")
    df = pd.read_csv(path, header=None, na_values="?")
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:].copy()
    X.columns = _make_feature_names("hepatitis", X.shape[1])
    return X, y, str(path), []


def load_uci_parkinsons(spec: DatasetSpec, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    path = download_file(spec.urls[0], raw_dir / spec.key / "parkinsons.data")
    df = pd.read_csv(path)
    y = df.pop("status")
    dropped = []
    if "name" in df.columns:
        df = df.drop(columns=["name"])
        dropped.append("name")
    return df, y, str(path), dropped


def load_uci_splice(spec: DatasetSpec, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    path = download_file(spec.urls[0], raw_dir / spec.key / "splice.data")
    df = pd.read_csv(path, header=None, names=["class", "instance_id", "sequence"], skipinitialspace=True)
    y = df["class"].astype(str).str.strip()
    seq = df["sequence"].astype(str).str.strip().str.upper()
    max_len = int(seq.str.len().max())
    if max_len <= 0:
        raise ValueError("Splice sequences are empty")
    rows = []
    for s in seq:
        chars = list(s)
        if len(chars) < max_len:
            chars.extend(["N"] * (max_len - len(chars)))
        rows.append(chars[:max_len])
    X = pd.DataFrame(rows, columns=[f"dna_pos_{i:02d}" for i in range(max_len)])
    return X, y, str(path), ["instance_id"]


def load_openml_dataset(spec: DatasetSpec, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    cache = raw_dir / "openml_cache"
    cache.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"as_frame": True, "cache": True, "data_home": str(cache)}
    if spec.openml_data_id is not None:
        kwargs["data_id"] = int(spec.openml_data_id)
    elif spec.openml_name is not None:
        kwargs["name"] = spec.openml_name
        if spec.openml_version is not None:
            kwargs["version"] = int(spec.openml_version)
    else:
        raise ValueError(f"{spec.key} has neither OpenML data_id nor name")
    try:
        bunch = fetch_openml(parser="auto", **kwargs)
    except TypeError:
        bunch = fetch_openml(**kwargs)
    X = bunch.data
    y = bunch.target
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"OpenML target for {spec.key} has {y.shape[1]} columns")
        y = y.iloc[:, 0]
    if y is None:
        raise ValueError(f"OpenML dataset {spec.key} has no default target")
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    y_series = pd.Series(y, name="target")
    detail = f"openml_data_id={spec.openml_data_id}" if spec.openml_data_id is not None else f"openml_name={spec.openml_name}"
    return X, y_series, detail, []


def _find_mat_xy(mat: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    keys = {k.lower(): k for k in mat.keys() if not k.startswith("__")}
    x_key = keys.get("x") or keys.get("data") or keys.get("fea") or keys.get("features")
    y_key = keys.get("y") or keys.get("label") or keys.get("labels") or keys.get("gnd") or keys.get("target")
    if x_key is None or y_key is None:
        public_keys = [k for k in mat.keys() if not k.startswith("__")]
        raise ValueError(f"Could not identify X/y arrays in .mat file. Keys={public_keys}")
    X = np.asarray(mat[x_key])
    y = np.asarray(mat[y_key]).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X in .mat must be 2-D, got shape={X.shape}")
    if X.shape[0] != y.shape[0] and X.shape[1] == y.shape[0]:
        X = X.T
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y mismatch in .mat: X={X.shape}, y={y.shape}")
    return X, y


def load_asu_mat(spec: DatasetSpec, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    if not spec.asu_mat_name:
        raise ValueError(f"{spec.key} has no ASU .mat file name")
    path = raw_dir / spec.key / spec.asu_mat_name
    urls = [f"{ASU_GITHUB_RAW}/{spec.asu_mat_name}", f"{ASU_WEB}/{spec.asu_mat_name}"]
    errors: list[str] = []
    for url in urls:
        try:
            download_file(url, path, timeout=240)
            break
        except Exception as exc:  # network dependent
            errors.append(f"{url}: {exc!r}")
            if path.exists() and path.stat().st_size == 0:
                path.unlink(missing_ok=True)
    else:
        raise RuntimeError("Failed to download ASU .mat dataset: " + " | ".join(errors))
    X, y = _find_mat_xy(loadmat(path))
    X_df = pd.DataFrame(X, columns=[f"f{i:05d}" for i in range(X.shape[1])])
    return X_df, pd.Series(y, name="target"), str(path), []


def load_raw(spec: DatasetSpec, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    loaders: dict[str, Callable[[DatasetSpec, Path], tuple[pd.DataFrame, pd.Series, str, list[str]]]] = {
        "uci_lymphography": load_uci_lymphography,
        "uci_sonar": load_uci_sonar,
        "uci_ionosphere": load_uci_ionosphere,
        "uci_heart": load_uci_heart,
        "uci_australian_credit": load_uci_australian_credit,
        "uci_vehicle": load_uci_vehicle,
        "uci_hepatitis": load_uci_hepatitis,
        "uci_parkinsons": load_uci_parkinsons,
        "uci_splice": load_uci_splice,
        "openml": load_openml_dataset,
        "asu_mat": load_asu_mat,
    }
    if spec.loader not in loaders:
        raise ValueError(f"Unknown loader {spec.loader!r} for {spec.key}")
    return loaders[spec.loader](spec, raw_dir)


def encode_target(y: pd.Series) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    y_series = pd.Series(y)
    if y_series.isna().any():
        raise ValueError("target contains missing values")
    enc = LabelEncoder()
    if pd.api.types.is_numeric_dtype(y_series):
        values = pd.to_numeric(y_series, errors="raise").to_numpy()
        y_enc = enc.fit_transform(values).astype(np.int64)
        class_mapping = {str(label): int(i) for i, label in enumerate(enc.classes_.tolist())}
    else:
        values = y_series.astype(str).str.strip().to_numpy()
        y_enc = enc.fit_transform(values).astype(np.int64)
        class_mapping = {str(label): int(i) for i, label in enumerate(enc.classes_.tolist())}
    if np.unique(y_enc).size < 2:
        raise ValueError("target has fewer than two classes")
    counts = pd.Series(y_enc).value_counts().sort_index().astype(int).to_dict()
    return y_enc, class_mapping, {str(k): int(v) for k, v in counts.items()}


def make_one_hot(max_categories: int | None) -> OneHotEncoder:
    kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
    if max_categories is not None and max_categories > 0:
        kwargs["max_categories"] = int(max_categories)
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)


def split_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    X = X.copy()
    X.columns = [str(c) for c in X.columns]
    all_missing = [str(c) for c in X.columns[X.isna().all()].tolist()]
    if all_missing:
        X = X.drop(columns=all_missing)
    numeric_cols = [str(c) for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [str(c) for c in X.columns if str(c) not in numeric_cols]
    return X, numeric_cols, categorical_cols, all_missing


def preprocess_dataframe(
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    *,
    spec: DatasetSpec,
    dropped_identifier_columns: list[str],
    standardize_numeric: bool,
    max_categories: int | None,
    categorical_missing_strategy: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, int], dict[str, int], dict[str, Any]]:
    X = X_raw.copy()
    X.columns = [str(c) for c in X.columns]
    y, class_mapping, class_distribution = encode_target(y_raw)

    if spec.loader == "asu_mat":
        X_num = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        dropped_all_missing = [str(c) for c in X_num.columns[X_num.isna().all()].tolist()]
        if dropped_all_missing:
            X_num = X_num.drop(columns=dropped_all_missing)
        missing = int(X_num.isna().sum().sum())
        if missing:
            med = X_num.median(axis=0, skipna=True).fillna(0.0)
            X_num = X_num.fillna(med)
        arr = X_num.to_numpy(dtype=np.float64, copy=True)
        if not np.all(np.isfinite(arr)):
            raise ValueError("high-dimensional X contains non-finite values after imputation")
        if standardize_numeric:
            arr = StandardScaler().fit_transform(arr)
        base_names = [str(c) for c in X_num.columns]
        numeric_cols = base_names
        categorical_cols: list[str] = []
    else:
        X, numeric_cols, categorical_cols, dropped_all_missing = split_columns(X)
        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if numeric_cols:
            numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
            if standardize_numeric:
                numeric_steps.append(("scaler", StandardScaler()))
            transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent") if categorical_missing_strategy == "most_frequent" else SimpleImputer(strategy="constant", fill_value="__missing__")
            transformers.append(("cat", Pipeline([("imputer", cat_imputer), ("onehot", make_one_hot(max_categories))]), categorical_cols))
        if not transformers:
            raise ValueError("no usable input columns remain")
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)
        arr = preprocessor.fit_transform(X)
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        arr = np.asarray(arr, dtype=np.float64)
        try:
            base_names = [str(n) for n in preprocessor.get_feature_names_out()]
        except Exception:
            base_names = [f"feature_{i}" for i in range(arr.shape[1])]
        missing = int(pd.DataFrame(X).isna().sum().sum())

    if not np.all(np.isfinite(arr)):
        raise ValueError("processed X contains NaN/Inf before variance filtering")
    selector = VarianceThreshold(threshold=0.0)
    X_proc = selector.fit_transform(arr)
    support = selector.get_support()
    dropped_constant = [name for name, keep in zip(base_names, support) if not keep]
    kept_names = [name for name, keep in zip(base_names, support) if keep]

    if X_proc.ndim != 2 or X_proc.shape[0] != y.shape[0]:
        raise ValueError(f"processed X/y shape mismatch: X={X_proc.shape}, y={y.shape}")
    if X_proc.shape[1] == 0:
        raise ValueError("no features remain after preprocessing")
    if not np.all(np.isfinite(X_proc)):
        raise ValueError("processed X contains non-finite values")

    feature_records = [
        {"feature_index": int(i), "feature_id": f"f{i:05d}", "transformed_name": str(name)}
        for i, name in enumerate(kept_names)
    ]
    audit = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "dropped_identifier_columns": dropped_identifier_columns,
        "dropped_all_missing_columns": dropped_all_missing,
        "dropped_constant_columns": dropped_constant,
        "imputed_missing_values": missing,
    }
    return X_proc, y, feature_records, class_mapping, class_distribution, audit


def shape_warnings(spec: DatasetSpec, X: np.ndarray, y: np.ndarray) -> list[str]:
    warnings: list[str] = []
    if X.shape[0] != spec.expected_n_samples:
        warnings.append(f"n_samples={X.shape[0]} differs from table expected_n_samples={spec.expected_n_samples}")
    n_classes = int(np.unique(y).size)
    if n_classes != spec.expected_n_classes:
        warnings.append(f"n_classes={n_classes} differs from table expected_n_classes={spec.expected_n_classes}")
    if X.shape[1] < 1:
        warnings.append("no processed features")
    # n_features after preprocessing may legitimately differ from table raw d because of one-hot and variance filtering.
    return warnings


def save_prepared(
    spec: DatasetSpec,
    out_root: Path,
    processed_root: Path,
    X: np.ndarray,
    y: np.ndarray,
    feature_records: list[dict[str, Any]],
    class_mapping: dict[str, int],
    class_distribution: dict[str, int],
    audit: dict[str, Any],
    args: argparse.Namespace,
    source_detail: str,
) -> dict[str, Any]:
    out_dir = out_root / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.float32 if args.float_dtype == "float32" else np.float64
    np.savez_compressed(
        out_dir / "dataset.npz",
        X=X.astype(dtype, copy=False),
        y=y.astype(np.int64, copy=False),
        n_classes=np.array([np.unique(y).size], dtype=np.int64),
    )
    write_json(out_dir / "feature_names.json", feature_records)
    write_json(out_dir / "class_mapping.json", class_mapping)

    if args.save_processed_csv:
        proc_dir = processed_root / spec.key
        proc_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(X, columns=[r["transformed_name"] for r in feature_records]).to_csv(proc_dir / "X_preprocessed.csv", index=False)
        pd.DataFrame({"target": y}).to_csv(proc_dir / "y.csv", index=False)

    warnings = shape_warnings(spec, X, y)
    meta = PreparedDatasetMetadata(
        dataset_key=spec.key,
        group=spec.group,
        display_name=spec.display_name,
        source=spec.source,
        source_detail=source_detail,
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        n_features_raw_expected=int(spec.expected_n_features_raw),
        n_classes=int(np.unique(y).size),
        class_distribution=class_distribution,
        expected_n_samples=int(spec.expected_n_samples),
        expected_n_features_raw=int(spec.expected_n_features_raw),
        expected_n_classes=int(spec.expected_n_classes),
        dtype=args.float_dtype,
        preprocessing_version="fs-prep-v2-table-30datasets",
        preprocessing_scope="global_unsupervised_fit_transform_for_current_blur_ga_matrix_interface",
        standardize_numeric=bool(args.standardize_numeric),
        numeric_columns=[str(c) for c in audit.get("numeric_columns", [])],
        categorical_columns=[str(c) for c in audit.get("categorical_columns", [])],
        dropped_identifier_columns=[str(c) for c in audit.get("dropped_identifier_columns", [])],
        dropped_all_missing_columns=[str(c) for c in audit.get("dropped_all_missing_columns", [])],
        dropped_constant_columns=[str(c) for c in audit.get("dropped_constant_columns", [])],
        imputed_missing_values=int(audit.get("imputed_missing_values", 0)),
        categorical_missing_strategy=args.categorical_missing_strategy,
        max_categories=(None if args.max_categories <= 0 else int(args.max_categories)),
        shape_warnings=warnings,
        notes=[
            spec.notes,
            "Processed features are numeric, finite, variance-filtered, and saved as dataset.npz.",
            "Use this folder with run_blur_ga.py --data-dir.",
            "Feature count after preprocessing can differ from table d because categorical features may be one-hot encoded and constants are removed.",
        ],
    )
    write_json(out_dir / "metadata.json", asdict(meta))
    return {
        "dataset_key": spec.key,
        "group": spec.group,
        "data_path": str(out_dir / "dataset.npz"),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_features_raw_expected": int(spec.expected_n_features_raw),
        "n_classes": int(np.unique(y).size),
        "source": spec.source,
        "display_name": spec.display_name,
        "expected_n_samples": int(spec.expected_n_samples),
        "expected_n_features_raw": int(spec.expected_n_features_raw),
        "expected_n_classes": int(spec.expected_n_classes),
        "dtype": args.float_dtype,
        "shape_warnings": " | ".join(warnings),
    }


def prepare_existing_processed_folder(spec: DatasetSpec, src: Path, dst: Path, *, float_dtype: str) -> dict[str, Any]:
    x_path = src / "X_preprocessed.csv"
    y_path = src / "y.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"missing X_preprocessed.csv/y.csv in {src}")
    X_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    if y_df.shape[1] != 1:
        raise ValueError(f"Expected one target column in {y_path}, got {y_df.shape[1]}")
    X_num = X_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing = int(X_num.isna().sum().sum())
    if missing:
        med = X_num.median(axis=0, skipna=True).fillna(0.0)
        X_num = X_num.fillna(med)
    X = X_num.to_numpy(dtype=np.float64, copy=True)
    y, class_mapping, class_distribution = encode_target(y_df.iloc[:, 0])
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y mismatch in {src}: X={X.shape}, y={y.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"Non-finite values remain in {src}")

    dst.mkdir(parents=True, exist_ok=True)
    dtype = np.float32 if float_dtype == "float32" else np.float64
    np.savez_compressed(dst / "dataset.npz", X=X.astype(dtype, copy=False), y=y.astype(np.int64), n_classes=np.array([np.unique(y).size], dtype=np.int64))

    if (src / "feature_names.json").exists():
        shutil.copy2(src / "feature_names.json", dst / "feature_names.json")
    else:
        write_json(dst / "feature_names.json", [
            {"feature_index": int(i), "feature_id": f"f{i:05d}", "transformed_name": str(col)}
            for i, col in enumerate(X_df.columns)
        ])
    if (src / "class_mapping.json").exists():
        shutil.copy2(src / "class_mapping.json", dst / "class_mapping.json")
    else:
        write_json(dst / "class_mapping.json", class_mapping)

    meta = {
        "dataset_key": spec.key,
        "group": spec.group,
        "display_name": spec.display_name,
        "source": spec.source,
        "source_detail": f"existing_processed_folder={src}",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_features_raw_expected": int(spec.expected_n_features_raw),
        "n_classes": int(np.unique(y).size),
        "class_distribution": class_distribution,
        "expected_n_samples": spec.expected_n_samples,
        "expected_n_features_raw": spec.expected_n_features_raw,
        "expected_n_classes": spec.expected_n_classes,
        "dtype": float_dtype,
        "preprocessing_version": "fs-prep-v2-existing-conversion",
        "preprocessing_scope": "existing_processed_csv_validated_and_converted_to_blur_ga_npz",
        "notes": [
            "Converted from an existing X_preprocessed.csv/y.csv folder without downloading.",
            "Use --mode download for a fully reproducible raw-download preprocessing run.",
        ],
    }
    write_json(dst / "metadata.json", meta)
    return {
        "dataset_key": spec.key,
        "group": spec.group,
        "data_path": str(dst / "dataset.npz"),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_features_raw_expected": int(spec.expected_n_features_raw),
        "n_classes": int(np.unique(y).size),
        "source": spec.source,
        "display_name": spec.display_name,
        "expected_n_samples": spec.expected_n_samples,
        "expected_n_features_raw": spec.expected_n_features_raw,
        "expected_n_classes": spec.expected_n_classes,
        "dtype": float_dtype,
        "shape_warnings": "",
    }


def copy_existing_prepared(specs: Iterable[DatasetSpec], out_root: Path, existing_root: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not existing_root.exists():
        raise FileNotFoundError(f"existing prepared dir not found: {existing_root}")
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    manifest_by_key: dict[str, dict[str, Any]] = {}
    manifest_path = existing_root / "manifest.csv"
    if manifest_path.exists():
        for row in pd.read_csv(manifest_path).to_dict("records"):
            key = str(row.get("dataset_key") or row.get("dataset") or "")
            if key:
                manifest_by_key[key] = row
    for spec in specs:
        src = existing_root / spec.key
        dst = out_root / spec.key
        if not src.exists():
            missing.append({"dataset_key": spec.key, "group": spec.group, "reason": "not_found_in_existing_dir"})
            print(f"SKIP {spec.key}: not found in {existing_root}")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        if (src / "dataset.npz").exists():
            shutil.copytree(src, dst)
            row = manifest_by_key.get(spec.key, {}).copy()
            if not row:
                with np.load(dst / "dataset.npz", allow_pickle=False) as data:
                    X = np.asarray(data["X"])
                    y = np.asarray(data["y"])
                row = {
                    "dataset_key": spec.key,
                    "data_path": str(dst / "dataset.npz"),
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "n_classes": int(np.unique(y).size),
                }
            row.update({
                "dataset_key": spec.key,
                "group": spec.group,
                "data_path": str(dst / "dataset.npz"),
                "source": spec.source,
                "display_name": spec.display_name,
                "expected_n_samples": spec.expected_n_samples,
                "expected_n_features_raw": spec.expected_n_features_raw,
                "expected_n_classes": spec.expected_n_classes,
                "dtype": row.get("dtype", args.float_dtype),
            })
            rows.append(row)
            print(f"OK   {spec.key}: copied existing prepared dataset")
        elif (src / "X_preprocessed.csv").exists() and (src / "y.csv").exists():
            row = prepare_existing_processed_folder(spec, src, dst, float_dtype=args.float_dtype)
            rows.append(row)
            print(f"OK   {spec.key}: converted existing processed CSV folder")
        else:
            missing.append({"dataset_key": spec.key, "group": spec.group, "reason": "no_dataset_npz_or_processed_csv"})
            print(f"SKIP {spec.key}: no dataset.npz or X_preprocessed.csv/y.csv in {src}")
    return rows, missing


def write_manifests(out_root: Path, manifest_rows: list[dict[str, Any]], skipped_rows: list[dict[str, Any]], missing_rows: list[dict[str, Any]]) -> None:
    cols = [
        "dataset_key", "group", "data_path", "n_samples", "n_features", "n_features_raw_expected", "n_classes", "source", "display_name",
        "expected_n_samples", "expected_n_features_raw", "expected_n_classes", "dtype", "shape_warnings",
    ]
    manifest = pd.DataFrame(manifest_rows)
    if not manifest.empty:
        for col in cols:
            if col not in manifest.columns:
                manifest[col] = ""
        manifest = manifest[cols + [c for c in manifest.columns if c not in cols]].sort_values(["group", "n_samples", "dataset_key"])
    manifest.to_csv(out_root / "manifest.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(out_root / "skipped_datasets.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(out_root / "missing_datasets.csv", index=False)

    group_dir = out_root / "dataset_groups"
    group_dir.mkdir(parents=True, exist_ok=True)
    for group in CANONICAL_GROUPS:
        keys = manifest.loc[manifest["group"] == group, "dataset_key"].astype(str).tolist() if not manifest.empty else []
        (group_dir / f"{group}.txt").write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")
    for alias, groups in GROUP_ALIASES.items():
        keys: list[str] = []
        for group in groups:
            if not manifest.empty:
                keys.extend(manifest.loc[manifest["group"] == group, "dataset_key"].astype(str).tolist())
        (group_dir / f"{alias}.txt").write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")

    pd.DataFrame([asdict(s) for s in DATASET_SPECS]).to_csv(out_root / "dataset_registry.csv", index=False)


def prepare_download(args: argparse.Namespace, specs: list[DatasetSpec], out_root: Path, raw_root: Path, processed_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for i, spec in enumerate(specs, start=1):
        print(f"[{i:02d}/{len(specs):02d}] {spec.group:17s} {spec.key}")
        try:
            X_raw, y_raw, source_detail, dropped_identifier_columns = load_raw(spec, raw_root)
            X, y, feature_records, class_mapping, class_distribution, audit = preprocess_dataframe(
                X_raw,
                y_raw,
                spec=spec,
                dropped_identifier_columns=dropped_identifier_columns,
                standardize_numeric=bool(args.standardize_numeric),
                max_categories=(None if args.max_categories <= 0 else args.max_categories),
                categorical_missing_strategy=args.categorical_missing_strategy,
            )
            row = save_prepared(spec, out_root, processed_root, X, y, feature_records, class_mapping, class_distribution, audit, args, source_detail)
            manifest_rows.append(row)
            shape_note = f"X={row['n_samples']}x{row['n_features']} raw_d_expected={row['n_features_raw_expected']} C={row['n_classes']}"
            warn = f" WARN: {row['shape_warnings']}" if row.get("shape_warnings") else ""
            print(f"    OK {shape_note}{warn}")
        except Exception as exc:
            skipped = {"dataset_key": spec.key, "group": spec.group, "reason": "error", "detail": repr(exc)}
            skipped_rows.append(skipped)
            print(f"    SKIP {spec.key}: {exc!r}")
            if args.fail_fast:
                raise
    return manifest_rows, skipped_rows


def print_registry(specs: list[DatasetSpec]) -> None:
    df = pd.DataFrame([asdict(s) for s in specs])
    cols = ["group", "key", "display_name", "source", "expected_n_samples", "expected_n_features_raw", "expected_n_classes", "loader", "openml_data_id", "asu_mat_name"]
    print(df[cols].to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = selected_specs(args)
    if args.list_datasets:
        print_registry(specs)
        return 0

    out_root = Path(args.output_dir)
    raw_root = Path(args.raw_cache_dir)
    processed_root = Path(args.processed_dir)

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    if args.save_processed_csv:
        processed_root.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Output dir: {out_root.resolve()}")
    print(f"Selected datasets: {len(specs)}")
    print(f"Groups: {', '.join(sorted(resolve_groups(args.groups)))}")

    skipped_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    if args.mode == "existing":
        if not args.existing_prepared_dir:
            raise ValueError("--mode existing requires --existing-prepared-dir")
        manifest_rows, missing_rows = copy_existing_prepared(specs, out_root, Path(args.existing_prepared_dir), args)
    else:
        manifest_rows, skipped_rows = prepare_download(args, specs, out_root, raw_root, processed_root)

    write_manifests(out_root, manifest_rows, skipped_rows, missing_rows)
    print("\nDone.")
    print(f"Prepared datasets: {len(manifest_rows)}")
    print(f"Skipped datasets: {len(skipped_rows)}")
    print(f"Missing existing datasets: {len(missing_rows)}")
    print(f"Manifest: {out_root / 'manifest.csv'}")
    print(f"Dataset groups: {out_root / 'dataset_groups'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Example PowerShell command:

python scripts/prepare_feature_selection_datasets.py `
  --mode download `
  --output-dir ../Dataset/prepared_feature_selection `
  --raw-cache-dir ../Dataset/dataset_cache/raw `
  --processed-dir ../Dataset/dataset_cache/processed `
  --groups all `
  --overwrite

Prepare only a subset:

python scripts/prepare_feature_selection_datasets.py `
  --mode download `
  --output-dir ../Dataset/prepared_feature_selection `
  --raw-cache-dir ../Dataset/dataset_cache/raw `
  --groups small_tabular medium_tabular `
  --overwrite
"""
