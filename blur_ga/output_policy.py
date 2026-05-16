from __future__ import annotations

"""Dataset-driven output policy for HPC evaluation runs.

A BLuR-GA dataset can request one or more evaluation output groups.  This keeps
HPC tasks uniform (dataset x method x outer_fold x repeat) while still allowing
special real-case datasets to emit linkage, building-block, and performance
artifacts in the same run when needed.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import csv

VALID_OUTPUT_GROUPS = {"linkage", "building_block", "performance", "interpretability", "ablation", "convergence"}


@dataclass(frozen=True)
class DatasetOutputPolicy:
    dataset_key: str
    eval_group: str = "performance"
    output_groups: tuple[str, ...] = ("performance",)
    evaluator_type: str = "supervised_feature_selection"
    split_mode: str = "outer_cv"

    def includes(self, group: str) -> bool:
        return group in set(self.output_groups)


def _normalise_groups(value: object, fallback: str) -> tuple[str, ...]:
    if value in (None, ""):
        groups = [fallback]
    elif isinstance(value, str):
        groups = [g.strip() for g in value.replace(";", ",").split(",") if g.strip()]
    else:
        groups = [str(g).strip() for g in value if str(g).strip()]
    out: list[str] = []
    for group in groups:
        if group not in VALID_OUTPUT_GROUPS:
            raise ValueError(f"Unknown output group {group!r}; valid groups are {sorted(VALID_OUTPUT_GROUPS)}")
        if group not in out:
            out.append(group)
    return tuple(out or [fallback])


def policy_from_metadata(metadata: dict[str, object]) -> DatasetOutputPolicy:
    eval_group = str(metadata.get("eval_group") or "performance")
    dataset_key = str(metadata.get("dataset_key") or metadata.get("dataset_name") or metadata.get("name") or "dataset")
    return DatasetOutputPolicy(
        dataset_key=dataset_key,
        eval_group=eval_group,
        output_groups=_normalise_groups(metadata.get("output_groups"), eval_group),
        evaluator_type=str(metadata.get("evaluator_type") or "supervised_feature_selection"),
        split_mode=str(metadata.get("split_mode") or "outer_cv"),
    )


def load_dataset_policy(dataset_dir: str | Path, *, default_dataset_key: str | None = None) -> DatasetOutputPolicy:
    folder = Path(dataset_dir)
    meta_path = folder / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return policy_from_metadata(meta)
    return DatasetOutputPolicy(dataset_key=default_dataset_key or folder.name)


def read_manifest_row(dataset_root: str | Path, dataset_key: str) -> dict[str, str]:
    manifest = Path(dataset_root) / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("dataset_key") == dataset_key or row.get("dataset") == dataset_key:
                return dict(row)
    raise KeyError(f"Dataset {dataset_key!r} not found in {manifest}")
