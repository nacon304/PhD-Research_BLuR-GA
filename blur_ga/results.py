from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import ArtifactLayout
from .engine import RunResult


@dataclass
class EvaluatedRun:
    dataset: str
    classifier_type: int
    ga_type: int
    repeat_id: int
    outer_fold: int
    run_id: int
    seed: int
    best_inner_fitness: float
    outer_score: float
    subset_size: int
    n_edges: int
    generations: int
    runtime_seconds: float
    eval_count: int
    chromosome: np.ndarray
    run_result: RunResult | None = None


def _chromosome_string(chromosome: np.ndarray | Sequence[int] | str) -> str:
    if isinstance(chromosome, str):
        return chromosome
    return "".join(str(int(v)) for v in chromosome)


def _selected_feature_string(chromosome: np.ndarray | Sequence[int] | str) -> str:
    if isinstance(chromosome, str):
        return ";".join(str(i) for i, v in enumerate(chromosome.strip()) if v == "1")
    arr = np.asarray(chromosome, dtype=int)
    return ";".join(str(int(i)) for i in np.flatnonzero(arr == 1))


@dataclass
class ResultWriter:
    output_dir: Path
    dataset: str
    classifier_type: int
    ga_type: int
    artifact_layout: ArtifactLayout = "both"
    rows: list[EvaluatedRun] = field(default_factory=list)
    prefix_override: str | None = None

    @property
    def prefix(self) -> str:
        # Dataset and method are already encoded in the directory path:
        #   <output>/<dataset>/<method>/...
        # Keep filenames compact and Windows-safe.
        return self.prefix_override or f"c{self.classifier_type}_a{self.ga_type}"

    def _run_tag(self, row: EvaluatedRun) -> str:
        return f"rep{row.repeat_id:02d}_fold{row.outer_fold:02d}_run{row.run_id:03d}"

    def _artifact_name(self, base: str, row: EvaluatedRun | None = None, *, root_level: bool = False) -> str:
        if root_level and row is not None:
            return f"{base}_{self.prefix}_{self._run_tag(row)}.csv"
        return f"{base}.csv"

    @property
    def _write_nested(self) -> bool:
        return self.artifact_layout in {"both", "nested"}

    @property
    def _write_root(self) -> bool:
        return self.artifact_layout in {"both", "root"}

    def add(self, row: EvaluatedRun) -> None:
        self.rows.append(row)

    def write_all(self) -> None:
        """Write aggregate/root-level CSVs for the rows currently in memory.

        For local sequential runs this preserves the old workflow.  For NeSI
        array jobs, pass write_aggregate_outputs=false to the experiment so each
        task writes only per-run artifacts; run `python run_blur_ga.py aggregate`
        afterwards to rebuild these same files safely.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_summary()
        self._write_legacy_vectors()
        self._write_selected_features()
        self._write_generation_traces()
        self._write_aggregate_building_block_artifacts()

    def _run_dir(self, row: EvaluatedRun) -> Path:
        return self.output_dir / f"rep{row.repeat_id:02d}" / f"fold{row.outer_fold:02d}" / f"run{row.run_id:03d}"

    def save_run_artifacts(self, row: EvaluatedRun) -> None:
        """Save per-run artifacts.  This is the HPC-safe output layer."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = self._run_dir(row)
        if self._write_nested:
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_run_summary(row, run_dir / self._artifact_name("run_summary"))
            self._write_run_selected_features(row, run_dir / self._artifact_name("selected_features"))
            self._write_run_generation_trace(row, run_dir / self._artifact_name("generation_trace"))

        if self._write_root:
            self._write_run_summary(row, self.output_dir / self._artifact_name("run_summary", row, root_level=True))
            self._write_run_selected_features(row, self.output_dir / self._artifact_name("selected_features", row, root_level=True))
            self._write_run_generation_trace(row, self.output_dir / self._artifact_name("generation_trace", row, root_level=True))

        if row.run_result is None or row.run_result.evig is None:
            return

        graph_dirs: list[Path] = []
        if self._write_nested:
            graph_dirs.append(run_dir)
        if self._write_root:
            graph_dirs.append(self.output_dir)

        for target_dir in graph_dirs:
            root_level = target_dir == self.output_dir
            # Dense d x d CSV matrices become very large for high-dimensional FS
            # datasets (e.g. 5966^2 text entries).  The edge CSVs below contain
            # the same active linkage information used by downstream graph/BB
            # analysis, so skip dense matrix dumps when d is large.
            n_features = int(getattr(row.run_result.evig, "n_features", 0))
            if n_features <= 500:
                row.run_result.evig.save_matrix(target_dir / self._artifact_name("eVIG", row, root_level=root_level))
                if hasattr(row.run_result.evig, "save_coefficient_matrix"):
                    row.run_result.evig.save_coefficient_matrix(target_dir / self._artifact_name("eVIG_coefficients", row, root_level=root_level))
            else:
                marker = target_dir / self._artifact_name("eVIG_matrix_skipped", row, root_level=root_level).replace(".csv", ".txt")
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(
                    f"Dense eVIG/eVIG_coefficients matrices were skipped for n_features={n_features}. "
                    "Use eVIG_edges.csv and eVIG_tested_pairs.csv for the active sparse graph.\n",
                    encoding="utf-8",
                )
            row.run_result.evig.save_edges(target_dir / self._artifact_name("eVIG_edges", row, root_level=root_level))
            row.run_result.evig.save_tested_pairs(target_dir / self._artifact_name("eVIG_tested_pairs", row, root_level=root_level))

        self._write_linkage_events(row, run_dir)
        self._write_graph_snapshots(row, run_dir)
        self._write_building_block_artifacts(row, run_dir)

    def _write_run_summary(self, row: EvaluatedRun, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset",
                "classifier_type",
                "ga_type",
                "repeat_id",
                "outer_fold",
                "run_id",
                "seed",
                "best_inner_fitness",
                "outer_score",
                "subset_size",
                "n_edges",
                "generations",
                "runtime_seconds",
                "eval_count",
                "best_chromosome",
                "best_selected_features",
            ])
            writer.writerow([
                row.dataset,
                row.classifier_type,
                row.ga_type,
                row.repeat_id,
                row.outer_fold,
                row.run_id,
                row.seed,
                f"{row.best_inner_fitness:.14f}",
                f"{row.outer_score:.14f}",
                row.subset_size,
                row.n_edges,
                row.generations,
                f"{row.runtime_seconds:.6f}",
                row.eval_count,
                _chromosome_string(row.chromosome),
                _selected_feature_string(row.chromosome),
            ])

    def _write_run_selected_features(self, row: EvaluatedRun, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["repeat_id", "outer_fold", "run_id", "feature_index"])
            for feature_idx in np.flatnonzero(row.chromosome == 1):
                writer.writerow([row.repeat_id, row.outer_fold, row.run_id, int(feature_idx)])

    def _trace_header(self, rows: Iterable[EvaluatedRun] | None = None) -> list[str]:
        base = [
            "repeat_id",
            "outer_fold",
            "run_id",
            "generation",
            "elapsed_seconds",
            "best_population_fitness",
            "mean_population_fitness",
            "std_population_fitness",
            "worst_population_fitness",
            "best_so_far_fitness",
            "best_population_subset_size",
            "best_so_far_subset_size",
            "mean_subset_size",
            "min_subset_size",
            "max_subset_size",
            "population_unique_chromosomes",
            "n_edges",
            "n_tested_pairs",
            "eval_count",
            "archive_size",
            "best_population_chromosome",
            "best_so_far_chromosome",
            "best_population_selected_features",
            "best_so_far_selected_features",
        ]
        # Keep deterministic order while allowing future trace keys to pass through.
        extras: list[str] = []
        if rows is not None:
            for r in rows:
                if r.run_result is None:
                    continue
                for item in r.run_result.generation_trace:
                    for key in item.keys():
                        if key not in base and key not in extras:
                            extras.append(key)
        return base + extras

    def _format_trace_value(self, value: object) -> object:
        if isinstance(value, float):
            return f"{value:.14f}"
        return value

    def _write_run_generation_trace(self, row: EvaluatedRun, path: Path) -> None:
        if row.run_result is None or not row.run_result.generation_trace:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        header = self._trace_header([row])
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for item in row.run_result.generation_trace:
                writer.writerow([
                    row.repeat_id if col == "repeat_id" else
                    row.outer_fold if col == "outer_fold" else
                    row.run_id if col == "run_id" else
                    self._format_trace_value(item.get(col, ""))
                    for col in header
                ])

    def _write_linkage_events(self, row: EvaluatedRun, run_dir: Path) -> None:
        events = row.run_result.linkage_events if row.run_result is not None else []
        if not events:
            return
        header = [
            "repeat_id",
            "outer_fold",
            "run_id",
            "generation",
            "eval_count",
            "feature_i",
            "feature_j",
            "omega",
            "signed_omega",
            "raw_signed_omega",
            "sign",
            "raw_sign",
            "is_active",
            "is_positive",
            "was_new_edge",
            "edge_weight_after",
            "signed_weight_after",
            "raw_signed_weight_after",
            "active_count_after",
            "positive_count_after",
            "negative_count_after",
            "tested_count_after",
            "weight_sum_after",
            "signed_weight_sum_after",
            "raw_signed_weight_sum_after",
            "first_seen_generation",
            "last_updated_generation",
            "parent_fitness",
            "fitness_g",
            "fitness_h",
            "fitness_gh",
            "best_so_far_fitness",
            "subset_size_parent",
            "subset_size_best",
        ]
        paths: list[Path] = []
        if self._write_nested:
            paths.append(run_dir / self._artifact_name("linkage_events"))
        if self._write_root:
            paths.append(self.output_dir / self._artifact_name("linkage_events", row, root_level=True))
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for event in events:
                    writer.writerow([
                        row.repeat_id,
                        row.outer_fold,
                        row.run_id,
                        int(event["generation"]),
                        int(event["eval_count"]),
                        int(event["feature_i"]),
                        int(event["feature_j"]),
                        f"{float(event['omega']):.14f}",
                        f"{float(event.get('signed_omega', event['omega'])):.14f}",
                        f"{float(event.get('raw_signed_omega', event.get('signed_omega', event['omega']))):.14f}",
                        int(event.get("sign", 1 if float(event.get('signed_omega', event['omega'])) > 0 else (-1 if float(event.get('signed_omega', event['omega'])) < 0 else 0))),
                        int(event.get("raw_sign", 1 if float(event.get('raw_signed_omega', event.get('signed_omega', event['omega']))) > 0 else (-1 if float(event.get('raw_signed_omega', event.get('signed_omega', event['omega']))) < 0 else 0))),
                        int(bool(event.get("is_active", event.get("is_positive", False)))),
                        int(bool(event["is_positive"])),
                        int(bool(event["was_new_edge"])),
                        f"{float(event['edge_weight_after']):.14f}",
                        f"{float(event.get('signed_weight_after', event['edge_weight_after'])):.14f}",
                        f"{float(event.get('raw_signed_weight_after', event.get('signed_weight_after', event['edge_weight_after']))):.14f}",
                        int(event.get("active_count_after", event.get("positive_count_after", 0))),
                        int(event["positive_count_after"]),
                        int(event.get("negative_count_after", 0)),
                        int(event["tested_count_after"]),
                        f"{float(event['weight_sum_after']):.14f}",
                        f"{float(event.get('signed_weight_sum_after', event['weight_sum_after'])):.14f}",
                        f"{float(event.get('raw_signed_weight_sum_after', event.get('signed_weight_sum_after', event['weight_sum_after']))):.14f}",
                        int(event["first_seen_generation"]),
                        int(event["last_updated_generation"]),
                        f"{float(event['parent_fitness']):.14f}",
                        f"{float(event['fitness_g']):.14f}",
                        f"{float(event['fitness_h']):.14f}",
                        f"{float(event['fitness_gh']):.14f}",
                        f"{float(event['best_so_far_fitness']):.14f}",
                        int(event["subset_size_parent"]),
                        int(event["subset_size_best"]),
                    ])

    def _write_graph_snapshots(self, row: EvaluatedRun, run_dir: Path) -> None:
        snapshots = row.run_result.graph_snapshots if row.run_result is not None else []
        if not snapshots:
            return
        header = [
            "repeat_id",
            "outer_fold",
            "run_id",
            "generation",
            "feature_i",
            "feature_j",
            "weight",
            "signed_weight",
            "sign",
            "raw_signed_weight",
            "raw_sign",
            "positive_count",
            "tested_count",
            "weight_sum",
            "first_seen_generation",
            "last_updated_generation",
        ]
        paths: list[Path] = []
        if self._write_nested:
            paths.append(run_dir / self._artifact_name("graph_snapshots"))
        if self._write_root:
            paths.append(self.output_dir / self._artifact_name("graph_snapshots", row, root_level=True))
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for item in snapshots:
                    writer.writerow([
                        row.repeat_id,
                        row.outer_fold,
                        row.run_id,
                        int(item["generation"]),
                        int(item["feature_i"]),
                        int(item["feature_j"]),
                        f"{float(item['weight']):.14f}",
                        f"{float(item.get('signed_weight', item['weight'])):.14f}",
                        int(item.get("sign", 1 if float(item.get('signed_weight', item['weight'])) > 0 else (-1 if float(item.get('signed_weight', item['weight'])) < 0 else 0))),
                        f"{float(item.get('raw_signed_weight', item.get('signed_weight', item['weight']))):.14f}",
                        int(item.get("raw_sign", 1 if float(item.get('raw_signed_weight', item.get('signed_weight', item['weight']))) > 0 else (-1 if float(item.get('raw_signed_weight', item.get('signed_weight', item['weight']))) < 0 else 0))),
                        int(item["positive_count"]),
                        int(item["tested_count"]),
                        f"{float(item['weight_sum']):.14f}",
                        int(item["first_seen_generation"]),
                        int(item["last_updated_generation"]),
                    ])

    def _write_building_block_artifacts(self, row: EvaluatedRun, run_dir: Path) -> None:
        if row.run_result is None:
            return
        summaries = row.run_result.building_block_summaries
        blocks = row.run_result.building_blocks
        if not summaries and not blocks:
            return
        paths: list[tuple[Path, Path]] = []
        if self._write_nested:
            paths.append((run_dir / self._artifact_name("building_block_summary"), run_dir / self._artifact_name("building_blocks")))
        if self._write_root:
            paths.append((
                self.output_dir / self._artifact_name("building_block_summary", row, root_level=True),
                self.output_dir / self._artifact_name("building_blocks", row, root_level=True),
            ))
        summary_header = [
            "repeat_id", "outer_fold", "run_id", "generation", "bb_method", "bb_weight_mode", "bb_search_mode",
            "n_features", "n_graph_edges", "n_candidates", "n_blocks", "mean_block_size",
            "max_block_size", "mean_score", "max_score", "mean_internal_abs",
            "mean_internal_positive", "mean_internal_negative_abs", "mean_signed_balance",
            "n_positive_edges_in_blocks", "n_negative_edges_in_blocks", "n_sign_conflicts_in_blocks",
            "bb_mix_trials", "bb_mix_accepts", "bb_mix_accept_rate", "bb_mix_fitness_gain", "bb_mix_size_reduction",
        ]
        block_header = [
            "repeat_id", "outer_fold", "run_id", "generation", "bb_method", "bb_weight_mode", "bb_search_mode",
            "block_id", "features", "size", "score", "density", "internal_abs_mean",
            "internal_positive_mean", "internal_negative_abs_mean", "external_abs_mean",
            "n_internal_edges", "n_positive_edges", "n_negative_edges",
            "signed_balance", "n_sign_conflicts", "polarity", "positive_group", "negative_group", "schema_hint",
            "source",
        ]
        for summary_path, block_path in paths:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            if summaries:
                with summary_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(summary_header)
                    for item in summaries:
                        writer.writerow([
                            row.repeat_id, row.outer_fold, row.run_id,
                            int(item.get("generation", 0)),
                            item.get("bb_method", "ltga"),
                            item.get("bb_weight_mode", ""),
                            item.get("bb_search_mode", "none"),
                            int(item.get("n_features", 0)),
                            int(item.get("n_graph_edges", 0)),
                            int(item.get("n_candidates", 0)),
                            int(item.get("n_blocks", 0)),
                            f"{float(item.get('mean_block_size', 0.0)):.14f}",
                            int(item.get("max_block_size", 0)),
                            f"{float(item.get('mean_score', 0.0)):.14f}",
                            f"{float(item.get('max_score', 0.0)):.14f}",
                            f"{float(item.get('mean_internal_abs', 0.0)):.14f}",
                            f"{float(item.get('mean_internal_positive', 0.0)):.14f}",
                            f"{float(item.get('mean_internal_negative_abs', 0.0)):.14f}",
                            f"{float(item.get('mean_signed_balance', 0.0)):.14f}",
                            int(item.get("n_positive_edges_in_blocks", 0)),
                            int(item.get("n_negative_edges_in_blocks", 0)),
                            int(item.get("n_sign_conflicts_in_blocks", 0)),
                            int(item.get("bb_mix_trials", 0)),
                            int(item.get("bb_mix_accepts", 0)),
                            f"{float(item.get('bb_mix_accept_rate', 0.0)):.14f}",
                            f"{float(item.get('bb_mix_fitness_gain', 0.0)):.14f}",
                            int(item.get("bb_mix_size_reduction", 0)),
                        ])
            if blocks:
                with block_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(block_header)
                    for item in blocks:
                        writer.writerow([
                            row.repeat_id, row.outer_fold, row.run_id,
                            int(item.get("generation", 0)),
                            item.get("bb_method", "ltga"),
                            item.get("bb_weight_mode", ""),
                            item.get("bb_search_mode", "none"),
                            int(item.get("block_id", 0)),
                            item.get("features", ""),
                            int(item.get("size", 0)),
                            f"{float(item.get('score', 0.0)):.14f}",
                            f"{float(item.get('density', 0.0)):.14f}",
                            f"{float(item.get('internal_abs_mean', 0.0)):.14f}",
                            f"{float(item.get('internal_positive_mean', 0.0)):.14f}",
                            f"{float(item.get('internal_negative_abs_mean', 0.0)):.14f}",
                            f"{float(item.get('external_abs_mean', 0.0)):.14f}",
                            int(item.get("n_internal_edges", 0)),
                            int(item.get("n_positive_edges", 0)),
                            int(item.get("n_negative_edges", 0)),
                            f"{float(item.get('signed_balance', 0.0)):.14f}",
                            int(item.get("n_sign_conflicts", 0)),
                            item.get("polarity", ""),
                            item.get("positive_group", ""),
                            item.get("negative_group", ""),
                            item.get("schema_hint", ""),
                            item.get("source", ""),
                        ])

    def _write_aggregate_building_block_artifacts(self) -> None:
        if not any(r.run_result is not None and (r.run_result.building_block_summaries or r.run_result.building_blocks) for r in self.rows):
            return
        summary_path = self.output_dir / f"building_block_summary_{self.prefix}.csv"
        block_path = self.output_dir / f"building_blocks_{self.prefix}.csv"
        summary_header = [
            "repeat_id", "outer_fold", "run_id", "generation", "bb_method", "bb_weight_mode", "bb_search_mode",
            "n_features", "n_graph_edges", "n_candidates", "n_blocks", "mean_block_size",
            "max_block_size", "mean_score", "max_score", "mean_internal_abs",
            "mean_internal_positive", "mean_internal_negative_abs", "mean_signed_balance",
            "n_positive_edges_in_blocks", "n_negative_edges_in_blocks", "n_sign_conflicts_in_blocks",
            "bb_mix_trials", "bb_mix_accepts", "bb_mix_accept_rate", "bb_mix_fitness_gain", "bb_mix_size_reduction",
        ]
        block_header = [
            "repeat_id", "outer_fold", "run_id", "generation", "bb_method", "bb_weight_mode", "bb_search_mode",
            "block_id", "features", "size", "score", "density", "internal_abs_mean",
            "internal_positive_mean", "internal_negative_abs_mean", "external_abs_mean",
            "n_internal_edges", "n_positive_edges", "n_negative_edges",
            "signed_balance", "n_sign_conflicts", "polarity", "positive_group", "negative_group", "schema_hint",
            "source",
        ]
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(summary_header)
            for row in sorted(self.rows, key=lambda x: (x.repeat_id, x.outer_fold, x.run_id)):
                if row.run_result is None:
                    continue
                for item in row.run_result.building_block_summaries:
                    writer.writerow([
                        row.repeat_id, row.outer_fold, row.run_id, int(item.get("generation", 0)),
                        item.get("bb_method", "ltga"), item.get("bb_weight_mode", ""), item.get("bb_search_mode", "none"),
                        int(item.get("n_features", 0)), int(item.get("n_graph_edges", 0)),
                        int(item.get("n_candidates", 0)), int(item.get("n_blocks", 0)),
                        f"{float(item.get('mean_block_size', 0.0)):.14f}", int(item.get("max_block_size", 0)),
                        f"{float(item.get('mean_score', 0.0)):.14f}", f"{float(item.get('max_score', 0.0)):.14f}",
                        f"{float(item.get('mean_internal_abs', 0.0)):.14f}",
                        f"{float(item.get('mean_internal_positive', 0.0)):.14f}",
                        f"{float(item.get('mean_internal_negative_abs', 0.0)):.14f}",
                        f"{float(item.get('mean_signed_balance', 0.0)):.14f}",
                        int(item.get("n_positive_edges_in_blocks", 0)),
                        int(item.get("n_negative_edges_in_blocks", 0)),
                        int(item.get("n_sign_conflicts_in_blocks", 0)),
                        int(item.get("bb_mix_trials", 0)),
                        int(item.get("bb_mix_accepts", 0)),
                        f"{float(item.get('bb_mix_accept_rate', 0.0)):.14f}",
                        f"{float(item.get('bb_mix_fitness_gain', 0.0)):.14f}",
                        int(item.get("bb_mix_size_reduction", 0)),
                    ])
        with block_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(block_header)
            for row in sorted(self.rows, key=lambda x: (x.repeat_id, x.outer_fold, x.run_id)):
                if row.run_result is None:
                    continue
                for item in row.run_result.building_blocks:
                    writer.writerow([
                        row.repeat_id, row.outer_fold, row.run_id, int(item.get("generation", 0)),
                        item.get("bb_method", "ltga"), item.get("bb_weight_mode", ""), item.get("bb_search_mode", "none"),
                        int(item.get("block_id", 0)), item.get("features", ""), int(item.get("size", 0)),
                        f"{float(item.get('score', 0.0)):.14f}", f"{float(item.get('density', 0.0)):.14f}",
                        f"{float(item.get('internal_abs_mean', 0.0)):.14f}",
                        f"{float(item.get('internal_positive_mean', 0.0)):.14f}",
                        f"{float(item.get('internal_negative_abs_mean', 0.0)):.14f}",
                        f"{float(item.get('external_abs_mean', 0.0)):.14f}",
                        int(item.get("n_internal_edges", 0)), int(item.get("n_positive_edges", 0)),
                        int(item.get("n_negative_edges", 0)),
                        f"{float(item.get('signed_balance', 0.0)):.14f}",
                        int(item.get("n_sign_conflicts", 0)),
                        item.get("polarity", ""), item.get("positive_group", ""),
                        item.get("negative_group", ""), item.get("schema_hint", ""),
                        item.get("source", ""),
                    ])

    def _write_summary(self) -> None:
        path = self.output_dir / f"nested_summary_{self.prefix}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset",
                "classifier_type",
                "ga_type",
                "repeat_id",
                "outer_fold",
                "run_id",
                "seed",
                "best_inner_fitness",
                "outer_score",
                "subset_size",
                "n_edges",
                "generations",
                "runtime_seconds",
                "eval_count",
            ])
            for r in sorted(self.rows, key=lambda x: (x.repeat_id, x.outer_fold, x.run_id)):
                writer.writerow([
                    r.dataset,
                    r.classifier_type,
                    r.ga_type,
                    r.repeat_id,
                    r.outer_fold,
                    r.run_id,
                    r.seed,
                    f"{r.best_inner_fitness:.14f}",
                    f"{r.outer_score:.14f}",
                    r.subset_size,
                    r.n_edges,
                    r.generations,
                    f"{r.runtime_seconds:.6f}",
                    r.eval_count,
                ])

    def _write_legacy_vectors(self) -> None:
        rows = sorted(self.rows, key=lambda x: (x.repeat_id, x.outer_fold, x.run_id))

        def write_vector(name: str, values: Iterable[str]) -> None:
            (self.output_dir / f"{name}_{self.prefix}.csv").write_text(", ".join(values), encoding="utf-8")

        write_vector("bfi", (f"{r.best_inner_fitness:.14f}" for r in rows))
        write_vector("test_score", (f"{r.outer_score:.14f}" for r in rows))
        write_vector("time", (f"{r.runtime_seconds:.2f}" for r in rows))
        write_vector("gen", (str(r.generations) for r in rows))
        write_vector("evals", (str(r.eval_count) for r in rows))
        if self.ga_type != 0:
            write_vector("nedges", (f"{float(r.n_edges):.1f}" for r in rows))

        bind_lines = [", ".join(str(int(v)) for v in r.chromosome) for r in rows]
        (self.output_dir / f"bind_{self.prefix}.csv").write_text("\n".join(bind_lines), encoding="utf-8")

    def _write_selected_features(self) -> None:
        path = self.output_dir / f"selected_features_{self.prefix}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["repeat_id", "outer_fold", "run_id", "feature_index"])
            for r in sorted(self.rows, key=lambda x: (x.repeat_id, x.outer_fold, x.run_id)):
                for feature_idx in np.flatnonzero(r.chromosome == 1):
                    writer.writerow([r.repeat_id, r.outer_fold, r.run_id, int(feature_idx)])

    def _write_generation_traces(self) -> None:
        has_trace = any(r.run_result is not None and r.run_result.generation_trace for r in self.rows)
        if not has_trace:
            return
        trace_path = self.output_dir / f"generation_trace_{self.prefix}.csv"
        rows = sorted(self.rows, key=lambda x: (x.repeat_id, x.outer_fold, x.run_id))
        header = self._trace_header(rows)
        with trace_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in rows:
                if r.run_result is None:
                    continue
                for item in r.run_result.generation_trace:
                    writer.writerow([
                        r.repeat_id if col == "repeat_id" else
                        r.outer_fold if col == "outer_fold" else
                        r.run_id if col == "run_id" else
                        self._format_trace_value(item.get(col, ""))
                        for col in header
                    ])
