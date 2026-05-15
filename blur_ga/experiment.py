from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ExperimentConfig, GAConfig
from .data import Dataset, SupervisedSplit
from .engine import GeneticFeatureSelector
from .evaluator import FitnessEvaluator
from .results import EvaluatedRun, ResultWriter
from .splitting import KFoldSplitter, make_split, remap_local_to_global


@dataclass
class NestedCVExperiment:
    dataset: Dataset
    ga_config: GAConfig
    experiment_config: ExperimentConfig

    def run(self, *, outer_fold_id: int | None = None, repeat_id: int | None = None) -> ResultWriter:
        out = self.experiment_config.output_dir
        writer = ResultWriter(
            output_dir=out,
            dataset=self.dataset.name,
            classifier_type=self.ga_config.classifier_type,
            ga_type=self.ga_config.ga_type,
            artifact_layout=self.experiment_config.artifact_layout,
        )
        repeats = [repeat_id] if repeat_id is not None else list(range(self.experiment_config.repeats))
        for rep in repeats:
            outer_splitter = KFoldSplitter(
                self.experiment_config.outer_folds,
                shuffle=self.experiment_config.shuffle,
                seed=self.experiment_config.seed,
                stratified=True,
            )
            outer_folds = outer_splitter.split(self.dataset, repeat_id=rep)
            if outer_fold_id is not None:
                outer_folds = [fold for fold in outer_folds if fold.fold_id == outer_fold_id]
                if not outer_folds:
                    raise ValueError(f"outer_fold_id={outer_fold_id} is outside 0..{self.experiment_config.outer_folds - 1}")

            for outer_fold in outer_folds:
                row = self._run_one_outer_fold(rep, outer_fold.fold_id, outer_fold.train_indices, outer_fold.test_indices)
                writer.add(row)
                writer.save_run_artifacts(row)
        if self.experiment_config.write_aggregate_outputs:
            writer.write_all()
        return writer

    def _run_one_outer_fold(
        self, repeat_id: int, outer_fold_id: int, outer_train_idx: np.ndarray, outer_test_idx: np.ndarray
    ) -> EvaluatedRun:
        # Inner CV is created inside the outer training pool only. This is the
        # critical leakage fix: the outer test fold is never touched by fitness.
        outer_train_dataset = self.dataset.subset(outer_train_idx)
        inner_splitter = KFoldSplitter(
            self.experiment_config.inner_folds,
            shuffle=self.experiment_config.shuffle,
            seed=self.experiment_config.seed + 7919,
            stratified=True,
        )
        inner_folds_local = inner_splitter.split(outer_train_dataset, repeat_id=repeat_id)
        inner_splits: list[SupervisedSplit] = []
        for inner in inner_folds_local:
            train_global = remap_local_to_global(inner.train_indices, outer_train_idx)
            val_global = remap_local_to_global(inner.test_indices, outer_train_idx)
            inner_splits.append(make_split(self.dataset, train_global, val_global))

        evaluator = FitnessEvaluator(inner_splits, self.ga_config)
        seed = self.experiment_config.seed + 100000 * repeat_id + 1000 * outer_fold_id
        run_id = repeat_id * self.experiment_config.outer_folds + outer_fold_id
        selector = GeneticFeatureSelector(self.ga_config, evaluator, seed=seed, run_id=run_id)
        result = selector.run()

        final_split = make_split(self.dataset, outer_train_idx, outer_test_idx)
        outer_score = evaluator.metric_score(final_split, result.best_chromosome)
        return EvaluatedRun(
            dataset=self.dataset.name,
            classifier_type=self.ga_config.classifier_type,
            ga_type=self.ga_config.ga_type,
            repeat_id=repeat_id,
            outer_fold=outer_fold_id,
            run_id=run_id,
            seed=seed,
            best_inner_fitness=result.best_inner_fitness,
            outer_score=outer_score,
            subset_size=result.subset_size,
            n_edges=result.n_edges,
            generations=result.generations,
            runtime_seconds=result.runtime_seconds,
            eval_count=result.eval_count,
            chromosome=result.best_chromosome,
            run_result=result,
        )
