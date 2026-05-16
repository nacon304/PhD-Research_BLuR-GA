from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blur_ga.config import GAConfig
from blur_ga.engine import GeneticFeatureSelector, RunResult
from blur_ga.benchmarks import load_problem


@dataclass
class SyntheticLandscapeEvaluator:
    problem: object
    eval_count: int = 0

    @property
    def n_features(self) -> int:
        return int(self.problem.n_features)

    def evaluate(self, chromosome) -> float:
        self.eval_count += 1
        return float(self.problem.evaluate(chromosome))


@dataclass
class SyntheticLandscapeRun:
    dataset: str
    ga_type: int
    repeat_id: int
    outer_fold: int
    run_id: int
    seed: int
    run_result: RunResult


def run_synthetic_landscape(
    *,
    problem_json: str | Path,
    ga_config: GAConfig,
    repeat_id: int = 0,
    outer_fold: int = 0,
    base_seed: int = 1,
) -> SyntheticLandscapeRun:
    problem = load_problem(problem_json)
    evaluator = SyntheticLandscapeEvaluator(problem)
    seed = int(base_seed) + 100000 * int(repeat_id) + 1000 * int(outer_fold)
    run_id = int(repeat_id) * 1 + int(outer_fold)
    selector = GeneticFeatureSelector(ga_config, evaluator, seed=seed, run_id=run_id)
    result = selector.run()
    return SyntheticLandscapeRun(
        dataset=str(problem.dataset_key),
        ga_type=int(ga_config.ga_type),
        repeat_id=int(repeat_id),
        outer_fold=int(outer_fold),
        run_id=int(run_id),
        seed=int(seed),
        run_result=result,
    )
