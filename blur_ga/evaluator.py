from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .config import GAConfig
from .data import SupervisedSplit
from .knn import KNNScorer


@dataclass
class FitnessEvaluator:
    """Feature-subset fitness over one or more train/validation splits."""

    splits: list[SupervisedSplit]
    config: GAConfig
    scorer: KNNScorer = field(init=False)
    n_features: int = field(init=False)
    eval_count: int = 0
    cache: dict[tuple[int, ...], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.splits:
            raise ValueError("FitnessEvaluator requires at least one split.")
        self.scorer = KNNScorer(self.config.knn_k)
        self.n_features = int(self.splits[0].X_train.shape[1])

    def selected_indices(self, chromosome: Iterable[int]) -> np.ndarray:
        arr = np.asarray(list(chromosome), dtype=np.int8)
        return np.flatnonzero(arr == 1).astype(int)

    def evaluate(self, chromosome: Iterable[int]) -> float:
        key = tuple(int(g) for g in chromosome)
        if self.config.cache_fitness and key in self.cache:
            return self.cache[key]
        selected = np.flatnonzero(np.asarray(key, dtype=np.int8) == 1).astype(int)
        if selected.size == 0:
            fitness = 0.0
        else:
            scores = [self.scorer.score(split, selected) for split in self.splits]
            mean_score = float(np.mean(scores))
            sparsity = (self.n_features - selected.size) / self.n_features
            fitness = self.config.fitness_weight_score * mean_score + self.config.fitness_weight_sparsity * sparsity
        self.eval_count += 1
        if self.config.cache_fitness:
            self.cache[key] = fitness
        return fitness

    def metric_score(self, split: SupervisedSplit, chromosome: Iterable[int]) -> float:
        selected = self.selected_indices(chromosome)
        if selected.size == 0:
            return 0.0
        return self.scorer.score(split, selected)
