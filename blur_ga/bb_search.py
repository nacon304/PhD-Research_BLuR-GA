from __future__ import annotations

"""Building-block-aware variation operators for BLuR-GA.

This module is deliberately independent of datasets and evaluators.  It consumes
LTGA-style :class:`BuildingBlockSnapshot` objects and provides two search modes:

``uniform``
    Treat each selected block as an atomic mask during crossover.  In signed
    mode, a copied block is lightly repaired to respect the same/opposite
    polarity groups learned from the signed linkage graph.

``pattern_refine``
    Learn supported block-level patterns from the top archive solutions and try
    them as safe local improvements on a small fraction of offspring.  In signed
    mode, raw archive patterns are projected to the block's coherent signed
    schema before they are scored.
"""

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .building_blocks import BBWeightMode, BuildingBlock, BuildingBlockSnapshot

BBSearchMode = Literal["none", "uniform", "pattern_refine"]

_EPS = 1e-12


@dataclass
class BBSearchStats:
    """Cumulative diagnostics for BB-aware search decisions."""

    trials: int = 0
    accepts: int = 0
    fitness_gain: float = 0.0
    size_reduction: int = 0

    @property
    def accept_rate(self) -> float:
        return float(self.accepts / self.trials) if self.trials else 0.0

    def record(
        self,
        *,
        accepted: bool,
        old_fitness: float,
        new_fitness: float,
        old_chromosome: np.ndarray,
        new_chromosome: np.ndarray,
    ) -> None:
        self.trials += 1
        if not accepted:
            return
        self.accepts += 1
        self.fitness_gain += max(0.0, float(new_fitness - old_fitness))
        self.size_reduction += max(0, int(np.sum(old_chromosome)) - int(np.sum(new_chromosome)))


class BuildingBlockVariation:
    """Pure variation helper for block-wise crossover and pattern refinement."""

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        n_features: int,
        mode: BBSearchMode = "none",
        weight_mode: BBWeightMode = "absolute",
        pattern_top_fraction: float = 0.30,
        pattern_min_support: int = 2,
        accept_equal_sparser: bool = True,
        refine_fraction: float = 0.20,
        max_refine_trials_per_generation: int | None = None,
        shuffle_blocks: bool = True,
        signed_repair_probability: float = 0.75,
    ) -> None:
        self.rng = rng
        self.n_features = int(n_features)
        self.mode = self._canonical_mode(mode)
        self.weight_mode = weight_mode
        self.pattern_top_fraction = float(pattern_top_fraction)
        self.pattern_min_support = int(pattern_min_support)
        self.accept_equal_sparser = bool(accept_equal_sparser)
        self.refine_fraction = float(refine_fraction)
        self.max_refine_trials_per_generation = max_refine_trials_per_generation
        self.shuffle_blocks = bool(shuffle_blocks)
        self.signed_repair_probability = float(signed_repair_probability)
        self.stats = BBSearchStats()

    @staticmethod
    def _canonical_mode(mode: str) -> BBSearchMode:
        value = str(mode).strip().lower().replace("-", "_")
        if value in {"", "none", "off", "false", "0"}:
            return "none"
        if value in {"uniform", "bb_uniform", "block_uniform"}:
            return "uniform"
        if value in {"pattern", "pattern_refine", "bb_pattern_refine"}:
            return "pattern_refine"
        raise ValueError(f"Unsupported bb_search_mode={mode!r}.")

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def ordered_blocks(self, snapshot: BuildingBlockSnapshot | None) -> list[BuildingBlock]:
        if snapshot is None or not snapshot.blocks:
            return []
        blocks = list(snapshot.blocks)
        blocks.sort(key=lambda b: (b.size, -b.score, b.features))
        if self.shuffle_blocks and len(blocks) > 1:
            perm = self.rng.permutation(len(blocks))
            blocks = [blocks[int(i)] for i in perm]
        return blocks

    def block_uniform_crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        snapshot: BuildingBlockSnapshot | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Uniform crossover with selected LTGA blocks inherited atomically."""
        blocks = self.ordered_blocks(snapshot)
        if not blocks:
            return self._plain_uniform_crossover(p1, p2)

        mask = self.rng.integers(0, 2, size=self.n_features, dtype=bool)
        child1 = np.where(mask, p2, p1).astype(np.int8)
        child2 = np.where(mask, p1, p2).astype(np.int8)
        for block in blocks:
            idx = np.asarray(block.features, dtype=int)
            if idx.size == 0:
                continue
            if bool(self.rng.integers(0, 2)):
                child1[idx] = self._donor_block_values(block, p2)
                child2[idx] = self._donor_block_values(block, p1)
            else:
                child1[idx] = self._donor_block_values(block, p1)
                child2[idx] = self._donor_block_values(block, p2)
        self._ensure_non_empty(child1)
        self._ensure_non_empty(child2)
        return child1, child2

    def _plain_uniform_crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = self.rng.integers(0, 2, size=self.n_features, dtype=bool)
        child1 = np.where(mask, p2, p1).astype(np.int8)
        child2 = np.where(mask, p1, p2).astype(np.int8)
        self._ensure_non_empty(child1)
        self._ensure_non_empty(child2)
        return child1, child2

    def _donor_block_values(self, block: BuildingBlock, donor: np.ndarray) -> np.ndarray:
        idx = np.asarray(block.features, dtype=int)
        raw = np.asarray(donor[idx], dtype=np.int8).copy()
        if not self._use_signed_schema(block):
            return raw
        if self.rng.random() > self.signed_repair_probability:
            return raw
        return self._project_signed_pattern(block, donor)

    def archive_top_matrix(self, X_archive: np.ndarray, y_archive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if X_archive.size == 0 or y_archive.size == 0:
            return np.empty((0, self.n_features), dtype=np.int8), np.empty(0, dtype=float)
        n = int(X_archive.shape[0])
        n_top = max(self.pattern_min_support, int(np.ceil(self.pattern_top_fraction * n)))
        n_top = min(n, max(1, n_top))
        order = np.argsort(np.asarray(y_archive, dtype=float))[::-1][:n_top]
        return X_archive[order].astype(np.int8, copy=False), y_archive[order].astype(float, copy=False)

    def learn_block_pattern(
        self,
        block: BuildingBlock,
        X_top: np.ndarray,
        y_top: np.ndarray,
    ) -> np.ndarray | None:
        """Return the best supported pattern for one block from top archive rows."""
        if X_top.size == 0 or y_top.size == 0:
            return None
        idx = np.asarray(block.features, dtype=int)
        if idx.size == 0:
            return None

        patterns: dict[tuple[int, ...], list[float]] = {}
        for row, fitness in zip(X_top, y_top):
            pattern = self._project_signed_pattern(block, row) if self._use_signed_schema(block) else row[idx]
            key = tuple(int(v) for v in pattern)
            patterns.setdefault(key, []).append(float(fitness))

        min_support = min(self.pattern_min_support, max(1, X_top.shape[0]))
        supported = [(key, vals) for key, vals in patterns.items() if len(vals) >= min_support]
        if not supported:
            supported = list(patterns.items())
        if not supported:
            return None

        best_key, _ = max(
            supported,
            key=lambda item: (float(np.mean(item[1])), len(item[1]), -sum(item[0])),
        )
        return np.asarray(best_key, dtype=np.int8)

    def accept_block_mix(
        self,
        candidate_fitness: float,
        current_fitness: float,
        candidate: np.ndarray,
        current: np.ndarray,
    ) -> bool:
        if candidate_fitness > current_fitness + _EPS:
            return True
        if self.accept_equal_sparser and abs(candidate_fitness - current_fitness) <= _EPS:
            return int(np.sum(candidate)) < int(np.sum(current))
        return False

    def n_individuals_to_refine(self, population_size: int) -> int:
        return max(1, min(int(population_size), int(np.ceil(self.refine_fraction * population_size))))

    def max_trials_this_generation(self, population_size: int) -> int:
        if self.max_refine_trials_per_generation is None:
            return max(1, int(population_size))
        return max(1, int(self.max_refine_trials_per_generation))

    def _use_signed_schema(self, block: BuildingBlock) -> bool:
        return (
            self.weight_mode == "signed"
            and (bool(block.positive_group) or bool(block.negative_group))
            and float(block.signed_balance) > 0.0
        )

    def _project_signed_pattern(self, block: BuildingBlock, chromosome: np.ndarray) -> np.ndarray:
        """Project one block to the coherent same/opposite signed schema.

        The absolute 0/1 anchor is not fixed by the signed graph.  It is inferred
        from the donor/archive row: positive-group bits vote directly, while
        negative-group bits vote after complementing.  This preserves the learned
        relative relation while still allowing both ABC and ~A~B~C style states
        to be selected by the data/fitness archive.
        """
        idx = tuple(int(f) for f in block.features)
        pos, neg = self._signed_groups(block)
        if not idx:
            return np.empty(0, dtype=np.int8)
        if not pos and not neg:
            return np.asarray(chromosome[list(idx)], dtype=np.int8).copy()
        votes: list[int] = []
        for f in pos:
            votes.append(int(chromosome[f]))
        for f in neg:
            votes.append(1 - int(chromosome[f]))
        if not votes:
            return np.asarray(chromosome[list(idx)], dtype=np.int8).copy()
        mean_vote = float(np.mean(votes))
        anchor = 1 if mean_vote > 0.5 else (0 if mean_vote < 0.5 else int(self.rng.integers(0, 2)))
        values: list[int] = []
        pos_set = set(pos)
        neg_set = set(neg)
        for f in idx:
            if f in pos_set:
                values.append(anchor)
            elif f in neg_set:
                values.append(1 - anchor)
            else:
                values.append(int(chromosome[f]))
        return np.asarray(values, dtype=np.int8)

    def _signed_groups(self, block: BuildingBlock) -> tuple[tuple[int, ...], tuple[int, ...]]:
        features = tuple(int(f) for f in block.features)
        feature_set = set(features)
        pos = self._parse_feature_list(block.positive_group)
        neg = self._parse_feature_list(block.negative_group)
        pos = tuple(f for f in pos if f in feature_set)
        neg = tuple(f for f in neg if f in feature_set and f not in set(pos))
        covered = set(pos) | set(neg)
        if not covered:
            return features, ()
        # Any missing feature is kept in the positive phase so every feature in
        # the block remains represented by the returned pattern.
        missing = tuple(f for f in features if f not in covered)
        return tuple(pos + missing), neg

    @staticmethod
    def _parse_feature_list(text: str | Iterable[int]) -> tuple[int, ...]:
        if isinstance(text, str):
            if not text.strip():
                return ()
            out: list[int] = []
            for part in text.replace(",", ";").split(";"):
                part = part.strip()
                if not part:
                    continue
                try:
                    out.append(int(part))
                except ValueError:
                    continue
            return tuple(out)
        return tuple(int(v) for v in text)

    def _ensure_non_empty(self, chrom: np.ndarray) -> None:
        if not np.any(chrom):
            chrom[int(self.rng.integers(0, self.n_features))] = 1
