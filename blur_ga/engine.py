from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import GAConfig
from .building_blocks import (
    LTGABuildingBlockConfig,
    LTGABuildingBlockExtractor,
    LinkageGraphView,
    BuildingBlockSnapshot,
)
from .evaluator import FitnessEvaluator
from .evig import EmpiricalVIG
from .lr_linkage import RegressionLinkageLearner, RegressionVIG, ga_type_to_regression_stage

_EPS = 1e-12


@dataclass
class Individual:
    chromosome: np.ndarray
    fitness: float

    def copy(self) -> "Individual":
        return Individual(self.chromosome.copy(), float(self.fitness))


@dataclass
class RunResult:
    run_id: int
    seed: int
    best_chromosome: np.ndarray
    best_inner_fitness: float
    generations: int
    runtime_seconds: float
    eval_count: int
    evig: EmpiricalVIG | RegressionVIG | None
    generation_trace: list[dict[str, float | int | str]] = field(default_factory=list)
    linkage_events: list[dict[str, float | int | bool]] = field(default_factory=list)
    graph_snapshots: list[dict[str, float | int | str]] = field(default_factory=list)
    building_block_summaries: list[dict[str, float | int | str]] = field(default_factory=list)
    building_blocks: list[dict[str, float | int | str]] = field(default_factory=list)
    archive_chromosomes: np.ndarray | None = None
    archive_fitness: np.ndarray | None = None
    archive_generations: np.ndarray | None = None

    @property
    def subset_size(self) -> int:
        return int(np.sum(self.best_chromosome))

    @property
    def n_edges(self) -> int:
        return 0 if self.evig is None else self.evig.n_edges


class GeneticFeatureSelector:
    """Standard GA / linkage-aware optimizer for binary feature selection.

    The optimizer is intentionally independent of datasets. It only depends on a
    FitnessEvaluator, making it usable with holdout validation, inner CV, or any
    future research evaluator without touching the GA logic.
    """

    def __init__(self, config: GAConfig, evaluator: FitnessEvaluator, *, seed: int = 1, run_id: int = 0) -> None:
        self.config = config
        self.evaluator = evaluator
        self.seed = int(seed)
        self.run_id = int(run_id)
        self.rng = np.random.default_rng(self.seed)
        self.n_features = evaluator.n_features
        self.mutation_probability = (
            1.0 / self.n_features if config.mutation_probability is None else float(config.mutation_probability)
        )
        self.population: list[Individual] = []
        self.best_so_far: Individual | None = None
        self.generation_trace: list[dict[str, float | int | str]] = []
        self.linkage_events: list[dict[str, float | int | bool]] = []
        self.graph_snapshots: list[dict[str, float | int | str]] = []
        self.building_block_summaries: list[dict[str, float | int | str]] = []
        self.building_blocks: list[dict[str, float | int | str]] = []
        self._latest_building_block_snapshot: BuildingBlockSnapshot | None = None
        self._bb_extractor: LTGABuildingBlockExtractor | None = None
        if self.config.build_building_blocks:
            self._bb_extractor = LTGABuildingBlockExtractor(
                LTGABuildingBlockConfig(
                    enabled=True,
                    weight_mode=self.config.bb_weight_mode,
                    min_block_size=self.config.bb_min_block_size,
                    max_block_size=self.config.bb_max_block_size,
                    max_blocks=self.config.bb_max_blocks,
                    snapshot_interval=self.config.bb_snapshot_interval,
                    external_penalty=self.config.bb_external_penalty,
                    size_penalty=self.config.bb_size_penalty,
                )
            )
        self._current_generation = 0
        self._archive_seen: set[tuple[int, ...]] = set()
        self._archive_chromosomes: list[np.ndarray] = []
        self._archive_fitness: list[float] = []
        self._archive_generations: list[int] = []
        self._archive_cache_size = -1
        self._archive_X_cache: np.ndarray | None = None
        self._archive_y_cache: np.ndarray | None = None
        self._archive_gen_cache: np.ndarray | None = None
        self._last_lr_archive_size = 0
        self._regression_stage = ga_type_to_regression_stage(config.ga_type)
        self._lr_learner: RegressionLinkageLearner | None = None
        if self._regression_stage is not None:
            self._lr_learner = RegressionLinkageLearner(
                n_features=self.n_features,
                stage=self._regression_stage,
                sparse_alpha=config.lr_sparse_alpha,
                auto_alpha=config.lr_auto_alpha,
                alpha_c=config.lr_alpha_c,
                delta=config.lr_delta,
                stability_subsamples=config.lr_stability_subsamples,
                stability_fraction=config.lr_stability_fraction,
                random_state=self.seed,
            )

    def run(self) -> RunResult:
        start = time.perf_counter()
        if self.config.ga_type == 1:
            evig: EmpiricalVIG | RegressionVIG | None = EmpiricalVIG(self.n_features)
        elif self._regression_stage is not None:
            evig = RegressionVIG(self.n_features)
        else:
            evig = None
        self.population = self._initial_population()
        self._update_best_from_population()
        last_improvement_gen = 0
        last_best = self._best_population_fitness()
        generation = 0

        while True:
            generation += 1
            current_best = self._best_population_fitness()
            if current_best - last_best > _EPS:
                last_best = current_best
                last_improvement_gen = generation

            if generation - last_improvement_gen > self.config.tau_reset:
                self._restart_population()
                self._update_best_from_population()
                last_best = self._best_population_fitness()
                last_improvement_gen = generation

            self._current_generation = generation
            if self.config.ga_type == 1:
                assert isinstance(evig, EmpiricalVIG)
                self.population = self._next_generation_empirical_linkage(evig, generation=generation)
                if self.config.save_graph_snapshots and generation % self.config.graph_snapshot_interval == 0:
                    self._capture_graph_snapshot(evig, generation)
                # Empirical/GAwLL linkage is updated online by linkage mutation;
                # use a dedicated interval for LTGA diagnostics.
                self._capture_building_blocks(evig, generation)
            elif self._regression_stage is not None:
                assert isinstance(evig, RegressionVIG)
                self.population = self._next_generation_standard()
                lr_updated = False
                if generation % self.config.lr_gap_gen == 0:
                    lr_updated = self._fit_regression_graph(evig, generation)
                if self.config.save_graph_snapshots and generation % self.config.graph_snapshot_interval == 0:
                    self._capture_graph_snapshot(evig, generation)
                # For type 2/3, LTGA is only rebuilt when the LR linkage graph
                # has actually been refit and pushed into RegressionVIG.
                if lr_updated:
                    self._capture_building_blocks(evig, generation, force=True)
            else:
                self.population = self._next_generation_standard()
            self._update_best_from_population()

            if self.config.save_generation_trace:
                self.generation_trace.append(self._make_generation_trace_row(generation, start, evig))

            if self._should_stop(generation, start):
                break

        if self._regression_stage is not None and isinstance(evig, RegressionVIG):
            self._fit_regression_graph(evig, generation, force=True)
            if self.config.save_graph_snapshots:
                self.graph_snapshots = [
                    item for item in self.graph_snapshots if int(item.get("generation", -1)) != generation
                ]
                self._capture_graph_snapshot(evig, generation)
            self.building_block_summaries = [
                item for item in self.building_block_summaries if int(item.get("generation", -1)) != generation
            ]
            self.building_blocks = [
                item for item in self.building_blocks if int(item.get("generation", -1)) != generation
            ]
            self._capture_building_blocks(evig, generation, force=True)

        assert self.best_so_far is not None
        return RunResult(
            run_id=self.run_id,
            seed=self.seed,
            best_chromosome=self.best_so_far.chromosome.copy(),
            best_inner_fitness=float(self.best_so_far.fitness),
            generations=generation,
            runtime_seconds=float(time.perf_counter() - start),
            eval_count=int(self.evaluator.eval_count),
            evig=evig,
            generation_trace=self.generation_trace,
            linkage_events=self.linkage_events,
            graph_snapshots=self.graph_snapshots,
            building_block_summaries=self.building_block_summaries,
            building_blocks=self.building_blocks,
            archive_chromosomes=np.vstack(self._archive_chromosomes).astype(np.int8, copy=False) if self._archive_chromosomes else np.empty((0, self.n_features), dtype=np.int8),
            archive_fitness=np.asarray(self._archive_fitness, dtype=float),
            archive_generations=np.asarray(self._archive_generations, dtype=int),
        )

    def _evaluate(self, chrom: np.ndarray) -> float:
        arr = np.asarray(chrom, dtype=np.int8)
        fitness = float(self.evaluator.evaluate(arr))
        key = tuple(int(v) for v in arr)
        if key not in self._archive_seen:
            self._archive_seen.add(key)
            self._archive_chromosomes.append(arr.copy())
            self._archive_fitness.append(fitness)
            self._archive_generations.append(int(self._current_generation))
        return fitness

    def _effective_lr_min_samples(self) -> int:
        """Return the archive-size threshold used before fitting regression linkage.

        For ga_type 2/3, this implements the PSLE/MPSLE scaling
        n_min = ceil(c_n * s_hat * log(2 p_g / delta)), while keeping
        lr_min_samples as a user-specified lower bound for backward compatibility.
        Older regression stages keep the original fixed lr_min_samples behavior.
        """
        base = int(self.config.lr_min_samples)
        if not self.config.lr_auto_min_samples:
            return base
        if self._regression_stage not in {"pairwise_lasso", "main_pairwise_lasso"}:
            return base
        p_cols = max(1, self.n_features * (self.n_features - 1) // 2)
        s_hat = int(self.config.lr_expected_edges) if self.config.lr_expected_edges is not None else 1
        s_hat = max(1, min(s_hat, p_cols))
        delta = min(max(float(self.config.lr_delta), 1e-12), 1.0 - 1e-12)
        theory_min = int(np.ceil(float(self.config.lr_min_samples_c) * s_hat * np.log((2.0 * p_cols) / delta)))
        return max(2, base, theory_min)

    def _archive_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        archive_size = len(self._archive_chromosomes)
        if self._archive_cache_size != archive_size or self._archive_X_cache is None:
            self._archive_X_cache = np.vstack(self._archive_chromosomes).astype(np.int8, copy=False)
            self._archive_y_cache = np.asarray(self._archive_fitness, dtype=float)
            self._archive_gen_cache = np.asarray(self._archive_generations, dtype=int)
            self._archive_cache_size = archive_size
        assert self._archive_y_cache is not None
        assert self._archive_gen_cache is not None
        return self._archive_X_cache, self._archive_y_cache, self._archive_gen_cache

    def _fit_regression_graph(self, evig: RegressionVIG, generation: int, force: bool = False) -> bool:
        """Fit the regression linkage model and return True only on a real graph update."""
        if self._lr_learner is None:
            return False
        archive_size = len(self._archive_chromosomes)
        if archive_size - self._last_lr_archive_size < self._effective_lr_min_samples() and not force:
            return False
        if archive_size == self._last_lr_archive_size and evig.last_n_samples == archive_size:
            evig.touch_fit_metadata(generation, archive_size)
            return False
        X_archive, y_archive, gen_archive = self._archive_arrays()
        fit = self._lr_learner.fit(X_archive, y_archive, gen_archive, generation=generation)
        evig.update_from_fit(
            fit,
            min_abs_weight=self.config.lr_edge_min_weight,
            top_k=self.config.lr_edge_top_k,
        )
        self._last_lr_archive_size = archive_size
        return True

    def _make_generation_trace_row(
        self,
        generation: int,
        start_time: float,
        evig: EmpiricalVIG | RegressionVIG | None,
    ) -> dict[str, float | int | str]:
        """Collect per-generation search diagnostics without changing the GA state.

        The original trace only stored best fitness and edge counts.  For later
        analysis we also store runtime, population-level fitness/subset
        statistics, archive size, and the best chromosomes at this generation.
        These reads are deterministic and do not consume RNG, so enabling the
        trace does not change the produced GA result.
        """
        fitness = np.asarray([ind.fitness for ind in self.population], dtype=float)
        subset_sizes = np.asarray([int(np.sum(ind.chromosome)) for ind in self.population], dtype=int)
        best_population = max(self.population, key=lambda ind: ind.fitness)
        best_so_far = self.best_so_far if self.best_so_far is not None else best_population
        best_pop_chrom = best_population.chromosome.astype(int, copy=False)
        best_so_far_chrom = best_so_far.chromosome.astype(int, copy=False)
        unique_population = len({tuple(int(v) for v in ind.chromosome) for ind in self.population})

        return {
            "generation": int(generation),
            "elapsed_seconds": float(time.perf_counter() - start_time),
            "best_population_fitness": float(np.max(fitness)) if fitness.size else 0.0,
            "mean_population_fitness": float(np.mean(fitness)) if fitness.size else 0.0,
            "std_population_fitness": float(np.std(fitness)) if fitness.size else 0.0,
            "worst_population_fitness": float(np.min(fitness)) if fitness.size else 0.0,
            "best_so_far_fitness": float(best_so_far.fitness),
            "best_population_subset_size": int(np.sum(best_pop_chrom)),
            "best_so_far_subset_size": int(np.sum(best_so_far_chrom)),
            "mean_subset_size": float(np.mean(subset_sizes)) if subset_sizes.size else 0.0,
            "min_subset_size": int(np.min(subset_sizes)) if subset_sizes.size else 0,
            "max_subset_size": int(np.max(subset_sizes)) if subset_sizes.size else 0,
            "population_unique_chromosomes": int(unique_population),
            "n_edges": int(evig.n_edges if evig is not None else 0),
            "n_tested_pairs": int(evig.n_tested_pairs if evig is not None else 0),
            "eval_count": int(self.evaluator.eval_count),
            "archive_size": int(len(self._archive_chromosomes)),
            "bb_n_blocks": int(self._latest_building_block_snapshot.n_blocks) if self._latest_building_block_snapshot else 0,
            "bb_n_candidates": int(self._latest_building_block_snapshot.n_candidates) if self._latest_building_block_snapshot else 0,
            "bb_max_score": float(self._latest_building_block_snapshot.max_score) if self._latest_building_block_snapshot else 0.0,
            "bb_mean_block_size": float(self._latest_building_block_snapshot.mean_block_size) if self._latest_building_block_snapshot else 0.0,
            "bb_positive_edges_in_blocks": int(self._latest_building_block_snapshot.n_positive_edges_in_blocks) if self._latest_building_block_snapshot else 0,
            "bb_negative_edges_in_blocks": int(self._latest_building_block_snapshot.n_negative_edges_in_blocks) if self._latest_building_block_snapshot else 0,
            "bb_mean_signed_balance": float(self._latest_building_block_snapshot.mean_signed_balance) if self._latest_building_block_snapshot else 0.0,
            "bb_sign_conflicts_in_blocks": int(self._latest_building_block_snapshot.n_sign_conflicts_in_blocks) if self._latest_building_block_snapshot else 0,
            "best_population_chromosome": "".join(str(int(v)) for v in best_pop_chrom),
            "best_so_far_chromosome": "".join(str(int(v)) for v in best_so_far_chrom),
            "best_population_selected_features": ";".join(str(int(i)) for i in np.flatnonzero(best_pop_chrom == 1)),
            "best_so_far_selected_features": ";".join(str(int(i)) for i in np.flatnonzero(best_so_far_chrom == 1)),
        }

    def _capture_graph_snapshot(self, evig: EmpiricalVIG | RegressionVIG, generation: int) -> None:
        signed_matrix = None
        raw_signed_matrix = None
        if hasattr(evig, "signed_matrix"):
            signed_matrix = evig.signed_matrix()
        elif hasattr(evig, "coefficient_matrix"):
            signed_matrix = evig.coefficient_matrix()
        if hasattr(evig, "raw_signed_matrix"):
            raw_signed_matrix = evig.raw_signed_matrix()
        for (
            feature_i,
            feature_j,
            weight,
            positive_count,
            tested_count,
            weight_sum,
            first_seen,
            last_updated,
        ) in evig.edge_table(
            min_weight=self.config.graph_snapshot_min_weight,
            top_k=self.config.graph_snapshot_top_k,
        ):
            signed_weight = float(signed_matrix[feature_i, feature_j]) if signed_matrix is not None else float(weight)
            sign = 1 if signed_weight > 0 else (-1 if signed_weight < 0 else 0)
            raw_signed_weight = float(raw_signed_matrix[feature_i, feature_j]) if raw_signed_matrix is not None else float(signed_weight)
            raw_sign = 1 if raw_signed_weight > 0 else (-1 if raw_signed_weight < 0 else 0)
            self.graph_snapshots.append(
                {
                    "generation": int(generation),
                    "feature_i": int(feature_i),
                    "feature_j": int(feature_j),
                    "weight": float(weight),
                    "signed_weight": float(signed_weight),
                    "sign": int(sign),
                    "raw_signed_weight": float(raw_signed_weight),
                    "raw_sign": int(raw_sign),
                    "positive_count": int(positive_count),
                    "tested_count": int(tested_count),
                    "weight_sum": float(weight_sum),
                    "first_seen_generation": int(first_seen),
                    "last_updated_generation": int(last_updated),
                }
            )

    def _capture_building_blocks(
        self,
        evig: EmpiricalVIG | RegressionVIG,
        generation: int,
        *,
        force: bool = False,
    ) -> None:
        if self._bb_extractor is None:
            return
        if not force:
            if self.config.ga_type == 1:
                interval = self.config.bb_gawll_update_interval or self.config.bb_snapshot_interval
                if generation % int(interval) != 0:
                    return
            elif self._regression_stage is not None:
                # Regression variants call this method only after a successful LR
                # graph update, so no generation-periodic rebuild is needed here.
                pass
            else:
                return
        if evig.n_edges <= 0:
            self._latest_building_block_snapshot = None
            return
        snapshot = self._bb_extractor.extract(LinkageGraphView.from_evig(evig), generation=generation)
        self._latest_building_block_snapshot = snapshot
        self.building_block_summaries.append(snapshot.summary_row())
        for block in snapshot.blocks:
            self.building_blocks.append(
                {
                    "generation": int(generation),
                    "bb_method": "ltga",
                    "bb_weight_mode": snapshot.mode,
                    "block_id": int(block.block_id),
                    "features": block.features_str,
                    "size": int(block.size),
                    "score": float(block.score),
                    "density": float(block.density),
                    "internal_abs_mean": float(block.internal_abs_mean),
                    "internal_positive_mean": float(block.internal_positive_mean),
                    "internal_negative_abs_mean": float(block.internal_negative_abs_mean),
                    "external_abs_mean": float(block.external_abs_mean),
                    "n_internal_edges": int(block.n_internal_edges),
                    "n_positive_edges": int(block.n_positive_edges),
                    "n_negative_edges": int(block.n_negative_edges),
                    "signed_balance": float(block.signed_balance),
                    "n_sign_conflicts": int(block.n_sign_conflicts),
                    "polarity": block.polarity,
                    "positive_group": block.positive_group,
                    "negative_group": block.negative_group,
                    "schema_hint": block.schema_hint,
                    "source": block.source,
                }
            )

    def _should_stop(self, generation: int, start_time: float) -> bool:
        if self.config.stop == "gen":
            return generation >= self.config.max_gen
        if self.config.stop == "time":
            if self.config.max_time is None:
                raise ValueError("max_time must be provided when stop='time'.")
            return (time.perf_counter() - start_time) >= self.config.max_time
        if self.config.stop == "eval":
            if self.config.max_evals is None:
                raise ValueError("max_evals must be provided when stop='eval'.")
            return self.evaluator.eval_count >= self.config.max_evals
        raise ValueError(f"Unknown stop criterion: {self.config.stop}")

    def _initial_population(self) -> list[Individual]:
        population: list[Individual] = []
        for _ in range(self.config.popsize):
            chrom = self.rng.integers(0, 2, size=self.n_features, dtype=np.int8)
            if not np.any(chrom):
                chrom[self.rng.integers(0, self.n_features)] = 1
            fit = self._evaluate(chrom)
            ind = Individual(chrom, fit)
            if self.config.local_search:
                ind = self._local_search(ind)
            population.append(ind)
        return population

    def _restart_population(self) -> None:
        if not self.population:
            return
        elite = max(self.population, key=lambda ind: ind.fitness).copy()
        n_reset = min(self.config.popsize, int(self.config.resetpop_rate * self.config.popsize) + 1)
        new_pop = [elite]
        for _ in range(1, n_reset):
            chrom = self.rng.integers(0, 2, size=self.n_features, dtype=np.int8)
            if not np.any(chrom):
                chrom[self.rng.integers(0, self.n_features)] = 1
            ind = Individual(chrom, self._evaluate(chrom))
            if self.config.local_search:
                ind = self._local_search(ind)
            new_pop.append(ind)
        while len(new_pop) < self.config.popsize:
            new_pop.append(self._tournament().copy())
        self.population = new_pop

    def _best_population_fitness(self) -> float:
        return float(max(ind.fitness for ind in self.population))

    def _update_best_from_population(self) -> None:
        best = max(self.population, key=lambda ind: ind.fitness)
        if self.best_so_far is None or best.fitness > self.best_so_far.fitness:
            self.best_so_far = best.copy()

    def _tournament(self) -> Individual:
        best_idx = int(self.rng.integers(0, len(self.population)))
        for _ in range(1, self.config.tournament_size):
            idx = int(self.rng.integers(0, len(self.population)))
            if self.population[idx].fitness > self.population[best_idx].fitness:
                best_idx = idx
        return self.population[best_idx]

    def _uniform_crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = self.rng.integers(0, 2, size=self.n_features, dtype=bool)
        child1 = np.where(mask, p2, p1).astype(np.int8)
        child2 = np.where(mask, p1, p2).astype(np.int8)
        return child1, child2

    def _mutate(self, chrom: np.ndarray) -> np.ndarray:
        flips = self.rng.random(self.n_features) < self.mutation_probability
        out = chrom.copy()
        out[flips] = 1 - out[flips]
        if not np.any(out):
            out[self.rng.integers(0, self.n_features)] = 1
        return out

    def _make_individual(self, chrom: np.ndarray) -> Individual:
        return Individual(chrom.astype(np.int8, copy=False), self._evaluate(chrom))

    def _next_generation_standard(self) -> list[Individual]:
        elite = max(self.population, key=lambda ind: ind.fitness).copy()
        new_pop: list[Individual] = [elite]
        while len(new_pop) < self.config.popsize:
            p1 = self._tournament().chromosome
            if len(new_pop) < self.config.popsize - 1:
                p2 = self._tournament().chromosome
                if self.rng.random() < self.config.crossover_probability:
                    c1, c2 = self._uniform_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                new_pop.append(self._make_individual(self._mutate(c1)))
                new_pop.append(self._make_individual(self._mutate(c2)))
            else:
                new_pop.append(self._make_individual(self._mutate(p1)))
        return new_pop[: self.config.popsize]

    def _next_generation_empirical_linkage(self, evig: EmpiricalVIG, *, generation: int) -> list[Individual]:
        elite = max(self.population, key=lambda ind: ind.fitness).copy()
        new_pop: list[Individual] = [elite]
        crossover_quota = int(self.config.ll_crossover_ratio * self.config.popsize)
        while len(new_pop) < self.config.popsize:
            p1 = self._tournament()
            if len(new_pop) < crossover_quota and len(new_pop) < self.config.popsize - 1:
                p2 = self._tournament()
                c1, c2 = self._uniform_crossover(p1.chromosome, p2.chromosome)
                new_pop.append(self._make_individual(self._mutate(c1)))
                new_pop.append(self._make_individual(self._mutate(c2)))
            elif len(new_pop) < self.config.popsize - 2:
                children = self._linkage_mutation(p1, evig, generation=generation)
                new_pop.extend(children)
            else:
                new_pop.append(self._make_individual(self._mutate(p1.chromosome)))
        return new_pop[: self.config.popsize]

    def _linkage_mutation(self, parent: Individual, evig: EmpiricalVIG, *, generation: int) -> list[Individual]:
        base = parent.chromosome
        g, h = self.rng.choice(self.n_features, size=2, replace=False)
        xg = base.copy()
        xh = base.copy()
        xhg = base.copy()
        xg[g] = 1 - xg[g]
        xh[h] = 1 - xh[h]
        xhg[g] = xg[g]
        xhg[h] = xh[h]
        if not np.any(xg):
            xg[g] = 1
        if not np.any(xh):
            xh[h] = 1
        if not np.any(xhg):
            xhg[g] = 1

        fx = parent.fitness
        ind_g = self._make_individual(xg)
        ind_h = self._make_individual(xh)
        ind_hg = self._make_individual(xhg)
        raw_diff_in_diff = ind_hg.fitness - ind_h.fitness - ind_g.fitness + fx
        # Orient the local binary second difference to the canonical 0/1 coefficient
        # sign.  Without this correction, the sign of the same QUBO-style edge can
        # flip depending on whether the parent state is 00, 01, 10, or 11.
        # Effective directions are computed after the non-empty-subset guard above;
        # if a nominal flip became a no-op, the observation is treated as inactive.
        dir_g = int(xg[g]) - int(base[g])
        dir_h = int(xh[h]) - int(base[h])
        oriented_diff_in_diff = raw_diff_in_diff * dir_g * dir_h if dir_g != 0 and dir_h != 0 else 0.0
        obs = evig.add_observation(
            int(g),
            int(h),
            oriented_diff_in_diff,
            raw_observed_weight=raw_diff_in_diff,
            epsilon=self.config.edge_epsilon,
            generation=generation,
        )

        if self.config.save_linkage_events:
            best_so_far_fit = float(self.best_so_far.fitness if self.best_so_far else fx)
            self.linkage_events.append(
                {
                    "generation": int(generation),
                    "eval_count": int(self.evaluator.eval_count),
                    "feature_i": int(obs.feature_i),
                    "feature_j": int(obs.feature_j),
                    "omega": float(obs.observed_weight),
                    "signed_omega": float(obs.observed_signed_weight),
                    "raw_signed_omega": float(obs.raw_observed_signed_weight),
                    "raw_sign": int(obs.raw_sign),
                    "sign": int(obs.sign),
                    "is_active": bool(obs.is_active),
                    "is_positive": bool(obs.is_positive),
                    "was_new_edge": bool(obs.was_new_edge),
                    "edge_weight_after": float(obs.edge_weight_after),
                    "signed_weight_after": float(obs.signed_weight_after),
                    "raw_signed_weight_after": float(obs.raw_signed_weight_after),
                    "active_count_after": int(obs.active_count_after),
                    "positive_count_after": int(obs.positive_count_after),
                    "negative_count_after": int(obs.negative_count_after),
                    "tested_count_after": int(obs.tested_count_after),
                    "weight_sum_after": float(obs.weight_sum_after),
                    "signed_weight_sum_after": float(obs.signed_weight_sum_after),
                    "raw_signed_weight_sum_after": float(obs.raw_signed_weight_sum_after),
                    "first_seen_generation": int(obs.first_seen_generation),
                    "last_updated_generation": int(obs.last_updated_generation),
                    "parent_fitness": float(fx),
                    "fitness_g": float(ind_g.fitness),
                    "fitness_h": float(ind_h.fitness),
                    "fitness_gh": float(ind_hg.fitness),
                    "best_so_far_fitness": best_so_far_fit,
                    "subset_size_parent": int(np.sum(base)),
                    "subset_size_best": int(np.sum(self.best_so_far.chromosome)) if self.best_so_far else int(np.sum(base)),
                }
            )
        return [ind_g, ind_h, ind_hg]

    def _local_search(self, individual: Individual) -> Individual:
        chrom = individual.chromosome.copy()
        fitness = float(individual.fitness)
        no_improvement = 0
        n_restart = 0
        order = self.rng.permutation(self.n_features)
        pos = 0
        while no_improvement < self.n_features:
            gene = int(order[pos])
            candidate = chrom.copy()
            candidate[gene] = 1 - candidate[gene]
            if not np.any(candidate):
                fit = 0.0
            else:
                fit = self._evaluate(candidate)
            if fit > fitness + _EPS:
                chrom = candidate
                fitness = fit
                no_improvement = 0
                n_restart += 1
                if n_restart > 5 * self.n_features:
                    break
            else:
                no_improvement += 1
            pos = (pos + 1) % self.n_features
        return Individual(chrom, fitness)
