from __future__ import annotations

"""LTGA-style building-block extraction from signed linkage graphs.

The extractor is a diagnostic/model-building layer: it reads the current
linkage graph and returns compact building-block logs without changing the GA
operators.  Two modes are intentionally supported:

- ``absolute``: ignore the sign during LTGA clustering and scoring; positive and
  negative edges are still reported as diagnostics.
- ``signed``: use the same absolute edge strength for LTGA clustering, but infer
  a polarity assignment inside every block.  Positive edges prefer variables in
  the same polarity group (AB or ~A~B); negative edges prefer opposite polarity
  groups (A~B or ~AB).  Blocks whose signs are internally inconsistent receive a
  lower score.
- ``positive``: keep only positive/same-state linkages when building and scoring
  the LTGA tree.  Negative/opposite-state linkages are ignored for the tree in
  this mode.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

BBWeightMode = Literal["absolute", "signed", "positive"]
BBScoreMode = Literal["current", "current_stability"]
BBSelectionMode = Literal["count", "coverage_budget"]

_EPS = 1e-12


@dataclass(frozen=True)
class LinkageGraphView:
    """Immutable view of one linkage graph snapshot.

    Parameters
    ----------
    weight:
        Non-negative edge importance, usually abs(coefficient), stability-weighted
        abs(coefficient), or empirical average absolute diff-in-diff.
    signed_weight:
        Signed edge value.  Positive means same-state relation; negative means
        opposite-state relation.  If unavailable, the unsigned weight is treated
        as positive.
    """

    weight: np.ndarray
    signed_weight: np.ndarray

    @classmethod
    def from_evig(cls, evig: object) -> "LinkageGraphView":
        weight = np.asarray(getattr(evig, "weight"), dtype=float)
        if hasattr(evig, "signed_matrix"):
            signed = np.asarray(getattr(evig, "signed_matrix")(), dtype=float)
        elif hasattr(evig, "coefficient_matrix"):
            signed = np.asarray(getattr(evig, "coefficient_matrix")(), dtype=float)
        elif hasattr(evig, "coefficient"):
            signed = np.asarray(getattr(evig, "coefficient"), dtype=float)
        else:
            signed = weight.copy()
        if signed.shape != weight.shape:
            signed = np.zeros_like(weight, dtype=float)
        weight = np.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
        signed = np.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(weight, 0.0)
        np.fill_diagonal(signed, 0.0)
        return cls(weight=weight, signed_weight=signed)

    def strength_matrix(self, mode: BBWeightMode) -> np.ndarray:
        """Return the non-negative matrix used by LTGA clustering.

        ``absolute`` and ``signed`` both treat positive and negative linkages as
        evidence that variables belong near each other in the tree; ``signed``
        later checks whether those signs form a coherent same/opposite schema.
        ``positive`` is stricter: it keeps only positive same-state linkages.
        """
        if mode not in {"absolute", "signed", "positive"}:  # pragma: no cover - config guards this.
            raise ValueError(f"Unknown building-block weight mode: {mode!r}")
        weight = np.asarray(self.weight, dtype=float).copy()
        weight[weight < 0.0] = 0.0
        if mode == "positive":
            signed = np.asarray(self.signed_weight, dtype=float)
            weight = np.where(signed > _EPS, weight, 0.0)
        np.fill_diagonal(weight, 0.0)
        return weight


@dataclass(frozen=True)
class SignedPolarity:
    spins: tuple[int, ...]
    signed_balance: float
    n_sign_conflicts: int
    positive_group: tuple[int, ...]
    negative_group: tuple[int, ...]

    @property
    def polarity_str(self) -> str:
        parts: list[str] = []
        for feature, spin in zip(self.positive_group + self.negative_group, (1,) * len(self.positive_group) + (-1,) * len(self.negative_group)):
            parts.append(f"{int(feature)}:{'+' if spin > 0 else '-'}")
        return ";".join(parts)

    @property
    def positive_group_str(self) -> str:
        return ";".join(str(int(v)) for v in self.positive_group)

    @property
    def negative_group_str(self) -> str:
        return ";".join(str(int(v)) for v in self.negative_group)

    @property
    def schema_hint(self) -> str:
        if not self.negative_group:
            return f"same({self.positive_group_str})"
        return f"same({self.positive_group_str}) | same({self.negative_group_str}) | opposite_between_groups"


@dataclass(frozen=True)
class BuildingBlock:
    generation: int
    block_id: int
    features: tuple[int, ...]
    score: float
    density: float
    internal_abs_mean: float
    internal_positive_mean: float
    internal_negative_abs_mean: float
    external_abs_mean: float
    n_internal_edges: int
    n_positive_edges: int
    n_negative_edges: int
    signed_balance: float = 0.0
    n_sign_conflicts: int = 0
    polarity: str = ""
    positive_group: str = ""
    negative_group: str = ""
    schema_hint: str = ""
    source: str = "ltga_internal_node"
    raw_score: float = 0.0
    stability_count: int = 0
    stability_factor: float = 1.0

    @property
    def size(self) -> int:
        return len(self.features)

    @property
    def features_str(self) -> str:
        return ";".join(str(int(v)) for v in self.features)


@dataclass(frozen=True)
class BuildingBlockSnapshot:
    generation: int
    mode: BBWeightMode
    score_mode: BBScoreMode
    selection_mode: BBSelectionMode
    n_features: int
    n_graph_edges: int
    n_candidates: int
    n_qualified: int
    n_blocks: int
    coverage_budget: int
    selected_feature_coverage: int
    mean_block_size: float
    max_block_size: int
    mean_score: float
    max_score: float
    mean_internal_abs: float
    mean_internal_positive: float
    mean_internal_negative_abs: float
    mean_signed_balance: float
    n_positive_edges_in_blocks: int
    n_negative_edges_in_blocks: int
    n_sign_conflicts_in_blocks: int
    blocks: tuple[BuildingBlock, ...]

    def summary_row(self) -> dict[str, float | int | str]:
        return {
            "generation": int(self.generation),
            "bb_method": "ltga",
            "bb_weight_mode": self.mode,
            "bb_score_mode": self.score_mode,
            "bb_selection_mode": self.selection_mode,
            "n_features": int(self.n_features),
            "n_graph_edges": int(self.n_graph_edges),
            "n_candidates": int(self.n_candidates),
            "n_qualified": int(self.n_qualified),
            "n_blocks": int(self.n_blocks),
            "coverage_budget": int(self.coverage_budget),
            "selected_feature_coverage": int(self.selected_feature_coverage),
            "mean_block_size": float(self.mean_block_size),
            "max_block_size": int(self.max_block_size),
            "mean_score": float(self.mean_score),
            "max_score": float(self.max_score),
            "mean_internal_abs": float(self.mean_internal_abs),
            "mean_internal_positive": float(self.mean_internal_positive),
            "mean_internal_negative_abs": float(self.mean_internal_negative_abs),
            "mean_signed_balance": float(self.mean_signed_balance),
            "n_positive_edges_in_blocks": int(self.n_positive_edges_in_blocks),
            "n_negative_edges_in_blocks": int(self.n_negative_edges_in_blocks),
            "n_sign_conflicts_in_blocks": int(self.n_sign_conflicts_in_blocks),
        }


@dataclass(frozen=True)
class LTGABuildingBlockConfig:
    enabled: bool = False
    weight_mode: BBWeightMode = "absolute"
    score_mode: BBScoreMode = "current"
    selection_mode: BBSelectionMode = "coverage_budget"
    min_block_size: int = 2
    max_block_size: int | None = None
    max_blocks: int = 20
    max_blocks_cap: int = 128
    coverage_ratio: float = 0.50
    coverage_min: int = 10
    coverage_cap: int = 256
    snapshot_interval: int = 1
    external_penalty: float = 0.25
    size_penalty: float = 0.01


class LTGABuildingBlockExtractor:
    """Extract non-overlapping LTGA internal-node building blocks."""

    def __init__(self, config: LTGABuildingBlockConfig) -> None:
        self.config = config
        # Counts how often an identical candidate block was seen in previous
        # linkage updates of this run.  It is intentionally per-extractor/per-run.
        self._block_history: dict[tuple[int, ...], int] = {}

    def extract(self, view: LinkageGraphView, *, generation: int) -> BuildingBlockSnapshot:
        weight = view.strength_matrix(self.config.weight_mode)
        signed = np.asarray(view.signed_weight, dtype=float)
        n_features = int(weight.shape[0])
        n_graph_edges = int(np.count_nonzero(np.triu(weight > 0.0, k=1)))
        max_block_size = self.config.max_block_size or n_features
        max_block_size = max(self.config.min_block_size, min(int(max_block_size), n_features))

        candidates = self._internal_tree_nodes(weight)
        candidates = [c for c in candidates if self.config.min_block_size <= len(c) <= max_block_size]
        scored = [self._score_block(tuple(sorted(set(c))), weight, signed, generation=generation) for c in candidates]
        scored = [b for b in scored if b.score > 0.0]
        scored.sort(key=lambda b: (-b.score, b.size, b.features))
        n_qualified = int(len(scored))

        selected = self._select_blocks(scored, n_features)

        # Update stability history after selection/scoring so the first time a
        # block appears uses factor 1.0 and repeated future appearances are rewarded.
        for block in scored:
            self._block_history[block.features] = self._block_history.get(block.features, 0) + 1

        selected.sort(key=lambda b: (b.size, -b.score, b.features))
        selected = [self._with_block_id(b, i) for i, b in enumerate(selected)]
        return self._snapshot(generation, n_features, n_graph_edges, len(candidates), n_qualified, selected)

    def _coverage_budget(self, n_features: int) -> int:
        budget = int(np.ceil(float(self.config.coverage_ratio) * int(n_features)))
        budget = max(int(self.config.coverage_min), budget)
        budget = min(int(self.config.coverage_cap), budget)
        return max(int(self.config.min_block_size), min(budget, int(n_features)))

    def _select_blocks(self, scored: list[BuildingBlock], n_features: int) -> list[BuildingBlock]:
        selected: list[BuildingBlock] = []
        used: set[int] = set()
        hard_cap = int(self.config.max_blocks if self.config.selection_mode == "count" else self.config.max_blocks_cap)
        hard_cap = max(1, hard_cap)
        coverage_budget = self._coverage_budget(n_features) if self.config.selection_mode == "coverage_budget" else int(n_features)

        for block in scored:
            block_features = {int(f) for f in block.features}
            if block_features & used:
                continue
            if self.config.selection_mode == "coverage_budget" and selected:
                if len(used | block_features) > coverage_budget:
                    continue
            selected.append(self._with_block_id(block, len(selected)))
            used.update(block_features)
            if len(selected) >= hard_cap:
                break

        # Always keep the best non-overlapping block if there are positive-scored
        # candidates.  This avoids disabling BB crossover just because the coverage
        # budget is very small for a tiny dataset.
        if not selected and scored:
            selected.append(self._with_block_id(scored[0], 0))
        return selected

    @staticmethod
    def _with_block_id(block: BuildingBlock, block_id: int) -> BuildingBlock:
        return BuildingBlock(
            generation=block.generation,
            block_id=int(block_id),
            features=block.features,
            score=block.score,
            density=block.density,
            internal_abs_mean=block.internal_abs_mean,
            internal_positive_mean=block.internal_positive_mean,
            internal_negative_abs_mean=block.internal_negative_abs_mean,
            external_abs_mean=block.external_abs_mean,
            n_internal_edges=block.n_internal_edges,
            n_positive_edges=block.n_positive_edges,
            n_negative_edges=block.n_negative_edges,
            signed_balance=block.signed_balance,
            n_sign_conflicts=block.n_sign_conflicts,
            polarity=block.polarity,
            positive_group=block.positive_group,
            negative_group=block.negative_group,
            schema_hint=block.schema_hint,
            source=block.source,
            raw_score=float(block.raw_score),
            stability_count=int(block.stability_count),
            stability_factor=float(block.stability_factor),
        )

    def _internal_tree_nodes(self, weight: np.ndarray) -> list[tuple[int, ...]]:
        """Return LTGA-style candidate masks without recursive tree traversal.

        This implementation is safe for high-dimensional feature-selection data.
        It avoids recursive scipy tree traversal, builds trees only inside active
        connected components of the learned linkage graph, and falls back to
        top-edge neighbourhood masks when a component is too large for dense
        hierarchical clustering.
        """
        n = int(weight.shape[0])
        if n < 2:
            return []
        wmat = np.asarray(weight, dtype=float)
        if wmat.shape[0] != wmat.shape[1]:
            return []

        # Extract active undirected edges.  This creates only a boolean mask plus
        # edge-index vectors and avoids copying the full float matrix.
        active_upper = np.triu(wmat > _EPS, k=1)
        edge_i, edge_j = np.nonzero(active_upper)
        if edge_i.size == 0:
            return []

        components = self._active_components_from_edges(n, edge_i, edge_j)
        if not components:
            return []

        max_hclust_component_size = 1500
        out: list[tuple[int, ...]] = []
        for component in components:
            m = int(component.size)
            if m < 2:
                continue
            if m == 2:
                out.append(tuple(sorted(int(v) for v in component)))
                continue
            if m > max_hclust_component_size:
                out.extend(self._large_component_candidates_from_edges(wmat, component, edge_i, edge_j))
                continue

            local_weight = np.asarray(wmat[np.ix_(component, component)], dtype=float)
            local_weight = np.nan_to_num(local_weight, nan=0.0, posinf=0.0, neginf=0.0)
            local_weight = np.maximum(local_weight, local_weight.T)
            np.fill_diagonal(local_weight, 0.0)
            max_w = float(np.max(local_weight))
            if max_w <= _EPS:
                continue
            similarity = np.clip(local_weight / max_w, 0.0, 1.0)
            dist = 1.0 - similarity
            np.fill_diagonal(dist, 0.0)
            condensed = squareform(dist, checks=False)
            if condensed.size == 0:
                continue
            optimal_ordering = bool(m <= 512)
            Z = linkage(condensed, method="average", optimal_ordering=optimal_ordering)
            root = to_tree(Z, rd=False)
            out.extend(self._iter_tree_items(root, component))

        seen: set[tuple[int, ...]] = set()
        unique: list[tuple[int, ...]] = []
        for item in out:
            if len(item) < 2:
                continue
            key = tuple(sorted(set(int(v) for v in item)))
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            unique.append(key)
        return unique

    @staticmethod
    def _active_components_from_edges(n: int, edge_i: np.ndarray, edge_j: np.ndarray) -> list[np.ndarray]:
        """Connected components from active undirected edge vectors."""
        adjacency: list[list[int]] = [[] for _ in range(int(n))]
        for a_raw, b_raw in zip(edge_i, edge_j):
            a = int(a_raw)
            b = int(b_raw)
            adjacency[a].append(b)
            adjacency[b].append(a)
        visited = np.zeros(int(n), dtype=bool)
        components: list[np.ndarray] = []
        starts = [i for i, nbrs in enumerate(adjacency) if nbrs]
        for start in starts:
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            comp: list[int] = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adjacency[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            if len(comp) >= 2:
                components.append(np.asarray(sorted(comp), dtype=int))
        components.sort(key=lambda c: (int(c.size), int(c[0])))
        return components

    @staticmethod
    def _iter_tree_items(root: object, component: np.ndarray) -> list[tuple[int, ...]]:
        """Iteratively collect internal-node feature sets from a scipy tree."""
        out: list[tuple[int, ...]] = []
        leaves_by_node_id: dict[int, tuple[int, ...]] = {}
        stack: list[tuple[object, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            node_id = int(node.id)
            if node.is_leaf():
                leaves_by_node_id[node_id] = (int(component[node_id]),)
                continue
            if not expanded:
                stack.append((node, True))
                stack.append((node.right, False))
                stack.append((node.left, False))
                continue
            left = leaves_by_node_id.pop(int(node.left.id))
            right = leaves_by_node_id.pop(int(node.right.id))
            items = tuple(sorted(left + right))
            leaves_by_node_id[node_id] = items
            if len(items) >= 2:
                out.append(items)
        return out

    def _large_component_candidates_from_edges(
        self,
        weight: np.ndarray,
        component: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
    ) -> list[tuple[int, ...]]:
        """Safe deterministic fallback for unexpectedly dense large components.

        The method works from active edge vectors, not from a full component
        distance matrix.  It ranks only a bounded number of strongest edges and
        builds compact neighbourhood candidates around them.
        """
        comp = np.asarray(component, dtype=int)
        comp_mask = np.zeros(int(weight.shape[0]), dtype=bool)
        comp_mask[comp] = True
        in_comp = comp_mask[edge_i] & comp_mask[edge_j]
        if not np.any(in_comp):
            return []
        rows = edge_i[in_comp].astype(int, copy=False)
        cols = edge_j[in_comp].astype(int, copy=False)
        vals = np.asarray(weight[rows, cols], dtype=float)
        valid = np.isfinite(vals) & (vals > _EPS)
        if not np.any(valid):
            return []
        rows = rows[valid]
        cols = cols[valid]
        vals = vals[valid]

        edge_budget = max(100, 10 * int(self.config.max_blocks))
        edge_budget = min(edge_budget, int(vals.size))
        if vals.size > edge_budget:
            top_idx = np.argpartition(vals, -edge_budget)[-edge_budget:]
            top_idx = top_idx[np.argsort(vals[top_idx])[::-1]]
        else:
            top_idx = np.argsort(vals)[::-1]

        max_size = self.config.max_block_size or min(64, int(comp.size))
        max_size = max(self.config.min_block_size, min(int(max_size), int(comp.size)))

        # Build adjacency weights for only the strongest candidate edges.  This
        # keeps the fallback linear in the retained edge budget.
        nbrs: dict[int, list[tuple[float, int]]] = {}
        for idx_raw in top_idx:
            idx = int(idx_raw)
            u = int(rows[idx])
            v = int(cols[idx])
            val = float(vals[idx])
            nbrs.setdefault(u, []).append((val, v))
            nbrs.setdefault(v, []).append((val, u))
        for values in nbrs.values():
            values.sort(key=lambda item: (-item[0], item[1]))

        out: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for idx_raw in top_idx:
            idx = int(idx_raw)
            u = int(rows[idx])
            v = int(cols[idx])
            pair = tuple(sorted((u, v)))
            if pair not in seen:
                seen.add(pair)
                out.append(pair)

            block = [u, v]
            # Alternate strong neighbours around the two endpoints.
            for _, nb in (nbrs.get(u, []) + nbrs.get(v, [])):
                if nb not in block:
                    block.append(int(nb))
                if len(block) >= max_size:
                    break
            if len(block) >= self.config.min_block_size:
                key = tuple(sorted(block))
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

    def _score_block(self, features: tuple[int, ...], weight: np.ndarray, signed: np.ndarray, *, generation: int) -> BuildingBlock:
        idx = np.asarray(features, dtype=int)
        if idx.size < 2:
            return BuildingBlock(generation, 0, features, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)

        internal = weight[np.ix_(idx, idx)]
        signed_internal = signed[np.ix_(idx, idx)]
        tri = np.triu_indices(idx.size, k=1)
        vals = internal[tri]
        signed_vals = signed_internal[tri]
        active = vals > _EPS
        n_possible = int(idx.size * (idx.size - 1) // 2)
        n_active = int(np.count_nonzero(active))
        density = float(n_active / max(1, n_possible))
        internal_abs_mean = float(np.mean(vals[active])) if n_active else 0.0
        pos_vals = signed_vals[(signed_vals > _EPS) & active]
        neg_vals = signed_vals[(signed_vals < -_EPS) & active]
        internal_pos = float(np.mean(pos_vals)) if pos_vals.size else 0.0
        internal_neg_abs = float(np.mean(np.abs(neg_vals))) if neg_vals.size else 0.0

        outside = np.ones(weight.shape[0], dtype=bool)
        outside[idx] = False
        external_vals = weight[np.ix_(idx, np.flatnonzero(outside))].reshape(-1)
        external_active = external_vals[external_vals > _EPS]
        external_abs_mean = float(np.mean(external_active)) if external_active.size else 0.0

        polarity = self._infer_signed_polarity(features, weight, signed)
        size_term = float((idx.size - 1) / max(1, weight.shape[0] - 1))
        base_score = (
            internal_abs_mean * (0.5 + 0.5 * np.sqrt(max(0.0, density)))
            - self.config.external_penalty * external_abs_mean
            - self.config.size_penalty * size_term
        )
        if self.config.weight_mode == "signed":
            # A signed block should not only be strongly connected; its signs
            # should also admit a coherent same/opposite polarity assignment.
            base_score *= max(0.0, float(polarity.signed_balance))

        raw_score = float(max(0.0, base_score))
        history_count = int(self._block_history.get(features, 0))
        stability_factor = float(np.sqrt(1.0 + history_count)) if self.config.score_mode == "current_stability" else 1.0
        final_score = raw_score * stability_factor

        return BuildingBlock(
            generation=int(generation),
            block_id=0,
            features=features,
            score=float(max(0.0, final_score)),
            density=density,
            internal_abs_mean=internal_abs_mean,
            internal_positive_mean=internal_pos,
            internal_negative_abs_mean=internal_neg_abs,
            external_abs_mean=external_abs_mean,
            n_internal_edges=n_active,
            n_positive_edges=int(pos_vals.size),
            n_negative_edges=int(neg_vals.size),
            signed_balance=float(polarity.signed_balance),
            n_sign_conflicts=int(polarity.n_sign_conflicts),
            polarity=polarity.polarity_str if self.config.weight_mode == "signed" else "",
            positive_group=polarity.positive_group_str if self.config.weight_mode == "signed" else "",
            negative_group=polarity.negative_group_str if self.config.weight_mode == "signed" else "",
            schema_hint=polarity.schema_hint if self.config.weight_mode == "signed" else "",
            raw_score=float(raw_score),
            stability_count=int(history_count),
            stability_factor=float(stability_factor),
        )

    def _infer_signed_polarity(self, features: tuple[int, ...], weight: np.ndarray, signed: np.ndarray) -> SignedPolarity:
        """Infer same/opposite phases for a block's signed edges.

        Positive edge: spin_i == spin_j.  Negative edge: spin_i != spin_j.
        This is the weighted signed-graph balance problem.  We use a deterministic
        greedy initialization followed by local flip improvements, which is fast
        and stable for diagnostic logging.
        """
        features = tuple(int(f) for f in features)
        n = len(features)
        if n == 0:
            return SignedPolarity((), 0.0, 0, (), ())
        local = {f: i for i, f in enumerate(features)}
        edges: list[tuple[int, int, float, int]] = []
        total_weight = 0.0
        for p, i in enumerate(features):
            for q in range(p + 1, n):
                j = features[q]
                w = float(weight[i, j])
                s = float(signed[i, j])
                if w <= _EPS or abs(s) <= _EPS:
                    continue
                desired = 1 if s > 0 else -1
                edges.append((p, q, w, desired))
                total_weight += w
        if not edges:
            pos = tuple(features)
            return SignedPolarity(tuple([1] * n), 1.0, 0, pos, ())

        adjacency: list[list[tuple[int, float, int]]] = [[] for _ in range(n)]
        for u, v, w, desired in edges:
            adjacency[u].append((v, w, desired))
            adjacency[v].append((u, w, desired))
        for nbrs in adjacency:
            nbrs.sort(key=lambda x: (-x[1], x[0]))

        spins = np.zeros(n, dtype=int)
        starts = sorted(range(n), key=lambda u: (-sum(w for _, w, _ in adjacency[u]), features[u]))
        for start in starts:
            if spins[start] != 0:
                continue
            spins[start] = 1
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v, _, desired in adjacency[u]:
                    implied = int(spins[u] * desired)
                    if spins[v] == 0:
                        spins[v] = implied
                        queue.append(v)
        spins[spins == 0] = 1

        def objective(current: np.ndarray) -> tuple[float, int]:
            consistent_weight = 0.0
            conflicts = 0
            for u, v, w, desired in edges:
                ok = int(current[u] * current[v]) == desired
                if ok:
                    consistent_weight += w
                else:
                    conflicts += 1
            return consistent_weight, conflicts

        best_weight, _ = objective(spins)
        improved = True
        while improved:
            improved = False
            for u in range(n):
                candidate = spins.copy()
                candidate[u] *= -1
                cand_weight, _ = objective(candidate)
                if cand_weight > best_weight + _EPS:
                    spins = candidate
                    best_weight = cand_weight
                    improved = True

        if spins[0] < 0:
            spins *= -1
        consistent_weight, conflicts = objective(spins)
        balance = float(consistent_weight / total_weight) if total_weight > _EPS else 1.0
        pos = tuple(features[i] for i, s in enumerate(spins) if s > 0)
        neg = tuple(features[i] for i, s in enumerate(spins) if s < 0)
        return SignedPolarity(tuple(int(s) for s in spins), balance, int(conflicts), pos, neg)

    def _snapshot(
        self,
        generation: int,
        n_features: int,
        n_graph_edges: int,
        n_candidates: int,
        n_qualified: int,
        selected: list[BuildingBlock],
    ) -> BuildingBlockSnapshot:
        if selected:
            sizes = np.asarray([b.size for b in selected], dtype=float)
            scores = np.asarray([b.score for b in selected], dtype=float)
            abs_means = np.asarray([b.internal_abs_mean for b in selected], dtype=float)
            pos_means = np.asarray([b.internal_positive_mean for b in selected], dtype=float)
            neg_means = np.asarray([b.internal_negative_abs_mean for b in selected], dtype=float)
            balances = np.asarray([b.signed_balance for b in selected], dtype=float)
            max_size = int(np.max(sizes))
            mean_size = float(np.mean(sizes))
            mean_score = float(np.mean(scores))
            max_score = float(np.max(scores))
            mean_abs = float(np.mean(abs_means))
            mean_pos = float(np.mean(pos_means))
            mean_neg = float(np.mean(neg_means))
            mean_balance = float(np.mean(balances))
            n_pos = int(sum(b.n_positive_edges for b in selected))
            n_neg = int(sum(b.n_negative_edges for b in selected))
            n_conflicts = int(sum(b.n_sign_conflicts for b in selected))
        else:
            max_size = 0
            mean_size = mean_score = max_score = mean_abs = mean_pos = mean_neg = mean_balance = 0.0
            n_pos = n_neg = n_conflicts = 0
        return BuildingBlockSnapshot(
            generation=int(generation),
            mode=self.config.weight_mode,
            score_mode=self.config.score_mode,
            selection_mode=self.config.selection_mode,
            n_features=int(n_features),
            n_graph_edges=int(n_graph_edges),
            n_candidates=int(n_candidates),
            n_qualified=int(n_qualified),
            n_blocks=int(len(selected)),
            coverage_budget=int(self._coverage_budget(n_features) if self.config.selection_mode == "coverage_budget" else n_features),
            selected_feature_coverage=int(len({int(f) for b in selected for f in b.features})),
            mean_block_size=mean_size,
            max_block_size=max_size,
            mean_score=mean_score,
            max_score=max_score,
            mean_internal_abs=mean_abs,
            mean_internal_positive=mean_pos,
            mean_internal_negative_abs=mean_neg,
            mean_signed_balance=mean_balance,
            n_positive_edges_in_blocks=n_pos,
            n_negative_edges_in_blocks=n_neg,
            n_sign_conflicts_in_blocks=n_conflicts,
            blocks=tuple(selected),
        )
