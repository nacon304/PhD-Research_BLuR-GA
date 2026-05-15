from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
import networkx as nx
import numpy as np
import pandas as pd

from .filters import apply_view_mode, edges_for_generation, filter_edges


def generations_for_animation(
    snapshots: pd.DataFrame,
    trace: pd.DataFrame | None = None,
    *,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> list[int]:
    if trace is not None and not trace.empty and "generation" in trace.columns:
        generations = sorted(int(g) for g in trace["generation"].unique())
    else:
        generations = sorted(int(g) for g in snapshots["generation"].unique())
    if start is not None:
        generations = [g for g in generations if g >= int(start)]
    if end is not None:
        generations = [g for g in generations if g <= int(end)]
    if not generations:
        raise ValueError("No generations available for animation after applying start/end filters.")
    step = max(1, int(step))
    return generations[::step]


def build_union_graph(snapshots: pd.DataFrame, n_nodes: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    if not snapshots.empty:
        final_gen = int(snapshots["generation"].max())
        final_edges = snapshots[snapshots["generation"] == final_gen]
        for r in final_edges.itertuples(index=False):
            g.add_edge(int(r.feature_i), int(r.feature_j), weight=float(r.weight))
    return g


def compute_layout(snapshots: pd.DataFrame, n_nodes: int, *, layout: str = "spring", seed: int = 1) -> dict[int, np.ndarray]:
    g = build_union_graph(snapshots, n_nodes)
    layout = layout.lower()
    if layout == "spring":
        if g.number_of_edges() == 0:
            return nx.circular_layout(g)
        return nx.spring_layout(g, seed=seed, weight="weight", k=None, iterations=150)
    if layout == "circular":
        return nx.circular_layout(g)
    if layout == "spectral":
        if g.number_of_edges() == 0:
            return nx.circular_layout(g)
        return nx.spectral_layout(g, weight="weight")
    raise ValueError(f"Unknown layout={layout!r}")


def _trace_value(trace: pd.DataFrame | None, generation: int, column: str) -> float | None:
    if trace is None or trace.empty or column not in trace.columns:
        return None
    rows = trace[trace["generation"] == generation]
    if rows.empty:
        rows = trace[trace["generation"] <= generation]
        if rows.empty:
            return None
        return float(rows.iloc[-1][column])
    return float(rows.iloc[-1][column])


class GraphEvolutionRenderer:
    def __init__(
        self,
        snapshots: pd.DataFrame,
        *,
        n_nodes: int,
        labels: dict[int, str] | None = None,
        trace: pd.DataFrame | None = None,
        layout_name: str = "spring",
        layout_seed: int = 1,
        global_weight_scale: bool = True,
    ) -> None:
        self.snapshots = snapshots.copy()
        self.trace = trace
        self.n_nodes = int(n_nodes)
        self.labels = labels or {i: str(i) for i in range(self.n_nodes)}
        self.pos = compute_layout(self.snapshots, self.n_nodes, layout=layout_name, seed=layout_seed)
        self.global_max_weight = float(self.snapshots["weight"].max()) if not self.snapshots.empty else 1.0
        if self.global_max_weight <= 0:
            self.global_max_weight = 1.0
        self.global_weight_scale = bool(global_weight_scale)

    def frame_edges(
        self,
        generation: int,
        *,
        filter_mode: str = "topk",
        top_k: int = 40,
        threshold: float = 0.0,
        percentile: float = 95.0,
        view_mode: str = "cumulative",
        window: int = 20,
    ) -> pd.DataFrame:
        edges = edges_for_generation(self.snapshots, generation)
        edges = apply_view_mode(edges, generation, view_mode=view_mode, window=window)
        return filter_edges(edges, mode=filter_mode, top_k=top_k, threshold=threshold, percentile=percentile)

    def draw_frame(
        self,
        ax: plt.Axes,
        generation: int,
        *,
        filter_mode: str = "topk",
        top_k: int = 40,
        threshold: float = 0.0,
        percentile: float = 95.0,
        view_mode: str = "cumulative",
        window: int = 20,
        show_labels: str = "auto",
        title: str = "BLuR-GA linkage graph evolution",
    ) -> None:
        ax.clear()
        ax.set_axis_off()
        edges = self.frame_edges(
            generation,
            filter_mode=filter_mode,
            top_k=top_k,
            threshold=threshold,
            percentile=percentile,
            view_mode=view_mode,
            window=window,
        )
        raw_edges = edges_for_generation(self.snapshots, generation)
        max_w = self.global_max_weight if self.global_weight_scale else (float(edges["weight"].max()) if not edges.empty else 1.0)
        max_w = max(max_w, 1e-12)

        # Draw weighted edges manually so width and alpha can vary per edge.
        for row in edges.itertuples(index=False):
            i = int(row.feature_i)
            j = int(row.feature_j)
            w = float(row.weight)
            norm_w = max(0.0, min(1.0, w / max_w))
            width = 0.4 + 5.0 * norm_w
            alpha = 0.12 + 0.82 * norm_w
            x1, y1 = self.pos[i]
            x2, y2 = self.pos[j]
            ax.plot([x1, x2], [y1, y2], linewidth=width, alpha=alpha, color="black", zorder=1)

        # Node size by weighted degree among shown edges.
        weighted_degree = {i: 0.0 for i in range(self.n_nodes)}
        for row in edges.itertuples(index=False):
            weighted_degree[int(row.feature_i)] += float(row.weight)
            weighted_degree[int(row.feature_j)] += float(row.weight)
        degree_values = np.array([weighted_degree[i] for i in range(self.n_nodes)], dtype=float)
        if degree_values.max() > 0:
            node_sizes = 140 + 520 * degree_values / degree_values.max()
        else:
            node_sizes = np.full(self.n_nodes, 180.0)

        xy = np.array([self.pos[i] for i in range(self.n_nodes)])
        ax.scatter(xy[:, 0], xy[:, 1], s=node_sizes, color="white", edgecolors="black", linewidths=1.2, zorder=2)

        should_label = show_labels == "on" or (show_labels == "auto" and self.n_nodes <= 40)
        if should_label:
            for i in range(self.n_nodes):
                x, y = self.pos[i]
                ax.text(x, y, self.labels.get(i, str(i)), fontsize=8, ha="center", va="center", zorder=3)

        best = _trace_value(self.trace, generation, "best_so_far_fitness")
        n_total = int(len(raw_edges))
        n_shown = int(len(edges))
        if edges.empty:
            max_edge = 0.0
            mean_edge = 0.0
        else:
            max_edge = float(edges["weight"].max())
            mean_edge = float(edges["weight"].mean())
        subtitle = (
            f"generation={generation} | shown_edges={n_shown} | total_edges={n_total} | "
            f"mean_w={mean_edge:.4f} | max_w={max_edge:.4f}"
        )
        if best is not None:
            subtitle += f" | best={best:.6f}"
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
        ax.margins(0.12)

    def save_animation(
        self,
        output: str | Path,
        generations: Iterable[int],
        *,
        fps: int = 6,
        filter_mode: str = "topk",
        top_k: int = 40,
        threshold: float = 0.0,
        percentile: float = 95.0,
        view_mode: str = "cumulative",
        window: int = 20,
        show_labels: str = "auto",
        title: str = "BLuR-GA linkage graph evolution",
        dpi: int = 140,
        width: float = 9.0,
        height: float = 7.0,
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        generations = list(int(g) for g in generations)
        if not generations:
            raise ValueError("Cannot save animation with no generations.")

        fig, ax = plt.subplots(figsize=(width, height))
        suffix = output.suffix.lower()
        if suffix == ".gif":
            writer = PillowWriter(fps=int(fps))
        elif suffix in {".mp4", ".m4v", ".mov"}:
            writer = FFMpegWriter(fps=int(fps), metadata={"artist": "BLuR-GA graph evolution renderer"})
        else:
            raise ValueError("Output extension must be .mp4, .m4v, .mov, or .gif")

        with writer.saving(fig, str(output), dpi=dpi):
            for gen in generations:
                self.draw_frame(
                    ax,
                    gen,
                    filter_mode=filter_mode,
                    top_k=top_k,
                    threshold=threshold,
                    percentile=percentile,
                    view_mode=view_mode,
                    window=window,
                    show_labels=show_labels,
                    title=title,
                )
                writer.grab_frame()
        plt.close(fig)
        return output

    def save_frame(
        self,
        output: str | Path,
        generation: int,
        **kwargs: object,
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        self.draw_frame(ax, generation, **kwargs)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return output
