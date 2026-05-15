#!/usr/bin/env python3
from __future__ import annotations

"""Render ground-truth linkage graphs from prepared BLuR-GA benchmark folders.

This version keeps the original CLI behaviour for normal datasets, but adds an
``--auto-special-style`` layer for a few benchmark families whose ground-truth
graphs otherwise look poor with a single global layout:

* large ``pairwise_qubo`` graphs are drawn with a Kamada-Kawai profile;
* ``gametes_style_epistasis`` graphs are drawn in a linked-node focus view;
* large ``planted_xor_parity`` graphs are drawn in a linked-node focus view.

The focus view hides isolated nodes visually, but the title and metrics still
report the original total number of features and hidden isolates.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


SUPPORTED_LAYOUTS = ["auto", "spring", "kamada", "spectral", "circular", "grid"]
SPECIAL_STYLE_CHOICES = ["auto", "off"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ground-truth linkage graphs for BLuR-GA benchmark datasets.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset-root", help="Root containing benchmark folders and manifest.csv, e.g. Dataset/prepared_blurga")
    group.add_argument("--dataset-dir", help="Single prepared dataset folder containing ground_truth_edges.csv")

    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--eval-group", default="linkage")
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None, help="Optional dataset_key filter")
    parser.add_argument("--output-dir", default=None, help="Where to save plots. Defaults to <dataset-root>/ground_truth_plots")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png")

    parser.add_argument("--layout", choices=SUPPORTED_LAYOUTS, default="auto")
    parser.add_argument("--layout-seed", type=int, default=7)
    parser.add_argument("--spring-k", type=float, default=None, help="Larger values spread spring-layout nodes further apart.")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--scale", type=float, default=1.0, help="Multiply coordinates after layout; useful when labels overlap.")

    parser.add_argument("--fig-width", type=float, default=10.0)
    parser.add_argument("--fig-height", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--node-size", type=float, default=170.0)
    parser.add_argument("--label-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--max-label-nodes", type=int, default=60)
    parser.add_argument("--font-size", type=float, default=7.0)

    parser.add_argument("--edge-filter", choices=["all", "topk", "threshold", "percentile"], default="all")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--min-abs-weight", type=float, default=0.0)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--signed", action="store_true", help="Use separate positive/negative edge colours.")
    parser.add_argument("--write-metrics", action="store_true")

    # New: keep normal behaviour for most datasets, but automatically override
    # only the known problematic cases described in the benchmark plotting notes.
    parser.add_argument(
        "--auto-special-style",
        choices=SPECIAL_STYLE_CHOICES,
        default="auto",
        help=(
            "auto = apply family-specific plot profiles for large pairwise_qubo, "
            "gametes_style_epistasis, and large planted_xor_parity. "
            "off = use exactly the user-provided layout options for every dataset."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(dataset_dir: Path, n_features: int) -> dict[int, str]:
    node_labels = dataset_dir / "node_labels.csv"
    if node_labels.exists():
        df = pd.read_csv(node_labels)
        if {"node_id", "label"}.issubset(df.columns):
            return {int(r.node_id): str(r.label) for r in df.itertuples(index=False)}
    feature_json = dataset_dir / "feature_names.json"
    if feature_json.exists():
        try:
            records = json.loads(feature_json.read_text(encoding="utf-8"))
            if isinstance(records, list):
                return {int(r["feature_index"]): str(r.get("feature_name", r["feature_index"])) for r in records}
        except Exception:
            pass
    return {i: f"f{i}" for i in range(n_features)}


def load_feature_positions(dataset_dir: Path) -> dict[int, tuple[float, float]] | None:
    feature_json = dataset_dir / "feature_names.json"
    if not feature_json.exists():
        return None
    records = json.loads(feature_json.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        return None
    if not {"grid_row", "grid_col"}.issubset(records[0].keys()):
        return None
    return {int(r["feature_index"]): (float(r["grid_col"]), -float(r["grid_row"])) for r in records}


def filter_edges(edges: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = edges.copy()
    if df.empty:
        return df
    df["abs_weight"] = df["abs_weight"].astype(float)
    if args.edge_filter == "threshold":
        df = df[df["abs_weight"] >= float(args.min_abs_weight)]
    elif args.edge_filter == "topk":
        df = df.sort_values("abs_weight", ascending=False).head(int(args.top_k))
    elif args.edge_filter == "percentile":
        cutoff = float(np.percentile(df["abs_weight"].to_numpy(), float(args.percentile)))
        df = df[df["abs_weight"] >= cutoff]
    else:
        if args.min_abs_weight > 0:
            df = df[df["abs_weight"] >= float(args.min_abs_weight)]
    return df.reset_index(drop=True)


def build_graph(n_features: int, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(n_features)))
    for row in edges.itertuples(index=False):
        graph.add_edge(
            int(row.feature_i),
            int(row.feature_j),
            weight=float(row.abs_weight),
            true_weight=float(row.true_weight),
            true_sign=int(row.true_sign),
        )
    return graph


def infer_special_profile(
    metadata: dict[str, Any],
    n_features: int,
    n_edges: int,
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, str, bool]:
    """Return per-dataset plotting args, profile name, and focus flag.

    The original script applies one set of CLI options to all selected datasets.
    Here we copy the args and override only the cases that are visually known to
    be problematic.  Other datasets still use the user's command as-is.
    """
    local = copy.copy(args)
    if args.auto_special_style == "off":
        return local, "user", False

    family = str(metadata.get("problem_family", ""))
    dataset_key = str(metadata.get("dataset_key", ""))

    # Pairwise QUBO with 100+ variables: spring/auto tends to compress the main
    # component and push isolates far away. Kamada is clearer for this density.
    if family == "pairwise_qubo" and n_features >= 100:
        local.layout = "kamada"
        local.scale = 3.0
        local.fig_width = 12.0
        local.fig_height = 9.0
        local.node_size = 35.0
        local.label_mode = "off"
        local.font_size = 6.0
        return local, "special_pairwise_large_kamada", False

    # GAMETES-style epistasis ground truth is intentionally tiny relative to the
    # full feature space: loci2 -> 1 true edge, loci3 -> 3 true edges. Drawing all
    # isolated features hides the actual epistatic structure, so focus on linked
    # nodes and report hidden isolates in the title/metrics.
    if family == "gametes_style_epistasis":
        local.layout = "circular"  # used only as fallback; focus layout handles this case.
        local.scale = 1.8
        local.fig_width = 8.0 if n_edges <= 3 else 9.0
        local.fig_height = 6.0
        local.node_size = 900.0 if n_edges <= 3 else 650.0
        local.label_mode = "on"
        local.font_size = 10.0
        return local, "special_gametes_focus_linked", True

    # Large planted XOR/parity has a few planted interaction groups plus many
    # distractor features. For the d100/g4-4 setting the graph has 12 true edges,
    # so a focus view shows the two planted blocks much better.
    if family == "planted_xor_parity" and n_features >= 100:
        local.layout = "circular"  # used only as fallback; focus layout handles this case.
        local.scale = 1.8
        local.fig_width = 10.0
        local.fig_height = 6.5
        local.node_size = 650.0
        local.label_mode = "on"
        local.font_size = 9.0
        return local, "special_planted_xor_focus_linked", True

    return local, "user", False


def compute_layout(graph: nx.Graph, dataset_dir: Path, args: argparse.Namespace) -> dict[int, np.ndarray]:
    layout = args.layout
    if layout == "auto":
        grid_pos = load_feature_positions(dataset_dir)
        if grid_pos is not None:
            layout = "grid"
        elif graph.number_of_nodes() <= 80:
            layout = "kamada"
        else:
            layout = "spring"
    if layout == "grid":
        grid_pos = load_feature_positions(dataset_dir)
        if grid_pos is not None:
            return {k: np.asarray(v, dtype=float) * float(args.scale) for k, v in grid_pos.items()}
        layout = "spring"
    if layout == "kamada":
        if graph.number_of_edges() == 0:
            pos = nx.circular_layout(graph)
        else:
            pos = nx.kamada_kawai_layout(graph, weight="weight", scale=float(args.scale))
        return {int(k): np.asarray(v, dtype=float) for k, v in pos.items()}
    if layout == "spring":
        k = args.spring_k
        if k is None and graph.number_of_nodes() > 0:
            k = 2.2 / math.sqrt(graph.number_of_nodes())
        pos = nx.spring_layout(
            graph,
            seed=int(args.layout_seed),
            weight="weight",
            k=k,
            iterations=int(args.iterations),
            scale=float(args.scale),
        )
        return {int(k_): np.asarray(v, dtype=float) for k_, v in pos.items()}
    if layout == "spectral":
        if graph.number_of_edges() == 0:
            pos = nx.circular_layout(graph, scale=float(args.scale))
        else:
            pos = nx.spectral_layout(graph, weight="weight", scale=float(args.scale))
        return {int(k): np.asarray(v, dtype=float) for k, v in pos.items()}
    if layout == "circular":
        pos = nx.circular_layout(graph, scale=float(args.scale))
        return {int(k): np.asarray(v, dtype=float) for k, v in pos.items()}
    raise ValueError(f"Unsupported layout: {args.layout}")


def component_node_positions(nodes: list[int], radius: float) -> dict[int, np.ndarray]:
    """Small deterministic layout for one connected planted component."""
    nodes = sorted(nodes)
    m = len(nodes)
    if m == 1:
        return {nodes[0]: np.asarray([0.0, 0.0], dtype=float)}
    if m == 2:
        return {
            nodes[0]: np.asarray([-0.55 * radius, 0.0], dtype=float),
            nodes[1]: np.asarray([0.55 * radius, 0.0], dtype=float),
        }
    # Place 3+ nodes on a circle; rotate so triangles/squares look balanced.
    start_angle = math.pi / 2.0
    out: dict[int, np.ndarray] = {}
    for idx, node in enumerate(nodes):
        theta = start_angle + 2.0 * math.pi * idx / m
        out[node] = np.asarray([radius * math.cos(theta), radius * math.sin(theta)], dtype=float)
    return out


def compute_focus_linked_layout(graph: nx.Graph, scale: float = 1.8) -> tuple[dict[int, np.ndarray], list[int], int]:
    """Layout only nodes incident to at least one shown edge.

    This is intended for very sparse ground-truth graphs where most features are
    distractors/isolates.  It preserves the real edge structure while preventing
    dozens of isolated nodes from dominating the plot.
    """
    linked_nodes = sorted([node for node, deg in graph.degree() if deg > 0])
    if not linked_nodes:
        pos = nx.circular_layout(graph, scale=scale)
        return {int(k): np.asarray(v, dtype=float) for k, v in pos.items()}, list(graph.nodes()), 0

    sub = graph.subgraph(linked_nodes).copy()
    components = [sorted(c) for c in nx.connected_components(sub)]
    components.sort(key=lambda c: (len(c), c[0]), reverse=True)

    n_comp = len(components)
    # A simple row/grid of components works better than spring for planted XOR
    # blocks and GAMETES loci because the components are tiny cliques/edges.
    if n_comp <= 2:
        centers = []
        spacing = 2.4 * scale
        for idx in range(n_comp):
            x = (idx - (n_comp - 1) / 2.0) * spacing
            centers.append(np.asarray([x, 0.0], dtype=float))
    else:
        cols = int(math.ceil(math.sqrt(n_comp)))
        rows = int(math.ceil(n_comp / cols))
        spacing_x = 2.4 * scale
        spacing_y = 1.9 * scale
        centers = []
        for idx in range(n_comp):
            r = idx // cols
            c = idx % cols
            x = (c - (cols - 1) / 2.0) * spacing_x
            y = -((r - (rows - 1) / 2.0) * spacing_y)
            centers.append(np.asarray([x, y], dtype=float))

    pos: dict[int, np.ndarray] = {}
    base_radius = 0.55 * scale
    for comp, center in zip(components, centers):
        local = component_node_positions(comp, radius=base_radius)
        for node, xy in local.items():
            pos[node] = xy + center

    hidden_isolates = graph.number_of_nodes() - len(linked_nodes)
    return pos, linked_nodes, hidden_isolates


def edge_colour(true_weight: float, signed: bool) -> str:
    if not signed:
        return "black"
    return "tab:blue" if true_weight >= 0 else "tab:red"


def draw_dataset(dataset_dir: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(dataset_dir / "metadata.json")
    dataset_key = str(metadata.get("dataset_key", dataset_dir.name))
    n_features = int(metadata.get("n_features", 0))
    edges_path = dataset_dir / "ground_truth_edges.csv"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing {edges_path}")

    all_edges = pd.read_csv(edges_path)
    local_args, profile, focus_linked = infer_special_profile(metadata, n_features, len(all_edges), args)
    shown_edges = filter_edges(all_edges, local_args)
    graph = build_graph(n_features, shown_edges)
    labels = load_labels(dataset_dir, n_features)

    if focus_linked:
        pos, drawn_nodes, hidden_isolates = compute_focus_linked_layout(graph, scale=float(local_args.scale))
    else:
        pos = compute_layout(graph, dataset_dir, local_args)
        drawn_nodes = list(range(n_features))
        hidden_isolates = 0

    fig, ax = plt.subplots(figsize=(float(local_args.fig_width), float(local_args.fig_height)))
    ax.set_axis_off()

    max_w = float(shown_edges["abs_weight"].max()) if not shown_edges.empty else 1.0
    max_w = max(max_w, 1e-12)
    for row in shown_edges.itertuples(index=False):
        i = int(row.feature_i)
        j = int(row.feature_j)
        if i not in pos or j not in pos:
            continue
        norm_w = max(0.0, min(1.0, float(row.abs_weight) / max_w))
        width = 0.35 + 4.5 * norm_w
        alpha = 0.18 + 0.75 * norm_w
        color = edge_colour(float(row.true_weight), bool(local_args.signed))
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        ax.plot([x1, x2], [y1, y2], linewidth=width, alpha=alpha, color=color, zorder=1)

    degree = dict(graph.degree(weight="weight"))
    drawn_nodes = sorted(drawn_nodes)
    degree_values = np.asarray([degree.get(i, 0.0) for i in drawn_nodes], dtype=float)
    if degree_values.size > 0 and degree_values.max() > 0:
        node_sizes = float(local_args.node_size) * (0.75 + 0.85 * degree_values / degree_values.max())
    else:
        node_sizes = np.full(len(drawn_nodes), float(local_args.node_size))

    if drawn_nodes:
        xy = np.asarray([pos[i] for i in drawn_nodes], dtype=float)
        ax.scatter(xy[:, 0], xy[:, 1], s=node_sizes, color="white", edgecolors="black", linewidths=1.15, zorder=2)

    show_labels = local_args.label_mode == "on" or (
        local_args.label_mode == "auto" and len(drawn_nodes) <= int(local_args.max_label_nodes)
    )
    if show_labels:
        for i in drawn_nodes:
            x, y = pos[i]
            ax.text(x, y, labels.get(i, str(i)), fontsize=float(local_args.font_size), ha="center", va="center", zorder=3)

    family = metadata.get("problem_family", "unknown")
    focus_note = ""
    if hidden_isolates > 0:
        focus_note = f" | drawn_nodes={len(drawn_nodes)}/{n_features} | hidden_isolates={hidden_isolates}"

    ax.set_title(
        f"Ground-truth linkage graph: {dataset_key}\n"
        f"family={family} | nodes={n_features}{focus_note} | "
        f"shown_edges={len(shown_edges)} / true_edges={len(all_edges)} | profile={profile}",
        fontsize=12,
    )
    ax.margins(0.20 if focus_linked else 0.15)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"ground_truth_{dataset_key}.{local_args.format}"
    fig.savefig(out_path, dpi=int(local_args.dpi), bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "dataset_key": dataset_key,
        "problem_family": family,
        "profile": profile,
        "n_features": n_features,
        "n_true_edges": int(len(all_edges)),
        "n_shown_edges": int(len(shown_edges)),
        "n_drawn_nodes": int(len(drawn_nodes)),
        "n_hidden_isolates": int(hidden_isolates),
        "density": float(len(all_edges) / max(1, n_features * (n_features - 1) / 2)),
        "plot_path": str(out_path),
    }
    if profile != "user":
        print(f"[special-style] {dataset_key}: {profile}")
    print(f"saved {out_path}")
    return metrics


def resolve_dataset_dirs(args: argparse.Namespace) -> tuple[list[Path], Path]:
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "plots"
        return [dataset_dir], output_dir

    root = Path(args.dataset_root)
    manifest_path = root / args.manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    if "eval_group" in manifest.columns:
        manifest = manifest[manifest["eval_group"] == args.eval_group]
    if args.families is not None:
        manifest = manifest[manifest["problem_family"].isin(args.families)]
    if args.datasets is not None:
        manifest = manifest[manifest["dataset_key"].isin(args.datasets)]
    if manifest.empty:
        raise ValueError("No datasets selected from manifest.")
    dirs = [root / str(row.prepared_folder) for row in manifest.itertuples(index=False)]
    output_dir = Path(args.output_dir) if args.output_dir else root / "ground_truth_plots"
    return dirs, output_dir


def main() -> None:
    args = parse_args()
    dataset_dirs, output_dir = resolve_dataset_dirs(args)
    metrics = [draw_dataset(path, output_dir, args) for path in dataset_dirs]
    if args.write_metrics:
        out = output_dir / "ground_truth_plot_metrics.csv"
        pd.DataFrame(metrics).to_csv(out, index=False)
        print(f"wrote metrics: {out}")


if __name__ == "__main__":
    main()
