#!/usr/bin/env python3
from __future__ import annotations

"""Analyze and visualize BLuR-GA LTGA building-block logs.

This script keeps the original compact diagnostics from:
  - building_block_summary*.csv
  - building_blocks*.csv

For the generation-level visualization, the default output is now intentionally
simple and presentation-friendly:
  1. bb_dsm_blocks_gen*.png
     A signed DSM / heatmap reordered so selected LTGA blocks appear as compact
     diagonal groups. Rectangles mark the selected blocks.
  2. bb_schema_cards_gen*.png
     A top-K block-card view that explains each selected block as same/opposite
     or as a relative signed schema.
  3. building_block_edge_membership_gen*.csv
     A machine-readable check of whether graph edges fall inside selected blocks.

The old graph-overlay plot is still available through --include-legacy-graph,
but it is disabled by default because dense linkage graphs are hard to read.

Example:
python analysis_interactive/analyze_building_blocks.py `
    --summary ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_main_pairwise_lasso/building_block_summary_c2_a3.csv `
    --blocks ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_main_pairwise_lasso/building_blocks_c2_a3.csv `
    --graph-snapshots ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_main_pairwise_lasso/graph_snapshots_c2_a3_rep00_fold00_run000.csv `
    --generation latest `
    --repeat-id 0 --outer-fold 0 --run-id 0 `
    --output-dir ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_main_pairwise_lasso/bb_analysis

python analysis_interactive/analyze_building_blocks.py `
    --summary ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_pairwise_lasso/building_block_summary_c2_a2.csv `
    --blocks ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_pairwise_lasso/building_blocks_c2_a2.csv `
    --graph-snapshots ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_pairwise_lasso/graph_snapshots_c2_a2_rep00_fold00_run000.csv `
    --generation latest `
    --repeat-id 0 --outer-fold 0 --run-id 0 `
    --output-dir ../Results/results_analysis_test_v2/anneal_task363614/blur_ga_pairwise_lasso/bb_analysis

python analysis_interactive/analyze_building_blocks.py `
    --summary ../Results/results_analysis_test_v2/anneal_task363614/empirical_linkage_legacy/building_block_summary_c2_a1.csv `
    --blocks ../Results/results_analysis_test_v2/anneal_task363614/empirical_linkage_legacy/building_blocks_c2_a1.csv `
    --graph-snapshots ../Results/results_analysis_test_v2/anneal_task363614/empirical_linkage_legacy/graph_snapshots_c2_a1_rep00_fold00_run000.csv `
    --generation latest `
    --repeat-id 0 --outer-fold 0 --run-id 0 `
    --output-dir ../Results/results_analysis_test_v2/anneal_task363614/empirical_linkage_legacy/bb_analysis
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Generic IO / small plots
# ---------------------------------------------------------------------------


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def _save_line(df: pd.DataFrame, x: str, y: str, out: Path, *, ylabel: str | None = None) -> None:
    if x not in df.columns or y not in df.columns or df.empty:
        return
    grouped = df.groupby(x, as_index=False)[y].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grouped[x], grouped[y], marker="o")
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _save_hist(df: pd.DataFrame, col: str, out: Path, *, xlabel: str | None = None) -> None:
    if col not in df.columns or df.empty:
        return
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=min(20, max(5, int(values.nunique()))))
    ax.set_xlabel(xlabel or col)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parsing / filtering helpers
# ---------------------------------------------------------------------------


def _parse_features(value: object) -> tuple[int, ...]:
    if pd.isna(value):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    out: list[int] = []
    for token in text.replace(",", ";").split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return tuple(dict.fromkeys(out))


def _parse_polarity(value: object) -> dict[int, int]:
    """Parse a relative polarity string like '27:+;74:+;73:-'."""
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    out: dict[int, int] = {}
    for token in text.replace(",", ";").split(";"):
        token = token.strip()
        if not token or ":" not in token:
            continue
        left, right = token.split(":", 1)
        try:
            feat = int(left.strip())
        except ValueError:
            continue
        sign_text = right.strip()
        if sign_text.startswith("+"):
            out[feat] = 1
        elif sign_text.startswith("-"):
            out[feat] = -1
    return out


def _filter_run(df: pd.DataFrame, *, repeat_id: int | None, outer_fold: int | None, run_id: int | None) -> pd.DataFrame:
    out = df.copy()
    for col, value in [("repeat_id", repeat_id), ("outer_fold", outer_fold), ("run_id", run_id)]:
        if value is not None and col in out.columns:
            out = out.loc[pd.to_numeric(out[col], errors="coerce").eq(int(value))]
    return out


def _choose_generation(blocks: pd.DataFrame, snapshots: pd.DataFrame, requested: str) -> int:
    gens: set[int] = set()
    for df in (blocks, snapshots):
        if not df.empty and "generation" in df.columns:
            gens.update(int(g) for g in pd.to_numeric(df["generation"], errors="coerce").dropna().unique())
    if not gens:
        raise ValueError("No generation column values available for visualization.")
    if str(requested).lower() == "latest":
        return max(gens)
    target = int(requested)
    if target in gens:
        return target
    earlier = [g for g in gens if g <= target]
    return max(earlier) if earlier else min(gens)


def _sort_blocks(block_rows: pd.DataFrame) -> pd.DataFrame:
    if block_rows.empty:
        return block_rows.copy()
    by = [c for c in ["score", "size"] if c in block_rows.columns]
    if by == ["score", "size"]:
        return block_rows.sort_values(["score", "size"], ascending=[False, False]).copy()
    if by == ["score"]:
        return block_rows.sort_values("score", ascending=False).copy()
    if by == ["size"]:
        return block_rows.sort_values("size", ascending=False).copy()
    return block_rows.copy()


def _block_membership(block_rows: pd.DataFrame) -> tuple[dict[int, int], dict[int, tuple[int, ...]]]:
    node_to_block: dict[int, int] = {}
    block_to_nodes: dict[int, tuple[int, ...]] = {}
    ordered = _sort_blocks(block_rows)
    for fallback_id, row in enumerate(ordered.itertuples(index=False)):
        block_id = int(getattr(row, "block_id", fallback_id))
        feats = _parse_features(getattr(row, "features", ""))
        if not feats:
            continue
        block_to_nodes[block_id] = feats
        for f in feats:
            # Keep the highest-scoring block if overlaps exist.
            node_to_block.setdefault(int(f), block_id)
    return node_to_block, block_to_nodes


def _weight_columns(graph_rows: pd.DataFrame) -> tuple[str, str | None, str | None]:
    weight_col = "weight" if "weight" in graph_rows.columns else None
    signed_col = "signed_weight" if "signed_weight" in graph_rows.columns else None
    sign_col = "sign" if "sign" in graph_rows.columns else None
    if weight_col is None:
        raise ValueError("graph_snapshots CSV must contain a 'weight' column.")
    return weight_col, signed_col, sign_col


def _edge_sign(row: object, signed_col: str | None, sign_col: str | None) -> int:
    if sign_col is not None:
        try:
            sign = int(getattr(row, sign_col))
            if sign > 0:
                return 1
            if sign < 0:
                return -1
        except Exception:
            pass
    if signed_col is not None:
        try:
            signed_weight = float(getattr(row, signed_col))
            if signed_weight > 0:
                return 1
            if signed_weight < 0:
                return -1
        except Exception:
            pass
    return 0


# ---------------------------------------------------------------------------
# Human-readable schema helpers
# ---------------------------------------------------------------------------


def _schema_from_row(row: object) -> str:
    """Return a compact human-readable relative schema for one block.

    The + / ~ notation is relative, not an absolute chromosome assignment.
    Example: '+{27,74} | ~{73}' means 27 and 74 are on one side of the signed
    relation and 73 is on the opposite side.
    """
    feats = _parse_features(getattr(row, "features", ""))
    polarity = _parse_polarity(getattr(row, "polarity", ""))
    if polarity:
        pos = [f for f in feats if polarity.get(f, 1) > 0]
        neg = [f for f in feats if polarity.get(f, 1) < 0]
        parts: list[str] = []
        if pos:
            parts.append("{" + ",".join(str(f) for f in pos) + "}")
        if neg:
            parts.append("~{" + ",".join(str(f) for f in neg) + "}")
        return " | ".join(parts) if parts else "{" + ",".join(str(f) for f in feats) + "}"

    schema_hint = str(getattr(row, "schema_hint", "") or "").strip()
    if schema_hint and schema_hint.lower() != "nan":
        return schema_hint

    pos_edges = int(float(getattr(row, "n_positive_edges", 0) or 0))
    neg_edges = int(float(getattr(row, "n_negative_edges", 0) or 0))
    if neg_edges == 0 and pos_edges > 0:
        return "same(" + ";".join(str(f) for f in feats) + ")"
    if pos_edges == 0 and neg_edges > 0 and len(feats) == 2:
        return f"{feats[0]} opposite {feats[1]}"
    return "{" + ",".join(str(f) for f in feats) + "}"


def _write_schema_csv(block_rows: pd.DataFrame, out: Path, *, top_k: int) -> None:
    if block_rows.empty:
        return
    ordered = _sort_blocks(block_rows).head(top_k).copy()
    schemas: list[str] = []
    for row in ordered.itertuples(index=False):
        schemas.append(_schema_from_row(row))
    ordered["relative_schema"] = schemas
    keep_cols = [
        c for c in [
            "repeat_id", "outer_fold", "run_id", "generation", "block_id", "features",
            "relative_schema", "size", "score", "n_positive_edges", "n_negative_edges",
            "signed_balance", "n_sign_conflicts", "polarity", "positive_group", "negative_group",
            "schema_hint",
        ] if c in ordered.columns
    ]
    ordered[keep_cols].to_csv(out, index=False)


# ---------------------------------------------------------------------------
# Edge/block membership diagnostic
# ---------------------------------------------------------------------------


def _write_membership_csv(graph_rows: pd.DataFrame, block_rows: pd.DataFrame, out: Path, generation: int) -> None:
    node_to_block, _ = _block_membership(block_rows)
    rows: list[dict[str, object]] = []
    for r in graph_rows.itertuples(index=False):
        i, j = int(r.feature_i), int(r.feature_j)
        bi = node_to_block.get(i, -1)
        bj = node_to_block.get(j, -1)
        signed_weight = float(getattr(r, "signed_weight", getattr(r, "weight", 0.0)))
        sign = int(getattr(r, "sign", 1 if signed_weight > 0 else (-1 if signed_weight < 0 else 0)))
        rows.append({
            "generation": generation,
            "feature_i": i,
            "feature_j": j,
            "weight": float(getattr(r, "weight", 0.0)),
            "signed_weight": signed_weight,
            "sign": sign,
            "feature_i_block": bi,
            "feature_j_block": bj,
            "edge_inside_same_selected_block": int(bi >= 0 and bi == bj),
        })
    pd.DataFrame(rows).to_csv(out, index=False)


# ---------------------------------------------------------------------------
# New compact visualization 1: DSM + block rectangles
# ---------------------------------------------------------------------------


def _ordered_features_for_dsm(graph_rows: pd.DataFrame, block_rows: pd.DataFrame, *, max_nodes: int | None) -> list[int]:
    """Order nodes so selected blocks appear as diagonal groups.

    1. Put block features first, grouped by block score.
    2. Put remaining graph nodes afterward by weighted degree.
    """
    ordered_blocks = _sort_blocks(block_rows)
    seen: set[int] = set()
    ordered: list[int] = []

    for row in ordered_blocks.itertuples(index=False):
        feats = _parse_features(getattr(row, "features", ""))
        for f in feats:
            if f not in seen:
                ordered.append(f)
                seen.add(f)

    degree: dict[int, float] = {}
    if not graph_rows.empty:
        for row in graph_rows.itertuples(index=False):
            i, j = int(row.feature_i), int(row.feature_j)
            w = abs(float(getattr(row, "weight", 0.0)))
            degree[i] = degree.get(i, 0.0) + w
            degree[j] = degree.get(j, 0.0) + w
    remaining = sorted((f for f in degree if f not in seen), key=lambda f: (-degree[f], f))
    ordered.extend(remaining)

    if max_nodes is not None and max_nodes > 0 and len(ordered) > max_nodes:
        # Keep all block nodes when possible, then fill with high-degree non-block nodes.
        block_nodes = ordered[: len(seen)]
        capacity = max(0, int(max_nodes) - len(block_nodes))
        ordered = block_nodes + remaining[:capacity]
    return ordered


def _build_signed_matrix(graph_rows: pd.DataFrame, ordered_features: list[int]) -> np.ndarray:
    idx = {f: k for k, f in enumerate(ordered_features)}
    mat = np.zeros((len(ordered_features), len(ordered_features)), dtype=float)
    weight_col, signed_col, sign_col = _weight_columns(graph_rows)

    for row in graph_rows.itertuples(index=False):
        i, j = int(row.feature_i), int(row.feature_j)
        if i not in idx or j not in idx:
            continue
        w = float(getattr(row, weight_col))
        sign = _edge_sign(row, signed_col, sign_col)
        value = sign * abs(w) if sign != 0 else 0.0
        a, b = idx[i], idx[j]
        mat[a, b] = value
        mat[b, a] = value
    return mat


def _block_spans_in_order(block_rows: pd.DataFrame, ordered_features: list[int]) -> list[tuple[int, int, int, tuple[int, ...]]]:
    pos = {f: k for k, f in enumerate(ordered_features)}
    spans: list[tuple[int, int, int, tuple[int, ...]]] = []
    for fallback_id, row in enumerate(_sort_blocks(block_rows).itertuples(index=False)):
        block_id = int(getattr(row, "block_id", fallback_id))
        feats = tuple(f for f in _parse_features(getattr(row, "features", "")) if f in pos)
        if not feats:
            continue
        locations = sorted(pos[f] for f in feats)
        start, end = locations[0], locations[-1]
        # Only draw a diagonal rectangle if the block is contiguous under the chosen ordering.
        if end - start + 1 == len(feats):
            spans.append((block_id, start, end + 1, feats))
    return spans


def _draw_block_dsm(
    graph_rows: pd.DataFrame,
    block_rows: pd.DataFrame,
    out: Path,
    *,
    generation: int,
    max_nodes: int | None,
) -> None:
    if graph_rows.empty or block_rows.empty:
        return
    ordered_features = _ordered_features_for_dsm(graph_rows, block_rows, max_nodes=max_nodes)
    if len(ordered_features) < 2:
        return

    mat = _build_signed_matrix(graph_rows, ordered_features)
    max_abs = float(np.nanmax(np.abs(mat))) if mat.size else 0.0
    if max_abs <= 0:
        max_abs = 1.0

    size = max(6.5, min(13.5, 0.26 * len(ordered_features) + 4.0))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("signed linkage strength")

    # Light grid lines for readability.
    ax.set_xticks(np.arange(len(ordered_features)))
    ax.set_yticks(np.arange(len(ordered_features)))
    show_all_labels = len(ordered_features) <= 45
    if show_all_labels:
        ax.set_xticklabels([str(f) for f in ordered_features], rotation=90, fontsize=7)
        ax.set_yticklabels([str(f) for f in ordered_features], fontsize=7)
    else:
        step = max(1, len(ordered_features) // 30)
        labels = [str(f) if i % step == 0 else "" for i, f in enumerate(ordered_features)]
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)

    spans = _block_spans_in_order(block_rows, ordered_features)
    for local_idx, (block_id, start, end, feats) in enumerate(spans):
        width = end - start
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), width, width, fill=False, linewidth=2.2))
        if local_idx < 25:
            ax.text(
                start + width / 2 - 0.5,
                start - 0.85,
                f"B{block_id}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                clip_on=False,
            )

    ax.set_title(f"LTGA building blocks as reordered DSM, generation {generation}")
    ax.set_xlabel("features reordered by selected blocks, then weighted degree")
    ax.set_ylabel("features reordered by selected blocks, then weighted degree")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# New compact visualization 2: block schema cards
# ---------------------------------------------------------------------------


def _draw_schema_cards(block_rows: pd.DataFrame, out: Path, *, generation: int, top_k: int) -> None:
    if block_rows.empty:
        return
    rows = list(_sort_blocks(block_rows).head(top_k).itertuples(index=False))
    if not rows:
        return

    n = len(rows)
    height = max(3.5, 0.55 * n + 1.4)
    fig, ax = plt.subplots(figsize=(12, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 1)
    ax.axis("off")

    ax.text(0.02, n + 0.6, f"Top {n} LTGA building blocks, generation {generation}", fontsize=14, fontweight="bold")
    ax.text(0.02, n + 0.2, "Relative schema: ~ means opposite polarity inside the block, not necessarily bit value 0.", fontsize=9)

    header_y = n - 0.15
    ax.text(0.03, header_y, "Block", fontsize=9, fontweight="bold")
    ax.text(0.16, header_y, "Features / relative schema", fontsize=9, fontweight="bold")
    ax.text(0.62, header_y, "Sign mix", fontsize=9, fontweight="bold")
    ax.text(0.78, header_y, "Score", fontsize=9, fontweight="bold")
    ax.text(0.89, header_y, "Conflicts", fontsize=9, fontweight="bold")

    for k, row in enumerate(rows):
        y = n - k - 0.85
        block_id = int(getattr(row, "block_id", k))
        size = int(float(getattr(row, "size", len(_parse_features(getattr(row, "features", "")))) or 0))
        score = float(getattr(row, "score", np.nan))
        pos_edges = int(float(getattr(row, "n_positive_edges", 0) or 0))
        neg_edges = int(float(getattr(row, "n_negative_edges", 0) or 0))
        conflicts = int(float(getattr(row, "n_sign_conflicts", 0) or 0))
        schema = _schema_from_row(row)

        # Alternating card background.  Do not encode meaning by color here;
        # keep the plot clean and readable when printed in grayscale.
        if k % 2 == 0:
            ax.add_patch(Rectangle((0.015, y - 0.26), 0.97, 0.46, alpha=0.06))

        ax.text(0.03, y, f"B{block_id}\nsize={size}", fontsize=8, va="center")
        ax.text(0.16, y, schema, fontsize=9, va="center")
        ax.text(0.62, y, f"+{pos_edges} / -{neg_edges}", fontsize=8, va="center")
        ax.text(0.78, y, "" if np.isnan(score) else f"{score:.4g}", fontsize=8, va="center")
        ax.text(0.91, y, str(conflicts), fontsize=8, va="center")

    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Legacy dense graph overlay: kept for debugging, disabled by default
# ---------------------------------------------------------------------------


def _draw_building_block_graph(
    graph_rows: pd.DataFrame,
    block_rows: pd.DataFrame,
    out: Path,
    *,
    generation: int,
    title_suffix: str = "",
) -> None:
    if graph_rows.empty:
        return
    G = nx.Graph()
    for r in graph_rows.itertuples(index=False):
        i, j = int(r.feature_i), int(r.feature_j)
        weight = float(getattr(r, "weight", 0.0))
        signed_weight = float(getattr(r, "signed_weight", weight))
        sign = int(getattr(r, "sign", 1 if signed_weight > 0 else (-1 if signed_weight < 0 else 0)))
        G.add_edge(i, j, weight=weight, signed_weight=signed_weight, sign=sign)
    if G.number_of_edges() == 0:
        return

    node_to_block, block_to_nodes = _block_membership(block_rows)
    pos = nx.spring_layout(G, seed=7, weight="weight", k=None, iterations=100)

    block_ids = sorted(set(node_to_block.values()))
    palette = plt.cm.tab20(np.linspace(0, 1, max(1, len(block_ids))))
    block_color = {bid: palette[idx % len(palette)] for idx, bid in enumerate(block_ids)}
    node_colors = [block_color.get(node_to_block.get(n, -999), (0.85, 0.85, 0.85, 1.0)) for n in G.nodes]
    node_sizes = [520 if n in node_to_block else 300 for n in G.nodes]

    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if int(d.get("sign", 0)) > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if int(d.get("sign", 0)) < 0]
    zero_edges = [(u, v) for u, v, d in G.edges(data=True) if int(d.get("sign", 0)) == 0]
    max_w = max(1e-12, float(pd.to_numeric(graph_rows["weight"], errors="coerce").max()))
    widths = {tuple(sorted((u, v))): 0.5 + 3.0 * float(d.get("weight", 0.0)) / max_w for u, v, d in G.edges(data=True)}

    fig, ax = plt.subplots(figsize=(10, 8))
    if pos_edges:
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=[widths[tuple(sorted(e))] for e in pos_edges], edge_color="#2ca02c", alpha=0.65, ax=ax)
    if neg_edges:
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=[widths[tuple(sorted(e))] for e in neg_edges], edge_color="#d62728", style="dashed", alpha=0.65, ax=ax)
    if zero_edges:
        nx.draw_networkx_edges(G, pos, edgelist=zero_edges, width=[widths[tuple(sorted(e))] for e in zero_edges], edge_color="#999999", alpha=0.4, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="black", linewidths=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    if G.number_of_edges() <= 45:
        edge_labels = {
            (u, v): ("+" if d.get("sign", 0) > 0 else ("−" if d.get("sign", 0) < 0 else "0"))
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    legend_lines = [f"B{bid}: " + ";".join(str(v) for v in nodes) for bid, nodes in sorted(block_to_nodes.items())]
    if legend_lines:
        ax.text(
            1.02,
            0.98,
            "Selected LTGA blocks\n" + "\n".join(legend_lines[:25]),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    ax.set_title(f"Legacy graph overlay, generation {generation}{title_suffix}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generation visualization wrapper
# ---------------------------------------------------------------------------


def _visualize_generation(
    *,
    snapshots: pd.DataFrame,
    blocks: pd.DataFrame,
    outdir: Path,
    generation_arg: str,
    repeat_id: int | None,
    outer_fold: int | None,
    run_id: int | None,
    top_edges: int | None,
    top_blocks: int,
    dsm_max_nodes: int | None,
    include_legacy_graph: bool,
) -> None:
    snapshots = _filter_run(snapshots, repeat_id=repeat_id, outer_fold=outer_fold, run_id=run_id)
    blocks = _filter_run(blocks, repeat_id=repeat_id, outer_fold=outer_fold, run_id=run_id)
    if snapshots.empty or blocks.empty:
        print("No matching graph/block rows after repeat/fold/run filtering; skipped generation visualization.")
        return

    generation = _choose_generation(blocks, snapshots, generation_arg)
    graph_rows = snapshots.loc[pd.to_numeric(snapshots["generation"], errors="coerce").eq(generation)].copy()
    block_rows = blocks.loc[pd.to_numeric(blocks["generation"], errors="coerce").eq(generation)].copy()
    if graph_rows.empty or block_rows.empty:
        print(f"No graph/block rows available at generation {generation}; skipped generation visualization.")
        return

    if top_edges is not None and top_edges > 0 and "weight" in graph_rows.columns:
        graph_rows = graph_rows.sort_values("weight", ascending=False).head(int(top_edges)).copy()

    _write_membership_csv(graph_rows, block_rows, outdir / f"building_block_edge_membership_gen{generation}.csv", generation)
    _write_schema_csv(block_rows, outdir / f"building_block_schemas_gen{generation}.csv", top_k=top_blocks)
    _draw_block_dsm(
        graph_rows,
        block_rows,
        outdir / f"bb_dsm_blocks_gen{generation}.png",
        generation=generation,
        max_nodes=dsm_max_nodes,
    )
    _draw_schema_cards(
        block_rows,
        outdir / f"bb_schema_cards_gen{generation}.png",
        generation=generation,
        top_k=top_blocks,
    )

    if include_legacy_graph:
        _draw_building_block_graph(
            graph_rows,
            block_rows,
            outdir / f"legacy_building_block_graph_gen{generation}.png",
            generation=generation,
            title_suffix="" if top_edges is None else f" (top {top_edges} edges)",
        )


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze LTGA building-block logs.")
    parser.add_argument("--summary", required=True, help="building_block_summary CSV")
    parser.add_argument("--blocks", default=None, help="building_blocks CSV")
    parser.add_argument("--graph-snapshots", default=None, help="Optional graph_snapshots CSV for generation visualization")
    parser.add_argument("--generation", default="latest", help="Generation to visualize, or 'latest' (default)")
    parser.add_argument("--repeat-id", type=int, default=None)
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--graph-top-edges", type=int, default=80, help="Use only top-N graph edges for DSM/legacy graph; <=0 means all")
    parser.add_argument("--top-blocks", type=int, default=20, help="Number of blocks shown in schema cards / schema CSV")
    parser.add_argument("--dsm-max-nodes", type=int, default=70, help="Maximum nodes shown in DSM; <=0 means all")
    parser.add_argument("--include-legacy-graph", action="store_true", help="Also write the old dense graph-overlay plot")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(args.summary)
    blocks = _read_csv(args.blocks) if args.blocks else pd.DataFrame()

    _save_line(summary, "generation", "n_blocks", out / "bb_n_blocks_by_generation.png", ylabel="mean selected blocks")
    _save_line(summary, "generation", "max_score", out / "bb_max_score_by_generation.png", ylabel="mean max block score")
    _save_line(summary, "generation", "mean_internal_abs", out / "bb_abs_strength_by_generation.png")
    _save_line(summary, "generation", "mean_internal_positive", out / "bb_positive_strength_by_generation.png")
    _save_line(summary, "generation", "mean_internal_negative_abs", out / "bb_negative_strength_by_generation.png")
    _save_line(summary, "generation", "mean_signed_balance", out / "bb_signed_balance_by_generation.png")
    _save_line(summary, "generation", "n_sign_conflicts_in_blocks", out / "bb_sign_conflicts_by_generation.png")

    if not blocks.empty:
        _save_hist(blocks, "size", out / "bb_block_size_histogram.png", xlabel="block size")
        _save_hist(blocks, "score", out / "bb_block_score_histogram.png", xlabel="block score")
        top_cols = [
            c for c in [
                "repeat_id", "outer_fold", "run_id", "generation", "block_id", "features", "size", "score",
                "internal_abs_mean", "internal_positive_mean", "internal_negative_abs_mean",
                "n_positive_edges", "n_negative_edges", "signed_balance", "n_sign_conflicts",
                "polarity", "positive_group", "negative_group", "schema_hint",
            ] if c in blocks.columns
        ]
        top = blocks.sort_values("score", ascending=False).head(50)[top_cols]
        top.to_csv(out / "top_building_blocks.csv", index=False)

    if args.graph_snapshots and args.blocks:
        snapshots = _read_csv(args.graph_snapshots)
        top_edges = None if args.graph_top_edges is not None and args.graph_top_edges <= 0 else args.graph_top_edges
        dsm_max_nodes = None if args.dsm_max_nodes is not None and args.dsm_max_nodes <= 0 else args.dsm_max_nodes
        _visualize_generation(
            snapshots=snapshots,
            blocks=blocks,
            outdir=out,
            generation_arg=str(args.generation),
            repeat_id=args.repeat_id,
            outer_fold=args.outer_fold,
            run_id=args.run_id,
            top_edges=top_edges,
            top_blocks=int(args.top_blocks),
            dsm_max_nodes=dsm_max_nodes,
            include_legacy_graph=bool(args.include_legacy_graph),
        )

    print(f"Wrote building-block analysis to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
