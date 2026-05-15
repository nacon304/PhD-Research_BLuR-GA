from __future__ import annotations

import argparse
import html
import json
import math
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_graph_evolution
from .importance import (
    auto_related_paths,
    choose_score_column,
    compute_edge_importance,
    compute_feature_importance,
    load_run_summary,
)
import networkx as nx




def generations_for_animation(
    snapshots: pd.DataFrame,
    trace: pd.DataFrame | None = None,
    *,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> list[int]:
    """Return saved generations for the dependency-light HTML UI.

    This duplicates the small helper from renderer.py so make_graph_ui.py does
    not import matplotlib. That matters on headless HPC nodes, where matplotlib
    font-cache creation can be slow or unavailable.
    """
    if trace is not None and not trace.empty and "generation" in trace.columns:
        generations = sorted(int(g) for g in trace["generation"].unique())
    else:
        generations = sorted(int(g) for g in snapshots["generation"].unique())
    if start is not None:
        generations = [g for g in generations if g >= int(start)]
    if end is not None:
        generations = [g for g in generations if g <= int(end)]
    if not generations:
        raise ValueError("No generations available after applying start/end filters.")
    step = max(1, int(step))
    return generations[::step]


def compute_layout(
    snapshots: pd.DataFrame,
    n_nodes: int,
    *,
    layout: str = "spring",
    seed: int = 1,
    layout_k_scale: float = 1.8,
    layout_iterations: int = 300,
) -> dict[int, np.ndarray]:
    """Compute node positions without importing matplotlib.

    The video renderer still uses matplotlib, but the HTML UI only needs a graph
    layout. Keeping this path matplotlib-free makes the analysis step safer on
    HPC login/compute nodes.
    """
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    if not snapshots.empty:
        final_gen = int(snapshots["generation"].max())
        final_edges = snapshots[snapshots["generation"] == final_gen]
        for r in final_edges.itertuples(index=False):
            g.add_edge(int(r.feature_i), int(r.feature_j), weight=float(r.weight))
    layout = layout.lower()
    if layout == "spring":
        if g.number_of_edges() == 0:
            return nx.circular_layout(g)
        # NetworkX default k=1/sqrt(n) is often too compact for 50+ feature
        # graphs.  A larger k spreads communities apart without touching the
        # underlying eVIG weights.  Pass --layout-k-scale 1.0 to recover the
        # old default spacing.
        k = float(layout_k_scale) / max(float(n_nodes) ** 0.5, 1.0)
        return nx.spring_layout(g, seed=seed, weight="weight", k=k, iterations=int(layout_iterations))
    if layout == "circular":
        return nx.circular_layout(g)
    if layout == "spectral":
        if g.number_of_edges() == 0:
            return nx.circular_layout(g)
        return nx.spectral_layout(g, weight="weight")
    raise ValueError(f"Unknown layout={layout!r}")

def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if not math.isfinite(x):
        return default
    return x


def _filter_dataframe(df: pd.DataFrame, *, run_id: int | None = None, repeat_id: int | None = None, outer_fold: int | None = None) -> pd.DataFrame:
    out = df.copy()
    if run_id is not None and "run_id" in out.columns:
        out = out[out["run_id"] == int(run_id)].copy()
    if repeat_id is not None and "repeat_id" in out.columns:
        out = out[out["repeat_id"] == int(repeat_id)].copy()
    if outer_fold is not None and "outer_fold" in out.columns:
        out = out[out["outer_fold"] == int(outer_fold)].copy()
    return out.reset_index(drop=True)


def load_selected_features(path: str | Path | None, *, run_id: int | None = None, repeat_id: int | None = None, outer_fold: int | None = None) -> set[int]:
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Selected-features file not found: {p}")
    df = pd.read_csv(p)
    df = _filter_dataframe(df, run_id=run_id, repeat_id=repeat_id, outer_fold=outer_fold)
    if "feature_index" in df.columns:
        return {int(x) for x in df["feature_index"].dropna().unique()}
    if "node_id" in df.columns:
        return {int(x) for x in df["node_id"].dropna().unique()}
    return set()


def _normalise_positions(pos: dict[int, np.ndarray], n_nodes: int) -> dict[int, tuple[float, float]]:
    xy = np.array([pos[i] for i in range(n_nodes)], dtype=float)
    if xy.size == 0:
        return {}
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-12)
    norm = (xy - min_xy) / span
    # Flip y so circular/spring layouts feel natural in browser coordinates.
    norm[:, 1] = 1.0 - norm[:, 1]
    return {i: (float(norm[i, 0]), float(norm[i, 1])) for i in range(n_nodes)}


def _prepare_payload(
    snapshots: pd.DataFrame,
    trace: pd.DataFrame | None,
    *,
    n_nodes: int,
    labels: dict[int, str],
    selected_features: set[int],
    layout_name: str,
    layout_seed: int,
    layout_k_scale: float,
    layout_iterations: int,
    initial_spacing: float,
    start: int | None,
    end: int | None,
    step: int,
    title: str,
    feature_importance: list[dict[str, Any]] | None = None,
    edge_importance: list[dict[str, Any]] | None = None,
    importance_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshots = snapshots.copy()
    if snapshots.empty:
        raise ValueError("Snapshot file is empty; cannot build UI.")

    gens = generations_for_animation(snapshots, trace, start=start, end=end, step=max(1, int(step)))
    snapshots = snapshots[snapshots["generation"].isin(gens)].copy()
    pos = compute_layout(
        snapshots,
        n_nodes,
        layout=layout_name,
        seed=layout_seed,
        layout_k_scale=layout_k_scale,
        layout_iterations=layout_iterations,
    )
    pos_norm = _normalise_positions(pos, n_nodes)

    nodes = []
    for i in range(n_nodes):
        x, y = pos_norm.get(i, (0.5, 0.5))
        nodes.append(
            {
                "id": int(i),
                "label": str(labels.get(i, str(i))),
                "x": x,
                "y": y,
                "selected": bool(i in selected_features),
            }
        )

    edges_by_generation: dict[str, list[dict[str, Any]]] = {}
    max_weight = 0.0
    max_edges = 0
    for gen, grp in snapshots.groupby("generation", sort=True):
        rows = []
        for r in grp.itertuples(index=False):
            w = _finite_float(getattr(r, "weight", 0.0))
            max_weight = max(max_weight, w)
            rows.append(
                {
                    "i": int(getattr(r, "feature_i")),
                    "j": int(getattr(r, "feature_j")),
                    "w": w,
                    "pc": int(getattr(r, "positive_count", 0)),
                    "tc": int(getattr(r, "tested_count", 0)),
                    "ws": _finite_float(getattr(r, "weight_sum", 0.0)),
                    "first": int(getattr(r, "first_seen_generation", gen)),
                    "last": int(getattr(r, "last_updated_generation", gen)),
                }
            )
        rows.sort(key=lambda e: e["w"], reverse=True)
        edges_by_generation[str(int(gen))] = rows
        max_edges = max(max_edges, len(rows))
    max_weight = max(max_weight, 1e-12)

    trace_by_generation: dict[str, dict[str, Any]] = {}
    if trace is not None and not trace.empty and "generation" in trace.columns:
        for _, row in trace.iterrows():
            gen = int(row["generation"])
            clean: dict[str, Any] = {}
            for col, value in row.items():
                if pd.isna(value):
                    continue
                if isinstance(value, (np.integer, int)):
                    clean[col] = int(value)
                elif isinstance(value, (np.floating, float)):
                    clean[col] = _finite_float(value)
                else:
                    clean[col] = str(value)
            trace_by_generation[str(gen)] = clean

    # A tiny timeline for the bottom sparkline.
    timeline = []
    for g in gens:
        edge_count = len(edges_by_generation.get(str(g), []))
        trace_g = trace_by_generation.get(str(g), {})
        timeline.append(
            {
                "generation": int(g),
                "n_edges": edge_count,
                "best": trace_g.get("best_so_far_fitness"),
            }
        )

    return {
        "title": title,
        "generations": [int(g) for g in gens],
        "nodes": nodes,
        "edgesByGeneration": edges_by_generation,
        "traceByGeneration": trace_by_generation,
        "timeline": timeline,
        "maxWeight": max_weight,
        "maxEdges": max_edges,
        "nNodes": n_nodes,
        "initialSpacing": float(initial_spacing),
        "featureImportance": feature_importance or [],
        "edgeImportance": edge_importance or [],
        "importanceMeta": importance_meta or {},
    }


def render_html(payload: dict[str, Any], *, source_hint: str = "") -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    safe_title = html.escape(str(payload.get("title", "BLuR-GA Graph Evolution Viewer")))
    safe_hint = html.escape(source_hint)
    # The UI is deliberately dependency-free: one HTML file with embedded data and vanilla JS/SVG.
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{safe_title}</title>
<style>
  :root {{
    --bg: #f7f7f8;
    --panel: #ffffff;
    --ink: #1f2937;
    --muted: #6b7280;
    --line: #e5e7eb;
    --accent: #2563eb;
    --accent-soft: #dbeafe;
    --selected: #f59e0b;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
  .app {{ display: grid; grid-template-columns: 310px minmax(520px, 1fr); grid-template-rows: auto 1fr auto; height: 100vh; gap: 12px; padding: 12px; }}
  header {{ grid-column: 1 / 3; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
  header h1 {{ margin: 0; font-size: 18px; font-weight: 750; }}
  header .sub {{ font-size: 12px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  aside {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 12px; overflow: auto; }}
  main {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; min-height: 0; position: relative; overflow: hidden; display: flex; flex-direction: column; }}
  footer {{ grid-column: 1 / 3; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 10px 14px; }}
  .section {{ border-bottom: 1px solid var(--line); padding-bottom: 12px; margin-bottom: 12px; }}
  .section:last-child {{ border-bottom: 0; margin-bottom: 0; }}
  .section-title {{ font-size: 12px; font-weight: 750; color: #374151; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 8px; }}
  label {{ display: block; font-size: 12px; font-weight: 650; margin: 8px 0 4px; color: #374151; }}
  select, input[type="number"], input[type="text"] {{ width: 100%; border: 1px solid var(--line); border-radius: 10px; padding: 7px 8px; background: white; color: var(--ink); }}
  input[type="range"] {{ width: 100%; }}
  .row {{ display: grid; grid-template-columns: 1fr 82px; gap: 8px; align-items: center; }}
  .button-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 8px; }}
  button {{ border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; background: #fff; cursor: pointer; font-weight: 650; color: var(--ink); }}
  button.primary {{ background: var(--accent); color: white; border-color: var(--accent); }}
  button:hover {{ filter: brightness(.98); }}
  .checkbox-row {{ display: flex; align-items: center; gap: 8px; margin-top: 8px; font-size: 13px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(10, 1fr); gap: 8px; }}
  .stat {{ border: 1px solid var(--line); background: #fbfbfc; border-radius: 12px; padding: 8px 10px; min-width: 0; }}
  .stat .k {{ font-size: 11px; color: var(--muted); }}
  .stat .v {{ font-weight: 800; font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .tabs {{ display: flex; gap: 8px; padding: 10px 10px 0; border-bottom: 1px solid var(--line); background: #fff; }}
  .tab-btn {{ padding: 8px 12px; border-radius: 10px 10px 0 0; border-bottom-color: transparent; font-size: 13px; }}
  .tab-btn.active {{ background: var(--accent-soft); border-color: #bfdbfe; color: #1d4ed8; }}
  .tab-panel {{ display: none; min-height: 0; flex: 1; overflow: auto; position: relative; }}
  .tab-panel.active {{ display: block; }}
  #graphPanel.active {{ display: block; }}
  #graphSvg {{ width: 100%; height: 100%; min-height: 520px; display: block; background: radial-gradient(circle at 50% 40%, #ffffff 0%, #fbfbfd 55%, #f3f4f6 100%); cursor: grab; touch-action: none; }}
  #graphSvg.dragging {{ cursor: grabbing; }}
  .table-wrap {{ padding: 14px; }}
  .formula {{ font-size: 12px; color: #374151; background: #f9fafb; border: 1px solid var(--line); border-radius: 12px; padding: 10px; margin-bottom: 12px; line-height: 1.5; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; background: white; }}
  th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; white-space: nowrap; }}
  th {{ position: sticky; top: 0; background: #f9fafb; z-index: 1; font-weight: 750; color: #374151; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  #tooltip {{ position: fixed; display: none; pointer-events: none; background: rgba(17,24,39,.94); color: #fff; padding: 8px 10px; border-radius: 9px; font-size: 12px; max-width: 280px; z-index: 10; box-shadow: 0 8px 30px rgba(0,0,0,.2); }}
  .cmd {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 11px; color: #374151; background: #f3f4f6; border: 1px solid var(--line); border-radius: 10px; padding: 8px; white-space: pre-wrap; overflow-wrap: anywhere; }}
  .mini {{ color: var(--muted); font-size: 11px; line-height: 1.35; }}
  .legend {{ position: absolute; top: 12px; right: 12px; background: rgba(255,255,255,.92); border: 1px solid var(--line); border-radius: 12px; padding: 8px 10px; font-size: 12px; color: var(--muted); box-shadow: 0 4px 20px rgba(0,0,0,.05); }}
  #timelineCanvas {{ width: 100%; height: 54px; display: block; margin-top: 8px; }}
  @media (max-width: 1100px) {{ .stats-grid {{ grid-template-columns: repeat(5, 1fr); }} }}
  @media (max-width: 880px) {{ .app {{ grid-template-columns: 1fr; height: auto; }} header, footer {{ grid-column: 1; }} main {{ height: 72vh; }} .stats-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>
<div class="app">
  <header>
    <div>
      <h1>{safe_title}</h1>
      <div class="sub">Interactive SVG replay of generation-by-generation eVIGw snapshots. {safe_hint}</div>
    </div>
    <div class="mini" id="headerInfo"></div>
  </header>

  <aside>
    <div class="section">
      <div class="section-title">Playback</div>
      <label>Generation</label>
      <div class="row">
        <input id="generationSlider" type="range" min="0" max="0" step="1" value="0" />
        <input id="generationBox" type="number" min="0" value="0" />
      </div>
      <div class="button-row">
        <button id="prevBtn">◀ Prev</button>
        <button id="playBtn" class="primary">▶ Play</button>
        <button id="nextBtn">Next ▶</button>
      </div>
      <div class="button-row" style="grid-template-columns: repeat(3, 1fr);">
        <button id="resetBtn">Reset Gen</button>
        <button id="fitViewBtn">Fit View</button>
        <button id="saveSvgBtn">Save SVG</button>
      </div>
      <label>Speed</label>
      <select id="speedSelect">
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
        <option value="5">5x</option>
        <option value="10">10x</option>
      </select>
      <label>Step per frame</label>
      <input id="stepBox" type="number" value="1" min="1" max="100" />
      <div class="checkbox-row"><input id="loopCheck" type="checkbox" checked /> <span>Loop playback</span></div>
    </div>

    <div class="section">
      <div class="section-title">Edge filter</div>
      <label>Filter mode</label>
      <select id="filterMode">
        <option value="topk" selected>Top-k strongest edges</option>
        <option value="topk_per_node">Top-k incident edges per node</option>
        <option value="threshold">Weight threshold</option>
        <option value="percentile">Percentile threshold</option>
        <option value="all">All edges</option>
      </select>
      <label>Top-k / k per node</label>
      <div class="row">
        <input id="topKSlider" type="range" min="1" max="100" value="30" />
        <input id="topKBox" type="number" min="1" value="30" />
      </div>
      <label>Threshold</label>
      <div class="row">
        <input id="thresholdSlider" type="range" min="0" max="1" step="0.001" value="0" />
        <input id="thresholdBox" type="number" min="0" step="0.001" value="0" />
      </div>
      <label>Percentile</label>
      <input id="percentileBox" type="number" min="0" max="100" step="1" value="95" />
    </div>

    <div class="section">
      <div class="section-title">View</div>
      <label>View mode</label>
      <select id="viewMode">
        <option value="cumulative" selected>Cumulative up to generation t</option>
        <option value="new">Only newly discovered edges</option>
        <option value="window">Sliding window</option>
      </select>
      <label>Window size</label>
      <input id="windowBox" type="number" min="1" value="20" />
      <div class="checkbox-row"><input id="showLabels" type="checkbox" checked /> <span>Show node labels</span></div>
      <div class="checkbox-row"><input id="showIsolated" type="checkbox" checked /> <span>Show isolated nodes</span></div>
      <div class="checkbox-row"><input id="edgeWidth" type="checkbox" checked /> <span>Edge width ∝ weight</span></div>
      <div class="checkbox-row"><input id="edgeOpacity" type="checkbox" checked /> <span>Edge opacity ∝ weight</span></div>
      <label>Graph spacing</label>
      <div class="row">
        <input id="spacingSlider" type="range" min="0.6" max="3.0" step="0.05" value="1.15" />
        <input id="spacingBox" type="number" min="0.6" max="3.0" step="0.05" value="1.15" />
      </div>
      <label>Node size</label>
      <div class="row">
        <input id="nodeScaleSlider" type="range" min="0.5" max="2.5" step="0.05" value="1.0" />
        <input id="nodeScaleBox" type="number" min="0.5" max="2.5" step="0.05" value="1.0" />
      </div>
      <label>Focus feature id</label>
      <input id="focusNodeBox" type="text" placeholder="e.g. 35" />
      <div class="checkbox-row"><input id="focusIncidentOnly" type="checkbox" /> <span>Show only incident edges</span></div>
      <label>Weight scale</label>
      <select id="weightScale">
        <option value="global" selected>Global across run</option>
        <option value="frame">Per current frame</option>
      </select>
    </div>

    <div class="section">
      <div class="section-title">Export video command</div>
      <div class="mini">Static HTML cannot run Python directly, but this command mirrors the current UI settings.</div>
      <div id="commandBox" class="cmd"></div>
    </div>
  </aside>

  <main>
    <div class="tabs">
      <button class="tab-btn active" data-tab="graphPanel">Graph replay</button>
      <button class="tab-btn" data-tab="featurePanel">Feature importance</button>
      <button class="tab-btn" data-tab="edgePanel">Edge importance</button>
    </div>
    <section id="graphPanel" class="tab-panel active">
      <svg id="graphSvg" role="img" aria-label="BLuR-GA graph evolution"></svg>
      <div class="legend">
        <div><b>Nodes</b>: features</div>
        <div><b>Edges</b>: interaction weight</div>
        <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--selected);"></span> selected feature</div>
      </div>
    </section>
    <section id="featurePanel" class="tab-panel">
      <div class="table-wrap">
        <div id="featureFormula" class="formula"></div>
        <table id="featureTable"></table>
      </div>
    </section>
    <section id="edgePanel" class="tab-panel">
      <div class="table-wrap">
        <div id="edgeFormula" class="formula"></div>
        <table id="edgeTable"></table>
      </div>
    </section>
  </main>

  <footer>
    <div class="stats-grid">
      <div class="stat"><div class="k">Generation</div><div class="v" id="statGen">-</div></div>
      <div class="stat"><div class="k">Shown edges</div><div class="v" id="statShown">-</div></div>
      <div class="stat"><div class="k">Total edges</div><div class="v" id="statTotal">-</div></div>
      <div class="stat"><div class="k">Mean weight</div><div class="v" id="statMean">-</div></div>
      <div class="stat"><div class="k">Max weight</div><div class="v" id="statMax">-</div></div>
      <div class="stat"><div class="k">Best fitness</div><div class="v" id="statBest">-</div></div>
      <div class="stat"><div class="k">Elapsed</div><div class="v" id="statElapsed">-</div></div>
      <div class="stat"><div class="k">Eval count</div><div class="v" id="statEval">-</div></div>
      <div class="stat"><div class="k">Best subset</div><div class="v" id="statSubset">-</div></div>
      <div class="stat"><div class="k">Unique pop</div><div class="v" id="statUnique">-</div></div>
    </div>
    <canvas id="timelineCanvas" height="54"></canvas>
  </footer>
</div>
<div id="tooltip"></div>
<script>
const DATA = {payload_json};
const NS = 'http://www.w3.org/2000/svg';
const svg = document.getElementById('graphSvg');
const tooltip = document.getElementById('tooltip');
let timer = null;
let currentIndex = 0;
let view = {{x: 0, y: 0, scale: 1}};
let isPanning = false;
let panStart = {{x: 0, y: 0, vx: 0, vy: 0}};

const els = {{
  slider: document.getElementById('generationSlider'),
  genBox: document.getElementById('generationBox'),
  prev: document.getElementById('prevBtn'),
  play: document.getElementById('playBtn'),
  next: document.getElementById('nextBtn'),
  reset: document.getElementById('resetBtn'),
  fitView: document.getElementById('fitViewBtn'),
  saveSvg: document.getElementById('saveSvgBtn'),
  speed: document.getElementById('speedSelect'),
  step: document.getElementById('stepBox'),
  loop: document.getElementById('loopCheck'),
  filterMode: document.getElementById('filterMode'),
  topKSlider: document.getElementById('topKSlider'),
  topKBox: document.getElementById('topKBox'),
  thresholdSlider: document.getElementById('thresholdSlider'),
  thresholdBox: document.getElementById('thresholdBox'),
  percentile: document.getElementById('percentileBox'),
  viewMode: document.getElementById('viewMode'),
  window: document.getElementById('windowBox'),
  showLabels: document.getElementById('showLabels'),
  showIsolated: document.getElementById('showIsolated'),
  edgeWidth: document.getElementById('edgeWidth'),
  edgeOpacity: document.getElementById('edgeOpacity'),
  spacingSlider: document.getElementById('spacingSlider'),
  spacingBox: document.getElementById('spacingBox'),
  nodeScaleSlider: document.getElementById('nodeScaleSlider'),
  nodeScaleBox: document.getElementById('nodeScaleBox'),
  focusNodeBox: document.getElementById('focusNodeBox'),
  focusIncidentOnly: document.getElementById('focusIncidentOnly'),
  weightScale: document.getElementById('weightScale')
}};

function initControls() {{
  els.slider.max = Math.max(0, DATA.generations.length - 1);
  els.genBox.min = DATA.generations[0] ?? 0;
  els.genBox.max = DATA.generations[DATA.generations.length - 1] ?? 0;
  const maxTopK = Math.max(1, Math.min(DATA.maxEdges || 100, 1000));
  els.topKSlider.max = maxTopK;
  els.topKSlider.value = Math.min(30, maxTopK);
  els.topKBox.max = maxTopK;
  els.topKBox.value = Math.min(30, maxTopK);
  const maxW = DATA.maxWeight || 1;
  els.thresholdSlider.max = String(maxW);
  els.thresholdSlider.step = String(Math.max(maxW / 1000, 1e-6));
  els.thresholdBox.step = String(Math.max(maxW / 1000, 1e-6));
  const spacing = DATA.initialSpacing || 1.15;
  els.spacingSlider.value = spacing;
  els.spacingBox.value = spacing;
  document.getElementById('headerInfo').textContent = `${{DATA.nNodes}} nodes | ${{DATA.generations.length}} saved generations | max weight=${{fmt(DATA.maxWeight)}}`;
}}

function fmt(x) {{
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '-';
  const v = Number(x);
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(2);
  return v.toFixed(4);
}}

function generation() {{ return DATA.generations[currentIndex]; }}
function rawEdgesForCurrent() {{ return DATA.edgesByGeneration[String(generation())] || []; }}
function focusNodeId() {{
  const raw = (els.focusNodeBox?.value || '').trim();
  if (!raw) return null;
  const id = Number.parseInt(raw, 10);
  if (!Number.isFinite(id) || id < 0 || id >= DATA.nNodes) return null;
  return id;
}}
function isIncident(e, nodeId) {{ return nodeId === null || e.i === nodeId || e.j === nodeId; }}

function edgesAfterView(raw) {{
  const gen = generation();
  const mode = els.viewMode.value;
  const window = Math.max(1, parseInt(els.window.value || '20'));
  if (mode === 'new') return raw.filter(e => e.first === gen);
  if (mode === 'window') return raw.filter(e => e.last >= gen - window + 1);
  return raw;
}}

function filteredEdges() {{
  let edges = edgesAfterView(rawEdgesForCurrent()).slice();
  const focus = focusNodeId();
  if (focus !== null && els.focusIncidentOnly.checked) edges = edges.filter(e => isIncident(e, focus));
  edges.sort((a, b) => b.w - a.w);
  const mode = els.filterMode.value;
  if (mode === 'topk') {{
    const k = Math.max(1, parseInt(els.topKBox.value || '30'));
    return edges.slice(0, k);
  }}
  if (mode === 'topk_per_node') {{
    const k = Math.max(1, parseInt(els.topKBox.value || '3'));
    const byNode = new Map();
    DATA.nodes.forEach(n => byNode.set(n.id, []));
    edges.forEach(e => {{
      if (!byNode.has(e.i)) byNode.set(e.i, []);
      if (!byNode.has(e.j)) byNode.set(e.j, []);
      byNode.get(e.i).push(e);
      byNode.get(e.j).push(e);
    }});
    const keep = new Map();
    byNode.forEach(list => {{
      list.sort((a,b) => b.w - a.w).slice(0, k).forEach(e => keep.set(`${{Math.min(e.i,e.j)}}-${{Math.max(e.i,e.j)}}`, e));
    }});
    return Array.from(keep.values()).sort((a,b) => b.w - a.w);
  }}
  if (mode === 'threshold') {{
    const th = Number(els.thresholdBox.value || 0);
    return edges.filter(e => e.w >= th);
  }}
  if (mode === 'percentile') {{
    if (!edges.length) return edges;
    const p = Math.max(0, Math.min(100, Number(els.percentile.value || 95)));
    const sorted = edges.map(e => e.w).sort((a,b) => a-b);
    const idx = Math.floor((p / 100) * (sorted.length - 1));
    const q = sorted[idx];
    return edges.filter(e => e.w >= q);
  }}
  return edges;
}}

function svgPoint(node, width, height) {{
  const margin = 52;
  const spacing = Number(els.spacingBox?.value || DATA.initialSpacing || 1.0);
  const nx = 0.5 + (node.x - 0.5) * spacing;
  const ny = 0.5 + (node.y - 0.5) * spacing;
  return [margin + nx * (width - 2 * margin), margin + ny * (height - 2 * margin)];
}}
function transformString() {{ return `translate(${{view.x}} ${{view.y}}) scale(${{view.scale}})`; }}
function fitView() {{ view = {{x: 0, y: 0, scale: 1}}; render(); }}
function zoomAt(clientX, clientY, factor) {{
  const rect = svg.getBoundingClientRect();
  const px = clientX - rect.left;
  const py = clientY - rect.top;
  const oldScale = view.scale;
  const newScale = Math.max(0.2, Math.min(8.0, oldScale * factor));
  const contentX = (px - view.x) / oldScale;
  const contentY = (py - view.y) / oldScale;
  view.x = px - contentX * newScale;
  view.y = py - contentY * newScale;
  view.scale = newScale;
  render();
}}

function clearSvg() {{ while (svg.firstChild) svg.removeChild(svg.firstChild); }}
function add(tag, attrs, parent=svg) {{
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {{}})) el.setAttribute(k, String(v));
  parent.appendChild(el);
  return el;
}}

function showTip(event, htmlText) {{
  tooltip.style.display = 'block';
  tooltip.innerHTML = htmlText;
  tooltip.style.left = (event.clientX + 14) + 'px';
  tooltip.style.top = (event.clientY + 14) + 'px';
}}
function hideTip() {{ tooltip.style.display = 'none'; }}
function activateTab(tabId) {{
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tabId));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.toggle('active', panel.id === tabId));
  if (tabId === 'graphPanel') render();
}}

function renderImportanceTables() {{
  const fMeta = DATA.importanceMeta?.feature || {{}};
  const eMeta = DATA.importanceMeta?.edge || {{}};
  const fFormula = document.getElementById('featureFormula');
  const eFormula = document.getElementById('edgeFormula');
  fFormula.innerHTML = `<b>Formula</b>: FI<sub>i</sub> = Σ<sub>r</sub> 1(i ∈ S<sub>r</sub>) · score<sub>r</sub> / Σ<sub>r</sub> score<sub>r</sub><br>score column: <b>${{escapeHtml(fMeta.scoreColumn || 'uniform')}}</b>; runs: <b>${{fMeta.nRuns || 0}}</b>. This is a fitness-weighted selection frequency.`;
  eFormula.innerHTML = `<b>Formula</b>: EI<sub>ij</sub> = (1/R) Σ<sub>r</sub> w<sub>ij,r</sub> / max<sub>e</sub>(w<sub>e,r</sub>), absent edge = 0<br>runs: <b>${{eMeta.nRuns || 0}}</b>. This rewards edges that are strong and repeatedly appear across runs.`;

  const fRows = DATA.featureImportance || [];
  const eRows = DATA.edgeImportance || [];
  const featureTable = document.getElementById('featureTable');
  const edgeTable = document.getElementById('edgeTable');
  featureTable.innerHTML = `<thead><tr><th>Rank</th><th>Feature</th><th>Label</th><th class="num">Importance</th><th class="num">Selected count</th><th class="num">Selection rate</th><th class="num">Mean selected score</th></tr></thead><tbody>${{fRows.map((r, idx) => `<tr><td>${{idx+1}}</td><td>${{r.feature}}</td><td>${{escapeHtml(r.label)}}</td><td class="num">${{fmt(r.importance)}}</td><td class="num">${{r.selected_count}}</td><td class="num">${{fmt(r.selection_rate)}}</td><td class="num">${{fmt(r.mean_score_when_selected)}}</td></tr>`).join('')}}</tbody>`;
  edgeTable.innerHTML = `<thead><tr><th>Rank</th><th>Edge</th><th>Label</th><th class="num">Importance</th><th class="num">Occurrence</th><th class="num">Occurrence rate</th><th class="num">Mean weight</th><th class="num">Max weight</th></tr></thead><tbody>${{eRows.map((r, idx) => `<tr><td>${{idx+1}}</td><td>${{r.edge}}</td><td>${{escapeHtml(r.label)}}</td><td class="num">${{fmt(r.importance)}}</td><td class="num">${{r.occurrence_count}}</td><td class="num">${{fmt(r.occurrence_rate)}}</td><td class="num">${{fmt(r.mean_weight_present)}}</td><td class="num">${{fmt(r.max_weight)}}</td></tr>`).join('')}}</tbody>`;
}}


function render() {{
  const width = svg.clientWidth || 800;
  const height = svg.clientHeight || 600;
  svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
  clearSvg();
  const raw = rawEdgesForCurrent();
  const shown = filteredEdges();
  const nodeById = new Map(DATA.nodes.map(n => [n.id, n]));
  const visibleNodes = new Set();
  shown.forEach(e => {{ visibleNodes.add(e.i); visibleNodes.add(e.j); }});

  const frameMax = shown.length ? Math.max(...shown.map(e => e.w)) : 1;
  const denom = Math.max(els.weightScale.value === 'global' ? DATA.maxWeight : frameMax, 1e-12);

  const weightedDegree = new Map(DATA.nodes.map(n => [n.id, 0]));
  shown.forEach(e => {{
    weightedDegree.set(e.i, (weightedDegree.get(e.i) || 0) + e.w);
    weightedDegree.set(e.j, (weightedDegree.get(e.j) || 0) + e.w);
  }});
  const maxDegree = Math.max(0, ...Array.from(weightedDegree.values()));

  const viewportGroup = add('g', {{id: 'viewportGroup', transform: transformString()}});
  const focus = focusNodeId();
  const edgeGroup = add('g', {{'class': 'edges'}}, viewportGroup);
  shown.forEach(e => {{
    const ni = nodeById.get(e.i), nj = nodeById.get(e.j);
    if (!ni || !nj) return;
    const [x1, y1] = svgPoint(ni, width, height);
    const [x2, y2] = svgPoint(nj, width, height);
    const norm = Math.max(0, Math.min(1, e.w / denom));
    const strokeWidth = els.edgeWidth.checked ? (0.5 + 6.0 * norm) : 1.3;
    let opacity = els.edgeOpacity.checked ? (0.08 + 0.84 * norm) : 0.55;
    const incident = isIncident(e, focus);
    if (focus !== null && !incident) opacity *= 0.16;
    const stroke = focus !== null && incident ? '#2563eb' : '#111827';
    const line = add('line', {{x1, y1, x2, y2, stroke, 'stroke-width': strokeWidth, 'stroke-opacity': opacity, 'stroke-linecap': 'round'}}, edgeGroup);
    line.addEventListener('mousemove', ev => showTip(ev, `<b>Edge ${{e.i}}–${{e.j}}</b><br>weight=${{fmt(e.w)}}<br>positive_count=${{e.pc}}<br>tested_count=${{e.tc}}<br>first_seen=${{e.first}}<br>last_updated=${{e.last}}`));
    line.addEventListener('mouseleave', hideTip);
  }});

  const nodeGroup = add('g', {{'class': 'nodes'}}, viewportGroup);
  DATA.nodes.forEach(n => {{
    if (!els.showIsolated.checked && !visibleNodes.has(n.id)) return;
    const [x, y] = svgPoint(n, width, height);
    const d = weightedDegree.get(n.id) || 0;
    const nodeScale = Number(els.nodeScaleBox?.value || 1.0);
    const r = (maxDegree > 0 ? 8 + 12 * Math.sqrt(d / maxDegree) : 8) * nodeScale;
    const isFocus = focus !== null && n.id === focus;
    const isNeighbor = focus !== null && shown.some(e => (e.i === focus && e.j === n.id) || (e.j === focus && e.i === n.id));
    const fill = isFocus ? '#ef4444' : (n.selected ? '#f59e0b' : '#ffffff');
    const stroke = isFocus ? '#991b1b' : (isNeighbor ? '#2563eb' : (n.selected ? '#92400e' : '#111827'));
    const nodeOpacity = focus !== null && !isFocus && !isNeighbor ? 0.45 : 1.0;
    const c = add('circle', {{cx: x, cy: y, r, fill, stroke, 'stroke-width': isFocus ? 2.6 : 1.4, opacity: nodeOpacity}}, nodeGroup);
    c.addEventListener('mousemove', ev => showTip(ev, `<b>Node ${{n.id}}</b>: ${{escapeHtml(n.label)}}<br>shown degree weight=${{fmt(d)}}<br>selected=${{n.selected ? 'yes' : 'no'}}`));
    c.addEventListener('mouseleave', hideTip);
    if (els.showLabels.checked) {{
      const t = add('text', {{x, y: y + 3, 'text-anchor': 'middle', 'font-size': DATA.nNodes <= 40 ? 10 : 8, 'font-weight': 650, 'pointer-events': 'none', fill: '#111827'}}, nodeGroup);
      t.textContent = n.label;
    }}
  }});

  updateStats(raw, shown);
  updateGenerationControls();
  drawTimeline();
  updateCommandBox();
}}

function escapeHtml(s) {{ return String(s).replace(/[&<>'"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#039;','"':'&quot;'}}[c])); }}

function updateStats(raw, shown) {{
  const gen = generation();
  const weights = shown.map(e => e.w);
  const mean = weights.length ? weights.reduce((a,b)=>a+b,0) / weights.length : 0;
  const maxW = weights.length ? Math.max(...weights) : 0;
  const tr = DATA.traceByGeneration[String(gen)] || {{}};
  document.getElementById('statGen').textContent = gen;
  document.getElementById('statShown').textContent = shown.length;
  document.getElementById('statTotal').textContent = raw.length;
  document.getElementById('statMean').textContent = fmt(mean);
  document.getElementById('statMax').textContent = fmt(maxW);
  document.getElementById('statBest').textContent = tr.best_so_far_fitness === undefined ? '-' : fmt(tr.best_so_far_fitness);
  document.getElementById('statElapsed').textContent = tr.elapsed_seconds === undefined ? '-' : `${{fmt(tr.elapsed_seconds)}}s`;
  document.getElementById('statEval').textContent = tr.eval_count === undefined ? '-' : tr.eval_count;
  document.getElementById('statSubset').textContent = tr.best_so_far_subset_size === undefined ? '-' : tr.best_so_far_subset_size;
  document.getElementById('statUnique').textContent = tr.population_unique_chromosomes === undefined ? '-' : tr.population_unique_chromosomes;
}}

function updateGenerationControls() {{
  els.slider.value = currentIndex;
  els.genBox.value = generation();
}}

function setIndex(idx) {{
  currentIndex = Math.max(0, Math.min(DATA.generations.length - 1, idx));
  render();
}}
function setGenerationValue(gen) {{
  let idx = DATA.generations.findIndex(g => g >= gen);
  if (idx < 0) idx = DATA.generations.length - 1;
  setIndex(idx);
}}

function play() {{
  if (timer) return;
  els.play.textContent = '⏸ Pause';
  const tick = () => {{
    const step = Math.max(1, parseInt(els.step.value || '1'));
    let nxt = currentIndex + step;
    if (nxt >= DATA.generations.length) {{
      if (els.loop.checked) nxt = 0;
      else {{ pause(); return; }}
    }}
    setIndex(nxt);
  }};
  const delay = Math.max(25, 500 / Number(els.speed.value || 1));
  timer = setInterval(tick, delay);
}}
function pause() {{
  if (timer) clearInterval(timer);
  timer = null;
  els.play.textContent = '▶ Play';
}}
function togglePlay() {{ timer ? pause() : play(); }}

function linkNumberAndSlider(slider, box, callback) {{
  slider.addEventListener('input', () => {{ box.value = slider.value; callback(); }});
  box.addEventListener('input', () => {{ slider.value = box.value; callback(); }});
}}

function updateCommandBox() {{
  const mode = els.filterMode.value;
  let filterArgs = `--filter-mode ${{mode}}`;
  if (mode === 'topk' || mode === 'topk_per_node') filterArgs += ` --top-k ${{els.topKBox.value}}`;
  if (mode === 'threshold') filterArgs += ` --threshold ${{els.thresholdBox.value}}`;
  if (mode === 'percentile') filterArgs += ` --percentile ${{els.percentile.value}}`;
  let viewArgs = `--view-mode ${{els.viewMode.value}}`;
  if (els.viewMode.value === 'window') viewArgs += ` --window ${{els.window.value}}`;
  document.getElementById('commandBox').textContent = `python analysis_interactive/make_graph_video.py \\\n  --snapshots path/to/graph_snapshots_*.csv \\\n  --trace path/to/generation_trace_*.csv \\\n  --output graph_evolution.mp4 \\\n  ${{filterArgs}} \\\n  ${{viewArgs}} \\\n  --fps 6 --step ${{els.step.value}}`;
}}

function drawTimeline() {{
  const canvas = document.getElementById('timelineCanvas');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(54 * dpr));
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width, h = 54;
  ctx.clearRect(0, 0, w, h);
  const data = DATA.timeline;
  if (!data.length) return;
  const vals = data.map(d => d.n_edges || 0);
  const maxV = Math.max(...vals, 1);
  ctx.strokeStyle = '#d1d5db'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, h - 10); ctx.lineTo(w, h - 10); ctx.stroke();
  ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((d, idx) => {{
    const x = data.length === 1 ? 0 : idx / (data.length - 1) * w;
    const y = h - 10 - (d.n_edges || 0) / maxV * (h - 18);
    if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.stroke();
  const xcur = DATA.generations.length === 1 ? 0 : currentIndex / (DATA.generations.length - 1) * w;
  ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(xcur, 2); ctx.lineTo(xcur, h - 2); ctx.stroke();
  ctx.fillStyle = '#6b7280'; ctx.font = '11px sans-serif';
  ctx.fillText('edge count timeline', 4, 12);
}}

function saveCurrentSvg() {{
  const serializer = new XMLSerializer();
  const source = serializer.serializeToString(svg);
  const blob = new Blob([source], {{type: 'image/svg+xml;charset=utf-8'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `BLuR-GA_graph_gen_${{generation()}}.svg`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}

initControls();
renderImportanceTables();
document.querySelectorAll('.tab-btn').forEach(btn => btn.addEventListener('click', () => activateTab(btn.dataset.tab)));
linkNumberAndSlider(els.topKSlider, els.topKBox, render);
linkNumberAndSlider(els.thresholdSlider, els.thresholdBox, render);
linkNumberAndSlider(els.spacingSlider, els.spacingBox, render);
linkNumberAndSlider(els.nodeScaleSlider, els.nodeScaleBox, render);
els.slider.addEventListener('input', () => setIndex(parseInt(els.slider.value || '0')));
els.genBox.addEventListener('input', () => setGenerationValue(parseInt(els.genBox.value || '0')));
els.prev.addEventListener('click', () => setIndex(currentIndex - 1));
els.next.addEventListener('click', () => setIndex(currentIndex + 1));
els.reset.addEventListener('click', () => setIndex(0));
els.fitView.addEventListener('click', fitView);
els.play.addEventListener('click', togglePlay);
els.saveSvg.addEventListener('click', saveCurrentSvg);
['speedSelect','stepBox','filterMode','percentileBox','viewMode','windowBox','showLabels','showIsolated','edgeWidth','edgeOpacity','focusNodeBox','focusIncidentOnly','weightScale','loopCheck'].forEach(id => {{
  document.getElementById(id).addEventListener('input', () => {{ if (id === 'speedSelect' && timer) {{ pause(); play(); }} render(); }});
  document.getElementById(id).addEventListener('change', () => {{ if (id === 'speedSelect' && timer) {{ pause(); play(); }} render(); }});
}});
svg.addEventListener('wheel', ev => {{
  ev.preventDefault();
  zoomAt(ev.clientX, ev.clientY, ev.deltaY < 0 ? 1.12 : 1 / 1.12);
}}, {{passive: false}});
svg.addEventListener('mousedown', ev => {{
  if (ev.button !== 0) return;
  isPanning = true;
  panStart = {{x: ev.clientX, y: ev.clientY, vx: view.x, vy: view.y}};
  svg.classList.add('dragging');
}});
window.addEventListener('mousemove', ev => {{
  if (!isPanning) return;
  view.x = panStart.vx + ev.clientX - panStart.x;
  view.y = panStart.vy + ev.clientY - panStart.y;
  render();
}});
window.addEventListener('mouseup', () => {{ isPanning = false; svg.classList.remove('dragging'); }});
svg.addEventListener('dblclick', fitView);
document.getElementById('timelineCanvas').addEventListener('click', ev => {{
  const rect = ev.currentTarget.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (ev.clientX - rect.left) / Math.max(rect.width, 1)));
  setIndex(Math.round(frac * (DATA.generations.length - 1)));
}});
window.addEventListener('resize', render);
render();
</script>
</body>
</html>"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a compact interactive HTML UI for BLuR-GA graph evolution snapshots.")
    p.add_argument("--snapshots", required=True, help="graph_snapshots_*.csv produced by the BLuR-GA runner")
    p.add_argument("--trace", default=None, help="Optional generation_trace_*.csv for fitness overlays and timeline")
    p.add_argument("--selected-features", default=None, help="Optional selected_features_*.csv to highlight selected nodes and compute feature importance. Auto-discovered when omitted if possible.")
    p.add_argument("--run-summary", default=None, help="Optional nested_summary_*.csv containing run-level fitness/outer_score for feature importance. Auto-discovered when omitted if possible.")
    p.add_argument("--score-column", default=None, help="Fitness/score column in --run-summary. Default auto: outer_score, then best_inner_fitness, etc.")
    p.add_argument("--importance-snapshots", nargs="*", default=None, help="Snapshot CSVs used for multi-run edge importance. Default auto-discovers graph_snapshots_*_r*.csv next to --snapshots.")
    p.add_argument("--run-id", type=int, default=None, help="Optional run_id filter")
    p.add_argument("--repeat-id", type=int, default=None, help="Optional repeat_id filter")
    p.add_argument("--outer-fold", type=int, default=None, help="Optional outer_fold filter")
    p.add_argument("--n-nodes", type=int, default=None, help="Number of feature nodes. Usually inferred from snapshots.")
    p.add_argument("--labels", default=None, help="Optional CSV with node_id,label or feature_index,feature_name columns")
    p.add_argument("--layout", choices=["spring", "circular", "spectral"], default="spring")
    p.add_argument("--layout-seed", type=int, default=1)
    p.add_argument("--layout-k-scale", type=float, default=1.8, help="Spring-layout spacing multiplier. 1.0 approximates NetworkX default; larger values separate components more.")
    p.add_argument("--layout-iterations", type=int, default=300, help="Number of spring-layout optimization iterations.")
    p.add_argument("--initial-spacing", type=float, default=1.15, help="Initial browser-side spacing multiplier; can still be changed in the UI.")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--step", type=int, default=1, help="Pre-subsample saved generations for lighter HTML files.")
    p.add_argument("--title", default="BLuR-GA Interactive Graph Evolution Viewer")
    p.add_argument("--output", required=True, help="Output .html path")
    p.add_argument("--open", action="store_true", help="Open the generated HTML in the default browser")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    related = auto_related_paths(args.snapshots)
    selected_features_path = args.selected_features or related.get("selected_features")
    run_summary_path = args.run_summary or related.get("run_summary")
    importance_snapshots = args.importance_snapshots or related.get("importance_snapshots") or [args.snapshots]

    data = load_graph_evolution(
        args.snapshots,
        trace_path=args.trace,
        n_nodes=args.n_nodes,
        labels_path=args.labels,
        run_id=args.run_id,
    )
    snapshots = _filter_dataframe(data.snapshots, run_id=args.run_id, repeat_id=args.repeat_id, outer_fold=args.outer_fold)
    trace = _filter_dataframe(data.trace, run_id=args.run_id, repeat_id=args.repeat_id, outer_fold=args.outer_fold) if data.trace is not None else None
    selected = load_selected_features(selected_features_path, run_id=args.run_id, repeat_id=args.repeat_id, outer_fold=args.outer_fold)
    selected_df = pd.read_csv(selected_features_path) if selected_features_path is not None and Path(selected_features_path).exists() else None
    run_summary = load_run_summary(run_summary_path)
    score_column = choose_score_column(run_summary, args.score_column)
    feature_importance, feature_meta = compute_feature_importance(
        selected_df,
        n_nodes=data.n_nodes,
        labels=data.labels,
        run_summary=run_summary,
        score_column=score_column,
    )
    edge_importance, edge_meta = compute_edge_importance(importance_snapshots, labels=data.labels)
    importance_meta = {"feature": feature_meta, "edge": edge_meta}
    if snapshots.empty:
        raise ValueError("No snapshot rows remain after applying run/repeat/fold filters.")

    payload = _prepare_payload(
        snapshots,
        trace,
        n_nodes=data.n_nodes,
        labels=data.labels,
        selected_features=selected,
        layout_name=args.layout,
        layout_seed=args.layout_seed,
        layout_k_scale=args.layout_k_scale,
        layout_iterations=args.layout_iterations,
        initial_spacing=args.initial_spacing,
        start=args.start,
        end=args.end,
        step=args.step,
        title=args.title,
        feature_importance=feature_importance,
        edge_importance=edge_importance,
        importance_meta=importance_meta,
    )
    source_hint = f"snapshots={Path(args.snapshots).name}"
    html_text = render_html(payload, source_hint=source_hint)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    print(f"Saved interactive UI: {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
