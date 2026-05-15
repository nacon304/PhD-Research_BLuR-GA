# BLuR-GA Graph Evolution Analysis

This folder renders the generation-by-generation eVIGw snapshots exported by the
BLuR-GA runner.

## 1. Produce graph snapshots during optimization

Example:

```bash
python run_blur_ga.py zoo 1 1 \
  --data-dir . \
  --output-dir demo_graph_logs \
  --outer-folds 2 \
  --inner-folds 2 \
  --repeats 1 \
  --outer-fold-id 0 \
  --repeat-id 0 \
  --popsize 30 \
  --max-gen 30 \
  --save-generation-trace true \
  --save-linkage-events true \
  --save-graph-snapshots true \
  --graph-snapshot-interval 1
```

Important exported files:

- `graph_snapshots_*.csv`: cumulative edge list at each saved generation.
- `linkage_events_*.csv`: one row per tested pair in linkage mutation.
- `eVIG_edges_*.csv`: final edge list.
- `generation_trace_*.csv`: fitness and edge counts over generations.

## 2. Render a video

```bash
python analysis_interactive/make_graph_video.py \
  --snapshots demo_graph_logs/graph_snapshots_zoo_c1_a1_r0.csv \
  --trace demo_graph_logs/generation_trace_zoo_c1_a1.csv \
  --run-id 0 \
  --output demo_graph_logs/graph_evolution_topk.mp4 \
  --filter-mode topk \
  --top-k 30 \
  --fps 6 \
  --step 1 \
  --layout spring
```

Useful options:

- `--filter-mode topk --top-k 30`: show only the strongest 30 edges.
- `--filter-mode threshold --threshold 0.05`: show all edges with weight >= 0.05.
- `--view-mode cumulative`: graph accumulated up to generation t.
- `--view-mode new`: only edges first discovered at generation t.
- `--view-mode window --window 20`: only recently updated edges.
- `--fps`: playback speed.
- `--step`: generations skipped per frame. Larger values make the video faster.
- `--frame-output frame.png`: additionally save one static frame.

## 3. Build a compact interactive HTML viewer

This is the lightweight UI version. It does **not** require Dash or Streamlit.
It creates one self-contained `.html` file with embedded graph data and a vanilla
JavaScript/SVG viewer.

```bash
python analysis_interactive/make_graph_ui.py \
  --snapshots demo_graph_logs/graph_snapshots_zoo_c1_a1_r0.csv \
  --trace demo_graph_logs/generation_trace_zoo_c1_a1.csv \
  --selected-features demo_graph_logs/selected_features_zoo_c1_a1.csv \
  --run-id 0 \
  --repeat-id 0 \
  --outer-fold 0 \
  --output demo_graph_logs/graph_evolution_ui.html \
  --layout spring
```

The UI supports:

- generation slider
- Play / Pause / Reset replay
- playback speed and step-per-frame controls
- edge filter by `top-k`, `threshold`, `percentile`, or `all`
- view mode: cumulative, new edges only, or sliding window
- edge width and opacity proportional to interaction weight
- selected feature highlighting when `selected_features_*.csv` is provided
- current-frame SVG export from the browser
- a generated video-export command matching the current UI settings

For large runs, use `--step 5` or `--step 10` to pre-subsample generations and
keep the HTML file lighter.

## 4. Keep video export as an option

The interactive UI is for exploration; `make_graph_video.py` remains the stable
way to export MP4/GIF videos:

```bash
python analysis_interactive/make_graph_video.py \
  --snapshots demo_graph_logs/graph_snapshots_zoo_c1_a1_r0.csv \
  --trace demo_graph_logs/generation_trace_zoo_c1_a1.csv \
  --run-id 0 \
  --output demo_graph_logs/graph_evolution_topk.mp4 \
  --filter-mode topk \
  --top-k 30 \
  --fps 6 \
  --step 1 \
  --layout spring
```

## Update: compact UI tabs and per-node top-k filtering

### Main graph tab

The graph replay now supports an additional edge filter:

```bash
--filter-mode topk_per_node --top-k 2
```

This keeps the union of the strongest `k` incident edges for every node in the current frame. It is useful when the global top-k filter hides peripheral nodes, while `all` or low thresholds make the graph too dense.

In the HTML UI, select:

```text
Filter mode = Top-k incident edges per node
Top-k / k per node = k
```

### Feature importance tab

The feature-importance score is computed from the selected-feature file and the run-level summary file:

```text
FI_i = sum_r 1(i in S_r) * score_r / sum_r score_r
```

where `S_r` is the selected subset in run `r`, and `score_r` is selected automatically from `outer_score`, then `best_inner_fitness`, then other common score columns. You can override it with:

```bash
--score-column outer_score
```

If no run-summary score is available, the UI falls back to uniform `score_r = 1`, so the score becomes the normal selection frequency.

### Edge importance tab

The edge-importance score is computed from final graph snapshots across runs:

```text
EI_ij = (1/R) * sum_r w_ij,r / max_e(w_e,r)
```

A missing edge in a run contributes zero. This score rewards edges that are both strong and repeatedly observed across runs.

By default, the UI auto-discovers files matching:

```text
graph_snapshots_<dataset>_c<classifier>_a<ga_type>_r*.csv
selected_features_<dataset>_c<classifier>_a<ga_type>.csv
nested_summary_<dataset>_c<classifier>_a<ga_type>.csv
```

You can override the multi-run snapshot set manually:

```bash
python analysis_interactive/make_graph_ui.py \
  --snapshots demo_graph_logs/graph_snapshots_zoo_c1_a1_r0.csv \
  --importance-snapshots demo_graph_logs/graph_snapshots_zoo_c1_a1_r*.csv \
  --output graph_evolution_ui.html
```
