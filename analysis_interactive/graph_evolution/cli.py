from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_graph_evolution
from .metrics import snapshot_metrics
from .renderer import GraphEvolutionRenderer, generations_for_animation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render BLuR-GA linkage graph snapshots as an MP4/GIF animation.")
    p.add_argument("--snapshots", required=True, help="graph_snapshots_*.csv produced by the BLuR-GA runner")
    p.add_argument("--trace", default=None, help="Optional generation_trace_*.csv for fitness overlays and full generation range")
    p.add_argument("--run-id", type=int, default=None, help="Optional run_id filter for the trace file")
    p.add_argument("--n-nodes", type=int, default=None, help="Number of feature nodes. Usually inferred from snapshots.")
    p.add_argument("--labels", default=None, help="Optional CSV with node_id,label or feature_index,feature_name columns")
    p.add_argument("--output", required=True, help="Output animation path: .mp4 or .gif")
    p.add_argument("--frame-output", default=None, help="Optional PNG path for saving one frame instead of/alongside animation")
    p.add_argument("--frame-generation", type=int, default=None, help="Generation to save as PNG when --frame-output is provided")

    p.add_argument("--filter-mode", choices=["topk", "topk_per_node", "threshold", "percentile", "all"], default="topk")
    p.add_argument("--top-k", type=int, default=40, help="Global top-k edges, or k incident edges per node when --filter-mode topk_per_node.")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--view-mode", choices=["cumulative", "new", "window"], default="cumulative")
    p.add_argument("--window", type=int, default=20)

    p.add_argument("--layout", choices=["spring", "circular", "spectral"], default="spring")
    p.add_argument("--layout-seed", type=int, default=1)
    p.add_argument("--weight-scale", choices=["global", "frame"], default="global")
    p.add_argument("--show-labels", choices=["auto", "on", "off"], default="auto")

    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--step", type=int, default=1, help="Generation step per rendered frame. Larger = faster playback.")
    p.add_argument("--fps", type=int, default=6, help="Video frames per second. Larger = faster playback.")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--width", type=float, default=9.0)
    p.add_argument("--height", type=float, default=7.0)
    p.add_argument("--title", default="BLuR-GA linkage graph evolution")
    p.add_argument("--metrics-output", default=None, help="Optional CSV path for per-snapshot graph metrics")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data = load_graph_evolution(
        args.snapshots,
        trace_path=args.trace,
        n_nodes=args.n_nodes,
        labels_path=args.labels,
        run_id=args.run_id,
    )
    renderer = GraphEvolutionRenderer(
        data.snapshots,
        n_nodes=data.n_nodes,
        labels=data.labels,
        trace=data.trace,
        layout_name=args.layout,
        layout_seed=args.layout_seed,
        global_weight_scale=args.weight_scale == "global",
    )
    generations = generations_for_animation(data.snapshots, data.trace, start=args.start, end=args.end, step=args.step)
    out = renderer.save_animation(
        args.output,
        generations,
        fps=args.fps,
        filter_mode=args.filter_mode,
        top_k=args.top_k,
        threshold=args.threshold,
        percentile=args.percentile,
        view_mode=args.view_mode,
        window=args.window,
        show_labels=args.show_labels,
        title=args.title,
        dpi=args.dpi,
        width=args.width,
        height=args.height,
    )
    print(f"Saved animation: {out}")

    if args.frame_output:
        frame_generation = args.frame_generation if args.frame_generation is not None else generations[-1]
        frame = renderer.save_frame(
            args.frame_output,
            frame_generation,
            filter_mode=args.filter_mode,
            top_k=args.top_k,
            threshold=args.threshold,
            percentile=args.percentile,
            view_mode=args.view_mode,
            window=args.window,
            show_labels=args.show_labels,
            title=args.title,
        )
        print(f"Saved frame: {frame}")

    if args.metrics_output:
        metrics = snapshot_metrics(data.snapshots)
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
