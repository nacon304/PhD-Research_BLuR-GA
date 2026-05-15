# Generate Data

## Benchmark for Linkage Learning

### 1. Generate benchmark datasets

```bash
python scripts/prepare_linkage_benchmarks.py `
  --output-root ../Dataset/prepared_linkage_benchmark `
  --preset main `
  --families all `
  --seeds 0 1 2 3 4 5 6 7 8 9 `
  --overwrite-manifest
```

### 2. Plot ground-truth graphs

```bash
python scripts/plot_ground_truth_graphs.py `
  --dataset-root ../Dataset/prepared_linkage_benchmark `
  --layout auto `
  --scale 1.8 `
  --signed `
  --write-metrics `
  --format png
```

## Output Structure

```text
prepared_blurga/
  manifest.csv
  linkage/
    <dataset_key>/
      metadata.json
      problem.json
      feature_names.json
      node_labels.csv
      ground_truth_edges.csv
      ground_truth_matrix.csv
      ground_truth_graph_snapshots.csv
      dataset.npz   # only for supervised synthetic datasets
```

## Supported Benchmark Families

- Pairwise pseudo-Boolean / QUBO
- Ising / Spin-glass
- Planted XOR / Parity
- NK Landscape
- MAX-SAT
- GAMETES-style Epistasis

## Useful Graph Plot Options

```bash
--layout auto|spring|kamada|spectral|circular|grid
--scale 1.8
--spring-k 0.5
--iterations 300
--edge-filter all|topk|threshold|percentile
--top-k 100
--min-abs-weight 0.3
--label-mode auto|on|off
--signed
```

## Notes

- QUBO / Ising / NK / MAX-SAT are landscape problems using `problem.json`
- Planted XOR and GAMETES-style datasets also include `dataset.npz`
- Ground-truth graph snapshots follow the same schema as BLuR-GA graph snapshots
