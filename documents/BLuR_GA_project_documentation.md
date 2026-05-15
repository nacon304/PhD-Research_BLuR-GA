# BLuR-GA Project Documentation

Tài liệu này giải thích project BLuR-GA theo hướng đọc code: command chạy ra config nào, các package/file gọi nhau ra sao, từng tham số chính có ý nghĩa gì, output nào được sinh ra, và từng hàm trong mỗi file đang làm nhiệm vụ gì.

---

## 1. Command bạn đưa sẽ chạy với config nào?

Command:

```powershell
python run_blur_ga.py batch `
  --preset analysis `
  --data-dir ../Dataset/prepared_tabarena `
  --output-root results_analysis_test_v2 `
  --datasets anneal_task363614 `
  --classifiers 2 `
  --ga-types 2
```

### 1.1. Cách command được xử lý

Luồng xử lý trong `run_blur_ga.py`:

```text
main()
  -> build_parser().parse_args()
  -> cmd_batch(args)
       -> apply_preset(args)
       -> fill_defaults(args)
       -> build_jobs(split_folds=False)
       -> common_oop_args(args, job)
       -> run_blur_ga_oop.main(oop_args)
```

Vì bạn dùng subcommand `batch`, code sẽ chạy tuần tự trong một Python process. Nó **không** tách từng fold/repeat thành task riêng. Tách fold/repeat chỉ xảy ra với `make-tasks` khi `split_folds=True`.

### 1.2. Config cuối cùng


| Nhóm | Giá trị thực tế |
|---|---|
| Subcommand | `batch` |
| Preset | `analysis` |
| Dataset được chọn | `anneal_task363614` |
| Data dir | `../Dataset/prepared_tabarena` |
| Output root | `results_analysis_test_v2` |
| Classifier | `2` = KNN-5 theo `GAConfig.knn_k` |
| GA type | `2` = `pairwise_lr`, thư mục method `blur_ga_stage1_pairwise` |
| Số dataset-method jobs | `1` job, vì batch không tách fold/repeat thành task riêng |
| Output dir của job | `results_analysis_test_v2/anneal_task363614/blur_ga_stage1_pairwise` |
| Repeats | `2` từ preset `analysis` |
| Outer folds | `2` từ preset `analysis` |
| Inner folds | `3` từ preset `analysis` |
| Tổng evaluated runs | `2 repeats × 2 outer folds = 4 runs` |
| Run IDs | `r0=(repeat0, fold0)`, `r1=(repeat0, fold1)`, `r2=(repeat1, fold0)`, `r3=(repeat1, fold1)` |
| Seeds thực tế | `1`, `1001`, `100001`, `101001` theo công thức `seed + 100000*repeat + 1000*outer_fold` |
| Population size | `100` |
| Stop criterion | `gen` |
| Max generations | `200` |
| Standard crossover probability | `0.4` |
| Linkage-learning crossover ratio | không truyền nên OOP runner đặt bằng `p_cross = 0.4` |
| Mutation probability | không truyền, engine dùng `1 / n_features` |
| Tournament size | `3` |
| Tau reset | `50` generations không cải thiện thì restart population |
| Reset population rate | `0.99` |
| Local search | `False` |
| Fitness cache | bật, vì không có `--no-fitness-cache` |
| Fitness formula | `0.98 * mean_inner_KNN_score + 0.02 * sparsity` |
| LR gap generation | `5` |
| LR min samples | `20` unique evaluated chromosomes trước khi fit graph |
| LR ridge alpha | `1e-6` |
| Sparse alpha / L1 ratio | có giá trị default nhưng không thực sự quan trọng cho `ga_type=2` |
| LR edge min weight | `0.0` |
| LR edge top-k | `None`, không giới hạn số edge cuối bằng top-k |
| Save generation trace | `True` |
| Save graph snapshots | `True` |
| Graph snapshot interval | `1`, lưu mỗi generation |
| Graph snapshot top-k | `None`, không giới hạn top-k snapshot |
| Graph snapshot min weight | `0.0` |
| Save linkage events | `False` |
| Artifact layout theo file paste | `nested` |
| Write aggregate outputs theo file paste | `False` |


### 1.3. Command thực tế được chuyển sang `run_blur_ga_oop.py`

```bash
python run_blur_ga_oop.py anneal_task363614 2 2 \
  --data-dir ../Dataset/prepared_tabarena \
  --output-dir results_analysis_test_v2/anneal_task363614/blur_ga_stage1_pairwise \
  --outer-folds 2 \
  --inner-folds 3 \
  --repeats 2 \
  --seed 1 \
  --shuffle true \
  --popsize 100 \
  --p-cross 0.4 \
  --tournament-size 3 \
  --tau-reset 50 \
  --resetpop-rate 0.99 \
  --local-search false \
  --stop gen \
  --max-gen 200 \
  --edge-epsilon 1e-06 \
  --save-generation-trace true \
  --save-linkage-events false \
  --save-graph-snapshots true \
  --graph-snapshot-interval 1 \
  --graph-snapshot-min-weight 0.0 \
  --lr-gap-gen 5 \
  --lr-min-samples 20 \
  --lr-ridge-alpha 1e-06 \
  --lr-sparse-alpha 0.001 \
  --lr-l1-ratio 0.95 \
  --lr-stability-subsamples 0 \
  --lr-stability-fraction 0.75 \
  --lr-edge-min-weight 0.0 \
  --lr-excess-window 5 \
  --artifact-layout nested \
  --write-aggregate-outputs false
```

### 1.4. Output dự kiến của command này

Vì `artifact_layout=nested` và `write_aggregate_outputs=false`, sau khi chạy `batch` bạn sẽ chủ yếu có output trong thư mục nested per-run:

```text
results_analysis_test_v2/
└── anneal_task363614/
    └── blur_ga_stage1_pairwise/
        ├── repeat_00/
        │   ├── outer_fold_00/run_000/
        │   └── outer_fold_01/run_001/
        └── repeat_01/
            ├── outer_fold_00/run_002/
            └── outer_fold_01/run_003/
```

Mỗi `run_xxx/` sẽ có các file quan trọng như:

```text
run_summary_anneal_task363614_c2_a2_rX.csv
selected_features_anneal_task363614_c2_a2_rX.csv
generation_trace_anneal_task363614_c2_a2_rX.csv
graph_snapshots_anneal_task363614_c2_a2_rX.csv
eVIG_anneal_task363614_c2_a2_rX.csv
eVIG_coefficients_anneal_task363614_c2_a2_rX.csv
eVIG_edges_anneal_task363614_c2_a2_rX.csv
eVIG_tested_pairs_anneal_task363614_c2_a2_rX.csv
```

Do `write_aggregate_outputs=false`, root-level aggregate files như `nested_summary_*.csv`, `generation_trace_*.csv`, `selected_features_*.csv` **chưa có ngay**. Muốn có chúng thì chạy sau cùng:

```bash
python run_blur_ga.py aggregate --results-root results_analysis_test_v2
```

Lưu ý: `aggregate` gom lại summary/vector/selected_features/generation_trace ở root-level, nhưng **không gom lại `graph_snapshots_*_r*.csv`**. Graph snapshots vẫn nằm trong từng thư mục nested run.

---

## 2. Kiến trúc tổng quan project

Luồng chạy chính:

```text
run_blur_ga.py                 # launcher: batch/make-tasks/run-task/aggregate
  -> run_blur_ga_oop.py         # runner trực tiếp cho một dataset-method setting
      -> blur_ga.data           # đọc dataset
      -> blur_ga.config         # tạo GAConfig/ExperimentConfig
      -> blur_ga.experiment     # nested CV
          -> blur_ga.splitting  # outer/inner folds
          -> blur_ga.evaluator  # wrapper fitness trên inner CV
          -> blur_ga.engine     # GA / linkage-aware GA core
              -> blur_ga.evig hoặc blur_ga.lr_linkage
          -> blur_ga.results    # ghi output CSV

analysis_interactive/           # đọc graph_snapshots + trace để vẽ UI/video
scripts/prepare_existing_tabarena.py # chuẩn bị dataset.npz từ dữ liệu TabArena/OpenML đã tải
hpc/                            # shell scripts cho SLURM/NeSI
```

---

## 3. Ý nghĩa các `ga_type`


| `ga_type` | Method folder | Graph object | Linkage source | Ý nghĩa |
|---:|---|---|---|---|
| 0 | `standard_ga` | None | Không có | GA baseline: tournament + uniform crossover + mutation. |
| 1 | `empirical_linkage_legacy` | `EmpiricalVIG` | Linkage mutation test cặp feature | Học edge từ omega interaction khi thử g/h/gh quanh parent. |
| 2 | `blur_ga_stage1_pairwise` | `RegressionVIG` | Pairwise regression | Fit `fitness ~ x_i x_j`, lấy `|β_ij|` làm weight. |
| 3 | `blur_ga_stage2_main_pairwise` | `RegressionVIG` | Main + pairwise regression | Fit `fitness ~ x_i + x_i x_j`, giảm omitted main-effect bias. |
| 4 | `blur_ga_stage3_sparse` | `RegressionVIG` | ElasticNet/LASSO | Sparse model để giữ ít edge đáng tin hơn. |
| 5 | `blur_ga_stage4_sparse_excess` | `RegressionVIG` | Sparse + excess response | Fit response đã baseline-standardized theo rolling generation window. |


---

## 4. Toàn bộ nhóm tham số command-line quan trọng

### 4.1. Shared options của `run_blur_ga.py batch` và `make-tasks`

| Tham số | Ý nghĩa |
|---|---|
| `--preset` | Preset `smoke`, `analysis`, `nesi_full`; chỉ điền vào option còn None. |
| `--data-dir` | Thư mục chứa dataset prepared folders hoặc .dat. |
| `--output-root` | Root output cho batch/make-tasks. |
| `--datasets` | Danh sách dataset hoặc `all`. |
| `--dataset-file` | File text/CSV chứa danh sách dataset. |
| `--exclude-datasets` | Loại một số dataset. |
| `--limit-datasets` | Giới hạn số dataset đầu tiên sau resolve. |
| `--classifiers` | Danh sách classifier id: 1=KNN-3, 2=KNN-5. |
| `--ga-types` | Danh sách method id 0..5. |
| `--repeats` | Số lần lặp outer CV. |
| `--outer-folds` | Số outer folds. |
| `--inner-folds` | Số inner folds dùng tính fitness. |
| `--seed` | Base seed. |
| `--shuffle` | Có shuffle trước khi split hay không. |
| `--popsize` | Population size. |
| `--p-cross` | Xác suất crossover chuẩn. |
| `--ll-crossover-ratio` | Tỷ lệ offspring dùng crossover trong empirical linkage; None thì OOP dùng p-cross. |
| `--mutation-prob` | Xác suất flip bit; None thì engine dùng 1/n_features. |
| `--tournament-size` | Kích thước tournament selection. |
| `--tau-reset` | Số generation không cải thiện trước khi restart. |
| `--resetpop-rate` | Tỷ lệ population reset khi restart. |
| `--local-search` | Bật greedy local search hay không. |
| `--stop` | Tiêu chí dừng: gen/time/eval. |
| `--max-gen` | Giới hạn generation. |
| `--max-time` | Giới hạn giây khi stop=time. |
| `--max-evals` | Giới hạn evaluations khi stop=eval. |
| `--edge-epsilon` | Ngưỡng nhỏ để xem edge positive trong empirical linkage. |
| `--save-generation-trace` | Ghi generation_trace CSV. |
| `--save-linkage-events` | Ghi từng event linkage mutation; có thể rất lớn. |
| `--save-graph-snapshots` | Ghi graph snapshots để vẽ graph evolution. |
| `--graph-snapshot-interval` | Lưu snapshot mỗi N generation. |
| `--graph-snapshot-top-k` | Chỉ lưu top-K edge mỗi snapshot. |
| `--graph-snapshot-min-weight` | Chỉ lưu edge snapshot có weight >= ngưỡng. |
| `--no-fitness-cache` | Tắt cache fitness nếu flag xuất hiện. |
| `--lr-gap-gen` | Fit/update LR graph mỗi N generations. |
| `--lr-min-samples` | Archive unique minimum trước khi fit LR graph. |
| `--lr-ridge-alpha` | Ridge penalty cho ga_type 2/3. |
| `--lr-sparse-alpha` | ElasticNet/LASSO alpha cho ga_type 4/5. |
| `--lr-l1-ratio` | ElasticNet L1 ratio cho ga_type 4/5. |
| `--lr-stability-subsamples` | Số subsample refits cho stability selection. |
| `--lr-stability-fraction` | Tỷ lệ archive mỗi subsample. |
| `--lr-edge-min-weight` | Bỏ LR edge dưới weight này. |
| `--lr-edge-top-k` | Giữ top-K LR edge cuối. |
| `--lr-excess-window` | Rolling window cho baseline-standardized excess response. |
| `--artifact-layout` | Ghi per-run artifacts ở root/nested/both. |
| `--write-aggregate-outputs` | Ghi root aggregate ngay trong run hay để aggregate sau. |

### 4.2. Các field chính trong `GAConfig`

| Field | Ý nghĩa |
|---|---|
| `classifier_type` | 1=KNN-3, 2=KNN-5. |
| `ga_type` | 0=standard, 1=empirical linkage, 2=pairwise LR, 3=main+pairwise LR, 4=sparse, 5=sparse+excess. |
| `popsize` | Số cá thể. |
| `crossover_probability` | Xác suất uniform crossover trong standard GA. |
| `ll_crossover_ratio` | Quota crossover trong empirical linkage GA. |
| `mutation_probability` | Xác suất flip bit; None => 1/n_features. |
| `tournament_size` | Số cá thể trong tournament. |
| `tau_reset` | Ngưỡng stagnation restart. |
| `resetpop_rate` | Tỷ lệ reset population. |
| `local_search` | Bật local search. |
| `stop` | gen/time/eval. |
| `max_gen` | Số generation tối đa. |
| `max_time` | Thời gian tối đa. |
| `max_evals` | Số fitness eval tối đa. |
| `edge_epsilon` | Ngưỡng interaction empirical positive. |
| `save_generation_trace` | Lưu diagnostics theo generation. |
| `save_linkage_events` | Lưu event pair test empirical linkage. |
| `save_graph_snapshots` | Lưu graph evolution snapshots. |
| `graph_snapshot_interval` | Khoảng generation giữa snapshots. |
| `graph_snapshot_top_k` | Top-K edge snapshot. |
| `graph_snapshot_min_weight` | Ngưỡng weight snapshot. |
| `fitness_weight_score` | Trọng số score trong fitness, default 0.98. |
| `fitness_weight_sparsity` | Trọng số sparsity trong fitness, default 0.02. |
| `cache_fitness` | Cache chromosome fitness. |
| `lr_gap_gen` | Chu kỳ fit LR graph. |
| `lr_min_samples` | Số mẫu archive tối thiểu. |
| `lr_ridge_alpha` | Ridge alpha. |
| `lr_sparse_alpha` | Sparse alpha. |
| `lr_l1_ratio` | L1 ratio. |
| `lr_stability_subsamples` | Số stability refits. |
| `lr_stability_fraction` | Tỷ lệ subsample. |
| `lr_edge_min_weight` | Filter edge weight. |
| `lr_edge_top_k` | Top-K final LR edges. |
| `lr_excess_window` | Rolling window excess response. |

### 4.3. Các field chính trong `ExperimentConfig`

| Field | Ý nghĩa |
|---|---|
| `outer_folds` | Số outer folds. |
| `inner_folds` | Số inner folds. |
| `repeats` | Số repeats. |
| `seed` | Base seed. |
| `shuffle` | Shuffle split. |
| `output_dir` | Thư mục output của một method. |
| `run_prefix` | Prefix bổ sung, hiện ít dùng. |
| `artifact_layout` | both/nested/root. |
| `write_aggregate_outputs` | Ghi root aggregate trong run hay không. |

---

## 5. Giải thích package/file chính

### 5.1. `run_blur_ga.py`

Đây là unified launcher. File này không chạy thuật toán GA trực tiếp mà điều phối workflow. Nó có 4 subcommands:

- `batch`: chạy tuần tự một tập dataset/method trên local hoặc một node.
- `make-tasks`: tạo manifest CSV cho SLURM array jobs.
- `run-task`: chạy một row trong manifest.
- `aggregate`: gom per-run nested outputs thành root-level summary files.

File này cũng giữ backward compatibility: nếu gọi kiểu cũ `python run_blur_ga.py anneal_task363614 1 2 ...`, nó forward thẳng sang `run_blur_ga_oop.main(...)`.

### 5.2. `run_blur_ga_oop.py`

Đây là runner trực tiếp cho một setting `(dataset, classifier, ga_type)`. Nó nhận positional args `problem classifier ga_type`, đọc dataset, tạo `GAConfig`, tạo `ExperimentConfig`, rồi gọi `NestedCVExperiment.run()`.

### 5.3. `blur_ga/config.py`

Chứa cấu hình typed dataclass:

- `GAConfig`: tham số thuật toán, logging graph, LR linkage, stopping condition.
- `ExperimentConfig`: nested CV, seed, output directory, artifact layout.

`__post_init__` validate các tham số để tránh config sai như `popsize < 4`, `lr_gap_gen < 1`, `outer_folds < 2`, v.v.

### 5.4. `blur_ga/data.py`

Chứa layer đọc dữ liệu. Có hai hướng:

- prepared OpenML/TabArena: folder chứa `dataset.npz`, direct `.npz`, hoặc folder có `X_preprocessed.csv` và `y.csv`.
- `.dat` format cũ.

Điểm quan trọng: parser không tự tạo split 70/30. Nó trả về full dataset để tầng `experiment.py` quyết định nested CV.

### 5.5. `blur_ga/splitting.py`

Tạo K-fold deterministic. Với classification, splitter stratify theo class để mỗi fold có phân phối nhãn cân bằng hơn.

### 5.6. `blur_ga/knn.py`

KNN scorer tự implement bằng NumPy. Nó dùng squared Euclidean distance theo batch, tránh tạo tensor 3D quá nặng. Classification trả accuracy; regression trả `1 - MSE`.

### 5.7. `blur_ga/evaluator.py`

Tính wrapper fitness cho chromosome. Một chromosome là vector nhị phân chọn feature. Fitness hiện tại:

```text
fitness = fitness_weight_score * mean_inner_score
        + fitness_weight_sparsity * sparsity

sparsity = (n_features - selected_size) / n_features
```

Default trong `GAConfig` là `0.98` cho score và `0.02` cho sparsity.

### 5.8. `blur_ga/experiment.py`

Điều phối nested CV. Outer test fold chỉ dùng sau cùng để đánh giá best subset. Trong quá trình GA, fitness chỉ được tính bằng inner folds nằm trong outer train. Đây là phần tránh leakage.

### 5.9. `blur_ga/engine.py`

Lõi thuật toán. Nó quản lý population, best-so-far, archive nghiệm đã evaluate, crossover/mutation/restart/local search, và update graph linkage.

Flow trong một run:

```text
init population random
while not stop:
    nếu stagnation > tau_reset: restart population
    nếu ga_type=0: standard GA next generation
    nếu ga_type=1: empirical linkage next generation + update EmpiricalVIG
    nếu ga_type=2..5: standard GA next generation + fit RegressionVIG mỗi lr_gap_gen
    nếu save_generation_trace: ghi trace row
    nếu save_graph_snapshots: chụp edge table theo interval
return RunResult
```

### 5.10. `blur_ga/evig.py`

Empirical Variable Interaction Graph cho `ga_type=1`. Nó lưu:

- `tested_count`: pair đã được test bao nhiêu lần.
- `count`: pair có positive interaction bao nhiêu lần.
- `weight_sum`: tổng observed positive weights.
- `weight`: trung bình positive weight.
- `first_seen_generation`, `last_updated_generation`.

### 5.11. `blur_ga/lr_linkage.py`

Regression linkage cho `ga_type=2..5`. Nó lấy archive nghiệm đã evaluate trong GA:

```text
X_archive = binary chromosomes
Y_archive = fitness values
```

Sau đó tạo design matrix:

- stage 2: chỉ pairwise `x_i x_j`.
- stage 3: main effects `x_i` + pairwise `x_i x_j`.
- stage 4: sparse ElasticNet/LASSO.
- stage 5: sparse + baseline-standardized excess response.

Final graph dùng `weight = abs(coefficient) * stability`.

### 5.12. `blur_ga/results.py`

Ghi output CSV. Có hai tầng output:

- per-run artifacts: an toàn cho HPC, mỗi run/fold ghi file riêng.
- aggregate/root files: gộp nhiều runs để tiện analysis.

### 5.13. `analysis_interactive/`

Đọc `graph_snapshots`, `generation_trace`, `selected_features` để tạo HTML UI hoặc video. File hay dùng nhất là:

```bash
python analysis_interactive/make_graph_ui.py   --snapshots path/to/graph_snapshots_*.csv   --trace path/to/generation_trace_*.csv   --selected-features path/to/selected_features_*.csv   --run-id 0   --output graph_evolution_ui.html   --layout spring
```

### 5.14. `scripts/prepare_existing_tabarena.py`

Chuẩn bị dữ liệu TabArena/OpenML đã có sẵn thành format nhẹ cho HPC:

```text
prepared_tabarena/<dataset_key>/dataset.npz
prepared_tabarena/<dataset_key>/metadata.json
prepared_tabarena/<dataset_key>/feature_names.json
prepared_tabarena/manifest.csv
prepared_tabarena/skipped_datasets.csv
```

### 5.15. `hpc/`

Chứa script SLURM:

- `job_slurm_blur_ga_task.sh`: chạy một row trong manifest do `make-tasks` sinh ra.
- `job_slurm_blur_ga_nested_fold.sh`: chạy một dataset/method/fold, repeat lấy từ SLURM array id.
- `submit_tabarena_manifest.sh`: submit array jobs theo manifest dataset.

---

## 6. Output files: file nào dùng để làm gì?


| File | Sinh khi nào | Dùng để làm gì |
|---|---|---|
| `run_summary_<prefix>_r<id>.csv` | Luôn per-run | Một dòng summary của run/fold: score, subset size, chromosome. |
| `selected_features_<prefix>_r<id>.csv` | Luôn per-run | Feature index được chọn trong best chromosome của run. |
| `generation_trace_<prefix>_r<id>.csv` | Khi `save_generation_trace=True` | Vẽ convergence, fitness, subset size, archive, n_edges theo generation. |
| `eVIG_<prefix>_r<id>.csv` | Khi có graph, ga_type 1..5 | Ma trận weight adjacency. |
| `eVIG_coefficients_<prefix>_r<id>.csv` | Regression graph ga_type 2..5 | Ma trận coefficient signed β_ij. |
| `eVIG_edges_<prefix>_r<id>.csv` | Khi có graph | Edge list positive/final, dễ phân tích linkage. |
| `eVIG_tested_pairs_<prefix>_r<id>.csv` | Khi có graph | Pair đã test/fit, gồm cả pair không positive. |
| `linkage_events_<prefix>_r<id>.csv` | Khi `save_linkage_events=True`, chủ yếu ga_type=1 | Debug empirical linkage từng pair event. |
| `graph_snapshots_<prefix>_r<id>.csv` | Khi `save_graph_snapshots=True` | Input chính cho graph evolution UI/video. |
| `nested_summary_<prefix>.csv` | Khi `write_aggregate_outputs=True` hoặc sau `aggregate` | Summary root-level của mọi runs. |
| `selected_features_<prefix>.csv` | Aggregate root | Selected features của mọi runs. |
| `generation_trace_<prefix>.csv` | Aggregate root nếu có per-run traces | Trace gộp mọi runs. |
| `bfi/test_score/time/gen/evals/nedges/bind_<prefix>.csv` | Aggregate root | Legacy vector outputs cho script cũ. |


---

## 7. Output nên giữ / có thể bỏ

### Nên giữ cho analysis chính

- `run_summary_*_r*.csv`
- `nested_summary_*.csv` sau aggregate
- `selected_features_*_r*.csv` và/hoặc `selected_features_*.csv`
- `generation_trace_*_r*.csv` và/hoặc `generation_trace_*.csv`
- `eVIG_edges_*_r*.csv`
- `eVIG_coefficients_*_r*.csv` cho ga_type 2..5

### Nên giữ khi cần graph evolution

- `graph_snapshots_*_r*.csv`

### Chỉ nên bật khi debug

- `linkage_events_*_r*.csv`
- `eVIG_tested_pairs_*_r*.csv`

### Legacy hoặc optional

- `bfi_*`, `test_score_*`, `time_*`, `gen_*`, `evals_*`, `nedges_*`, `bind_*`

---

## 8. Cần file nào để vẽ đồ thị?

| Muốn vẽ gì | File cần |
|---|---|
| Convergence fitness theo generation | `generation_trace_*.csv` |
| Subset size theo generation | `generation_trace_*.csv` |
| n_edges/archive size theo generation | `generation_trace_*.csv` |
| So sánh outer score giữa runs/methods | `nested_summary_*.csv` |
| Feature selection frequency | `selected_features_*.csv` |
| Final linkage graph | `eVIG_edges_*_r*.csv` hoặc `eVIG_*.csv` |
| Signed regression interactions | `eVIG_coefficients_*_r*.csv` |
| Graph evolution UI/video | `graph_snapshots_*_r*.csv` + `generation_trace_*.csv` + `selected_features_*.csv` |

---

## 9. Function-by-function reference

### `analysis_interactive/graph_evolution/cli.py`

| Thành phần | Nhiệm vụ |
|---|---|
| `build_parser()` | CLI parser cho make_graph_video/frame. |
| `main(argv: list[str] | None=None)` | Load graph evolution và render frame/video bằng GraphEvolutionRenderer. |

### `analysis_interactive/graph_evolution/data.py`

| Thành phần | Nhiệm vụ |
|---|---|
| class `GraphEvolutionData` | Container cho snapshots, trace, n_nodes, labels. |
| `load_snapshots(path: str | Path)` | Đọc graph_snapshots CSV và chuẩn hóa kiểu dữ liệu. |
| `load_trace(path: str | Path | None, *, run_id: int | None=None)` | Đọc generation_trace CSV nếu có. |
| `infer_n_nodes(snapshots: pd.DataFrame, n_nodes: int | None=None)` | Suy ra số node từ snapshots hoặc input n_nodes. |
| `load_labels(path: str | Path | None, n_nodes: int)` | Đọc label node từ file nếu có. |
| `load_graph_evolution(snapshots_path: str | Path, *, trace_path: str | Path | None=None, n_nodes: int | None=None, labels_path: str | Path | None=None, run_id: int | None=None)` | Load snapshots và related data cho renderer/UI. |

### `analysis_interactive/graph_evolution/filters.py`

| Thành phần | Nhiệm vụ |
|---|---|
| `edges_for_generation(snapshots: pd.DataFrame, generation: int)` | Lấy snapshot edge mới nhất tại hoặc trước generation. |
| `apply_view_mode(edges: pd.DataFrame, generation: int, *, view_mode: str, window: int=20)` | Chọn cumulative/delta/new edge view mode. |
| `filter_edges(edges: pd.DataFrame, *, mode: str='topk', top_k: int=40, threshold: float=0.0, percentile: float=95.0)` | Lọc edge theo min_weight/top_k/top_k_per_node. |

### `analysis_interactive/graph_evolution/importance.py`

| Thành phần | Nhiệm vụ |
|---|---|
| `run_key_columns(df: pd.DataFrame)` | Xác định cột khóa repeat/outer/run có trong dataframe. |
| `normalise_pair(i: int, j: int)` | Sắp xếp pair feature_i<feature_j. |
| `auto_related_paths(snapshots_path: str | Path)` | Tự tìm selected_features/run_summary/graph_snapshots liên quan. |
| `load_run_summary(path: str | Path | None)` | Đọc summary nếu file tồn tại. |
| `choose_score_column(run_summary: pd.DataFrame | None, requested: str | None=None)` | Chọn score column để weight importance. |
| `_make_run_id(df: pd.DataFrame)` | Tạo run_id text từ repeat/outer/run. |
| `_score_table(run_summary: pd.DataFrame | None, score_column: str | None)` | Tạo table score theo run. |
| `compute_feature_importance(selected_features: pd.DataFrame | None, *, n_nodes: int, labels: dict[int, str] | None=None, run_summary: pd.DataFrame | None=None, score_column: str | None=None)` | Tính feature importance từ tần suất selected feature có weight theo score. |
| `_final_edges_by_run(snapshot_paths: Iterable[str | Path])` | Lấy final edge table theo từng run từ snapshot paths. |
| `compute_edge_importance(snapshot_paths: Iterable[str | Path], *, labels: dict[int, str] | None=None)` | Tính edge importance đa run từ final edge weights. |

### `analysis_interactive/graph_evolution/metrics.py`

| Thành phần | Nhiệm vụ |
|---|---|
| `snapshot_metrics(snapshots: pd.DataFrame)` | Tính metrics theo từng snapshot generation. |
| `metrics_at_generation(snapshots: pd.DataFrame, generation: int)` | Tính metrics graph tại một generation. |

### `analysis_interactive/graph_evolution/renderer.py`

| Thành phần | Nhiệm vụ |
|---|---|
| `generations_for_animation(snapshots: pd.DataFrame, trace: pd.DataFrame | None=None, *, start: int | None=None, end: int | None=None, step: int=1)` | Chọn danh sách generation để render animation. |
| `build_union_graph(snapshots: pd.DataFrame, n_nodes: int)` | Tạo NetworkX graph union từ snapshots. |
| `compute_layout(snapshots: pd.DataFrame, n_nodes: int, *, layout: str='spring', seed: int=1)` | Tính layout node bằng NetworkX. |
| `_trace_value(trace: pd.DataFrame | None, generation: int, column: str)` | Lấy giá trị trace tại generation. |
| class `GraphEvolutionRenderer` | Renderer matplotlib/networkx cho frame hoặc video. |
| `GraphEvolutionRenderer.__init__(self, snapshots: pd.DataFrame, *, n_nodes: int, labels: dict[int, str] | None=None, trace: pd.DataFrame | None=None, layout_name: str='spring', layout_seed: int=1, global_weight_scale: bool=True)` | Khởi tạo renderer, layout, visual options. |
| `GraphEvolutionRenderer.frame_edges(self, generation: int, *, filter_mode: str='topk', top_k: int=40, threshold: float=0.0, percentile: float=95.0, view_mode: str='cumulative', window: int=20)` | Lấy edges cho một frame. |
| `GraphEvolutionRenderer.draw_frame(self, ax: plt.Axes, generation: int, *, filter_mode: str='topk', top_k: int=40, threshold: float=0.0, percentile: float=95.0, view_mode: str='cumulative', window: int=20, show_labels: str='auto', title: str='BLuR-GA linkage graph evolution')` | Vẽ một frame lên matplotlib axes. |
| `GraphEvolutionRenderer.save_animation(self, output: str | Path, generations: Iterable[int], *, fps: int=6, filter_mode: str='topk', top_k: int=40, threshold: float=0.0, percentile: float=95.0, view_mode: str='cumulative', window: int=20, show_labels: str='auto', title: str='BLuR-GA linkage graph evolution', dpi: int=140, width: float=9.0, height: float=7.0)` | Xuất GIF/MP4 animation. |
| `GraphEvolutionRenderer.save_frame(self, output: str | Path, generation: int, **kwargs: object)` | Xuất một frame ảnh tĩnh. |

### `analysis_interactive/graph_evolution/ui.py`

| Thành phần | Nhiệm vụ |
|---|---|
| `generations_for_animation(snapshots: pd.DataFrame, trace: pd.DataFrame | None=None, *, start: int | None=None, end: int | None=None, step: int=1)` | Trả saved generations cho HTML UI. |
| `compute_layout(snapshots: pd.DataFrame, n_nodes: int, *, layout: str='spring', seed: int=1, layout_k_scale: float=1.8, layout_iterations: int=300)` | Tính layout node không cần matplotlib. |
| `_finite_float(value: Any, default: float=0.0)` | Ép float hữu hạn với default. |
| `_filter_dataframe(df: pd.DataFrame, *, run_id: int | None=None, repeat_id: int | None=None, outer_fold: int | None=None)` | Clean NaN/Inf trước khi serialize JSON. |
| `load_selected_features(path: str | Path | None, *, run_id: int | None=None, repeat_id: int | None=None, outer_fold: int | None=None)` | Đọc selected_features CSV. |
| `_normalise_positions(pos: dict[int, np.ndarray], n_nodes: int)` | Scale layout positions về canvas. |
| `_prepare_payload(snapshots: pd.DataFrame, trace: pd.DataFrame | None, *, n_nodes: int, labels: dict[int, str], selected_features: set[int], layout_name: str, layout_seed: int, layout_k_scale: float, layout_iterations: int, initial_spacing: float, start: int | None, end: int | None, step: int, title: str, feature_importance: list[dict[str, Any]] | None=None, edge_importance: list[dict[str, Any]] | None=None, importance_meta: dict[str, Any] | None=None)` | Chuẩn bị JSON payload cho HTML UI. |
| `render_html(payload: dict[str, Any], *, source_hint: str='')` | Render HTML string có JS/CSS interactive. |
| `build_parser()` | CLI parser cho make_graph_ui. |
| `main(argv: list[str] | None=None)` | Load data, build payload, ghi HTML và optional open browser. |

### `blur_ga/config.py`

| Thành phần | Nhiệm vụ |
|---|---|
| class `GAConfig` | Dataclass chứa toàn bộ cấu hình thuật toán GA/BLuR-GA cho một run. |
| `GAConfig.__post_init__(self)` | Validate giới hạn tham số GA/LR/output sau khi khởi tạo config. |
| `GAConfig.knn_k(self)` | Property map classifier_type=1 thành KNN-3, classifier_type=2 thành KNN-5. |
| class `ExperimentConfig` | Dataclass chứa cấu hình tầng experiment: nested CV, seed, output layout. |
| `ExperimentConfig.__post_init__(self)` | Validate giới hạn tham số GA/LR/output sau khi khởi tạo config. |

### `blur_ga/data.py`

| Thành phần | Nhiệm vụ |
|---|---|
| class `Dataset` | Container cho toàn bộ dataset X/y và metadata classification/regression. |
| `Dataset.n_samples(self)` | Số dòng/mẫu trong X. |
| `Dataset.n_features(self)` | Số cột/feature trong X. |
| `Dataset.subset(self, indices: Iterable[int])` | Tạo Dataset con theo chỉ số dòng. |
| class `SupervisedSplit` | Container cho một split train/eval dùng cho inner CV hoặc outer test. |
| `SupervisedSplit.n_train(self)` | Số mẫu train. |
| `SupervisedSplit.n_eval(self)` | Số mẫu eval. |
| `_safe_dataset_name(path: Path, metadata_name: object | None=None)` | Chuẩn hóa tên dataset an toàn cho filename/path. |
| `_read_metadata_name(path: Path)` | Đọc metadata.json để lấy dataset_key/dataset_name/name nếu có. |
| `_coerce_classification_labels(y: np.ndarray)` | Đảm bảo nhãn classification là integer contiguous 0..K-1. |
| class `ProcessedTabularParser` | Đọc prepared TabArena/OpenML dataset: dataset.npz, .npz, hoặc X_preprocessed.csv+y.csv. |
| `ProcessedTabularParser.read(cls, path: str | Path)` | Đọc dataset từ folder/file tùy parser. |
| `ProcessedTabularParser._read_npz(path: Path, *, name: str)` | Đọc X/y/n_classes từ npz, validate numeric finite và label. |
| `ProcessedTabularParser._read_processed_csv(folder: Path)` | Đọc X_preprocessed.csv/y.csv, ép numeric và label integer. |
| class `DatParser` | Parser thống nhất cho .dat cũ và prepared processed folder/npz. |
| `DatParser._tokens(line: str)` | Tách token trong file .dat. |
| `DatParser.read(cls, problem: str | Path, search_dir: str | Path='.')` | Đọc dataset từ folder/file tùy parser. |

### `blur_ga/engine.py`

| Thành phần | Nhiệm vụ |
|---|---|
| class `Individual` | Một cá thể GA gồm chromosome binary và fitness. |
| `Individual.copy(self)` | Deep copy chromosome và fitness. |
| class `RunResult` | Kết quả một inner GA run gồm best chromosome, evig, traces, snapshots. |
| `RunResult.subset_size(self)` | Số feature được chọn trong best chromosome. |
| `RunResult.n_edges(self)` | Số edge trong evig nếu có. |
| class `GeneticFeatureSelector` | Lõi GA/BLuR-GA cho binary feature selection. |
| `GeneticFeatureSelector.__init__(self, config: GAConfig, evaluator: FitnessEvaluator, *, seed: int=1, run_id: int=0)` | Nhận config/evaluator, tạo RNG, archive, learner nếu ga_type=2..5. |
| `GeneticFeatureSelector.run(self)` | Main GA loop: init population, tạo thế hệ, update linkage, log trace, stop, trả RunResult. |
| `GeneticFeatureSelector._evaluate(self, chrom: np.ndarray)` | Gọi evaluator, đồng thời lưu nghiệm unique vào archive cho regression linkage. |
| `GeneticFeatureSelector._fit_regression_graph(self, evig: RegressionVIG, generation: int)` | Fit RegressionLinkageLearner khi archive đủ lr_min_samples và update RegressionVIG. |
| `GeneticFeatureSelector._make_generation_trace_row(self, generation: int, start_time: float, evig: EmpiricalVIG | RegressionVIG | None)` | Tạo một dòng diagnostics theo generation. |
| `GeneticFeatureSelector._capture_graph_snapshot(self, evig: EmpiricalVIG | RegressionVIG, generation: int)` | Chụp edge table của graph tại generation hiện tại. |
| `GeneticFeatureSelector._should_stop(self, generation: int, start_time: float)` | Kiểm tra stop theo gen/time/eval. |
| `GeneticFeatureSelector._initial_population(self)` | Sinh population random khác rỗng và evaluate. |
| `GeneticFeatureSelector._restart_population(self)` | Reset population khi stagnation quá tau_reset, giữ elite. |
| `GeneticFeatureSelector._best_population_fitness(self)` | Fitness tốt nhất trong population hiện tại. |
| `GeneticFeatureSelector._update_best_from_population(self)` | Cập nhật best_so_far nếu population có cá thể tốt hơn. |
| `GeneticFeatureSelector._tournament(self)` | Tournament selection. |
| `GeneticFeatureSelector._uniform_crossover(self, p1: np.ndarray, p2: np.ndarray)` | Uniform crossover hai parent. |
| `GeneticFeatureSelector._mutate(self, chrom: np.ndarray)` | Bit-flip mutation, đảm bảo không rỗng feature subset. |
| `GeneticFeatureSelector._make_individual(self, chrom: np.ndarray)` | Tạo Individual từ chromosome sau evaluate. |
| `GeneticFeatureSelector._next_generation_standard(self)` | Sinh thế hệ mới bằng elitism + tournament + crossover/mutation. |
| `GeneticFeatureSelector._next_generation_empirical_linkage(self, evig: EmpiricalVIG, *, generation: int)` | Sinh thế hệ mới cho ga_type=1: quota crossover, phần còn lại linkage mutation. |
| `GeneticFeatureSelector._linkage_mutation(self, parent: Individual, evig: EmpiricalVIG, *, generation: int)` | Test pair g/h/gh quanh parent, tính omega interaction và cập nhật EmpiricalVIG. |
| `GeneticFeatureSelector._local_search(self, individual: Individual)` | Greedy local search thử flip từng bit để cải thiện fitness. |

### `blur_ga/evaluator.py`

| Thành phần | Nhiệm vụ |
|---|---|
| class `FitnessEvaluator` | Tính fitness của một chromosome trên một hoặc nhiều inner validation splits. |
| `FitnessEvaluator.__post_init__(self)` | Tạo KNNScorer và n_features. |
| `FitnessEvaluator.selected_indices(self, chromosome: Iterable[int])` | Lấy index feature có gene=1. |
| `FitnessEvaluator.evaluate(self, chromosome: Iterable[int])` | Tính wrapper fitness = 0.98*mean_score + 0.02*sparsity, có cache. |
| `FitnessEvaluator.metric_score(self, split: SupervisedSplit, chromosome: Iterable[int])` | Tính score thuần trên một split, dùng cho outer test. |

### `blur_ga/evig.py`

| Thành phần | Nhiệm vụ |
|---|---|
| class `EdgeObservation` | Kết quả một lần test tương tác giữa hai feature trong empirical linkage. |
| class `EmpiricalVIG` | Weighted Variable Interaction Graph học bằng linkage mutation empirical. |
| `EmpiricalVIG.__post_init__(self)` | Khởi tạo các ma trận weight/count/tested_count/... |
| `EmpiricalVIG.add_observation(self, a: int, b: int, observed_weight: float, *, epsilon: float=0.0, generation: int=-1)` | Cập nhật thống kê edge sau một observation omega. |
| `EmpiricalVIG.n_edges(self)` | Số edge positive đang có. |
| `EmpiricalVIG.n_tested_pairs(self)` | Số pair từng được test. |
| `EmpiricalVIG.adjacency_matrix(self)` | Trả ma trận weight hiện tại. |
| `EmpiricalVIG.edge_table(self, *, min_weight: float=0.0, top_k: int | None=None)` | Trả danh sách edge positive, sort theo weight giảm dần. |
| `EmpiricalVIG.tested_pair_table(self)` | Trả mọi pair đã test kể cả không positive. |
| `EmpiricalVIG.save_matrix(self, path: str | Path)` | Ghi adjacency matrix eVIG. |
| `EmpiricalVIG.save_edges(self, path: str | Path)` | Ghi edge list positive. |
| `EmpiricalVIG.save_tested_pairs(self, path: str | Path)` | Ghi bảng pair đã test. |

### `blur_ga/experiment.py`

| Thành phần | Nhiệm vụ |
|---|---|
| class `NestedCVExperiment` | Điều phối nested CV: outer test chỉ dùng cuối, inner CV dùng làm fitness. |
| `NestedCVExperiment.run(self, *, outer_fold_id: int | None=None, repeat_id: int | None=None)` | Loop repeat/outer folds, gọi _run_one_outer_fold, lưu artifacts, có thể write aggregate. |
| `NestedCVExperiment._run_one_outer_fold(self, repeat_id: int, outer_fold_id: int, outer_train_idx: np.ndarray, outer_test_idx: np.ndarray)` | Tạo inner splits từ outer train, chạy GeneticFeatureSelector, evaluate best subset trên outer test. |

### `blur_ga/knn.py`

| Thành phần | Nhiệm vụ |
|---|---|
| class `KNNScorer` | KNN nhẹ phụ thuộc NumPy, dùng batch distance để tránh 3D broadcast nặng. |
| `KNNScorer.__init__(self, k: int, *, batch_size: int=256)` | Cấu hình k và batch_size. |
| `KNNScorer.score(self, split: SupervisedSplit, selected_features: np.ndarray)` | Tính accuracy cho classification hoặc 1-MSE cho regression trên selected_features. |
| `KNNScorer._nearest_indices(self, X_train: np.ndarray, X_eval: np.ndarray, k: int)` | Tính k nearest train rows cho từng eval row bằng squared Euclidean distance. |
| `KNNScorer._predict_classification(self, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, k: int, n_classes: int)` | Vote class bằng np.bincount trên hàng xóm gần nhất. |
| `KNNScorer._predict_regression(self, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, k: int)` | Dự đoán regression bằng trung bình y của hàng xóm. |

### `blur_ga/lr_linkage.py`

| Thành phần | Nhiệm vụ |
|---|---|
| class `RegressionLinkageFit` | Kết quả một lần fit regression-linkage tại một generation. |
| class `_DenseLinearModel` | Model tuyến tính tối giản cho dense ridge/OLS-like fit. |
| class `RegressionVIG` | Graph interaction học từ regression trên archive nghiệm GA. |
| `RegressionVIG.__post_init__(self)` | Khởi tạo ma trận weight/coefficient/stability/tested_count. |
| `RegressionVIG.n_edges(self)` | Số edge có weight dương. |
| `RegressionVIG.n_tested_pairs(self)` | Số pair có tested_count > 0. |
| `RegressionVIG.adjacency_matrix(self)` | Trả ma trận weight. |
| `RegressionVIG.coefficient_matrix(self)` | Trả ma trận hệ số signed β_ij. |
| `RegressionVIG.update_from_fit(self, fit: RegressionLinkageFit, *, min_abs_weight: float=0.0, top_k: int | None=None)` | Cập nhật graph từ RegressionLinkageFit, áp min_abs_weight và top_k. |
| `RegressionVIG.edge_table(self, *, min_weight: float=0.0, top_k: int | None=None)` | Trả edge table gồm weight, coefficient, stability, selected_count, tested_count. |
| `RegressionVIG.tested_pair_table(self)` | Trả pair table cho toàn bộ pair trong lần fit. |
| `RegressionVIG.save_matrix(self, path: str | Path)` | Ghi weight matrix. |
| `RegressionVIG.save_coefficient_matrix(self, path: str | Path)` | Ghi coefficient matrix signed. |
| `RegressionVIG.save_edges(self, path: str | Path)` | Ghi edge list. |
| `RegressionVIG.save_tested_pairs(self, path: str | Path)` | Ghi tested pairs. |
| class `RegressionLinkageLearner` | Fit pairwise linkage từ archive chromosomes/fitness/generation. |
| `RegressionLinkageLearner.__init__(self, *, n_features: int, stage: RegressionStage, ridge_alpha: float=1e-06, sparse_alpha: float=0.001, l1_ratio: float=0.95, stability_subsamples: int=0, stability_fraction: float=0.75, random_state: int=1, excess_window: int=5)` | Tạo tất cả pairs, pair index arrays, cấu hình stage/regularization/stability. |
| `RegressionLinkageLearner.include_main_effects(self)` | True cho stage main_pairwise_lr/sparse/sparse_excess. |
| `RegressionLinkageLearner.use_sparse_model(self)` | True cho sparse/sparse_excess. |
| `RegressionLinkageLearner.use_excess_response(self)` | True cho sparse_excess. |
| `RegressionLinkageLearner.fit(self, chromosomes: np.ndarray, fitness: np.ndarray, generations: np.ndarray, *, generation: int)` | Tạo design matrix, biến đổi y nếu cần, fit model, lấy pair coefficients/weights/stability. |
| `RegressionLinkageLearner._design_matrix(self, X_bits: np.ndarray)` | Tạo X_design gồm main effects và/hoặc pairwise x_i*x_j. |
| `RegressionLinkageLearner._fit_model(self, X_design: np.ndarray, y: np.ndarray)` | Chọn ElasticNet nếu sparse, ngược lại dense ridge dual. |
| `RegressionLinkageLearner._fit_dense_ridge_dual(self, X_design: np.ndarray, y: np.ndarray)` | Giải ridge dạng dual n×n cho p≫n. |
| `RegressionLinkageLearner._stability_selection(self, X_design: np.ndarray, y: np.ndarray, pair_start: int)` | Refit trên subsamples để tính tần suất edge nonzero. |
| `RegressionLinkageLearner._baseline_standardized_excess(self, y: np.ndarray, generations: np.ndarray)` | Chuẩn hóa response theo rolling median/MAD theo generation. |
| `ga_type_to_regression_stage(ga_type: int)` | Map ga_type 2..5 sang stage string. |
| `regression_stage_label(ga_type: int)` | Trả label stage hoặc none. |

### `blur_ga/results.py`

| Thành phần | Nhiệm vụ |
|---|---|
| class `EvaluatedRun` | Một dòng kết quả sau khi một outer fold được evaluate. |
| `_chromosome_string(chromosome: np.ndarray | Sequence[int] | str)` | Chuyển chromosome array sang chuỗi 0101. |
| `_selected_feature_string(chromosome: np.ndarray | Sequence[int] | str)` | Chuyển chromosome sang chuỗi index feature được chọn, ngăn cách bởi ;. |
| class `ResultWriter` | Ghi per-run artifacts và aggregate/root-level CSV. |
| `ResultWriter.prefix(self)` | Tên prefix dataset_c<classifier>_a<ga_type>. |
| `ResultWriter._write_nested(self)` | True nếu artifact_layout ghi nested. |
| `ResultWriter._write_root(self)` | True nếu artifact_layout ghi root. |
| `ResultWriter.add(self, row: EvaluatedRun)` | Thêm EvaluatedRun vào writer.rows. |
| `ResultWriter.write_all(self)` | Ghi root-level aggregate summary/vector/selected_features/generation_trace. |
| `ResultWriter._run_dir(self, row: EvaluatedRun)` | Tạo path repeat_xx/outer_fold_xx/run_xxx. |
| `ResultWriter.save_run_artifacts(self, row: EvaluatedRun)` | Ghi per-run summary, selected_features, generation_trace, eVIG, events/snapshots. |
| `ResultWriter._write_run_summary(self, row: EvaluatedRun, path: Path)` | Ghi một CSV summary cho run. |
| `ResultWriter._write_run_selected_features(self, row: EvaluatedRun, path: Path)` | Ghi selected features của một run. |
| `ResultWriter._trace_header(self, rows: Iterable[EvaluatedRun] | None=None)` | Định nghĩa cột generation_trace và thêm extra keys nếu có. |
| `ResultWriter._format_trace_value(self, value: object)` | Format float trong trace. |
| `ResultWriter._write_run_generation_trace(self, row: EvaluatedRun, path: Path)` | Ghi trace riêng cho một run. |
| `ResultWriter._write_linkage_events(self, row: EvaluatedRun, run_dir: Path)` | Ghi linkage_events nếu có. |
| `ResultWriter._write_graph_snapshots(self, row: EvaluatedRun, run_dir: Path)` | Ghi graph_snapshots nếu có. |
| `ResultWriter._write_summary(self)` | Ghi nested_summary root-level. |
| `ResultWriter._write_legacy_vectors(self)` | Ghi bfi/test_score/time/gen/evals/nedges/bind legacy files. |
| `ResultWriter._write_selected_features(self)` | Ghi selected_features aggregate root-level. |
| `ResultWriter._write_generation_traces(self)` | Ghi generation_trace aggregate root-level. |

### `blur_ga/splitting.py`

| Thành phần | Nhiệm vụ |
|---|---|
| class `Fold` | Dataclass chứa train_indices/test_indices/fold_id/repeat_id. |
| class `KFoldSplitter` | Tạo deterministic K-fold, có stratification cho classification. |
| `KFoldSplitter.__init__(self, n_splits: int, *, shuffle: bool=True, seed: int=1, stratified: bool=True)` | Lưu n_splits/shuffle/seed/stratified và validate. |
| `KFoldSplitter.split(self, dataset: Dataset, *, repeat_id: int=0, indices: np.ndarray | None=None)` | Sinh list Fold; classification thì chia đều theo class. |
| `make_split(dataset: Dataset, train_indices: np.ndarray, eval_indices: np.ndarray)` | Tạo SupervisedSplit từ Dataset và train/eval indices. |
| `remap_local_to_global(local_indices: np.ndarray, global_pool: np.ndarray)` | Map chỉ số local trong outer_train_dataset về chỉ số global của dataset gốc. |

### `run_blur_ga.py`

| Thành phần | Nhiệm vụ |
|---|---|
| `str2bool(value: str | bool)` | Chuyển chuỗi command-line như true/false/yes/no/1/0 thành bool. |
| class `Job` | Dataclass mô tả một job: dataset, classifier, ga_type, output_dir, optional repeat/fold id. |
| `method_folder_name(ga_type: int)` | Map ga_type sang tên thư mục method dễ đọc dưới output-root. |
| `available_datasets(data_dir: Path)` | Quét data-dir để tìm dataset dạng folder có dataset.npz hoặc file .dat. |
| `read_dataset_file(path: Path)` | Đọc danh sách dataset từ file text/CSV; lấy field đầu tiên mỗi dòng. |
| `resolve_datasets(args: argparse.Namespace)` | Quyết định danh sách dataset cuối cùng từ --datasets, --dataset-file, --exclude-datasets, --limit-datasets. |
| `apply_preset(args: argparse.Namespace)` | Áp preset smoke/analysis/nesi_full vào những argument còn None; CLI explicit luôn thắng preset. |
| `fill_defaults(args: argparse.Namespace)` | Điền default cuối cùng cho mọi argument còn None sau khi apply preset. |
| `build_jobs(args: argparse.Namespace, *, split_folds: bool)` | Tạo danh sách Job; batch tạo job theo dataset×classifier×ga_type, make-tasks có thể tách repeat/fold. |
| `common_oop_args(args: argparse.Namespace, job: Job)` | Biến Job + args thành danh sách argument truyền vào run_blur_ga_oop.py. |
| `run_oop_args(oop_args: list[str])` | Gọi trực tiếp run_blur_ga_oop.main(...) trong cùng process. |
| `add_shared_options(p: argparse.ArgumentParser)` | Đăng ký toàn bộ option chung cho batch và make-tasks. |
| `build_parser()` | Tạo argparse parser với subcommands batch/make-tasks/run-task/aggregate. |
| `cmd_batch(args: argparse.Namespace)` | Chạy tuần tự các job được chọn trong một Python command. |
| `cmd_make_tasks(args: argparse.Namespace)` | Tạo CSV manifest cho SLURM array jobs, mỗi row chứa command_json. |
| `cmd_run_task(args: argparse.Namespace)` | Đọc một row trong manifest theo task_id rồi chạy command_json. |
| `_read_one_csv(path: Path)` | Đọc một file CSV thành list[dict]. |
| `_is_nested_run_summary(path: Path)` | Nhận diện run_summary nằm trong repeat_xx/outer_fold_xx/run_xxx. |
| `_method_dir_from_run_summary(path: Path)` | Từ run_summary suy ra thư mục method để aggregate output. |
| `_prefix_from_summary_name(path: Path)` | Tách prefix dataset_cX_aY từ tên run_summary_..._rZ.csv. |
| `_chrom_to_bind(chrom: str)` | Chuyển chuỗi chromosome 0101 thành format legacy “0, 1, 0, 1”. |
| `cmd_aggregate(args: argparse.Namespace)` | Gom các per-run nested artifacts thành root summary/vector/trace files. |
| `main(argv: list[str] | None=None)` | Entry point: nếu positional command cũ thì forward sang OOP runner; nếu subcommand thì gọi handler tương ứng. |

### `run_blur_ga_oop.py`

| Thành phần | Nhiệm vụ |
|---|---|
| `str2bool(value: str | bool)` | Parser bool cho các flag của runner trực tiếp. |
| `build_parser()` | Định nghĩa positional args problem/classifier/ga_type và toàn bộ option GA/nested CV/output. |
| `main(argv: list[str] | None=None)` | Đọc dataset, tạo GAConfig và ExperimentConfig, chạy NestedCVExperiment, in kết quả trung bình. |

### `scripts/prepare_existing_tabarena.py`

| Thành phần | Nhiệm vụ |
|---|---|
| class `PreparedDatasetMetadata` | Metadata lưu kèm mỗi prepared dataset. |
| `parse_args()` | CLI parser cho data preparation. |
| `sanitize_key(text: Any, max_len: int=96)` | Chuẩn hóa key dataset ngắn gọn/an toàn. |
| `str2bool(value: str)` | Parser bool. |
| `load_json(path: Path)` | Đọc JSON nếu tồn tại. |
| `write_json(path: Path, obj: Any)` | Ghi JSON pretty. |
| `discover_dataset_folders(source_dir: Path)` | Tìm các folder dataset có X_preprocessed.csv/y.csv. |
| `read_xy(folder: Path)` | Đọc X_preprocessed.csv và y.csv. |
| `sanitize_X(X: pd.DataFrame, *, drop_constant: bool)` | Ép X numeric finite, impute missing, drop non-numeric/constant. |
| `encode_y(y: pd.Series)` | Encode label classification thành int. |
| `filtered_feature_names(folder: Path, kept_indices: list[int], n_features: int)` | Tạo feature_names.json sau khi lọc cột. |
| `process_folder(folder: Path, out_root: Path, args: argparse.Namespace)` | Xử lý một folder dataset thành dataset.npz + metadata. |
| `main()` | Loop toàn bộ folders, ghi manifest.csv và skipped_datasets.csv. |


---

## 10. Gợi ý đọc code theo thứ tự

Nếu mục tiêu là hiểu project từ input đến output, đọc theo thứ tự này:

```text
1. run_blur_ga.py
2. run_blur_ga_oop.py
3. blur_ga/config.py
4. blur_ga/data.py
5. blur_ga/experiment.py
6. blur_ga/splitting.py
7. blur_ga/evaluator.py
8. blur_ga/engine.py
9. blur_ga/evig.py
10. blur_ga/lr_linkage.py
11. blur_ga/results.py
12. analysis_interactive/make_graph_ui.py
```

Nếu mục tiêu là chỉnh thuật toán, tập trung vào:

```text
blur_ga/engine.py
blur_ga/lr_linkage.py
blur_ga/evig.py
```

Nếu mục tiêu là chỉnh output, tập trung vào:

```text
blur_ga/results.py
run_blur_ga.py aggregate
```

Nếu mục tiêu là chỉnh graph viewer, tập trung vào:

```text
analysis_interactive/graph_evolution/ui.py
analysis_interactive/graph_evolution/importance.py
```
