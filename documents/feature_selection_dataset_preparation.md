# Feature-selection dataset preparation

This project now has a reproducible dataset builder for the **Feature Selection Performance Evaluation** part of BLuR-GA.

## Main command

```bash
python scripts/prepare_feature_selection_datasets.py \
  --mode download \
  --output-dir ../Dataset/prepared_feature_selection \
  --raw-cache-dir ../Dataset/dataset_cache/raw \
  --processed-dir ../Dataset/dataset_cache/processed \
  --groups all \
  --overwrite
```

The output root is directly readable by the existing runner:

```bash
python run_blur_ga.py batch \
  --data-dir ../Dataset/prepared_feature_selection \
  --output-root ../Results/results_fs_small \
  --dataset-group small \
  --classifiers 2 \
  --ga-types 0 1 2 3 \
  --repeat 1 \
  --inner-folds 3 \
  --outer-folds 3 \
  --popsize 100 \
  --max-gen 100
```

`--dataset-group` reads names from:

```text
<data-dir>/dataset_groups/small.txt
<data-dir>/dataset_groups/medium.txt
<data-dir>/dataset_groups/highdim.txt
<data-dir>/dataset_groups/all.txt
```

You can still use the previous modes:

```bash
--datasets website_phishing_task363707 Marketing_Campaign_task363684
--dataset-file path/to/custom_dataset_list.txt
--datasets all
```

Precedence is:

1. `--dataset-file`
2. `--dataset-group`
3. `--datasets`

## Dataset groups

### small

- `heart_disease_uci`
- `parkinsons_uci`
- `website_phishing_task363707`
- `Is-this-a-good-customer_task363682`
- `Marketing_Campaign_task363684`
- `hazelnut-spread-contaminant-detection_task363674`

### medium

- `students_dropout_and_academic_success_task363704`
- `churn_task363623`
- `polish_companies_bankruptcy_task363694`
- `satellite_uci`

### highdim

- `Isolet_asu`
- `Carcinom_asu`
- `Lung_asu`
- `Prostate_GE_asu`
- `TOX_171_asu`

## Preprocessing policy

For UCI and OpenML/TabArena datasets:

- target labels are encoded to `0..K-1`;
- numeric columns use `SimpleImputer(strategy='median')` and `StandardScaler`;
- categorical columns use `SimpleImputer(strategy='constant', fill_value='__missing__')` and `OneHotEncoder(handle_unknown='ignore', max_categories=50)`;
- zero-variance processed columns are removed with `VarianceThreshold(0.0)`;
- output is saved as compressed `dataset.npz` with arrays `X`, `y`, and `n_classes`.

For ASU/scikit-feature high-dimensional `.mat` datasets:

- the numeric matrix is loaded from the `.mat` file;
- labels are encoded to `0..K-1`;
- missing or non-finite values are defensively imputed;
- features are standardized and zero-variance columns are removed;
- output is saved using the same `dataset.npz` format.

## Recovery from an existing prepared folder

For old archives that already contain prepared BLuR-GA datasets, use:

```bash
python scripts/prepare_feature_selection_datasets.py \
  --mode existing \
  --existing-prepared-dir ../Dataset/prepared_tabarena \
  --output-dir ../Dataset/prepared_feature_selection_from_existing \
  --groups all \
  --overwrite
```

This does not download anything. It only copies matching datasets and writes `dataset_groups/*.txt`, `manifest.csv`, and `missing_datasets.csv`.
