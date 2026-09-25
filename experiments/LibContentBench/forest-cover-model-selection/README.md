# Forest-Cover Model Selection

Select an XGBoost classifier for forest-cover prediction using Dask-ML GridSearchCV.
The workload uses a fixed, stratified 120,000-row Covertype subset, six model
configurations, and three cross-validation folds. Timing covers all 18 fits,
refitting, and prediction on the 24,000 held-out rows.

## Setup

Run from this directory on Linux with Python 3.10:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install numpy==2.2.6 scipy==1.15.3 scikit-learn==1.6.1 \
  xgboost-cpu==3.0.5 dask-ml==2025.1.0 dask==2025.3.0 \
  distributed==2025.3.0 pandas==2.2.3 threadpoolctl==3.6.0
export PYTHONHASHSEED=0
WORK=/var/tmp/libcontentbench/forest-cover-model-selection
python run.py prepare --data-dir "$WORK/data"
```

## Run

Baseline, followed by the configuration selected by CONCORD:

```bash
DASK_CV_WORKERS=0 DASK_XGB_THREADS=0 DASK_SCHEDULER=threads \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/baseline"

DASK_CV_WORKERS=26 DASK_XGB_THREADS=6 DASK_SCHEDULER=threads \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/tuned" \
  --reference "$WORK/baseline"
```

Each run prints `wall_time_sec`. The output directory contains predictions,
cross-validation scores, and `result.json`. The second run checks the selected
model, predicted labels, probabilities, and fold scores against the baseline.
Compare baseline time divided by tuned time; run the commands serially.

| Control | Range | Meaning |
| --- | --- | --- |
| `DASK_CV_WORKERS` | 0–48 | Concurrent fits; 0 uses all available CPUs |
| `DASK_XGB_THREADS` | 0–48 | Native threads per fit; 0 uses the library default |
| `DASK_SCHEDULER` | `threads`, `processes` | Local Dask scheduler |

The paper's configuration space has 4,802 settings. Keep the dataset, model grid,
and folds fixed when tuning. The selected settings were measured on a dual-socket
host with 24 physical cores and 48 hardware threads; timings depend on the host.
