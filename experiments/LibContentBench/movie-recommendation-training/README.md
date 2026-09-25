# Movie Recommendation Training

Train an implicit-feedback ALS model on all 20,000,263 MovieLens20M ratings and
recommend ten unseen movies to each of the 138,493 users. The model uses 128
factors and 15 iterations. Timing includes training and recommendations.

## Setup

Run from this directory on Linux with Python 3.10:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install implicit==0.7.2 numpy==2.2.6 scipy==1.15.3 \
  pandas==2.3.2 threadpoolctl==3.6.0
export PYTHONHASHSEED=0
WORK=/var/tmp/libcontentbench/movie-recommendation-training
python run.py prepare --data-dir "$WORK/data"
```

Preparation downloads MovieLens20M and builds a sparse ratings matrix. No ratings
are sampled or removed.

## Run

Baseline, followed by the configuration selected by CONCORD:

```bash
REC_THREADS=0 OPENBLAS_NUM_THREADS=48 OMP_PROC_BIND=false OMP_PLACES=cores \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/baseline"

REC_THREADS=24 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/tuned" \
  --reference "$WORK/baseline"
```

Each run prints `wall_time_sec` and saves model factors, recommendations, scores,
and `result.json` in its output directory. The second run checks numerical
outputs and recommendation sets against the baseline; ranking changes are
allowed only for scores tied within `1e-5`. Compare baseline time divided by
tuned time, running the commands serially.

| Control | Range | Meaning |
| --- | --- | --- |
| `REC_THREADS` | 0–48 | ALS workers; 0 uses all available CPUs |
| `OPENBLAS_NUM_THREADS` | 1–48 | BLAS threads |
| `OMP_PROC_BIND` | `false`, `close`, `spread` | OpenMP binding policy |
| `OMP_PLACES` | `cores`, `threads` | OpenMP placement units |

The paper's configuration space has 14,112 settings. Keep factors, iterations,
regularization, seed, and ratings fixed when tuning. The selected settings were
measured on a dual-socket host with 24 physical cores and 48 hardware threads;
timings depend on the host.
