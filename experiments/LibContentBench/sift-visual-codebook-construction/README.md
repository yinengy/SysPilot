# SIFT Visual Codebook Construction

Train a 1,024-center visual codebook with Spark MLlib on all 100,000 SIFT1M
learning descriptors, then save the model and assign every descriptor to a
cluster. Each run starts a fresh local JVM. Timing includes startup, input
loading, training, output writes, and shutdown.

## Setup

Run from this directory on x86-64 Linux with Python 3.10, a C compiler, and
OpenBLAS (`build-essential` and `libopenblas0-pthread` on Ubuntu):

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install numpy==2.2.6 scipy==1.15.3
export PYTHONHASHSEED=0
WORK=/var/tmp/libcontentbench/sift-visual-codebook-construction
python run.py prepare --data-dir "$WORK/data"
```

Preparation downloads Spark 4.1.1, Temurin JDK 21.0.12.1, and SIFT1M, verifies
checksums, and compiles the Java driver and BLAS observer. Existing installations
can be supplied with `--spark-home /path/to/spark --java-home /path/to/jdk`.
The observer reads BLAS settings without changing them.

## Run

Baseline, followed by the configuration selected by CONCORD:

```bash
SPARK_TASK_CPUS=0 SPARK_BLAS_THREADS=0 \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/baseline"

SPARK_TASK_CPUS=3 SPARK_BLAS_THREADS=1 \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/tuned" \
  --reference "$WORK/baseline"
```

Use a fresh output directory for each run. Each run prints `wall_time_sec` and
saves the Spark model, centers, assignments, and `result.json`. The second run
checks centers, assignments, iteration count, and costs against the baseline,
allowing a permutation of cluster labels. Compare baseline time divided by tuned
time, running the commands serially.

| Control | Range | Meaning |
| --- | --- | --- |
| `SPARK_TASK_CPUS` | 0–48 | CPU slots reserved per task; 0 keeps Spark's default of 1 |
| `SPARK_BLAS_THREADS` | 0–48 | OpenBLAS threads; 0 leaves the native default |

The paper's configuration space has 2,401 settings. Keep 48 input partitions,
the 4 GiB heap, seed, initialization, convergence settings, and dense-array AUTO
solver fixed when tuning. The selected settings were measured on a dual-socket
host with 24 physical cores and 48 hardware threads; timings depend on the host.
