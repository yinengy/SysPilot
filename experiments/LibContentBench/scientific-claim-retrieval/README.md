# Scientific Claim Retrieval

Encode 5,183 SciFact abstracts and 300 test queries with MiniLM, then retrieve the
ten nearest documents per query using exact FAISS inner-product search. Timing
includes CPU worker startup, encoding, shutdown, and retrieval. Loading the model
in the parent process is outside the timed region.

## Setup

Run from this directory on Linux with Python 3.10:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install sentence-transformers==3.4.1 transformers==4.48.3 \
  tokenizers==0.21.4 huggingface-hub==0.36.0 faiss-cpu==1.10.0 \
  numpy==1.26.4 scipy==1.15.3
export PYTHONHASHSEED=0
WORK=/var/tmp/libcontentbench/scientific-claim-retrieval
python run.py prepare --data-dir "$WORK/data"
```

Preparation downloads SciFact and the pinned `all-MiniLM-L6-v2` model revision.
Subsequent runs use local files without network access.

## Run

Baseline, followed by the configuration selected by CONCORD:

```bash
ST_PROCESSES=4 ST_INTRA_THREADS=0 ST_TOKENIZER_THREADS=0 \
ST_BATCH_SIZE=32 ST_CHUNK_SIZE=0 \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/baseline"

ST_PROCESSES=2 ST_INTRA_THREADS=0 ST_TOKENIZER_THREADS=0 \
ST_BATCH_SIZE=18 ST_CHUNK_SIZE=384 \
  python run.py run --data-dir "$WORK/data" --output-dir "$WORK/tuned" \
  --reference "$WORK/baseline"
```

Each run prints `wall_time_sec` and saves embeddings, rankings, scores, and
`result.json` in its output directory. The second run checks numerical outputs
and requires identical document rankings. Compare baseline time divided by tuned
time, running the commands serially.

| Control | Range | Meaning |
| --- | --- | --- |
| `ST_PROCESSES` | 1–8 | CPU encoder processes |
| `ST_INTRA_THREADS` | 0–48 | Native threads per process; 0 keeps library defaults |
| `ST_TOKENIZER_THREADS` | 0–48 | Tokenizer threads; 0 keeps the default |
| `ST_BATCH_SIZE` | 1–128 | Texts per encoder batch |
| `ST_CHUNK_SIZE` | 0–2048 | Texts per process-pool job; 0 uses the library heuristic |

The paper's configuration space has 5,037,720,576 settings. Keep model weights,
sequence length, normalization, input texts, and FAISS search fixed when tuning.
The selected settings were measured on a dual-socket host with 24 physical cores
and 48 hardware threads; timings depend on the host.
