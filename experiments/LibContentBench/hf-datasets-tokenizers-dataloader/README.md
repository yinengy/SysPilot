# HF-Datasets-Tokenizers-DataLoader

This benchmark creates local contention between Hugging Face `datasets`, Hugging Face `tokenizers`, and a real PyTorch `DataLoader` training loop.

The workload is synthetic on purpose: it generates a fixed text corpus and a fixed local BPE tokenizer artifact, but the contention comes from real `datasets.Dataset.map(num_proc=...)`, real `tokenizers.Tokenizer.encode_batch(...)`, and real PyTorch worker/prefetch behavior during model training.

## Layout

- `prepare.sh` — creates the local virtualenv, installs runtime dependencies, and records readiness/runtime metadata.
- archived case catalogs and reports — preserved for historical baseline/application/library-aware/oracle evidence.
- `run_hf_datasets_tokenizers_case.py` — single case runner.
- `build_report.py` — rebuilds `result_table.csv`, `comparison_report.json`, `acceptance_report.json`, and `summary.md` from archived run artifacts.
- `work/knob_visibility.json` — explicit split between application-only and library-aware tuning surfaces.

## Modes

- `baseline` — aggressive dataset multiprocessing plus tokenizer parallelism and high-worker DataLoader settings.
- `application_only` — changes only coarse model-shape knobs while keeping the same library settings.
- `library_aware` — coordinates `datasets.map(num_proc=...)`, tokenizer parallelism, and DataLoader worker behavior.
- `oracle` — bounded extra library-aware shapes.

## Runtime layout

- Repo-local evidence lives under `runs/`.
- Live dataset/tokenizer artifacts and per-case scratch files live under `/var/tmp/libcontentbench/hf-datasets-tokenizers-dataloader`.

## Usage

```bash
bash prepare.sh
artifact/venv/bin/python build_report.py
```

Run the packaged direct worker commands from `../summary_repro.md` when you want baseline or best archived repro commands.

## Fixed-baseline SysPilot tuning surface

To let `syspilot tune` keep the packaged baseline workload fixed while changing only real library knobs, the worker reads these optional environment overrides:

- `BENCHMARK_TORCH_NUM_THREADS`
- `BENCHMARK_DATASETS_NUM_PROC`
- `BENCHMARK_TOKENIZERS_PARALLELISM`
- `BENCHMARK_NUM_WORKERS`
- `BENCHMARK_PREFETCH_FACTOR`
- `BENCHMARK_PERSISTENT_WORKERS`

These map directly onto the benchmark's `datasets`, `tokenizers`, and PyTorch runtime settings. Workload-shape fields like corpus size, sequence length, tokenizer vocabulary target, batch size, and epoch count stay fixed unless you intentionally change the direct CLI workload shape.

For SysPilot tuning, pass the original `datasets` LPCL contract when you want to tune `BENCHMARK_DATASETS_NUM_PROC`. The `datasets` control intentionally uses this benchmark env var because it is the fixed-workload action surface for `datasets.Dataset.map(num_proc=...)`.
