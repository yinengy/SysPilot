#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.LibContentBench.libcontentbench.common import (
    load_json_object,
    performance_metrics_payload,
    write_json,
)
from experiments.LibContentBench.libcontentbench.monitoring import ProcessTreeSampler


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    label: str
    mode: str
    output_json: str
    runtime_root: str
    dataset_rows: int
    tokens_per_row: int
    max_length: int
    tokenizer_vocab_size: int
    hidden_dim: int
    num_classes: int
    batch_size: int
    epochs: int
    learning_rate: float
    torch_num_threads: int
    datasets_num_proc: int
    tokenize_batch_size: int
    tokenizers_parallelism: bool
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool
    timeout_seconds: int


INT_ENV_OVERRIDES: dict[str, tuple[str, int]] = {
    "torch_num_threads": ("BENCHMARK_TORCH_NUM_THREADS", 1),
    "datasets_num_proc": ("BENCHMARK_DATASETS_NUM_PROC", 1),
    "num_workers": ("BENCHMARK_NUM_WORKERS", 0),
    "prefetch_factor": ("BENCHMARK_PREFETCH_FACTOR", 1),
}

BOOL_ENV_OVERRIDES: dict[str, str] = {
    "tokenizers_parallelism": "BENCHMARK_TOKENIZERS_PARALLELISM",
    "persistent_workers": "BENCHMARK_PERSISTENT_WORKERS",
}

_TOKENIZER_CACHE: dict[tuple[str, int], Any] = {}


DEFAULTS = {
    "case_id": "baseline_direct",
    "label": "baseline direct",
    "mode": "baseline",
    "output_json": "work/direct_result.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one HF datasets tokenizers DataLoader case.")
    parser.add_argument("--case-config")
    parser.add_argument("--case-id", default=DEFAULTS["case_id"])
    parser.add_argument("--label", default=DEFAULTS["label"])
    parser.add_argument("--mode", default=DEFAULTS["mode"])
    parser.add_argument("--output-json", default=DEFAULTS["output_json"])
    parser.add_argument("--runtime-root")
    parser.add_argument("--dataset-rows", type=int)
    parser.add_argument("--tokens-per-row", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--tokenizer-vocab-size", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--tokenize-batch-size", type=int)
    parser.add_argument("--timeout-seconds", type=int)
    return parser.parse_args(argv)


def load_case(path: Path) -> CaseConfig:
    payload = load_json_object(path)
    return CaseConfig(
        case_id=str(payload["case_id"]),
        label=str(payload.get("label", payload["case_id"])),
        mode=str(payload.get("mode", "unknown")),
        output_json=str(payload["output_json"]),
        runtime_root=str(payload["runtime_root"]),
        dataset_rows=int(payload["dataset_rows"]),
        tokens_per_row=int(payload["tokens_per_row"]),
        max_length=int(payload["max_length"]),
        tokenizer_vocab_size=int(payload["tokenizer_vocab_size"]),
        hidden_dim=int(payload["hidden_dim"]),
        num_classes=int(payload["num_classes"]),
        batch_size=int(payload["batch_size"]),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        torch_num_threads=int(payload["torch_num_threads"]),
        datasets_num_proc=int(payload["datasets_num_proc"]),
        tokenize_batch_size=int(payload["tokenize_batch_size"]),
        tokenizers_parallelism=bool(payload["tokenizers_parallelism"]),
        num_workers=int(payload["num_workers"]),
        prefetch_factor=int(payload["prefetch_factor"]),
        persistent_workers=bool(payload["persistent_workers"]),
        timeout_seconds=int(payload["timeout_seconds"]),
    )


def config_from_args(args: argparse.Namespace) -> CaseConfig:
    missing = [
        option
        for option, value in (
            ("--runtime-root", args.runtime_root),
            ("--dataset-rows", args.dataset_rows),
            ("--tokens-per-row", args.tokens_per_row),
            ("--max-length", args.max_length),
            ("--tokenizer-vocab-size", args.tokenizer_vocab_size),
            ("--hidden-dim", args.hidden_dim),
            ("--num-classes", args.num_classes),
            ("--batch-size", args.batch_size),
            ("--epochs", args.epochs),
            ("--learning-rate", args.learning_rate),
            ("--tokenize-batch-size", args.tokenize_batch_size),
            ("--timeout-seconds", args.timeout_seconds),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Direct HF datasets/tokenizers runs require " + ", ".join(missing) + " when --case-config is not provided"
        )
    return CaseConfig(
        case_id=str(args.case_id),
        label=str(args.label),
        mode=str(args.mode),
        output_json=str(args.output_json),
        runtime_root=str(args.runtime_root),
        dataset_rows=int(args.dataset_rows),
        tokens_per_row=int(args.tokens_per_row),
        max_length=int(args.max_length),
        tokenizer_vocab_size=int(args.tokenizer_vocab_size),
        hidden_dim=int(args.hidden_dim),
        num_classes=int(args.num_classes),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        torch_num_threads=1,
        datasets_num_proc=1,
        tokenize_batch_size=int(args.tokenize_batch_size),
        tokenizers_parallelism=False,
        num_workers=0,
        prefetch_factor=1,
        persistent_workers=False,
        timeout_seconds=int(args.timeout_seconds),
    )


def _parse_bool_env(name: str, raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value, got {raw!r}")


def apply_env_overrides(config: CaseConfig) -> tuple[CaseConfig, dict[str, Any]]:
    updates: dict[str, Any] = {}
    applied_env: dict[str, Any] = {}
    for field_name, (env_name, minimum) in INT_ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        value = int(raw)
        if value < minimum:
            raise ValueError(f"{env_name} must be >= {minimum}, got {value}")
        updates[field_name] = value
        applied_env[env_name] = value
    for field_name, env_name in BOOL_ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        value = _parse_bool_env(env_name, raw)
        updates[field_name] = value
        applied_env[env_name] = value
    if not updates:
        return config, {}
    return replace(config, **updates), applied_env


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_readiness(workspace_root: Path) -> dict[str, Any]:
    readiness_path = workspace_root / "env/readiness.json"
    if not readiness_path.exists():
        return {"ready": False, "notes": "prepare.sh has not been run yet"}
    return load_json_object(readiness_path)


def corpus_root_for(config: CaseConfig) -> Path:
    return Path(config.runtime_root) / "datasets" / f"rows{config.dataset_rows}_tokens{config.tokens_per_row}_classes{config.num_classes}"


def tokenizer_root_for(config: CaseConfig) -> Path:
    return Path(config.runtime_root) / "tokenizers" / f"rows{config.dataset_rows}_vocab{config.tokenizer_vocab_size}_tokens{config.tokens_per_row}"


def build_dataloader_kwargs(config: CaseConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": config.num_workers,
        "pin_memory": False,
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        kwargs["prefetch_factor"] = config.prefetch_factor
    return kwargs


def _synth_text(index: int, tokens_per_row: int) -> str:
    words = []
    for offset in range(tokens_per_row):
        value = (index * 17 + offset * 13) % 97
        words.append(f"tok{value:02d}")
    return " ".join(words)


def ensure_corpus(config: CaseConfig, datasets_module: Any) -> Path:
    corpus_root = corpus_root_for(config)
    if corpus_root.exists():
        return corpus_root
    texts = [_synth_text(index, config.tokens_per_row) for index in range(config.dataset_rows)]
    labels = [int((index * 7) % config.num_classes) for index in range(config.dataset_rows)]
    dataset = datasets_module.Dataset.from_dict({"text": texts, "label": labels})
    dataset.save_to_disk(str(corpus_root))
    write_json(
        corpus_root / "manifest.json",
        {
            "dataset_rows": config.dataset_rows,
            "tokens_per_row": config.tokens_per_row,
            "num_classes": config.num_classes,
        },
    )
    return corpus_root


def ensure_tokenizer(config: CaseConfig, tokenizers_module: Any) -> Path:
    root = tokenizer_root_for(config)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_path = root / "tokenizer.json"
    if tokenizer_path.exists():
        return tokenizer_path

    models = importlib.import_module("tokenizers.models")
    normalizers = importlib.import_module("tokenizers.normalizers")
    pre_tokenizers = importlib.import_module("tokenizers.pre_tokenizers")
    trainers = importlib.import_module("tokenizers.trainers")

    tokenizer = tokenizers_module.Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=config.tokenizer_vocab_size,
        special_tokens=["[PAD]", "[UNK]"],
    )
    tokenizer.train_from_iterator(
        (_synth_text(index, config.tokens_per_row) for index in range(config.dataset_rows)),
        trainer=trainer,
        length=config.dataset_rows,
    )
    tokenizer.save(str(tokenizer_path), pretty=False)
    write_json(
        root / "manifest.json",
        {
            "dataset_rows": config.dataset_rows,
            "tokenizer_vocab_size": config.tokenizer_vocab_size,
            "tokens_per_row": config.tokens_per_row,
        },
    )
    return tokenizer_path


def _tokenizer_for(tokenizer_path: str, max_length: int) -> Any:
    key = (tokenizer_path, max_length)
    cached = _TOKENIZER_CACHE.get(key)
    if cached is not None:
        return cached
    tokenizers_module = importlib.import_module("tokenizers")
    tokenizer = tokenizers_module.Tokenizer.from_file(tokenizer_path)
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise ValueError("tokenizer is missing [PAD] token")
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(length=max_length, pad_id=int(pad_id), pad_token="[PAD]")
    _TOKENIZER_CACHE[key] = tokenizer
    return tokenizer


def tokenize_batch(batch: dict[str, Any], *, tokenizer_path: str, max_length: int) -> dict[str, Any]:
    tokenizer = _tokenizer_for(tokenizer_path, max_length)
    encodings = tokenizer.encode_batch(list(batch["text"]))
    return {
        "input_ids": [encoding.ids for encoding in encodings],
        "attention_mask": [encoding.attention_mask for encoding in encodings],
    }


def build_model(torch_module: Any, nn_module: Any, *, vocab_size: int, hidden_dim: int, num_classes: int) -> Any:
    class MeanPoolClassifier(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn_module.Embedding(vocab_size, hidden_dim)
            self.proj = nn_module.Linear(hidden_dim, num_classes)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            embedded = self.embedding(input_ids)
            mask = attention_mask.unsqueeze(-1).to(dtype=embedded.dtype)
            pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return self.proj(pooled)

    return MeanPoolClassifier()


def run_case(config: CaseConfig, case_dir: Path, env_overrides: dict[str, Any]) -> dict[str, Any]:
    workspace_root = Path(__file__).resolve().parent
    readiness = load_readiness(workspace_root)
    effective_env_path = case_dir / "effective_env_overrides.json"
    write_json(effective_env_path, env_overrides)
    if not readiness.get("ready"):
        return {
            "benchmark": "hf-datasets-tokenizers-dataloader",
            "resource": "dataset processing + tokenization + input pipeline",
            "contending_libraries": ["datasets", "tokenizers", "pytorch", "torch.utils.data"],
            "case_id": config.case_id,
            "label": config.label,
            "mode": config.mode,
            "status": "blocked",
            "success": False,
            "throughput": None,
            "wall_time_sec": None,
            "samples_per_sec": None,
            "steps_per_sec": None,
            "resource_signal_primary": None,
            "resource_signal_secondary": None,
            "config_summary": "torch, datasets, or tokenizers unavailable in workspace venv",
            "notes": "PyTorch, datasets, or tokenizers is unavailable in the workspace virtualenv, so the benchmark stays explicitly blocked.",
            "readiness": readiness,
            "config_snapshot": asdict(config),
            "env_overrides": env_overrides,
            "effective_env_path": str(effective_env_path),
        }

    os.environ["TOKENIZERS_PARALLELISM"] = "true" if config.tokenizers_parallelism else "false"

    datasets_module = importlib.import_module("datasets")
    torch_module = importlib.import_module("torch")
    nn_module = importlib.import_module("torch.nn")
    optim_module = importlib.import_module("torch.optim")
    tokenizers_module = importlib.import_module("tokenizers")

    corpus_root = ensure_corpus(config, datasets_module)
    tokenizer_path = ensure_tokenizer(config, tokenizers_module)

    if hasattr(torch_module, "set_num_threads"):
        torch_module.set_num_threads(max(1, config.torch_num_threads))
    if hasattr(torch_module, "set_num_interop_threads"):
        try:
            torch_module.set_num_interop_threads(1)
        except RuntimeError:
            pass
    torch_module.manual_seed(0)

    raw_dataset = datasets_module.load_from_disk(str(corpus_root))
    sampler = ProcessTreeSampler(os.getpid(), case_dir / "metrics_timeseries.csv")
    sampler.start()
    started = time.perf_counter()

    tokenized = raw_dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=config.tokenize_batch_size,
        num_proc=config.datasets_num_proc,
        remove_columns=["text"],
        fn_kwargs={"tokenizer_path": str(tokenizer_path), "max_length": config.max_length},
        desc="tokenize_corpus",
    )
    tokenized = tokenized.with_format("torch", columns=["input_ids", "attention_mask", "label"])

    tokenizer = _tokenizer_for(str(tokenizer_path), config.max_length)
    vocab_size = int(tokenizer.get_vocab_size())
    dataloader = torch_module.utils.data.DataLoader(tokenized, **build_dataloader_kwargs(config))
    model = build_model(
        torch_module,
        nn_module,
        vocab_size=vocab_size,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
    )
    optimizer = optim_module.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn_module.CrossEntropyLoss()

    step_durations: list[float] = []
    loss_trace: list[float] = []
    total_samples = 0
    step_index = 0
    timed_out = False
    timeout_deadline = started + config.timeout_seconds
    try:
        for _epoch in range(config.epochs):
            for batch in dataloader:
                if time.perf_counter() > timeout_deadline:
                    timed_out = True
                    raise TimeoutError(f"benchmark exceeded {config.timeout_seconds}s")
                step_started = time.perf_counter()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(outputs, batch["label"])
                loss.backward()
                optimizer.step()
                step_index += 1
                total_samples += int(batch["label"].shape[0])
                loss_trace.append(float(loss.detach().cpu().item()))
                step_durations.append(time.perf_counter() - step_started)
    finally:
        sampler_summary = sampler.stop()

    wall_time = time.perf_counter() - started
    steps_per_sec = step_index / wall_time if wall_time > 0 else None
    samples_per_sec = total_samples / wall_time if wall_time > 0 else None
    config_summary = (
        f"threads={config.torch_num_threads},map_proc={config.datasets_num_proc},tok_parallel={config.tokenizers_parallelism},"
        f"workers={config.num_workers},prefetch={config.prefetch_factor}"
    )
    notes = "Local datasets.map + tokenizers.encode_batch + PyTorch DataLoader benchmark."
    if env_overrides:
        notes += " Env overrides applied for fixed-baseline tuning."
    if timed_out:
        notes += " Run hit timeout budget."

    return {
        "benchmark": "hf-datasets-tokenizers-dataloader",
        "resource": "dataset processing + tokenization + input pipeline",
        "contending_libraries": ["datasets", "tokenizers", "pytorch", "torch.utils.data"],
        "case_id": config.case_id,
        "label": config.label,
        "mode": config.mode,
        "status": "ok" if not timed_out else "error",
        "success": not timed_out,
        "wall_time_sec": wall_time,
        "throughput": samples_per_sec,
        "samples_per_sec": samples_per_sec,
        "steps_per_sec": steps_per_sec,
        "dataset_rows": config.dataset_rows,
        "tokens_per_row": config.tokens_per_row,
        "max_length": config.max_length,
        "tokenizer_vocab_size": config.tokenizer_vocab_size,
        "hidden_dim": config.hidden_dim,
        "num_classes": config.num_classes,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "torch_num_threads": config.torch_num_threads,
        "datasets_num_proc": config.datasets_num_proc,
        "tokenize_batch_size": config.tokenize_batch_size,
        "tokenizers_parallelism": config.tokenizers_parallelism,
        "num_workers": config.num_workers,
        "prefetch_factor": config.prefetch_factor,
        "persistent_workers": config.persistent_workers,
        "latency_p95": percentile(step_durations, 0.95),
        "latency_p99": percentile(step_durations, 0.99),
        "resource_signal_primary": sampler_summary["write_bytes_delta"],
        "resource_signal_secondary": sampler_summary["peak_threads"],
        "config_summary": config_summary,
        "notes": notes,
        "corpus_root": str(corpus_root),
        "tokenizer_path": str(tokenizer_path),
        "loss_final": loss_trace[-1] if loss_trace else None,
        "loss_trace_head": loss_trace[:8],
        "sampler_summary": sampler_summary,
        "readiness": readiness,
        "config_snapshot": asdict(config),
        "env_overrides": env_overrides,
        "effective_env_path": str(effective_env_path),
    }


def main() -> int:
    args = parse_args()
    if args.case_config:
        config = load_case(Path(args.case_config).resolve())
    else:
        config = config_from_args(args)
    config, env_overrides = apply_env_overrides(config)
    output_json = Path(config.output_json).resolve()
    case_dir = output_json.parent
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = run_case(config, case_dir, env_overrides)
    except Exception as exc:  # pragma: no cover - exercised in subprocess
        payload = {
            "benchmark": "hf-datasets-tokenizers-dataloader",
            "case_id": config.case_id,
            "label": config.label,
            "mode": config.mode,
            "status": "error",
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
    write_json(output_json, payload)
    print(json.dumps(performance_metrics_payload(payload), sort_keys=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
