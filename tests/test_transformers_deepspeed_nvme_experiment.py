from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

import pytest


class ExperimentModule(Protocol):
    def load_case_config(self, path: Path) -> object: ...

    def build_deepspeed_config(self, config: object, paths: dict[str, Path]) -> dict[str, Any]: ...


def _workspace_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "Transformers"
        / "deepspeed-nvme-offload-io-contention"
    )


def _load_module() -> ExperimentModule:
    path = _workspace_dir() / "run_clm_case.py"
    spec = importlib.util.spec_from_file_location("transformers_ds_nvme_run_case", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(ExperimentModule, cast(object, module))


def _base_case_payload(output_json: Path) -> dict[str, Any]:
    return {
        "case_id": "smoke_case",
        "label": "smoke case",
        "output_json": str(output_json),
        "runtime_root": "/tmp/hf-ds-nvme-tests",
        "env_python": "/usr/bin/python3",
        "deepspeed_bin": "/usr/bin/deepspeed",
        "transformers_source_root": "/tmp/transformers",
        "run_clm_path": "/tmp/transformers/run_clm.py",
        "timeout_seconds": 60,
        "samplers": {
            "vmstat": False,
            "iostat": False,
            "pidstat": False,
            "nvidia_smi": False,
        },
        "runtime_layout": {
            "hf_home": "/tmp/hf-home",
            "hf_datasets_cache": "/tmp/hf-datasets",
            "huggingface_hub_cache": "/tmp/hf-hub",
            "transformers_cache": "/tmp/transformers-cache",
            "torch_extensions_dir": "/tmp/torch-extensions",
            "triton_cache_dir": "/tmp/triton-cache",
            "tmp_root": "/tmp/hf-tmp",
            "shared_io_root": "/tmp/shared-io",
            "isolated_offload_root": "/tmp/isolated-offload",
            "isolated_output_root": "/tmp/isolated-output",
            "isolated_scratch_root": "/tmp/isolated-scratch",
        },
        "fixed_workload": {
            "model_name_or_path": "gpt2",
            "dataset_name": "wikitext",
            "dataset_config_name": "wikitext-2-raw-v1",
            "block_size": 128,
            "max_steps": 4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "save_steps": 2,
            "save_total_limit": 1,
            "logging_steps": 1,
            "learning_rate": 5e-05,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "seed": 17,
            "num_gpus": 1,
            "fp16": True,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 1,
            "preprocessing_num_workers": 1,
            "disable_tqdm": True,
            "report_to": "none",
        },
        "tuning": {
            "zero_stage": 3,
            "offload_optimizer_device": "nvme",
            "offload_param_device": "nvme",
            "path_mode": "shared",
            "optimizer_buffer_count": 4,
            "optimizer_fast_init": False,
            "param_buffer_count": 5,
            "param_buffer_size": 100000000,
            "max_in_cpu_params": 100000000,
            "stage3_gather_16bit_weights_on_model_save": False,
            "aio_block_size": 1048576,
            "aio_queue_depth": 16,
            "aio_thread_count": 1,
            "aio_single_submit": False,
            "aio_overlap_events": True,
            "env_overrides": {},
        },
    }


def _write_case_config(tmp_path: Path, payload: dict[str, Any], *, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_build_deepspeed_config_for_zero2_omits_stage3_keys(tmp_path: Path) -> None:
    module = _load_module()
    payload = copy.deepcopy(_base_case_payload(tmp_path / "case" / "result.json"))
    payload["tuning"]["zero_stage"] = 2
    payload["tuning"]["offload_param_device"] = "none"

    config_path = _write_case_config(tmp_path, payload, name="zero2_case.json")
    config = module.load_case_config(config_path)
    ds_config = module.build_deepspeed_config(config, {"offload_root": Path("/tmp/offload")})
    zero_optimization = ds_config["zero_optimization"]

    assert zero_optimization["stage"] == 2
    assert zero_optimization["offload_optimizer"]["device"] == "nvme"
    assert "offload_param" not in zero_optimization
    assert "stage3_prefetch_bucket_size" not in zero_optimization
    assert "stage3_param_persistence_threshold" not in zero_optimization
    assert "stage3_gather_16bit_weights_on_model_save" not in zero_optimization


def test_build_deepspeed_config_for_zero3_preserves_stage3_settings(tmp_path: Path) -> None:
    module = _load_module()
    payload = copy.deepcopy(_base_case_payload(tmp_path / "case" / "result.json"))
    payload["tuning"]["stage3_gather_16bit_weights_on_model_save"] = True

    config_path = _write_case_config(tmp_path, payload, name="zero3_case.json")
    config = module.load_case_config(config_path)
    ds_config = module.build_deepspeed_config(config, {"offload_root": Path("/tmp/offload")})
    zero_optimization = ds_config["zero_optimization"]

    assert zero_optimization["stage"] == 3
    assert zero_optimization["stage3_prefetch_bucket_size"] == 50000000
    assert zero_optimization["stage3_param_persistence_threshold"] == 100000
    assert zero_optimization["stage3_gather_16bit_weights_on_model_save"] is True
    assert zero_optimization["offload_param"]["device"] == "nvme"
    assert zero_optimization["offload_param"]["nvme_path"] == "/tmp/offload/params"


def test_load_case_config_rejects_param_offload_outside_zero3(tmp_path: Path) -> None:
    module = _load_module()
    payload = copy.deepcopy(_base_case_payload(tmp_path / "case" / "result.json"))
    payload["tuning"]["zero_stage"] = 2
    payload["tuning"]["offload_param_device"] = "cpu"

    config_path = _write_case_config(tmp_path, payload, name="invalid_zero2_case.json")

    with pytest.raises(ValueError, match="offload_param_device requires tuning.zero_stage == 3"):
        module.load_case_config(config_path)
