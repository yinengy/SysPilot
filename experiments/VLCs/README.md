# VLCs Transformer Experiment

This workspace packages the compiled VLC transformer target under `experiments/VLCs/`.

## Packaged files

- `bin/parallel_vlc` — copied compiled target binary.
- `config/transformer.json` — transformer VLC configuration.
- `benchmarks/Transformer/` — shared library tree referenced by the config.
- `wikitext2_processed.pt` — dataset file expected from the runtime working directory.
- `run_vlcs.py` — wrapper that exposes one core-assignment flag for each of the 24 cores on this host.

## Wrapper usage

Each `--core-N` flag assigns core `N` to workflow `0` or workflow `1`. Omitted core flags are left unassigned. A core can only appear in one workflow because each core has at most one assignment flag. The wrapper converts the explicitly assigned cores into the two core-list arguments expected by `parallel_vlc`, captures the target program output internally, and prints only a final JSON object.

Both workflows must receive at least one core.

Run a two-core split with core 0 assigned to workflow 0 and core 1 assigned to workflow 1:

```bash
./run_vlcs.py --core-0 0 --core-1 1
```

Run a larger split:

```bash
./run_vlcs.py \
  --core-0 0 --core-1 0 --core-2 0 --core-3 0 \
  --core-12 1 --core-13 1 --core-14 1 --core-15 1
```

With `./run_vlcs.py --core-0 0 --core-1 1`, the generated target command receives only `0` and `1` as the two workflow core lists.

Expected output:

```json
{"walltime":41.576143,"workflow_0_walltime":41.0524,"workflow_1_walltime":22.6502}
```

`workflow_0_walltime` is parsed from `VLC 1(transformer) end time`, and `workflow_1_walltime` is parsed from `VLC 2(transformer) end time`.

## SysPilot tuning example

The packaged LPCL contract is `contracts/vlcs_libtorch_cpu/contract.yaml`. It models the target as a simplified CPU-only LibTorch transformer and exposes only the wrapper's `--core-0` through `--core-23` mapping controls.

```bash
python3 -m syspilot.cli tune \
  --workdir /home/yyan/SysPilot/experiments/VLCs \
  --ui-mode plain \
  --command "./run_vlcs.py --core-0 0 --core-1 1" \
  --contract vlcs_libtorch_cpu \
  --primary-objective walltime \
  --primary-objective-mode minimize
```

## SMAC3/OpenTuner autotuner adaptor

`run_vlcs_autotuners.py` lives in this experiment folder and tunes the same 24 `--core-N` assignment controls. It seeds the requested split with cores 0-11 assigned to workflow 0 and cores 12-23 assigned to workflow 1, minimizes the wrapper JSON `walltime`, and writes raw tuner logs, per-trial stdout/stderr, summaries, and best-so-far series under the repo-level `runs/` directory.

Run the requested 60-evaluation search for both tuners:

```bash
cd /home/yyan/SysPilot/experiments/VLCs
/home/yyan/miniconda3/envs/agent/bin/python run_vlcs_autotuners.py --budget 60 --tuner both
```

Useful validation commands:

```bash
/home/yyan/miniconda3/envs/agent/bin/python run_vlcs_autotuners.py --dry-run --budget 60
/home/yyan/miniconda3/envs/agent/bin/python run_vlcs_autotuners.py --budget 60 --tuner smac3
/home/yyan/miniconda3/envs/agent/bin/python run_vlcs_autotuners.py --budget 60 --tuner opentuner
```

The script requires `opentuner` for `--tuner opentuner` and `smac` plus `ConfigSpace` for `--tuner smac3`. If a selected tuner is missing, it exits before launching trials with an explicit dependency error.

Artifacts are written to `runs/vlcs_autotuners_budget<BUDGET>_<UTC timestamp>/` by default:

- `raw/<tuner>.log` — raw optimizer and trial progress output.
- `tuner_outputs/<tuner>/trials/trial_*/stdout.txt` and `stderr.txt` — raw VLC wrapper output per evaluation.
- `tuner_outputs/<tuner>/trials.csv` and `summary.json` — trial table and best configuration summary.
- `best_series/vlcs_<tuner>_b<BUDGET>.csv` — best-so-far walltime series.
- `summary.csv` — cross-tuner summary.
