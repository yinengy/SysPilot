# Toolbox virtual probes

`toolbox/` is the repo-level source of truth for LPCL `virtual_probe` support.

Mapping rule:

- `process.cpu.thread_count` → `toolbox/probes/process/cpu/thread_count.py`
- `cgroup.memory.events.high` → `toolbox/probes/cgroup/memory/events/high.py`

Each probe script is responsible for one `virtual_probe` name. If a contract declares a probe and the matching script is missing, `check-contract` should fail.

Probe scripts may expose either:

- `collect(payload) -> dict` for one-shot probes
- `begin(payload) -> Any` and `finish(payload, state) -> dict` for probes that need live sampling during the run

Standard result shape:

```json
{
  "observed": true,
  "source": "runtime_output",
  "value": 7,
  "excerpt": ["matching line"],
  "flat": {"memory.events.high": 7},
  "structured": {"status": "pressured"}
}
```
