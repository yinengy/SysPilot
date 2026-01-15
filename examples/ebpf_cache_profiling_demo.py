#!/usr/bin/env python3
"""
Demo: LLM-Generated eBPF for Real Cache Profiling

This demonstrates SysPilot's ability to dynamically generate BCC eBPF programs
and execute them for real cache miss profiling on a cache-intensive workload.

Run as USER: python3 examples/ebpf_cache_profiling_demo.py
The script will use sudo only for the eBPF execution part.
"""

import asyncio
import json
import os
import sys
import subprocess
import multiprocessing
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syspilot.core.config import LLMConfig, LLMProvider
from syspilot.core.llm import LLMInterface, Message


def load_llm_config():
    """Load LLM configuration from JSON file."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "llm_config.json"
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"LLM config not found at {config_path}. "
            "Please create config/llm_config.json with provider, base_url, api_key, and model."
        )
    with open(config_path, 'r') as f:
        return json.load(f)


def create_workload_script():
    """Create a Python script that generates cache-intensive workload."""
    return '''#!/usr/bin/env python3
"""
Cache-intensive workload generator.
Creates memory access patterns that cause cache misses.
"""
import random
import time
import os

def cache_friendly_access(data, iterations):
    """Sequential access - cache friendly."""
    total = 0
    for _ in range(iterations):
        for i in range(len(data)):
            total += data[i]
    return total

def cache_unfriendly_access(data, iterations):
    """Random access - causes cache misses."""
    total = 0
    size = len(data)
    for _ in range(iterations):
        for _ in range(size):
            idx = random.randint(0, size - 1)
            total += data[idx]
    return total

def matrix_multiply_naive(size):
    """Naive matrix multiply - poor cache utilization."""
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    B = [[random.random() for _ in range(size)] for _ in range(size)]
    C = [[0.0] * size for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            for k in range(size):
                C[i][j] += A[i][k] * B[k][j]
    return C

def linked_list_traverse(size):
    """Linked list traversal - pointer chasing causes cache misses."""
    class Node:
        def __init__(self, val):
            self.val = val
            self.next = None
    
    nodes = [Node(i) for i in range(size)]
    random.shuffle(nodes)
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    
    total = 0
    current = nodes[0]
    while current:
        total += current.val
        current = current.next
    return total

def main():
    print(f"PID: {os.getpid()}")
    print("Starting cache-intensive workload...")
    print()
    
    # Phase 1: Large array with random access
    print("[Phase 1] Random array access (cache unfriendly)...")
    large_array = list(range(10_000_000))
    random.shuffle(large_array)
    start = time.time()
    cache_unfriendly_access(large_array, 3)
    print(f"  Completed in {time.time() - start:.2f}s")
    
    # Phase 2: Matrix multiplication
    print("[Phase 2] Naive matrix multiplication...")
    start = time.time()
    matrix_multiply_naive(300)
    print(f"  Completed in {time.time() - start:.2f}s")
    
    # Phase 3: Linked list traversal
    print("[Phase 3] Linked list pointer chasing...")
    start = time.time()
    for _ in range(100):
        linked_list_traverse(100_000)
    print(f"  Completed in {time.time() - start:.2f}s")
    
    # Phase 4: Sequential access for comparison
    print("[Phase 4] Sequential array access (cache friendly)...")
    start = time.time()
    cache_friendly_access(large_array, 3)
    print(f"  Completed in {time.time() - start:.2f}s")
    
    print()
    print("Workload complete!")

if __name__ == "__main__":
    main()
'''


def create_ebpf_profiler_script():
    """Create the BCC eBPF profiler script."""
    return '''#!/usr/bin/env python3
"""
BCC eBPF Cache Profiler - Profiles cache performance for a target process.
"""
from bcc import BPF, PerfType, PerfHWConfig
import time
import sys
import os

TARGET_PID = int(sys.argv[1]) if len(sys.argv) > 1 else -1
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 10

bpf_text = """
#include <uapi/linux/ptrace.h>

BPF_ARRAY(ref_count, u64, 1);
BPF_ARRAY(miss_count, u64, 1);
BPF_HASH(per_cpu_refs, u32, u64);
BPF_HASH(per_cpu_misses, u32, u64);

int on_cache_ref(void *ctx) {
    u32 key = 0;
    u32 cpu = bpf_get_smp_processor_id();
    
    ref_count.atomic_increment(key);
    
    u64 *cpu_val = per_cpu_refs.lookup(&cpu);
    if (cpu_val) {
        (*cpu_val)++;
    } else {
        u64 init = 1;
        per_cpu_refs.update(&cpu, &init);
    }
    return 0;
}

int on_cache_miss(void *ctx) {
    u32 key = 0;
    u32 cpu = bpf_get_smp_processor_id();
    
    miss_count.atomic_increment(key);
    
    u64 *cpu_val = per_cpu_misses.lookup(&cpu);
    if (cpu_val) {
        (*cpu_val)++;
    } else {
        u64 init = 1;
        per_cpu_misses.update(&cpu, &init);
    }
    return 0;
}
"""

print("=" * 60)
print("BCC eBPF Cache Profiler")
print("=" * 60)

if TARGET_PID > 0:
    print(f"Target PID: {TARGET_PID}")
else:
    print("Target: System-wide")
print(f"Duration: {DURATION} seconds")
print()

b = BPF(text=bpf_text)

b.attach_perf_event(
    ev_type=PerfType.HARDWARE,
    ev_config=PerfHWConfig.CACHE_REFERENCES,
    fn_name="on_cache_ref",
    sample_period=100000,
    pid=TARGET_PID if TARGET_PID > 0 else -1
)

b.attach_perf_event(
    ev_type=PerfType.HARDWARE,
    ev_config=PerfHWConfig.CACHE_MISSES,
    fn_name="on_cache_miss",
    sample_period=100000,
    pid=TARGET_PID if TARGET_PID > 0 else -1
)

print(f"Profiling for {DURATION} seconds...")
print()

interval = 1
for i in range(DURATION):
    time.sleep(interval)
    refs = b["ref_count"][0].value
    misses = b["miss_count"][0].value
    rate = (misses / refs * 100) if refs > 0 else 0
    print(f"  [{i+1:2d}s] Refs: {refs:>12,}  Misses: {misses:>10,}  Rate: {rate:>6.2f}%")

print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

total_refs = b["ref_count"][0].value
total_misses = b["miss_count"][0].value
miss_rate = (total_misses / total_refs * 100) if total_refs > 0 else 0

print(f"Total Cache References: {total_refs:>15,}")
print(f"Total Cache Misses:     {total_misses:>15,}")
print(f"Cache Miss Rate:        {miss_rate:>14.2f}%")
print()

print("Per-CPU Statistics:")
print("-" * 40)
for cpu, refs in sorted(b["per_cpu_refs"].items()):
    cpu_misses = b["per_cpu_misses"].get(cpu, type('', (), {'value': 0})()).value
    cpu_rate = (cpu_misses / refs.value * 100) if refs.value > 0 else 0
    print(f"  CPU {cpu.value:2d}: Refs={refs.value:>10,}  Misses={cpu_misses:>8,}  Rate={cpu_rate:.2f}%")

print("=" * 60)
'''


async def demo_ebpf_generation():
    print("=" * 70)
    print("DYNAMIC eBPF GENERATION DEMO")
    print("LLM generates BCC Python script for REAL cache profiling")
    print("=" * 70)

    llm_cfg = load_llm_config()
    print(f"\nUsing: {llm_cfg['model']}")

    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=llm_cfg["model"],
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg["base_url"],
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 4000),
    )
    llm = LLMInterface(config)

    print("\n" + "=" * 70)
    print("[STEP 1] Creating cache-intensive workload script...")
    print("=" * 70)

    workload_path = "/tmp/syspilot_workload.py"
    with open(workload_path, 'w') as f:
        f.write(create_workload_script())
    os.chmod(workload_path, 0o755)
    print(f"   Workload script saved to: {workload_path}")

    print("\n" + "=" * 70)
    print("[STEP 2] Asking LLM to generate BCC eBPF profiler...")
    print("=" * 70)

    prompt = """Write a BCC Python script that profiles CPU cache performance for a target process.

Use this EXACT working BPF code pattern:

```c
BPF_ARRAY(ref_count, u64, 1);
BPF_ARRAY(miss_count, u64, 1);
BPF_HASH(per_cpu_refs, u32, u64);
BPF_HASH(per_cpu_misses, u32, u64);

int on_cache_ref(void *ctx) {
    u32 key = 0;
    u32 cpu = bpf_get_smp_processor_id();
    ref_count.atomic_increment(key);
    u64 *cpu_val = per_cpu_refs.lookup(&cpu);
    if (cpu_val) { (*cpu_val)++; }
    else { u64 init = 1; per_cpu_refs.update(&cpu, &init); }
    return 0;
}

int on_cache_miss(void *ctx) {
    u32 key = 0;
    u32 cpu = bpf_get_smp_processor_id();
    miss_count.atomic_increment(key);
    u64 *cpu_val = per_cpu_misses.lookup(&cpu);
    if (cpu_val) { (*cpu_val)++; }
    else { u64 init = 1; per_cpu_misses.update(&cpu, &init); }
    return 0;
}
```

Requirements:
1. Use `from bcc import BPF, PerfType, PerfHWConfig`
2. Accept TARGET_PID from sys.argv[1] (default -1 for system-wide)
3. Accept DURATION from sys.argv[2] (default 15 seconds)
4. Use `void *ctx` as function parameter (NOT struct bpf_perf_event_data)
5. Use atomic_increment for counters
6. Attach with: ev_type=PerfType.HARDWARE, ev_config=PerfHWConfig.CACHE_REFERENCES/CACHE_MISSES
7. Use sample_period=100000 and pid=TARGET_PID
8. Print live stats every second during profiling
9. Print final summary with total refs, misses, miss rate, and per-CPU breakdown

Output ONLY the complete Python code."""

    messages = [Message(role="user", content=prompt)]

    print("\n   Sending prompt to LLM...")
    response = await llm.chat(messages)
    generated_script = response.content

    if "```python" in generated_script:
        generated_script = generated_script.split("```python")[1].split("```")[0]
    elif "```" in generated_script:
        generated_script = generated_script.split("```")[1].split("```")[0]

    generated_script = generated_script.strip()

    print("\n   LLM-generated BCC script (first 50 lines):")
    print("   " + "-" * 60)
    lines = generated_script.split('\n')
    for line in lines[:50]:
        print(f"   {line}")
    if len(lines) > 50:
        print(f"   ... ({len(lines) - 50} more lines)")
    print("   " + "-" * 60)

    profiler_path = "/tmp/syspilot_cache_profiler.py"
    with open(profiler_path, 'w') as f:
        f.write(generated_script)
    os.chmod(profiler_path, 0o755)
    print(f"\n   Profiler script saved to: {profiler_path}")

    print("\n" + "=" * 70)
    print("[STEP 3] Starting workload and profiler...")
    print("=" * 70)

    print("\n   Starting workload in background...")
    workload_proc = subprocess.Popen(
        ['python3', workload_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    time.sleep(0.5)
    workload_pid = workload_proc.pid
    print(f"   Workload PID: {workload_pid}")

    print(f"\n   Starting eBPF profiler (targeting PID {workload_pid})...")
    print("   Enter sudo password when prompted:\n")

    profiler_result = subprocess.run(
        ['sudo', 'python3', profiler_path, str(workload_pid), '20'],
        timeout=120
    )

    print("\n   Waiting for workload to complete...")
    workload_output, _ = workload_proc.communicate(timeout=60)
    
    print("\n   === WORKLOAD OUTPUT ===")
    print(workload_output)

    print("\n" + "=" * 70)
    print("[SUMMARY]")
    print("=" * 70)
    print("""
   This demo showed:
   
   1. LLM dynamically generated a BCC eBPF profiler script
   2. A cache-intensive workload was created with:
      - Random array access (cache unfriendly)
      - Naive matrix multiplication (poor cache utilization)
      - Linked list pointer chasing (causes cache misses)
      - Sequential access (cache friendly, for comparison)
   3. Real hardware cache events were captured via eBPF
   4. Per-CPU cache statistics were collected
   
   The agent can generate ANY eBPF program on-demand for custom profiling.
""")


if __name__ == "__main__":
    asyncio.run(demo_ebpf_generation())
