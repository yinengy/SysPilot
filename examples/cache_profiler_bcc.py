#!/usr/bin/env python3
"""
LLM-Generated BCC script for cache profiling.
Run with: sudo python3 examples/cache_profiler_bcc.py
"""

from bcc import BPF
import time
import ctypes as ct

bpf_text = """
#include <uapi/linux/ptrace.h>

BPF_ARRAY(cache_refs, u64, 1);
BPF_ARRAY(cache_misses, u64, 1);

int on_cache_ref(struct bpf_perf_event_data *ctx) {
    u32 key = 0;
    u64 *val = cache_refs.lookup(&key);
    if (val) {
        (*val)++;
    }
    return 0;
}

int on_cache_miss(struct bpf_perf_event_data *ctx) {
    u32 key = 0;
    u64 *val = cache_misses.lookup(&key);
    if (val) {
        (*val)++;
    }
    return 0;
}
"""

b = BPF(text=bpf_text)

b.attach_perf_event(
    ev_type=BPF.PERF_TYPE_HARDWARE,
    ev_config=BPF.PERF_COUNT_HW_CACHE_REFERENCES,
    fn_name="on_cache_ref",
    sample_period=1000
)

b.attach_perf_event(
    ev_type=BPF.PERF_TYPE_HARDWARE,
    ev_config=BPF.PERF_COUNT_HW_CACHE_MISSES,
    fn_name="on_cache_miss",
    sample_period=1000
)

print("Profiling CPU cache performance for 3 seconds...")
time.sleep(3)

cache_refs = b["cache_refs"]
cache_misses = b["cache_misses"]

total_refs = cache_refs[ct.c_int(0)].value
total_misses = cache_misses[ct.c_int(0)].value

miss_rate = (total_misses / total_refs * 100) if total_refs > 0 else 0

print("\n=== CPU Cache Performance Summary ===")
print(f"Total cache references: {total_refs:,}")
print(f"Total cache misses:     {total_misses:,}")
print(f"Cache miss rate:        {miss_rate:.2f}%")
