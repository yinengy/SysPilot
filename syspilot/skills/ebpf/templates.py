from __future__ import annotations

from typing import Any


class EBPFTemplates:
    """Pre-built eBPF program templates for common profiling tasks."""
    
    SYSCALL_COUNT = '''#!/usr/bin/env bpftrace
// Count syscalls by process
tracepoint:raw_syscalls:sys_enter
/pid == {pid}/
{{
    @syscalls[comm, args->id] = count();
}}

interval:s:{duration}
{{
    print(@syscalls);
    clear(@syscalls);
    exit();
}}
'''

    CPU_PROFILE = '''#!/usr/bin/env bpftrace
// CPU profiling - sample stack traces
profile:hz:99
/pid == {pid}/
{{
    @stacks[ustack] = count();
}}

interval:s:{duration}
{{
    exit();
}}

END
{{
    print(@stacks, 10);
}}
'''

    MEMORY_ALLOC = '''#!/usr/bin/env bpftrace
// Track memory allocations
uprobe:/lib/x86_64-linux-gnu/libc.so.6:malloc
/pid == {pid}/
{{
    @alloc_sizes = hist(arg0);
    @total_allocs = count();
}}

uprobe:/lib/x86_64-linux-gnu/libc.so.6:free
/pid == {pid}/
{{
    @total_frees = count();
}}

interval:s:{duration}
{{
    exit();
}}
'''

    IO_LATENCY = '''#!/usr/bin/env bpftrace
// I/O latency histogram
tracepoint:block:block_rq_issue
/pid == {pid}/
{{
    @start[args->dev, args->sector] = nsecs;
}}

tracepoint:block:block_rq_complete
/@start[args->dev, args->sector]/
{{
    @io_latency_us = hist((nsecs - @start[args->dev, args->sector]) / 1000);
    delete(@start[args->dev, args->sector]);
}}

interval:s:{duration}
{{
    exit();
}}
'''

    FILE_OPENS = '''#!/usr/bin/env bpftrace
// Track file opens
tracepoint:syscalls:sys_enter_openat
/pid == {pid}/
{{
    @files[str(args->filename)] = count();
}}

interval:s:{duration}
{{
    exit();
}}

END
{{
    print(@files, 20);
}}
'''

    NETWORK_CONNECTIONS = '''#!/usr/bin/env bpftrace
// Track network connections
tracepoint:syscalls:sys_enter_connect
/pid == {pid}/
{{
    @connects = count();
}}

tracepoint:syscalls:sys_enter_accept,
tracepoint:syscalls:sys_enter_accept4
/pid == {pid}/
{{
    @accepts = count();
}}

interval:s:{duration}
{{
    exit();
}}
'''

    CONTEXT_SWITCHES = '''#!/usr/bin/env bpftrace
// Track context switches
tracepoint:sched:sched_switch
/args->prev_pid == {pid} || args->next_pid == {pid}/
{{
    @switches = count();
    @switch_reasons[args->prev_state] = count();
}}

interval:s:{duration}
{{
    exit();
}}
'''

    PAGE_FAULTS = '''#!/usr/bin/env bpftrace
// Track page faults
software:page-faults:1
/pid == {pid}/
{{
    @page_faults = count();
    @fault_addrs = hist(arg0);
}}

interval:s:{duration}
{{
    exit();
}}
'''

    NUMA_PAGE_MIGRATIONS = '''#!/usr/bin/env bpftrace
// Track NUMA page migrations via migrate_pages syscall
tracepoint:syscalls:sys_enter_migrate_pages
/pid == {pid}/
{{
    @migrations = count();
}}

tracepoint:exceptions:page_fault_user
/pid == {pid}/
{{
    @user_faults = count();
}}

tracepoint:exceptions:page_fault_kernel
/pid == {pid}/
{{
    @kernel_faults = count();
}}

interval:s:{duration}
{{
    exit();
}}
'''

    CACHE_MISSES = '''#!/usr/bin/env bpftrace
// Track hardware cache misses
hardware:cache-misses:1000
/pid == {pid}/
{{
    @cache_misses = count();
    @miss_stacks[ustack(5)] = count();
}}

hardware:cache-references:10000
/pid == {pid}/
{{
    @cache_refs = count();
}}

interval:s:{duration}
{{
    exit();
}}
'''

    BCC_SYSCALL_LATENCY = '''#!/usr/bin/env python3
from bcc import BPF
import time

bpf_text = """
#include <uapi/linux/ptrace.h>

BPF_HASH(start, u32);
BPF_HISTOGRAM(dist);

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != {pid}) return 0;
    u64 ts = bpf_ktime_get_ns();
    start.update(&pid, &ts);
    return 0;
}}

TRACEPOINT_PROBE(raw_syscalls, sys_exit) {{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != {pid}) return 0;
    u64 *tsp = start.lookup(&pid);
    if (tsp == 0) return 0;
    u64 delta = bpf_ktime_get_ns() - *tsp;
    dist.increment(bpf_log2l(delta / 1000));
    start.delete(&pid);
    return 0;
}}
"""

b = BPF(text=bpf_text)
print("Tracing syscall latency for PID {pid}... Duration: {duration}s")
time.sleep({duration})
print("\\nSyscall latency (us):")
b["dist"].print_log2_hist("latency")
'''

    @classmethod
    def get_template(cls, name: str) -> str | None:
        templates = {
            "syscall_count": cls.SYSCALL_COUNT,
            "cpu_profile": cls.CPU_PROFILE,
            "memory_alloc": cls.MEMORY_ALLOC,
            "io_latency": cls.IO_LATENCY,
            "file_opens": cls.FILE_OPENS,
            "network_connections": cls.NETWORK_CONNECTIONS,
            "context_switches": cls.CONTEXT_SWITCHES,
            "page_faults": cls.PAGE_FAULTS,
            "numa_page_migrations": cls.NUMA_PAGE_MIGRATIONS,
            "cache_misses": cls.CACHE_MISSES,
            "bcc_syscall_latency": cls.BCC_SYSCALL_LATENCY,
        }
        return templates.get(name)
    
    @classmethod
    def list_templates(cls) -> list[dict[str, str]]:
        return [
            {"name": "syscall_count", "description": "Count syscalls by type"},
            {"name": "cpu_profile", "description": "CPU profiling with stack traces"},
            {"name": "memory_alloc", "description": "Track memory allocations"},
            {"name": "io_latency", "description": "I/O latency histogram"},
            {"name": "file_opens", "description": "Track file open operations"},
            {"name": "network_connections", "description": "Track network connections"},
            {"name": "context_switches", "description": "Track context switches"},
            {"name": "page_faults", "description": "Track page faults"},
            {"name": "numa_page_migrations", "description": "Track NUMA page migrations and cross-node faults"},
            {"name": "cache_misses", "description": "Track hardware cache misses with stack traces"},
            {"name": "bcc_syscall_latency", "description": "Syscall latency with BCC"},
        ]
    
    @classmethod
    def render(cls, name: str, pid: int, duration: int = 5) -> str | None:
        template = cls.get_template(name)
        if template:
            return template.format(pid=pid, duration=duration)
        return None
