#!/usr/bin/env python3
"""
Example: Dynamic script generation for custom profiling.

This example shows how the agent can dynamically generate
scripts to process profiling data and extract key insights.
"""

import asyncio

from syspilot.skills.builtin import DynamicScriptSkill


STRACE_PROCESSOR_SCRIPT = '''
import sys
import re
from collections import Counter

def process_strace(input_text: str) -> dict:
    """Process strace output and extract key metrics."""
    syscalls = Counter()
    errors = Counter()
    slow_calls = []
    
    for line in input_text.strip().split("\\n"):
        # Match syscall pattern: syscall(args) = result <time>
        match = re.match(r"(\\w+)\\([^)]*\\)\\s*=\\s*(-?\\d+|\\?)(.*)", line)
        if match:
            syscall, result, rest = match.groups()
            syscalls[syscall] += 1
            
            # Check for errors
            if result == "-1" or "E" in rest:
                error_match = re.search(r"(E\\w+)", rest)
                if error_match:
                    errors[error_match.group(1)] += 1
            
            # Check for slow calls (>100ms)
            time_match = re.search(r"<([\\d.]+)>", rest)
            if time_match and float(time_match.group(1)) > 0.1:
                slow_calls.append((syscall, float(time_match.group(1))))
    
    return {
        "total_syscalls": sum(syscalls.values()),
        "top_syscalls": syscalls.most_common(10),
        "errors": dict(errors),
        "slow_calls": sorted(slow_calls, key=lambda x: -x[1])[:5],
    }

# Example usage with sample data
sample = """
read(3, "data", 1024) = 512 <0.000123>
write(1, "output", 6) = 6 <0.000045>
open("/etc/passwd", O_RDONLY) = 4 <0.000234>
read(4, "root:x:0:0", 4096) = 1024 <0.150000>
close(4) = 0 <0.000012>
stat("/tmp/test", 0x7fff...) = -1 ENOENT (No such file or directory) <0.000089>
"""

result = process_strace(sample)
print("=== Strace Analysis ===")
print(f"Total syscalls: {result['total_syscalls']}")
print(f"Top syscalls: {result['top_syscalls']}")
print(f"Errors: {result['errors']}")
print(f"Slow calls: {result['slow_calls']}")
'''


PERF_PROCESSOR_SCRIPT = '''
import sys
import re

def process_perf_report(input_text: str) -> dict:
    """Process perf report output and extract hotspots."""
    hotspots = []
    
    for line in input_text.strip().split("\\n"):
        # Match perf report format: overhead% command shared_object symbol
        match = re.match(r"\\s*([\\d.]+)%\\s+(\\S+)\\s+(\\S+)\\s+\\[.\\]\\s+(.*)", line)
        if match:
            overhead, command, shared_obj, symbol = match.groups()
            hotspots.append({
                "overhead": float(overhead),
                "command": command,
                "library": shared_obj,
                "function": symbol,
            })
    
    return {
        "total_hotspots": len(hotspots),
        "top_functions": hotspots[:10],
        "by_library": _group_by_library(hotspots),
    }

def _group_by_library(hotspots):
    libs = {}
    for h in hotspots:
        lib = h["library"]
        libs[lib] = libs.get(lib, 0) + h["overhead"]
    return sorted(libs.items(), key=lambda x: -x[1])[:5]

# Example
sample = """
    25.00%  myapp  myapp       [.] hot_function
    15.00%  myapp  libc.so.6   [.] malloc
    10.00%  myapp  libc.so.6   [.] free
     8.00%  myapp  myapp       [.] process_data
     5.00%  myapp  libpthread  [.] pthread_mutex_lock
"""

result = process_perf_report(sample)
print("=== Perf Analysis ===")
print(f"Hotspots found: {result['total_hotspots']}")
for f in result['top_functions'][:3]:
    print(f"  {f['overhead']:.1f}% - {f['function']} ({f['library']})")
print(f"By library: {result['by_library']}")
'''


async def main() -> None:
    skill = DynamicScriptSkill()
    
    print("=== Running Strace Processor Script ===")
    result = await skill.execute(
        script_type="python",
        script=STRACE_PROCESSOR_SCRIPT,
    )
    print(result.output)
    
    print("\n=== Running Perf Processor Script ===")
    result = await skill.execute(
        script_type="python",
        script=PERF_PROCESSOR_SCRIPT,
    )
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
