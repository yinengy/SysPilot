#!/usr/bin/env python3
"""
Example: Using eBPF skills for advanced profiling.

This example demonstrates how to:
1. Check eBPF availability
2. Use built-in eBPF templates
3. Write custom eBPF scripts
"""

import asyncio

from syspilot.skills.ebpf import EBPFSkill, EBPFTemplates


async def main() -> None:
    skill = EBPFSkill()
    
    print("=== Checking eBPF Availability ===")
    result = await skill.execute(action="check_availability")
    print(f"Available: {result.data}")
    
    if not result.data.get("any_available"):
        print("\neBPF tools not available. Install bpftrace or bcc.")
        print("  Ubuntu/Debian: sudo apt install bpftrace bpfcc-tools")
        print("  Fedora: sudo dnf install bpftrace bcc")
        return
    
    print("\n=== Available Templates ===")
    result = await skill.execute(action="list_templates")
    for template in result.data.get("templates", []):
        print(f"  - {template['name']}: {template['description']}")
    
    print("\n=== Template Preview: syscall_count ===")
    script = EBPFTemplates.render("syscall_count", pid=1234, duration=5)
    print(script)
    
    print("\n=== Custom bpftrace Script Example ===")
    custom_script = '''
tracepoint:syscalls:sys_enter_read,
tracepoint:syscalls:sys_enter_write
/pid == $target/
{
    @io_ops[probe] = count();
}

interval:s:5
{
    exit();
}
'''
    print("Script to trace read/write syscalls:")
    print(custom_script)
    
    print("\n=== BCC Python Script Example ===")
    bcc_script = EBPFTemplates.render("bcc_syscall_latency", pid=1234, duration=5)
    print("BCC script for syscall latency:")
    print(bcc_script[:500] + "...")


if __name__ == "__main__":
    asyncio.run(main())
