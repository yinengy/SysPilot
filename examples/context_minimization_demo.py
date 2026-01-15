#!/usr/bin/env python3
"""
Demo: Real Context Minimization with LLM-Generated Scripts

This script demonstrates the REAL flow where the SysPilot agent:
1. Collects raw profiling data (strace)
2. Asks the LLM to generate a processing script
3. Executes the LLM-generated script
4. Returns minimized context

Requirements:
- strace installed on the system

Run: python3 examples/context_minimization_demo.py
"""

import asyncio
import json
import subprocess
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syspilot.core.config import SysPilotConfig, LLMConfig, LLMProvider
from syspilot.core.llm import LLMInterface, Message
from syspilot.skills.builtin import DynamicScriptSkill


def load_llm_config():
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


async def demo_real_context_minimization():
    print("=" * 70)
    print("REAL CONTEXT MINIMIZATION DEMO")
    print("LLM generates the processing script dynamically")
    print("=" * 70)

    llm_cfg = load_llm_config()
    print(f"\nUsing proxy: {llm_cfg['base_url']}")
    print(f"Model: {llm_cfg['model']}")

    # Step 1: Generate real strace data
    print("\n" + "=" * 70)
    print("[STEP 1] Generating real strace data...")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        strace_file = f.name

    subprocess.run(
        ['strace', '-o', strace_file, '-T', 'ls', '-la', '/usr'],
        capture_output=True, text=True
    )

    with open(strace_file) as f:
        raw_strace = f.read()

    raw_lines = raw_strace.strip().split('\n')
    raw_bytes = len(raw_strace)

    print(f"\n   Raw strace output: {len(raw_lines)} lines, {raw_bytes:,} bytes")
    print(f"\n   Sample raw lines (first 5):")
    for line in raw_lines[:5]:
        print(f"   | {line[:75]}...")
    print(f"   | ... ({len(raw_lines) - 5} more lines)")

    # Step 2: Ask LLM to generate a processing script
    print("\n" + "=" * 70)
    print("[STEP 2] Asking LLM to generate a processing script...")
    print("=" * 70)

    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model=llm_cfg["model"],
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg["base_url"],
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 2000),
    )
    llm = LLMInterface(config)

    # Show a sample of the raw data to the LLM (not all of it - that's the point!)
    sample_lines = '\n'.join(raw_lines[:10])

    prompt = f"""Write a Python script to analyze strace output.

The strace data format looks like this (sample):
```
{sample_lines}
```

CRITICAL REQUIREMENTS:
- A variable named `raw_data` will ALREADY be defined before your script runs
- Your script must NOT define `raw_data` - just USE it
- The actual `raw_data` contains {len(raw_lines)} lines of strace output

Your script should:
1. Parse `raw_data` to count syscalls by name
2. Count errors (lines with "= -1")
3. Extract unique file paths from the data
4. Print a summary with: total syscalls, top 5 syscalls, error count, unique files count

Output ONLY the Python code. Do NOT include raw_data definition."""

    messages = [Message(role="user", content=prompt)]

    print("\n   Prompt sent to LLM (asking for processing script)...")
    print("   Waiting for LLM response...")

    response = await llm.chat(messages)
    generated_script = response.content

    # Clean up the script (remove markdown code blocks if present)
    if "```python" in generated_script:
        generated_script = generated_script.split("```python")[1].split("```")[0]
    elif "```" in generated_script:
        generated_script = generated_script.split("```")[1].split("```")[0]

    print("\n   LLM-generated script:")
    print("   " + "-" * 50)
    for line in generated_script.strip().split('\n')[:25]:
        print(f"   {line}")
    if len(generated_script.strip().split('\n')) > 25:
        print(f"   ... ({len(generated_script.strip().split(chr(10))) - 25} more lines)")
    print("   " + "-" * 50)

    # Step 3: Execute the LLM-generated script with real data
    print("\n" + "=" * 70)
    print("[STEP 3] Executing LLM-generated script with real data...")
    print("=" * 70)

    # Inject the raw data into the script
    full_script = f'''raw_data = """{raw_strace}"""

{generated_script}
'''

    skill = DynamicScriptSkill()
    result = await skill.execute(script_type="python", script=full_script)

    if not result.success:
        print(f"\n   Script execution failed: {result.error}")
        print(f"   Output: {result.output}")
        os.unlink(strace_file)
        return

    minimized_output = result.output.strip()
    minimized_bytes = len(minimized_output)

    print(f"\n   === MINIMIZED OUTPUT (what LLM would receive next) ===")
    print(f"\n{minimized_output}")

    # Step 4: Summary
    print("\n" + "=" * 70)
    print("[SUMMARY] Context Minimization Results")
    print("=" * 70)
    print(f"   Raw strace:        {raw_bytes:>8,} bytes  ({len(raw_lines)} lines)")
    print(f"   Minimized output:  {minimized_bytes:>8,} bytes")
    print(f"   Compression:       {raw_bytes / max(minimized_bytes, 1):>8.1f}x smaller")
    print(f"   Token savings:     ~{(raw_bytes - minimized_bytes) // 4:,} tokens saved")
    print()
    print("   KEY INSIGHT: The LLM generated the processing script itself!")
    print("   It decided what metrics matter and how to extract them.")
    print("   This is dynamic, not hardcoded.")

    os.unlink(strace_file)


if __name__ == "__main__":
    asyncio.run(demo_real_context_minimization())
