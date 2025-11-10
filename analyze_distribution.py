#!/usr/bin/env python3
"""Analyze the distribution of gas-containing subhalos across ID ranges."""
import json
from collections import defaultdict

# Read all API responses
responses = []

with open("api_responses.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            if "response" in data:
                responses.append(data)
        except:
            pass

print(f"Total API responses: {len(responses)}")
print()

# Analyze distribution by ID ranges
ranges = [
    (1, 2000, "1-2000"),
    (2001, 5000, "2001-5000"),
    (5001, 10000, "5001-10000"),
    (10001, 20000, "10001-20000"),
]

print("="*70)
print("DISTRIBUTION OF GAS-CONTAINING SUBHALOS BY ID RANGE")
print("="*70)

for start, end, label in ranges:
    in_range = [r for r in responses if start <= r["subhalo_id"] <= end]
    with_gas = [r for r in in_range if r["response"].get("mass_gas", 0) > 0]
    
    print(f"\n{label} (IDs {start}-{end}):")
    print(f"  Total checked: {len(in_range)}")
    if len(in_range) > 0:
        print(f"  With gas: {len(with_gas)} ({len(with_gas)/len(in_range)*100:.1f}% if checked)")
    else:
        print(f"  With gas: {len(with_gas)} (no checks in this range)")
    
    if with_gas:
        gas_masses = [r["response"]["mass_gas"] * 1e10 for r in with_gas]
        print(f"  Gas mass range: {min(gas_masses):.2e} to {max(gas_masses):.2e} Msun")
        print(f"  Average gas mass: {sum(gas_masses)/len(gas_masses):.2e} Msun")

# Check if we're actually checking all subhalos in sequence
print()
print("="*70)
print("CHECKING IF WE'RE CHECKING ALL SUBHALOS SEQUENTIALLY")
print("="*70)

checked_ids = sorted([r["subhalo_id"] for r in responses])
print(f"First 20 checked IDs: {checked_ids[:20]}")
print(f"Last 20 checked IDs: {checked_ids[-20:]}")

# Check for gaps
gaps = []
for i in range(len(checked_ids) - 1):
    if checked_ids[i+1] - checked_ids[i] > 1:
        gaps.append((checked_ids[i], checked_ids[i+1]))

if gaps:
    print(f"\nFound {len(gaps)} gaps in sequence:")
    for gap in gaps[:10]:  # Show first 10 gaps
        print(f"  Gap: {gap[0]} -> {gap[1]} (missing {gap[1] - gap[0] - 1} IDs)")
else:
    print("\nNo gaps found - checking sequentially")

# Analyze gas distribution more granularly
print()
print("="*70)
print("GAS DISTRIBUTION BY 500-ID BLOCKS")
print("="*70)

for block_start in range(1, 20001, 500):
    block_end = min(block_start + 499, 20000)
    in_block = [r for r in responses if block_start <= r["subhalo_id"] <= block_end]
    with_gas = [r for r in in_block if r["response"].get("mass_gas", 0) > 0]
    
    if len(in_block) > 0:
        print(f"IDs {block_start:5d}-{block_end:5d}: {len(with_gas):3d} with gas out of {len(in_block):4d} checked ({len(with_gas)/len(in_block)*100:5.1f}%)")

# Check which specific IDs have gas
print()
print("="*70)
print("ALL SUBHALO IDs WITH GAS (first 100)")
print("="*70)

with_gas_ids = sorted([r["subhalo_id"] for r in responses if r["response"].get("mass_gas", 0) > 0])
print(f"Total subhalos with gas: {len(with_gas_ids)}")
print(f"First 50 IDs with gas: {with_gas_ids[:50]}")
print(f"Last 50 IDs with gas: {with_gas_ids[-50:] if len(with_gas_ids) > 50 else with_gas_ids}")

# Check density - how many per 1000 IDs
print()
print("="*70)
print("DENSITY OF GAS-CONTAINING SUBHALOS")
print("="*70)

for range_start in range(1, 20001, 1000):
    range_end = min(range_start + 999, 20000)
    in_range = [r for r in responses if range_start <= r["subhalo_id"] <= range_end]
    with_gas = [r for r in in_range if r["response"].get("mass_gas", 0) > 0]
    
    if len(in_range) > 0:
        print(f"IDs {range_start:5d}-{range_end:5d}: {len(with_gas):2d} with gas ({len(with_gas)/len(in_range)*100:5.1f}%)")

