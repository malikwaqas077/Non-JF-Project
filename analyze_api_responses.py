#!/usr/bin/env python3
"""Analyze API responses to understand why we couldn't find more candidates."""
import json
from collections import defaultdict

# Read all API responses
responses = []
errors = []

with open("api_responses.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            if "error" in data:
                errors.append(data)
            elif "response" in data:
                responses.append(data)
        except:
            pass

print(f"Total API calls: {len(responses) + len(errors)}")
print(f"Successful responses: {len(responses)}")
print(f"Errors: {len(errors)}")
print()

# Analyze why subhalos don't meet criteria
criteria_levels = [
    {"name": "Level 0", "min_gas": 1e6, "max_gas": 1e10, "min_stellar": 5e7, "max_frac": 0.40},
    {"name": "Level 1", "min_gas": 5e5, "max_gas": 2e10, "min_stellar": 1e7, "max_frac": 0.50},
    {"name": "Level 2", "min_gas": 1e5, "max_gas": 5e10, "min_stellar": 1e6, "max_frac": 0.60},
    {"name": "Level 3", "min_gas": 1e4, "max_gas": 1e11, "min_stellar": 1e5, "max_frac": 0.70},
    {"name": "Level 4", "min_gas": 1e3, "max_gas": 1e12, "min_stellar": 1e4, "max_frac": 0.80},
]

reasons = defaultdict(int)
gas_masses = []
stellar_masses = []
gas_fractions = []
zero_gas = 0
zero_stellar = 0
zero_total = 0

for resp in responses:
    r = resp["response"]
    subhalo_id = resp["subhalo_id"]
    
    gas_mass = r.get("mass_gas", 0) * 1e10
    stellar_mass = r.get("mass_stars", 0) * 1e10
    total_mass = r.get("mass", 0) * 1e10
    gas_frac = gas_mass / total_mass if total_mass > 0 else 0
    
    gas_masses.append(gas_mass)
    stellar_masses.append(stellar_mass)
    gas_fractions.append(gas_frac)
    
    if gas_mass == 0:
        zero_gas += 1
    if stellar_mass == 0:
        zero_stellar += 1
    if total_mass == 0:
        zero_total += 1
    
    # Check against most lenient criteria (Level 4)
    level4 = criteria_levels[4]
    
    if gas_mass == 0:
        reasons["Zero gas mass"] += 1
    elif gas_mass < level4["min_gas"]:
        reasons[f"Gas mass too low (< {level4['min_gas']:.0e})"] += 1
    elif gas_mass > level4["max_gas"]:
        reasons[f"Gas mass too high (> {level4['max_gas']:.0e})"] += 1
    elif stellar_mass < level4["min_stellar"]:
        reasons[f"Stellar mass too low (< {level4['min_stellar']:.0e})"] += 1
    elif gas_frac >= level4["max_frac"]:
        reasons[f"Gas fraction too high (>= {level4['max_frac']:.0%})"] += 1
    else:
        reasons["Would qualify at Level 4"] += 1

print("="*70)
print("ANALYSIS OF WHY SUBHALOS DON'T QUALIFY")
print("="*70)
print(f"\nSubhalos with zero gas mass: {zero_gas}")
print(f"Subhalos with zero stellar mass: {zero_stellar}")
print(f"Subhalos with zero total mass: {zero_total}")
print()

print("Reasons for not qualifying (even at most lenient Level 4):")
for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f"  {reason}: {count}")

print()
print("="*70)
print("STATISTICS")
print("="*70)

if gas_masses:
    gas_masses = [g for g in gas_masses if g > 0]
    if gas_masses:
        print(f"\nGas Mass Statistics (non-zero only):")
        print(f"  Min: {min(gas_masses):.2e} Msun")
        print(f"  Max: {max(gas_masses):.2e} Msun")
        print(f"  Mean: {sum(gas_masses)/len(gas_masses):.2e} Msun")
        print(f"  Median: {sorted(gas_masses)[len(gas_masses)//2]:.2e} Msun")

if stellar_masses:
    stellar_masses = [s for s in stellar_masses if s > 0]
    if stellar_masses:
        print(f"\nStellar Mass Statistics (non-zero only):")
        print(f"  Min: {min(stellar_masses):.2e} Msun")
        print(f"  Max: {max(stellar_masses):.2e} Msun")
        print(f"  Mean: {sum(stellar_masses)/len(stellar_masses):.2e} Msun")
        print(f"  Median: {sorted(stellar_masses)[len(stellar_masses)//2]:.2e} Msun")

if gas_fractions:
    gas_fractions = [f for f in gas_fractions if f > 0]
    if gas_fractions:
        print(f"\nGas Fraction Statistics (non-zero only):")
        print(f"  Min: {min(gas_fractions):.2%}")
        print(f"  Max: {max(gas_fractions):.2%}")
        print(f"  Mean: {sum(gas_fractions)/len(gas_fractions):.2%}")
        print(f"  Median: {sorted(gas_fractions)[len(gas_fractions)//2]:.2%}")

# Check how many would qualify at each level
print()
print("="*70)
print("QUALIFICATION BY LEVEL")
print("="*70)

for level in criteria_levels:
    qualified = 0
    for resp in responses:
        r = resp["response"]
        gas_mass = r.get("mass_gas", 0) * 1e10
        stellar_mass = r.get("mass_stars", 0) * 1e10
        total_mass = r.get("mass", 0) * 1e10
        gas_frac = gas_mass / total_mass if total_mass > 0 else 0
        
        if (gas_mass > 0 and 
            level["min_gas"] <= gas_mass <= level["max_gas"] and
            stellar_mass >= level["min_stellar"] and
            gas_frac < level["max_frac"]):
            qualified += 1
    
    print(f"{level['name']}: {qualified} subhalos would qualify")

