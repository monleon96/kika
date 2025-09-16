#!/usr/bin/env python3
"""
Debug script to investigate multigroup vs continuous coefficient discrepancies.
"""

import numpy as np
import matplotlib.pyplot as plt
from mcnpy.endf.read_endf import read_endf
from mcnpy.cov.multigroup.mg_mf34_covmat import MGMF34CovMat
from mcnpy.cov.multigroup.MF34_to_MG import MF34_to_MG
from mcnpy.energy_grids import SCALE56

# Load ENDF data
jeff_Fe56 = '/soft_snc/lib/endf/jeff40/neutrons/13-Al-26g.txt'
jeff_endf = read_endf(jeff_Fe56)
zaid = 13026

# Get MF4 data
mf4_data = jeff_endf.mf[4].mt[2]

# Check energy ranges
print("=== ENERGY RANGE ANALYSIS ===")
print(f"MF4 energy range:")
if hasattr(mf4_data, 'legendre_energies'):
    mf4_energies = np.array(mf4_data.legendre_energies)
    print(f"  Min: {mf4_energies.min():.2e} eV")
    print(f"  Max: {mf4_energies.max():.2e} eV")
    print(f"  Number of points: {len(mf4_energies)}")
else:
    print("  No legendre_energies attribute found")

# Multigroup grid
SCALE56_eV = [e * 1e6 for e in SCALE56]
print(f"\nSCALE56 multigroup grid:")
print(f"  Min: {min(SCALE56_eV):.2e} eV")
print(f"  Max: {max(SCALE56_eV):.2e} eV")
print(f"  Number of groups: {len(SCALE56_eV)-1}")

# Check last few groups
print(f"\nLast 5 group boundaries (eV):")
for i, energy in enumerate(SCALE56_eV[-6:]):
    print(f"  Group boundary {len(SCALE56_eV)-6+i}: {energy:.2e} eV")

# Check if MG grid extends beyond MF4 data
if hasattr(mf4_data, 'legendre_energies'):
    mf4_max = mf4_energies.max()
    mg_max = max(SCALE56_eV)
    if mg_max > mf4_max:
        print(f"\n*** ISSUE FOUND ***")
        print(f"Multigroup grid extends beyond MF4 data!")
        print(f"MF4 max energy: {mf4_max:.2e} eV")
        print(f"MG max energy: {mg_max:.2e} eV")
        print(f"Extension ratio: {mg_max/mf4_max:.2f}")
        
        # Find which groups are affected
        affected_groups = []
        for i, energy in enumerate(SCALE56_eV[:-1]):  # Skip last boundary
            if energy > mf4_max:
                affected_groups.append(i)
        print(f"Affected groups (0-indexed): {affected_groups}")
        print(f"Total affected groups: {len(affected_groups)}")

# Extract coefficients to check behavior
print(f"\n=== COEFFICIENT EXTRACTION TEST ===")

# Test energies both inside and outside MF4 range
if hasattr(mf4_data, 'legendre_energies'):
    test_energies = np.array([
        mf4_energies.max() * 0.9,  # Inside range
        mf4_energies.max() * 1.1,  # Just outside
        mf4_energies.max() * 2.0,  # Far outside  
        max(SCALE56_eV) * 0.9,     # Near MG max
    ])
    
    print(f"Test energies:")
    for i, E in enumerate(test_energies):
        print(f"  {i}: {E:.2e} eV")
    
    # Extract coefficients
    coeffs_dict = mf4_data.extract_legendre_coefficients(
        test_energies, max_legendre_order=3, out_of_range="zero"
    )
    
    print(f"\nExtracted coefficients (out_of_range='zero'):")
    for order in [1, 2, 3]:
        if order in coeffs_dict:
            print(f"  L={order}: {coeffs_dict[order]}")
    
    # Test with "hold" option
    coeffs_dict_hold = mf4_data.extract_legendre_coefficients(
        test_energies, max_legendre_order=3, out_of_range="hold"
    )
    
    print(f"\nExtracted coefficients (out_of_range='hold'):")
    for order in [1, 2, 3]:
        if order in coeffs_dict_hold:
            print(f"  L={order}: {coeffs_dict_hold[order]}")

print(f"\n=== ANALYSIS COMPLETE ===")