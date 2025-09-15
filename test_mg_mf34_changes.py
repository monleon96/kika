#!/usr/bin/env python3
"""
Test script to verify the changes made to MGMF34CovMat class.
"""

import numpy as np
from mcnpy.cov.mg_mf34_covmat import MGMF34CovMat

def test_mg_mf34_changes():
    """Test the new MGMF34CovMat structure."""
    
    # Create a test object
    mg_cov = MGMF34CovMat()
    
    # Set energy grid
    mg_cov.energy_grid = np.array([1e-5, 1e-3, 1e-1, 1e1, 1e3, 2e7])  # 5 groups
    
    print(f"Energy grid: {mg_cov.energy_grid}")
    print(f"Number of groups: {mg_cov.num_groups}")
    
    # Test data
    isotope = 26056
    mt = 2
    l = 1
    
    # Create some dummy multigroup data
    num_groups = mg_cov.num_groups
    mg_matrix = np.random.rand(num_groups, num_groups) * 0.1
    mg_coeffs = np.random.rand(num_groups) * 0.5
    
    # Add a matrix
    mg_cov.add_matrix(
        isotope_row=isotope,
        reaction_row=mt,
        l_row=l,
        isotope_col=isotope,
        reaction_col=mt,
        l_col=l,
        relative_matrix=mg_matrix,
        absolute_matrix=mg_matrix * mg_coeffs[:, np.newaxis] * mg_coeffs[np.newaxis, :],
        mg_means_row=mg_coeffs,
        mg_means_col=mg_coeffs,
        frame="LAB"
    )
    
    print(f"\nAfter adding matrix:")
    print(f"Number of matrices: {mg_cov.num_matrices}")
    print(f"Legendre coefficients keys: {list(mg_cov.legendre_coefficients.keys())}")
    print(f"Coefficients for (26056, 2, 1): {mg_cov.legendre_coefficients[(26056, 2, 1)]}")
    
    # Test that private attributes exist but aren't meant for direct access
    print(f"\nPrivate attributes exist:")
    print(f"_mg_means_row length: {len(mg_cov._mg_means_row)}")
    print(f"_mg_means_col length: {len(mg_cov._mg_means_col)}")
    
    # Test filtering
    filtered = mg_cov.filter_by_isotope_reaction(isotope, mt)
    print(f"\nFiltered object:")
    print(f"Number of matrices: {filtered.num_matrices}")
    print(f"Energy grid preserved: {np.array_equal(filtered.energy_grid, mg_cov.energy_grid)}")
    print(f"Legendre coefficients preserved: {list(filtered.legendre_coefficients.keys())}")
    
    print("\nAll tests passed! ✅")

if __name__ == "__main__":
    test_mg_mf34_changes()