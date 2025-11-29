"""
Utility functions for comparing ACE objects.
Contains shared functions used across multiple comparison modules.
"""

import numpy as np
from typing import List

def compare_arrays(arr1: List[float], arr2: List[float], tolerance: float, 
                  name: str, verbose: bool) -> bool:
    """
    Compare two arrays of numeric values within a tolerance.
    
    Parameters
    ----------
    arr1 : List[float]
        First array
    arr2 : List[float]
        Second array
    tolerance : float
        Tolerance for comparison
    name : str
        Name of the array (for reporting)
    verbose : bool
        If True, print detailed information about any differences
        
    Returns
    -------
    bool
        True if arrays are equivalent within tolerance, False otherwise
    """
    # Check lengths
    if len(arr1) != len(arr2):
        if verbose:
            print(f"{name} mismatch: Length differs ({len(arr1)} vs {len(arr2)})")
        return False
    
    # Empty arrays match
    if len(arr1) == 0:
        return True
    
    # Convert to numpy arrays for easier comparison
    np_arr1 = np.array(arr1, dtype=float)
    np_arr2 = np.array(arr2, dtype=float)
    
    # Calculate relative differences
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.abs(np_arr1 - np_arr2) / np.maximum(np.abs(np_arr1), np.abs(np_arr2))
    
    # Handle zeros and NaNs
    rel_diff[np.isnan(rel_diff)] = 0
    rel_diff[np.isinf(rel_diff)] = 0
    
    # For values that are both close to zero, use absolute difference
    near_zero = (np.abs(np_arr1) < tolerance) & (np.abs(np_arr2) < tolerance)
    abs_diff = np.abs(np_arr1 - np_arr2)
    
    # Values match if relative difference is within tolerance OR
    # both values are near zero and absolute difference is within tolerance
    match = (rel_diff <= tolerance) | (near_zero & (abs_diff <= tolerance))
    
    if not np.all(match):
        mismatch_idx = np.where(~match)[0]
        if verbose:
            print(f"{name} mismatch: {len(mismatch_idx)} values differ beyond tolerance")
            if len(mismatch_idx) <= 10:  # Show details for up to 10 mismatches
                for idx in mismatch_idx:
                    print(f"  At index {idx}: {np_arr1[idx]} vs {np_arr2[idx]} "
                         f"(diff: {abs_diff[idx]}, rel_diff: {rel_diff[idx]})")
            else:
                print(f"  First 5 mismatches:")
                for idx in mismatch_idx[:5]:
                    print(f"  At index {idx}: {np_arr1[idx]} vs {np_arr2[idx]} "
                         f"(diff: {abs_diff[idx]}, rel_diff: {rel_diff[idx]})")
                print(f"  ... and {len(mismatch_idx) - 5} more")
                
            # Show statistics on differences
            print(f"  Max absolute difference: {np.max(abs_diff)}")
            print(f"  Max relative difference: {np.max(rel_diff)}")
        return False
    
    return True

def compare_floats(val1: float, val2: float, tolerance: float, name: str, verbose: bool) -> bool:
    """
    Compare two floating-point values within a tolerance.
    
    Parameters
    ----------
    val1 : float
        First value
    val2 : float
        Second value
    tolerance : float
        Tolerance for comparison
    name : str
        Name of the value (for reporting)
    verbose : bool
        If True, print detailed information about any differences
        
    Returns
    -------
    bool
        True if values are equivalent within tolerance, False otherwise
    """
    # Calculate relative difference
    if abs(val1) > tolerance or abs(val2) > tolerance:
        rel_diff = abs(val1 - val2) / max(abs(val1), abs(val2))
    else:
        # Both values near zero, use absolute difference
        rel_diff = 0
    
    # Calculate absolute difference
    abs_diff = abs(val1 - val2)
    
    # Values match if relative difference is within tolerance OR
    # both values are near zero and absolute difference is within tolerance
    match = (rel_diff <= tolerance) or (abs(val1) < tolerance and abs(val2) < tolerance and abs_diff <= tolerance)
    
    if not match and verbose:
        print(f"{name} mismatch: {val1} vs {val2} (diff: {abs_diff}, rel_diff: {rel_diff})")
    
    return match
