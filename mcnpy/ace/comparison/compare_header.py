"""
Module for comparing header data in ACE format.
"""

from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.comparison.compare_utils import compare_arrays, compare_floats

def compare_header(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare header information between two ACE objects."""
    if ace1.header is None and ace2.header is None:
        return True
    
    if ace1.header is None or ace2.header is None:
        if verbose:
            print("Header mismatch: One header is None")
        return False
    
    # Compare essential header attributes
    header_attrs = [
        "zaid", "atomic_weight_ratio", "temperature", "date", "matid", "nxs_array", "jxs_array"
    ]
    
    for attr in header_attrs:
        if not hasattr(ace1.header, attr) or not hasattr(ace2.header, attr):
            if verbose:
                print(f"Header mismatch: Missing attribute {attr}")
            return False
        
        val1 = getattr(ace1.header, attr)
        val2 = getattr(ace2.header, attr)
        
        if attr in ["nxs_array", "jxs_array"]:
            # Compare arrays
            if not compare_arrays(val1, val2, tolerance, f"Header {attr}", verbose):
                return False
        elif attr == "temperature":
            # Compare temperature with tolerance
            if not compare_floats(val1, val2, tolerance, "Header temperature", verbose):
                return False
        else:
            # Compare other values exactly
            if val1 != val2:
                if verbose:
                    print(f"Header mismatch: {attr} differs - {val1} vs {val2}")
                return False
    
    return True
