"""
Utility functions for working with energy grids.
"""


import numpy as np
from .grids import (
    VITAMINJ174, VITAMINJ175, SCALE44, SCALE56, SCALE238, SCALE252,
    WIMS69, ECCO33, ONEGROUP20
)

def _identify_energy_grid(energies, tolerance=1e-5):
    """
    Compare an energy grid with known predefined grids to identify which standard grid it matches.
    
    :param energies: List or array of energy bin boundaries
    :type energies: list or numpy.ndarray
    :param tolerance: Relative tolerance for floating point comparison, defaults to 1e-5
    :type tolerance: float, optional
    :returns: Name of the matching grid or None if no match is found
    :rtype: str or None
    """
    if energies is None or len(energies) < 2:
        return None
    
    # Dictionary mapping grid variables to their names
    grid_dict = {
        "VITAMINJ174": VITAMINJ174,
        "VITAMINJ175": VITAMINJ175, 
        "SCALE44": SCALE44,
        "SCALE56": SCALE56,
        "SCALE238": SCALE238,
        "SCALE252": SCALE252,
        "WIMS69": WIMS69,
        "ECCO33": ECCO33,
        "ONEGROUP20": ONEGROUP20
    }
    
    # Convert input to numpy array for comparison
    energy_array = np.array(energies)
    
    # Check each grid for match
    for grid_name, grid_values in grid_dict.items():
        # Skip if length doesn't match
        if len(energy_array) != len(grid_values):
            continue
        
        # Convert grid to numpy array
        grid_array = np.array(grid_values)
        
        # Calculate relative differences
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.abs((energy_array - grid_array) / grid_array)
            
        # Check if all values are within tolerance (or exactly equal for zeros)
        is_match = np.all(
            np.logical_or(
                rel_diff <= tolerance,
                np.logical_and(energy_array == 0, grid_array == 0)
            )
        )
        
        if is_match:
            return grid_name
    
    return None