"""
Module for comparing photon production data in ACE format.
"""

from kika.ace.classes.ace import Ace
from kika.ace.comparison.compare_gpd import compare_gpd
from kika.ace.comparison.compare_photon_xs import compare_photon_production_xs

def compare_photon_production(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare all photon production data between two ACE objects."""
    # Compare GPD block (total photon production cross section)
    if not compare_gpd(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare photon production cross sections (SIGP block)
    if not compare_photon_production_xs(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare photon yield multipliers if available
    if (hasattr(ace1, 'photon_yield_multipliers') and ace1.photon_yield_multipliers is not None and
        hasattr(ace2, 'photon_yield_multipliers') and ace2.photon_yield_multipliers is not None):
        
        from kika.ace.comparison.compare_yield_multipliers import compare_photon_yield_multipliers
        if not compare_photon_yield_multipliers(ace1, ace2, tolerance, verbose):
            return False
    
    # Compare photon production angular distributions
    if hasattr(ace1, 'angular_distributions') and hasattr(ace2, 'angular_distributions'):
        from kika.ace.comparison.compare_angular import compare_photon_angular
        if not compare_photon_angular(ace1, ace2, tolerance, verbose):
            return False
    
    # Compare photon production energy distributions
    if hasattr(ace1, 'energy_distributions') and hasattr(ace2, 'energy_distributions'):
        from kika.ace.comparison.compare_energy_dist import compare_photon_energy_dist
        if not compare_photon_energy_dist(ace1, ace2, tolerance, verbose):
            return False
    
    return True
