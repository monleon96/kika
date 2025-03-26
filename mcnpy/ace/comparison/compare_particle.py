"""
Module for comparing particle production data in ACE format.
"""

from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.comparison.compare_particle_types import compare_particle_types
from mcnpy.ace.comparison.compare_particle_locators import compare_particle_production_locators
from mcnpy.ace.comparison.compare_particle_xs import compare_particle_production_xs
from mcnpy.ace.comparison.compare_particle_release import compare_particle_release
from mcnpy.ace.comparison.compare_reaction_counts import compare_reaction_counts

def compare_particle_production(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare all particle production data between two ACE objects."""
    # Compare particle types (PTYPE block)
    if not compare_particle_types(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare particle production locators (IXS block)
    compare_particle_production_locators(ace1, ace2, tolerance, verbose)
    # Note: We don't return False here because different locator values might be acceptable
    
    # Compare particle production cross section data (HPD blocks)
    if not compare_particle_production_xs(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare particle production MT numbers (MTRH blocks)
    if hasattr(ace1, 'reaction_mt_data') and hasattr(ace2, 'reaction_mt_data'):
        from mcnpy.ace.comparison.compare_mtr import compare_particle_mt_data
        if not compare_particle_mt_data(ace1, ace2, tolerance, verbose):
            return False
    
    # Compare particle release data (TYRH blocks)
    if not compare_particle_release(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare particle reaction counts
    if not compare_reaction_counts(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare particle production angular distributions (ANDH blocks)
    if hasattr(ace1, 'angular_distributions') and hasattr(ace2, 'angular_distributions'):
        from mcnpy.ace.comparison.compare_angular import compare_particle_angular
        if not compare_particle_angular(ace1, ace2, tolerance, verbose):
            return False
    
    # Compare particle production energy distributions (DLWH blocks)
    if hasattr(ace1, 'energy_distributions') and hasattr(ace2, 'energy_distributions'):
        from mcnpy.ace.comparison.compare_energy_dist import compare_particle_energy_dist
        if not compare_particle_energy_dist(ace1, ace2, tolerance, verbose):
            return False
    
    # Compare particle yield multipliers if available
    if (hasattr(ace1, 'particle_yield_multipliers') and ace1.particle_yield_multipliers is not None and
        hasattr(ace2, 'particle_yield_multipliers') and ace2.particle_yield_multipliers is not None):
        
        from mcnpy.ace.comparison.compare_yield_multipliers import compare_particle_yield_multipliers
        if not compare_particle_yield_multipliers(ace1, ace2, tolerance, verbose):
            return False
    
    return True
