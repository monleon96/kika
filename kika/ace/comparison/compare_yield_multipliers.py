"""
Module for comparing yield multiplier data in ACE format.
"""

from kika.ace.classes.ace import Ace

def compare_yield_multipliers(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare yield multiplier data between two ACE objects."""
    # Compare photon yield multipliers
    if not compare_photon_yield_multipliers(ace1, ace2, tolerance, verbose):
        return False
    
    # Compare particle yield multipliers
    if not compare_particle_yield_multipliers(ace1, ace2, tolerance, verbose):
        return False
    
    return True

def compare_photon_yield_multipliers(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare photon yield multiplier data."""
    has_data1 = ace1.photon_yield_multipliers is not None and ace1.photon_yield_multipliers.has_data
    has_data2 = ace2.photon_yield_multipliers is not None and ace2.photon_yield_multipliers.has_data
    
    if not has_data1 and not has_data2:
        return True
    
    if has_data1 != has_data2:
        if verbose:
            print("Photon yield multipliers mismatch: Presence differs")
        return False
    
    # Compare MT numbers used as multipliers
    mts1 = ace1.photon_yield_multipliers.multiplier_mts
    mts2 = ace2.photon_yield_multipliers.multiplier_mts
    
    if len(mts1) != len(mts2):
        if verbose:
            print(f"Photon yield multipliers mismatch: Number of MT numbers differs ({len(mts1)} vs {len(mts2)})")
        return False
    
    # Compare MT values
    if mts1 != mts2:
        if verbose:
            print("Photon yield multipliers mismatch: Different MT numbers")
            print(f"MT values in first ACE: {mts1}")
            print(f"MT values in second ACE: {mts2}")
        return False
    
    return True

def compare_particle_yield_multipliers(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare particle yield multiplier data."""
    has_data1 = ace1.particle_yield_multipliers is not None and ace1.particle_yield_multipliers.has_data
    has_data2 = ace2.particle_yield_multipliers is not None and ace2.particle_yield_multipliers.has_data
    
    if not has_data1 and not has_data2:
        return True
    
    if has_data1 != has_data2:
        if verbose:
            print("Particle yield multipliers mismatch: Presence differs")
        return False
    
    # Compare MT numbers used as multipliers
    mts1 = ace1.particle_yield_multipliers.multiplier_mts
    mts2 = ace2.particle_yield_multipliers.multiplier_mts
    
    if len(mts1) != len(mts2):
        if verbose:
            print(f"Particle yield multipliers mismatch: Number of MT numbers differs ({len(mts1)} vs {len(mts2)})")
        return False
    
    # Compare MT values
    if mts1 != mts2:
        if verbose:
            print("Particle yield multipliers mismatch: Different MT numbers")
            print(f"MT values in first ACE: {mts1}")
            print(f"MT values in second ACE: {mts2}")
        return False
    
    # Compare particle type multipliers
    particles1 = set(ace1.particle_yield_multipliers.particle_multipliers.keys())
    particles2 = set(ace2.particle_yield_multipliers.particle_multipliers.keys())
    
    if particles1 != particles2:
        if verbose:
            print("Particle yield multipliers mismatch: Different particle types")
            print(f"Particle types only in first: {sorted(particles1 - particles2)}")
            print(f"Particle types only in second: {sorted(particles2 - particles1)}")
        return False
    
    # Compare multipliers for each particle type
    for particle_idx in sorted(particles1):
        mt_list1 = ace1.particle_yield_multipliers.particle_multipliers[particle_idx]
        mt_list2 = ace2.particle_yield_multipliers.particle_multipliers[particle_idx]
        
        if len(mt_list1) != len(mt_list2):
            if verbose:
                print(f"Particle type {particle_idx} yield multipliers mismatch: Number of MT numbers differs "
                      f"({len(mt_list1)} vs {len(mt_list2)})")
            return False
        
        if mt_list1 != mt_list2:
            if verbose:
                print(f"Particle type {particle_idx} yield multipliers mismatch: Different MT numbers")
                print(f"MT values in first ACE: {mt_list1}")
                print(f"MT values in second ACE: {mt_list2}")
            return False
    
    return True
