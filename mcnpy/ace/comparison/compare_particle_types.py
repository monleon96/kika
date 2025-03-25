"""
Module for comparing particle production types data in ACE format.
"""

from mcnpy.ace.ace import Ace

def compare_particle_types(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare particle production types between two ACE objects."""
    # Check if both objects have particle types data
    has_particle_types1 = (ace1.particle_types is not None and ace1.particle_types.has_data)
    has_particle_types2 = (ace2.particle_types is not None and ace2.particle_types.has_data)
    
    if not has_particle_types1 and not has_particle_types2:
        return True
    
    if has_particle_types1 != has_particle_types2:
        if verbose:
            print("Particle production types mismatch: Presence differs")
        return False
    
    # Compare number of particle types
    if len(ace1.particle_types.particle_ids) != len(ace2.particle_types.particle_ids):
        if verbose:
            print(f"Particle production types mismatch: Number of particle types differs "
                  f"({len(ace1.particle_types.particle_ids)} vs {len(ace2.particle_types.particle_ids)})")
        return False
    
    # Compare particle IDs
    particle_ids1 = ace1.particle_types.particle_ids
    particle_ids2 = ace2.particle_types.particle_ids
    
    if particle_ids1 != particle_ids2:
        if verbose:
            print("Particle production types mismatch: Different particle IDs")
            print(f"Particle IDs in first ACE: {particle_ids1}")
            print(f"Particle IDs in second ACE: {particle_ids2}")
        return False
    
    return True
