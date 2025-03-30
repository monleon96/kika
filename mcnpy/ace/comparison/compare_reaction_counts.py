"""
Module for comparing particle reaction counts in ACE format.
"""

from mcnpy.ace.classes.ace import Ace

def compare_reaction_counts(ace1: Ace, ace2: Ace, tolerance: float = 1e-6, verbose: bool = True) -> bool:
    """Compare particle reaction counts between two ACE objects."""
    has_counts1 = (ace1.secondary_particle_reactions is not None and ace1.secondary_particle_reactions.has_data)
    has_counts2 = (ace2.secondary_particle_reactions is not None and ace2.secondary_particle_reactions.has_data)
    
    if not has_counts1 and not has_counts2:
        return True
    
    if has_counts1 != has_counts2:
        if verbose:
            print("Particle reaction counts mismatch: Presence differs")
        return False
    
    # Compare number of reaction counts
    if len(ace1.secondary_particle_reactions.reaction_counts) != len(ace2.secondary_particle_reactions.reaction_counts):
        if verbose:
            print(f"Particle reaction counts mismatch: Number of entries differs "
                  f"({len(ace1.secondary_particle_reactions.reaction_counts)} vs "
                  f"{len(ace2.secondary_particle_reactions.reaction_counts)})")
        return False
    
    # Compare reaction counts for each particle type
    counts1 = ace1.secondary_particle_reactions.reaction_counts
    counts2 = ace2.secondary_particle_reactions.reaction_counts
    
    for i, (count1, count2) in enumerate(zip(counts1, counts2)):
        if count1 != count2:
            if verbose:
                print(f"Particle reaction counts mismatch: Particle type {i+1} count differs ({count1} vs {count2})")
            return False
    
    return True
