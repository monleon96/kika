from dataclasses import dataclass, field
from typing import List

@dataclass
class SecondaryParticleReactions:
    """
    Container for the number of reactions that produce each secondary particle type.
    
    This tells us how many different nuclear reactions produce each type of secondary
    particle. For example, it might tell us there are 20 reactions that produce
    neutrons, 5 that produce protons, etc.
    """
    reaction_counts: List[int] = field(default_factory=list)
    has_data: bool = False
    
    @property
    def num_particle_types(self) -> int:
        """
        Get the number of particle types for which reaction counts are available.
        
        Returns
        -------
        int
            Number of particle types
        """
        return len(self.reaction_counts)
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No secondary particle reaction data available"
        
        output = f"Number of Reactions per Secondary Particle Type:\n"
        output += "=" * 70 + "\n"
        output += "This data shows how many different reactions produce each type of secondary particle.\n\n"
        
        for i, count in enumerate(self.reaction_counts, 1):
            output += f"Particle Type {i}: {count} reactions\n"
        
        return output
