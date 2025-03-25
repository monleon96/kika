from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class ParticleReactionCounts:
    """
    Container for the number of reactions per particle type (NTRO block).
    
    This block specifies how many reactions produce each secondary particle type
    defined in the PTYPE block.
    """
    has_data: bool = False
    reaction_counts: List[int] = field(default_factory=list)  # Number of reactions per particle type
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No particle reaction count data available"
        
        output = f"Number of Reactions per Secondary Particle Type:\n"
        output += "=" * 70 + "\n"
        
        for i, count in enumerate(self.reaction_counts, 1):
            output += f"Particle Type {i}: {count} reactions\n"
        
        return output
