from dataclasses import dataclass, field
from typing import List, Dict, Optional
from typing import Tuple

@dataclass
class YieldMultipliers:
    """
    Base class for yield multiplier data (YP/YH blocks).
    
    These blocks contain MT identifiers used as yield multipliers in calculating
    production cross sections from yield data.
    """
    has_data: bool = False
    multiplier_mts: List[int] = field(default_factory=list)
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No yield multiplier data available"
        
        output = f"Yield Multiplier MTs ({len(self.multiplier_mts)} entries):\n"
        output += "=" * 50 + "\n"
        output += "These MT numbers are used as multipliers when calculating\n"
        output += "production cross sections from yield data.\n\n"
        output += ", ".join(map(str, self.multiplier_mts))
        return output

@dataclass
class PhotonYieldMultipliers(YieldMultipliers):
    """
    Container for photon yield multiplier data (YP block).
    
    This block contains MT identifiers used to calculate photon production
    cross sections from yield data.
    """
    pass

@dataclass
class SecondaryParticleYieldMultipliers(YieldMultipliers):
    """
    Container for secondary particle yield multiplier data (YH block).
    
    This block contains MT identifiers used to calculate secondary particle
    production cross sections from yield data. The data is organized by
    particle type.
    """
    # Dictionary mapping particle type index to list of MT multipliers
    particle_multipliers: Dict[int, List[int]] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No secondary particle yield multiplier data available"
        
        output = f"Secondary Particle Yield Multiplier Data\n"
        output += "=" * 50 + "\n"
        output += "These MT numbers are used as multipliers when calculating\n"
        output += "secondary particle production cross sections from yield data.\n\n"
        
        for particle_type, mts in sorted(self.particle_multipliers.items()):
            output += f"Particle Type {particle_type}: {len(mts)} multipliers (MT={', '.join(map(str, mts))})\n"
        
        return output

# For backward compatibility
ParticleYieldMultipliers = SecondaryParticleYieldMultipliers
