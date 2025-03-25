from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import IntEnum

class ParticleID(IntEnum):
    """Enumeration of particle types in ACE format."""
    NEUTRON = 1
    PROTON = 9
    DEUTERON = 31
    TRITON = 32
    HELION = 33   # He-3
    ALPHA = 34    # He-4

@dataclass
class ParticleProductionTypes:
    """
    Container for secondary particle production data (PTYPE block).
    
    This block lists the types of particles that can be produced in nuclear reactions.
    The ACE file contains production data (cross sections, angular distributions, 
    and energy distributions) for each of these secondary particle types.
    
    For example, if this container lists [neutron, proton, alpha], it means the 
    ACE file contains data for neutron-, proton-, and alpha-producing reactions.
    """
    has_data: bool = False
    particle_ids: List[int] = field(default_factory=list)  # List of particle identifiers
    
    @property
    def num_secondary_particles(self) -> int:
        """Return the number of secondary particle types that can be produced."""
        return len(self.particle_ids)
    
    def get_particle_name(self, particle_id: int) -> str:
        """
        Get the name of a particle based on its ID.
        
        Parameters
        ----------
        particle_id : int
            The particle ID
            
        Returns
        -------
        str
            The particle name
        """
        particle_names = {
            ParticleID.NEUTRON: "neutron",
            ParticleID.PROTON: "proton",
            ParticleID.DEUTERON: "deuteron",
            ParticleID.TRITON: "triton",
            ParticleID.HELION: "helion",
            ParticleID.ALPHA: "alpha"
        }
        return particle_names.get(particle_id, f"unknown ({particle_id})")
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No secondary particle production data available"
        
        output = f"Secondary Particles That Can Be Produced ({self.num_secondary_particles} types):\n"
        output += "=" * 70 + "\n"
        output += "This data defines which types of secondary particles can be produced\n"
        output += "in nuclear reactions. The ACE file contains production cross sections,\n"
        output += "angular distributions, and energy distributions for these particles.\n\n"
        
        for i, particle_id in enumerate(self.particle_ids, 1):
            particle_name = self.get_particle_name(particle_id)
            output += f"{i}. {particle_name.capitalize()} (ID: {particle_id})\n"
        
        return output
