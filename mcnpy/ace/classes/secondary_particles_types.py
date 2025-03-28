from dataclasses import dataclass, field
from typing import List
from enum import IntEnum

class ParticleID(IntEnum):
    """
    Standard particle identifiers used in ACE files.
    """
    NEUTRON = 1
    PHOTON = 2
    POSITRON = 3
    ELECTRON = 4
    PROTON = 9
    DEUTERON = 31
    TRITON = 32
    HELION = 33
    ALPHA = 34

@dataclass
class SecondaryParticleTypes:
    """
    Container for secondary particle types that can be produced in nuclear reactions.
    
    This data defines which types of particles (neutrons, protons, alphas, etc.)
    can be produced in nuclear reactions. The ACE file contains production data
    for these particles, including cross sections, angular distributions,
    and energy distributions.
    
    For example, if this contains [neutron, proton, alpha], it means the 
    nuclear data includes information about neutron-, proton-, and alpha-producing
    reactions.
    """
    particle_ids: List[int] = field(default_factory=list)
    has_data: bool = False
    
    @property
    def num_secondary_particles(self) -> int:
        """
        Get the number of secondary particle types.
        
        Returns
        -------
        int
            Number of secondary particle types
        """
        return len(self.particle_ids)
    
    def get_particle_name(self, particle_id: int) -> str:
        """
        Get a human-readable name for a particle ID.
        
        Parameters
        ----------
        particle_id : int
            The particle identifier
            
        Returns
        -------
        str
            Human-readable name for the particle
        """
        try:
            # Try to map to a standard particle name
            return {
                ParticleID.NEUTRON: "neutron",
                ParticleID.PHOTON: "photon",
                ParticleID.POSITRON: "positron",
                ParticleID.ELECTRON: "electron",
                ParticleID.PROTON: "proton",
                ParticleID.DEUTERON: "deuteron",
                ParticleID.TRITON: "triton",
                ParticleID.HELION: "helion",
                ParticleID.ALPHA: "alpha"
            }.get(particle_id, f"particle-{particle_id}")
        except (ValueError, TypeError):
            # If conversion fails, return a generic name
            return f"particle-{particle_id}"
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No secondary particle data available"
        
        output = f"Secondary Particles That Can Be Produced ({self.num_secondary_particles} types):\n"
        output += "=" * 70 + "\n"
        output += "This data defines which types of particles can be produced in nuclear reactions.\n"
        output += "The ACE file contains production cross sections, angular distributions, and\n"
        output += "energy distributions for these particles.\n\n"
        
        for i, particle_id in enumerate(self.particle_ids, 1):
            particle_name = self.get_particle_name(particle_id)
            output += f"{i}. {particle_name.capitalize()} (ID: {particle_id})\n"
        
        return output
