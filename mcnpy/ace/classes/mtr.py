from dataclasses import dataclass, field
from typing import List, Optional
from mcnpy.ace.parsers.xss import XssEntry

@dataclass
class ReactionMTData:
    """Container for MT reaction numbers from MTR, MTRP, and MTRH blocks."""
    incident_neutron: List[XssEntry] = field(default_factory=list)  # MTR Block - neutron reaction MT numbers
    photon_production: List[XssEntry] = field(default_factory=list)  # MTRP Block - photon production MT numbers
    particle_production: List[List[XssEntry]] = field(default_factory=list)  # MTRH Block - particle production MT numbers
    secondary_neutron_mt: List[XssEntry] = field(default_factory=list)  # MT numbers for reactions with secondary neutrons
    
    @property
    def has_neutron_mt_data(self) -> bool:
        """Check if neutron reaction MT numbers are available."""
        return len(self.incident_neutron) > 0
    
    @property
    def has_photon_production_mt_data(self) -> bool:
        """Check if photon production MT numbers are available."""
        return len(self.photon_production) > 0
    
    @property
    def has_particle_production_mt_data(self) -> bool:
        """Check if particle production MT numbers are available."""
        return len(self.particle_production) > 0
    
    @property
    def has_secondary_neutron_data(self) -> bool:
        """Check if secondary neutron MT numbers are available."""
        return len(self.secondary_neutron_mt) > 0
    
    def get_particle_production_mt_numbers(self, particle_idx: int = 0) -> Optional[List[XssEntry]]:
        """
        Get the list of particle production MT numbers for a specific particle type.
        
        Parameters
        ----------
        particle_idx : int
            Index of the particle type (0-based)
            
        Returns
        -------
        List[XssEntry] or None
            The list of MT numbers, or None if the particle type doesn't exist
        """
        if particle_idx < 0 or particle_idx >= len(self.particle_production):
            return None
        return self.particle_production[particle_idx]
    
    def get_num_particle_types(self) -> int:
        """Get the number of particle types with MT data."""
        return len(self.particle_production)
