from dataclasses import dataclass, field
from typing import List, Optional
from mcnpy.ace.parsers.xss import XssEntry

@dataclass
class EnergyDistributionLocators:
    """Container for energy distribution locators from LDLW, LDLWP, LDLWH, and DNEDL blocks."""
    incident_neutron: List[XssEntry] = field(default_factory=list)  # LDLW Block - neutron reaction energy locators
    photon_production: List[XssEntry] = field(default_factory=list)  # LDLWP Block - photon production energy locators
    particle_production: List[List[XssEntry]] = field(default_factory=list)  # LDLWH Block - particle production energy locators
    delayed_neutron: List[XssEntry] = field(default_factory=list)  # DNEDL Block - delayed neutron energy locators
    
    # Store the NXS values
    num_secondary_neutron_reactions: int = 0  # NXS(5)
    num_photon_production_reactions: int = 0  # NXS(6)
    num_particle_types: int = 0  # NXS(7)
    num_delayed_neutron_precursors: int = 0  # NXS(8)
    
    @property
    def has_neutron_data(self) -> bool:
        """Check if neutron reaction energy distribution locators are available."""
        return len(self.incident_neutron) > 0
    
    @property
    def has_photon_production_data(self) -> bool:
        """Check if photon production energy distribution locators are available."""
        return len(self.photon_production) > 0
    
    @property
    def has_particle_production_data(self) -> bool:
        """Check if particle production energy distribution locators are available."""
        return len(self.particle_production) > 0
    
    @property
    def has_delayed_neutron_data(self) -> bool:
        """Check if delayed neutron energy distribution locators are available."""
        return len(self.delayed_neutron) > 0
    
    def get_particle_production_locators(self, particle_idx: int = 0) -> Optional[List[XssEntry]]:
        """
        Get the list of particle production energy distribution locators for a specific particle type.
        
        Parameters
        ----------
        particle_idx : int
            Index of the particle type (0-based)
            
        Returns
        -------
        List[XssEntry] or None
            The list of locators, or None if the particle type doesn't exist
        """
        if particle_idx < 0 or particle_idx >= len(self.particle_production):
            return None
        return self.particle_production[particle_idx]
    
    def get_locator_value(self, locator: XssEntry) -> int:
        """
        Get the integer value of a locator.
        
        Parameters
        ----------
        locator : XssEntry
            The locator entry
            
        Returns
        -------
        int
            The integer value of the locator
        """
        return int(locator.value)
    
    def get_neutron_locator_values(self) -> List[int]:
        """Get the integer values of neutron reaction locators."""
        return [int(entry.value) for entry in self.incident_neutron]
    
    def get_photon_locator_values(self) -> List[int]:
        """Get the integer values of photon production locators."""
        return [int(entry.value) for entry in self.photon_production]
    
    def get_delayed_locator_values(self) -> List[int]:
        """Get the integer values of delayed neutron locators."""
        return [int(entry.value) for entry in self.delayed_neutron]
    
    def get_particle_locator_values(self, particle_idx: int = 0) -> Optional[List[int]]:
        """Get the integer values of particle production locators for a specific particle type."""
        locators = self.get_particle_production_locators(particle_idx)
        if locators is None:
            return None
        return [int(entry.value) for entry in locators]
