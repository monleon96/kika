from dataclasses import dataclass, field
from typing import List, Optional
from mcnpy.ace.classes.xss import XssEntry

@dataclass
class AngularDistributionLocators:
    """Container for angular distribution locators from LAND, LANDP, and LANDH blocks."""
    incident_neutron: List[XssEntry] = field(default_factory=list)  # LAND Block - neutron reaction angular locators
    elastic_scattering: Optional[XssEntry] = None  # First element of LAND block (special case)
    photon_production: List[XssEntry] = field(default_factory=list)  # LANDP Block - photon production angular locators
    particle_production: List[List[XssEntry]] = field(default_factory=list)  # LANDH Block - particle production angular locators
    
    # Store the NXS values
    num_neutron_reactions: int = 0  # NXS(4)
    num_secondary_neutron_reactions: int = 0  # NXS(5)
    num_photon_production_reactions: int = 0  # NXS(6)
    num_particle_types: int = 0  # NXS(7)
    
    @property
    def has_neutron_data(self) -> bool:
        """Check if neutron reaction angular distribution locators are available."""
        return len(self.incident_neutron) > 0
    
    @property
    def has_elastic_data(self) -> bool:
        """Check if elastic scattering angular distribution locator is available."""
        return self.elastic_scattering is not None and int(self.elastic_scattering.value) > 0
    
    @property
    def has_photon_production_data(self) -> bool:
        """Check if photon production angular distribution locators are available."""
        return len(self.photon_production) > 0
    
    @property
    def has_particle_production_data(self) -> bool:
        """Check if particle production angular distribution locators are available."""
        return len(self.particle_production) > 0
    
    def get_particle_production_locators(self, particle_idx: int = 0) -> Optional[List[XssEntry]]:
        """
        Get the list of particle production angular distribution locators for a specific particle type.
        
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
    
    def get_locator_value(self, locator: Optional[XssEntry]) -> int:
        """
        Get the integer value of a locator.
        
        Parameters
        ----------
        locator : XssEntry or None
            The locator entry
            
        Returns
        -------
        int
            The integer value of the locator, or 0 if the locator is None
        """
        if locator is None:
            return 0
        return int(locator.value)
