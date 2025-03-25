from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

@dataclass
class ParticleLocatorSet:
    """Container for the 10 locators for a single particle type."""
    hpd: int = 0    # Location of total particle production and heating data
    mtrh: int = 0   # Location of particle production MT array
    tyrh: int = 0   # Location of particle production TYR data
    lsigh: int = 0  # Location of particle production cross section locators
    sigh: int = 0   # Location of particle production cross sections
    landh: int = 0  # Location of particle production angular distribution locators
    andh: int = 0   # Location of particle production angular distributions
    ldlwh: int = 0  # Location of particle production energy distribution locators
    dlwh: int = 0   # Location of particle production energy distributions
    yh: int = 0     # Location of particle production yield multipliers

@dataclass
class ParticleProductionLocators:
    """
    Container for particle production locators (IXS block).
    
    This block provides locators to various data blocks for each secondary
    particle type, similar to the JXS array but specific to each particle.
    """
    has_data: bool = False
    locator_sets: List[ParticleLocatorSet] = field(default_factory=list)  # Locators for each particle type
    
    def get_locators(self, particle_index: int) -> Optional[ParticleLocatorSet]:
        """
        Get the locator set for a specific particle type.
        
        Parameters
        ----------
        particle_index : int
            1-based index of the particle type (as defined in PTYPE block)
            
        Returns
        -------
        Optional[ParticleLocatorSet]
            The locator set for the specified particle, or None if not available
        """
        if not self.has_data or particle_index < 1 or particle_index > len(self.locator_sets):
            return None
        
        return self.locator_sets[particle_index - 1]
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No particle production locator data available"
        
        output = f"Particle Production Locators for {len(self.locator_sets)} Particle Types:\n"
        output += "=" * 70 + "\n"
        output += "These locators point to data blocks specific to each secondary particle type,\n"
        output += "including cross sections, angular distributions, and energy distributions.\n\n"
        
        for i, locator_set in enumerate(self.locator_sets, 1):
            output += f"Particle Type {i}:\n"
            output += f"  HPD (production & heating): {locator_set.hpd}\n"
            output += f"  MTRH (MT numbers): {locator_set.mtrh}\n"
            output += f"  TYRH (particle release): {locator_set.tyrh}\n"
            output += f"  LSIGH (XS locators): {locator_set.lsigh}\n"
            output += f"  SIGH (cross sections): {locator_set.sigh}\n"
            output += f"  LANDH (angular dist. locators): {locator_set.landh}\n"
            output += f"  ANDH (angular distributions): {locator_set.andh}\n"
            output += f"  LDLWH (energy dist. locators): {locator_set.ldlwh}\n"
            output += f"  DLWH (energy distributions): {locator_set.dlwh}\n"
            output += f"  YH (yield multipliers): {locator_set.yh}\n\n"
        
        return output
