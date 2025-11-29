from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ParticleLocatorSet:
    """
    Container for the locators for a single particle type.
    
    Each particle type has 10 locators that point to different data blocks.
    """
    hpd: int = 0       # Cross section and heating data location
    mtrh: int = 0      # MT reaction numbers location
    tyrh: int = 0      # Reaction types location
    lsigh: int = 0     # Cross section locators location
    sigh: int = 0      # Cross section data location
    landh: int = 0     # Angular distribution locators location
    andh: int = 0      # Angular distribution data location
    ldlwh: int = 0     # Energy distribution locators location
    dlwh: int = 0      # Energy distribution data location
    yh: int = 0        # Yield multipliers location

@dataclass
class SecondaryParticleDataLocators:
    """
    Container for data locations for each secondary particle type.
    
    This contains pointers to where various data for each secondary particle type
    can be found. It includes locations for cross sections, angular distributions,
    energy distributions, and more, specific to each particle type.
    """
    locator_sets: List[ParticleLocatorSet] = field(default_factory=list)
    has_data: bool = False
    
    def get_locators(self, particle_index: int) -> Optional[ParticleLocatorSet]:
        """
        Get the locator set for a specific particle type by index.
        
        Parameters
        ----------
        particle_index : int
            The 1-based index of the particle type
            
        Returns
        -------
        ParticleLocatorSet or None
            The locator set for the particle type, or None if not found
        """
        if particle_index < 1 or particle_index > len(self.locator_sets):
            return None
        return self.locator_sets[particle_index - 1]
    
    def create_locator_set(self, **kwargs) -> ParticleLocatorSet:
        """
        Create a new locator set with the provided values.
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for ParticleLocatorSet constructor
            
        Returns
        -------
        ParticleLocatorSet
            The created locator set
        """
        return ParticleLocatorSet(**kwargs)
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No secondary particle data location information available"
        
        output = f"Data Locations for {len(self.locator_sets)} Secondary Particle Types:\n"
        output += "=" * 70 + "\n"
        output += "These locations point to specific data for each secondary particle type,\n"
        output += "including cross sections, angular distributions, and energy distributions.\n\n"
        
        for i, locator_set in enumerate(self.locator_sets, 1):
            output += f"Particle Type {i}:\n"
            output += f"  Production & Heating data: {locator_set.hpd}\n"
            output += f"  MT reaction numbers: {locator_set.mtrh}\n"
            output += f"  Reaction type data: {locator_set.tyrh}\n"
            output += f"  Cross section locators: {locator_set.lsigh}\n"
            output += f"  Cross section data: {locator_set.sigh}\n"
            output += f"  Angular dist. locators: {locator_set.landh}\n"
            output += f"  Angular distributions: {locator_set.andh}\n"
            output += f"  Energy dist. locators: {locator_set.ldlwh}\n"
            output += f"  Energy distributions: {locator_set.dlwh}\n"
            output += f"  Yield multipliers: {locator_set.yh}\n\n"
        
        return output
