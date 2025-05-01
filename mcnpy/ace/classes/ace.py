from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mcnpy.ace.classes.xss import XssEntry
from mcnpy.ace.classes.header import Header
from mcnpy.ace.classes.nubar.nubar import NuContainer
from mcnpy.ace.classes.delayed_neutron.delayed_neutron import DelayedNeutronData
from mcnpy.ace.classes.mt_reaction.mtr import ReactionMTData
from mcnpy.ace.classes.q_values import QValues
from mcnpy.ace.classes.particle_release.particle_release import ParticleRelease
from mcnpy.ace.classes.cross_section.cross_section_locators import CrossSectionLocators
from mcnpy.ace.classes.cross_section.cross_section_data import CrossSectionData
from mcnpy.ace.classes.angular_distribution.angular_locators import AngularDistributionLocators
from mcnpy.ace.classes.angular_distribution.container import AngularDistributionContainer
from mcnpy.ace.classes.energy_distribution.locators import EnergyDistributionLocators
from mcnpy.ace.classes.energy_distribution.container import EnergyDistributionContainer
from mcnpy.ace.classes.gpd import PhotonProductionData
from mcnpy.ace.classes.photon_production_xs import PhotonProductionCrossSections, ParticleProductionCrossSections
from mcnpy.ace.classes.yield_multipliers import PhotonYieldMultipliers, SecondaryParticleYieldMultipliers
from mcnpy.ace.classes.fission_xs import FissionCrossSection
from mcnpy.ace.classes.unresolved_resonance import UnresolvedResonanceTables
from mcnpy.ace.classes.secondary_particles.secondary_particle_cross_sections import SecondaryParticleCrossSections
from mcnpy.ace.classes.secondary_particles.secondary_particle_data_locators import SecondaryParticleDataLocators
from mcnpy.ace.classes.secondary_particles.secondary_particle_reactions import SecondaryParticleReactions
from mcnpy.ace.classes.secondary_particles.secondary_particles_types import SecondaryParticleTypes
from mcnpy.ace.classes.esz import EszBlock
from mcnpy.ace.classes.ace_repr import ace_repr


@dataclass
class Ace:
    """
    Class representing ACE format data.
    
    ACE (A Compact ENDF) is a format used in nuclear data libraries.
    """
    # Original filename
    filename: Optional[str] = None
    
    # Header information - loaded immediately
    header: Optional[Header] = None
    
    # XSS data array - loaded immediately
    xss_data: Optional[List[XssEntry]] = None  # Main data array
    
    # Standard ACE data blocks
    esz_block: Optional[EszBlock] = None
    nubar: Optional[NuContainer] = None
    delayed_neutron_data: Optional[DelayedNeutronData] = None
    reaction_mt_data: Optional[ReactionMTData] = None
    q_values: Optional[QValues] = None
    particle_release: Optional[ParticleRelease] = None
    xs_locators: Optional[CrossSectionLocators] = None
    cross_section: Optional[CrossSectionData] = None 
    angular_locators: Optional[AngularDistributionLocators] = None
    angular_distributions: Optional[AngularDistributionContainer] = None
    energy_distribution_locators: Optional[EnergyDistributionLocators] = None
    energy_distributions: Optional[EnergyDistributionContainer] = None
    photon_production_data: Optional[PhotonProductionData] = None
    photon_production_xs: Optional[PhotonProductionCrossSections] = None
    particle_production_xs: Optional[ParticleProductionCrossSections] = None
    photon_yield_multipliers: Optional[PhotonYieldMultipliers] = None
    particle_yield_multipliers: Optional[SecondaryParticleYieldMultipliers] = None
    fission_xs: Optional[FissionCrossSection] = None
    unresolved_resonance: Optional[UnresolvedResonanceTables] = None
    
    # User-friendly attribute names for secondary particle data
    secondary_particle_types: Optional[SecondaryParticleTypes] = None
    secondary_particle_reactions: Optional[SecondaryParticleReactions] = None
    secondary_particle_data_locations: Optional[SecondaryParticleDataLocators] = None
    secondary_particle_cross_sections: Optional[SecondaryParticleCrossSections] = None
    
    # Cache for energy distributions (still lazy-loaded)
    _cache: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    # Debug flag for parsers
    _debug: bool = False
    
    @property
    def zaid(self) -> int:
        """
        Get the ZAID (Z number, A number, isotope identifier) of the ACE file.
        
        :returns: ZAID as an integer
        :rtype: int
        """
        if self.header and self.header.zaid:
            return self.header.zaid
        return None

    @property
    def energies(self):
        """Energy grid - returns list of float values"""
        if self.esz_block and self.esz_block.energies:
            return [e.value for e in self.esz_block.energies]
        return None
    
    @property
    def total_xs(self):
        """Total cross section - returns list of float values"""
        if self.esz_block and self.esz_block.total_xs:
            return [xs.value for xs in self.esz_block.total_xs]
        return None
    
    @property
    def absorption_xs(self):
        """Absorption cross section - returns list of float values"""
        if self.esz_block and self.esz_block.absorption_xs:
            return [xs.value for xs in self.esz_block.absorption_xs]
        return None
    
    @property
    def elastic_xs(self):
        """Elastic cross section - returns list of float values"""
        if self.esz_block and self.esz_block.elastic_xs:
            return [xs.value for xs in self.esz_block.elastic_xs]
        return None
    
    def copy(self) -> 'Ace':
        """
        Create an exact copy of the Ace object.
        
        Returns:
        --------
        Ace
            A new Ace object that is an exact copy of this one
        """
        import copy
        return copy.deepcopy(self)
    
    @property
    def heating_numbers(self):
        """Heating numbers - returns list of float values"""
        if self.esz_block and self.esz_block.heating_numbers:
            return [h.value for h in self.esz_block.heating_numbers]
        return None
    
    @property
    def mt_numbers(self) -> List[int]:
        """
        Get a list of all available MT numbers, including standard ones (total, elastic, absorption).
        
        :returns: List of MT numbers in ascending order
        :rtype: List[int]
        """
        mt_list = [1, 2, 101]  # Standard MT numbers (total, elastic, absorption)
        
        # Add reaction-specific MT numbers if available
        if self.reaction_mt_data and self.reaction_mt_data.has_neutron_mt_data:
            # Extract integer values from XssEntry objects
            mt_list.extend([int(entry.value) for entry in self.reaction_mt_data.incident_neutron])
        
        # Remove duplicates and sort
        return sorted(list(set(mt_list)))
    
    __repr__ = ace_repr

    def get_cross_section(self, reaction: Union[int, List[int], None] = None) -> pd.DataFrame:
        """
        Get cross section data for a specific reaction or list of reactions.
        
        This method provides a unified way to access all cross sections using MT numbers:
        - MT=1: Total cross section
        - MT=2: Elastic scattering cross section
        - MT=101: Absorption cross section
        - Other MT numbers: Reaction-specific cross sections
        
        Examples:
        ---------
        # Get total cross section (MT=1)
        >>> xs = ace.get_cross_section(1)
        
        # Get cross sections for total, elastic, and fission (MT=18)
        >>> xs = ace.get_cross_section([1, 2, 18])
        
        # Get all available cross sections
        >>> all_xs = ace.get_cross_section()
        
        Parameters:
        -----------
        reaction : int or List[int], optional
            MT number(s) of the reaction(s). If None, returns all available cross sections.
                
        Returns:
        --------
        pd.DataFrame
            DataFrame with energies and cross sections
        
        Raises:
        -------
        ValueError
            If the requested reaction is not available in the data
        TypeError
            If reaction is not an integer or list of integers
        """
        
        # If reaction is None, get all available cross sections
        if reaction is None:
            # Get all available MT numbers, including standard ones and reaction-specific ones
            reaction_list = self.mt_numbers
        # Handle single value or list of reactions
        elif isinstance(reaction, int):
            reaction_list = [reaction]
        elif isinstance(reaction, list) and all(isinstance(r, int) for r in reaction):
            reaction_list = reaction
        else:
            raise TypeError("reaction must be an integer (MT number) or a list of integers")
        
        # For full energy range, return DataFrame
        result = {}
        
        # Add energy column
        if not self.esz_block or not self.esz_block.has_data:
            raise ValueError("Energy grid is not available")
        # Extract values from XssEntry objects
        energy_values = [e.value for e in self.esz_block.energies]
        result["Energy"] = energy_values
        
        # Add standard cross sections and reaction-specific cross sections
        for mt in reaction_list:
            try:
                if mt == 1:  # Total
                    if not self.esz_block.total_xs or len(self.esz_block.total_xs) != len(self.esz_block.energies):
                        raise ValueError(f"Cross section data for MT={mt} (Total) is not available")
                    result[f"MT={mt}"] = [xs.value for xs in self.esz_block.total_xs]
                elif mt == 2:  # Elastic
                    if not self.esz_block.elastic_xs or len(self.esz_block.elastic_xs) != len(self.esz_block.energies):
                        raise ValueError(f"Cross section data for MT={mt} (Elastic) is not available")
                    result[f"MT={mt}"] = [xs.value for xs in self.esz_block.elastic_xs]
                elif mt == 101:  # Absorption
                    if not self.esz_block.absorption_xs or len(self.esz_block.absorption_xs) != len(self.esz_block.energies):
                        raise ValueError(f"Cross section data for MT={mt} (Absorption) is not available")
                    result[f"MT={mt}"] = [xs.value for xs in self.esz_block.absorption_xs]
                else:
                    # Handle reaction-specific cross sections
                    if not self.cross_section or not self.cross_section.has_data:
                        raise ValueError(f"Cross section data for MT={mt} is not available")
                        
                    # FIX: Use the reaction dictionary directly instead of calling a non-existent method
                    reaction_xs = self.cross_section.reaction.get(mt)
                    if not reaction_xs:
                        raise ValueError(f"Cross section data for MT={mt} is not available")
                    
                    # Get the energy index and number of energies for this reaction
                    energy_idx = reaction_xs.energy_idx
                    num_energies = reaction_xs.num_energies
                    
                    # Note: energy_idx is stored as a 0-indexed value, but we need
                    # to use it as an index into the energy_values array
                    if energy_idx < 0 or energy_idx >= len(energy_values) or num_energies <= 0:
                        raise ValueError(f"Invalid energy index or number of energies for MT={mt}")
                    
                    # Extract the cross section values from XssEntry objects
                    rx_xs_values = [xs.value for xs in reaction_xs._xs_entries]
                    
                    # Verify the length matches the declared number of energies
                    if len(rx_xs_values) != num_energies:
                        raise ValueError(f"Mismatch in cross section data length for MT={mt}: "
                                        f"declared {num_energies}, found {len(rx_xs_values)}")
                    
                    # Create array with full energy grid length (all zeros)
                    xs_values = np.zeros(len(energy_values))
                    
                    # Check if we can fit the data into the array
                    # energy_idx is 0-indexed, so we don't need to convert it
                    if energy_idx + num_energies > len(energy_values):
                        raise ValueError(f"Cross section data for MT={mt} would extend beyond energy grid: "
                                        f"start={energy_idx}, length={num_energies}, "
                                        f"grid size={len(energy_values)}, sum={energy_idx+num_energies}")
                    
                    # Populate the array with cross section values
                    # Use energy_idx directly since it's already 0-indexed
                    xs_values[energy_idx:energy_idx+num_energies] = rx_xs_values
                    
                    result[f"MT={mt}"] = xs_values
            except ValueError as e:
                if reaction is not None:
                    raise e
                # If getting all available cross sections, ignore those with errors
        
        return pd.DataFrame(result)

    def plot_cross_section(self, reactions=None, energies=None, ax=None, **kwargs):
        """
        Plot cross sections for specified reactions.
        
        Examples:
        ---------
        # Plot total, elastic, and fission cross sections
        >>> ace.plot_cross_section([1, 2, 18])
        
        # Plot with custom styling
        >>> ace.plot_cross_section([1, 2], linewidth=2, color=['blue', 'red'])
        
        # Plot on an existing axis
        >>> fig, ax = plt.subplots()
        >>> ace.plot_cross_section([18], ax=ax)
        
        Parameters:
        -----------
        reactions : int or list of int, optional
            MT number(s) to plot. Default is [1, 2, 101] (total, elastic, absorption)
        energies : list, optional
            Energy range to plot, default is the full energy grid
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created
        **kwargs : dict
            Additional keyword arguments passed to plot function
            
        Returns:
        --------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot
        
        Raises:
        -------
        ValueError
            If any of the requested cross sections are not available
        """
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
                
        if reactions is None:
            reactions = [1, 2, 101]  # Default: total, elastic, absorption
        
        # Convert single reaction to list for consistent handling
        if isinstance(reactions, int):
            reactions = [reactions]
                
        if energies is None and self.energies:
            energies = self.energies
        
        # Get data for all reactions in one go
        try:
            xs_data = self.get_cross_section(reactions)
        except ValueError as e:
            # Add debug information
            print(f"Error in plot_cross_section for reactions {reactions}")
            print(f"Available MT numbers: {self.mt_numbers}")
            
            # If we have the specific reaction that's causing issues:
            if isinstance(reactions, (int, list)) and (isinstance(reactions, int) or len(reactions) == 1):
                mt = reactions if isinstance(reactions, int) else reactions[0]
                if self.cross_section and mt in self.cross_section.reaction:
                    rx = self.cross_section.reaction[mt]
                    print(f"MT={mt} energy_idx={rx.energy_idx}, num_energies={rx.num_energies}")
            
            raise ValueError(f"Could not plot: {str(e)}")
            
        # Plot each reaction
        for mt in reactions:
            ax.plot(xs_data["Energy"], xs_data[f"MT={mt}"], label=f"MT={mt}", **kwargs)
                            
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Cross Section (barns)')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
            
        return ax