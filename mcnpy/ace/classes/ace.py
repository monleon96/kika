from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mcnpy.ace.parsers.xss import XssEntry
from mcnpy.ace.classes.header import Header
from mcnpy.ace.classes.nubar.nubar import NuContainer
from mcnpy.ace.classes.delayed_neutron.delayed_neutron import DelayedNeutronData
from mcnpy.ace.classes.mt_reaction.mtr import ReactionMTData
from mcnpy.ace.classes.q_values import QValues
from mcnpy.ace.classes.particle_release import ParticleRelease
from mcnpy.ace.classes.xs_locators import CrossSectionLocators
from mcnpy.ace.classes.xs_data import CrossSectionData
from mcnpy.ace.classes.angular_distribution.angular_locators import AngularDistributionLocators
from mcnpy.ace.classes.angular_distribution.angular_distribution import AngularDistributionContainer
from mcnpy.ace.classes.energy_distribution_locators import EnergyDistributionLocators
from mcnpy.ace.classes.energy_distribution_container import EnergyDistributionContainer
from mcnpy.ace.classes.gpd import PhotonProductionData
from mcnpy.ace.classes.photon_production_xs import PhotonProductionCrossSections, ParticleProductionCrossSections
from mcnpy.ace.classes.yield_multipliers import PhotonYieldMultipliers, SecondaryParticleYieldMultipliers
from mcnpy.ace.classes.fission_xs import FissionCrossSection
from mcnpy.ace.classes.unresolved_resonance import UnresolvedResonanceTables
from mcnpy.ace.classes.secondary_particle_cross_sections import SecondaryParticleCrossSections
from mcnpy.ace.classes.secondary_particle_data_locators import SecondaryParticleDataLocators
from mcnpy.ace.classes.secondary_particle_reactions import SecondaryParticleReactions
from mcnpy.ace.classes.secondary_particles_types import SecondaryParticleTypes
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
    xs_data: Optional[CrossSectionData] = None
    angular_locators: Optional[AngularDistributionLocators] = None
    angular_distributions: Optional[AngularDistributionContainer] = None
    energy_distribution_locators: Optional[EnergyDistributionLocators] = None
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
    def energy_distributions(self):
        """Lazy-loaded energy distributions"""
        if "energy_distributions" not in self._cache:
            from mcnpy.ace.parsers.parse_energy_distributions import read_energy_distribution_blocks
            from mcnpy.ace.classes.energy_distribution_container import EnergyDistributionContainer
            self._cache["energy_distributions"] = EnergyDistributionContainer()
            try:
                read_energy_distribution_blocks(self, debug=self._debug)
            except ValueError:
                # Skip if we don't have energy locators yet
                pass
        return self._cache["energy_distributions"]
    
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

    def get_cross_section(self, reaction: Union[int, List[int], None] = None, energy: Optional[float] = None) -> Union[pd.DataFrame, float, None]:
        """
        Get cross section data for a specific reaction or list of reactions.
        
        This method provides a unified way to access all cross sections using MT numbers:
        - MT=1: Total cross section
        - MT=2: Elastic scattering cross section
        - MT=101: Absorption cross section
        - Other MT numbers: Reaction-specific cross sections
        
        :param reaction: MT number(s) of the reaction(s). If None, returns all available cross sections.
        :type reaction: int or List[int], optional
        :param energy: If provided, returns an interpolated cross section value at this energy,
                      otherwise returns a DataFrame with energies and cross sections
        :type energy: float, optional
                
        :returns: DataFrame with energies and cross sections, or interpolated value if energy is provided
        :rtype: pd.DataFrame, float, or None
        
        :raises ValueError: If the requested reaction is not available in the data
        :raises TypeError: If reaction is not an integer or list of integers
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
        
        # For single energy point, return interpolated value(s)
        if energy is not None:
            if not self.esz_block or not self.esz_block.has_data:
                return None
                
            if len(reaction_list) == 1:
                # Single reaction, single energy -> return float
                mt = reaction_list[0]
                
                if mt in [1, 2, 101]:
                    # Standard cross sections
                    energy_values = [e.value for e in self.esz_block.energies]
                    if mt == 1:
                        xs_values = [xs.value for xs in self.esz_block.total_xs]
                    elif mt == 2:
                        xs_values = [xs.value for xs in self.esz_block.elastic_xs]
                    else:  # mt == 101
                        xs_values = [xs.value for xs in self.esz_block.absorption_xs]
                        
                    if xs_values and len(xs_values) == len(energy_values):
                        return np.interp(energy, energy_values, xs_values)
                    else:
                        raise ValueError(f"Cross section data for MT={mt} is not available")
                else:
                    # Reaction-specific cross sections
                    if not self.xs_data or not self.xs_data.has_data:
                        raise ValueError(f"Cross section data for MT={mt} is not available")
                    
                    result = self.xs_data.get_interpolated_xs(mt, energy, [e.value for e in self.esz_block.energies])
                    if result is None:
                        raise ValueError(f"Cross section data for MT={mt} is not available")
                    return result
            else:
                # Multiple reactions, single energy -> return Series
                result = {}
                for mt in reaction_list:
                    if mt in [1, 2, 101]:
                        # Standard cross sections
                        energy_values = [e.value for e in self.esz_block.energies]
                        if mt == 1:
                            xs_values = [xs.value for xs in self.esz_block.total_xs]
                        elif mt == 2:
                            xs_values = [xs.value for xs in self.esz_block.elastic_xs]
                        else:  # mt == 101
                            xs_values = [xs.value for xs in self.esz_block.absorption_xs]
                            
                        if not xs_values or len(xs_values) != len(energy_values):
                            raise ValueError(f"Cross section data for MT={mt} is not available")
                        result[f"MT={mt}"] = np.interp(energy, energy_values, xs_values)
                    else:
                        # Reaction-specific cross sections
                        if not self.xs_data or not self.xs_data.has_data:
                            raise ValueError(f"Cross section data for MT={mt} is not available")
                        
                        value = self.xs_data.get_interpolated_xs(mt, energy, [e.value for e in self.esz_block.energies])
                        if value is None:
                            raise ValueError(f"Cross section data for MT={mt} is not available")
                        result[f"MT={mt}"] = value
                
                return pd.Series(result)
        
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
                    if not self.xs_data or not self.xs_data.has_data:
                        raise ValueError(f"Cross section data for MT={mt} is not available")
                        
                    reaction_xs = self.xs_data.get_reaction_xs(mt)
                    if not reaction_xs:
                        raise ValueError(f"Cross section data for MT={mt} is not available")
                    
                    # Create array with full energy grid length
                    xs_values = np.zeros_like(energy_values)  # Use the extracted values instead of XssEntry objects
                    rx_energies = reaction_xs.get_energies(self.esz_block.energies)
                    rx_energies_values = [e.value if hasattr(e, 'value') else e for e in rx_energies]  # Extract values if needed
                    
                    # Extract the values from XssEntry objects if needed
                    rx_xs_values = [xs.value if hasattr(xs, 'value') else xs for xs in reaction_xs.xs_values]
                    
                    # Find starting index - use numeric values for comparison
                    start_idx = np.searchsorted(energy_values, rx_energies_values[0])
                    end_idx = start_idx + len(rx_xs_values)
                    
                    # Populate the array with numeric values, not XssEntry objects
                    xs_values[start_idx:end_idx] = rx_xs_values
                    
                    result[f"MT={mt}"] = xs_values
            except ValueError as e:
                if reaction is not None:
                    raise e
        
        return pd.DataFrame(result)
    
    def plot_cross_section(self, reactions=None, energies=None, ax=None, **kwargs):
        """
        Plot cross sections for specified reactions.
        
        :param reactions: MT number(s) to plot. Default is [1, 2, 101] (total, elastic, absorption)
        :type reactions: int or list of int, optional
        :param energies: Energy range to plot, default is the full energy grid
        :type energies: list, optional
        :param ax: Axes to plot on, if None a new figure is created
        :type ax: matplotlib.axes.Axes, optional
        :param kwargs: Additional keyword arguments passed to plot function
            
        :returns: The matplotlib axes containing the plot
        :rtype: matplotlib.axes.Axes
        
        :raises ValueError: If any of the requested cross sections are not available
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