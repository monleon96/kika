from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mcnpy.ace.xss import XssEntry
from mcnpy.ace.classes.header import Header
from mcnpy.ace.classes.nubar.nubar import NuContainer
from mcnpy.ace.classes.delayed_neutron import DelayedNeutronData
from mcnpy.ace.classes.mtr import ReactionMTData
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
from mcnpy.ace.classes.yield_multipliers import PhotonYieldMultipliers, ParticleYieldMultipliers
from mcnpy.ace.classes.fission_xs import FissionCrossSection
from mcnpy.ace.classes.unresolved_resonance import UnresolvedResonanceTables
from mcnpy.ace.classes.particle_production import ParticleProductionTypes
from mcnpy.ace.classes.particle_reaction_counts import ParticleReactionCounts
from mcnpy.ace.classes.particle_production_locators import ParticleProductionLocators
from mcnpy.ace.classes.particle_production_xs_data import ParticleProductionXSContainer
from mcnpy.ace.classes.esz import EszBlock


@dataclass
class Ace:
    """
    Class representing ACE format data.
    
    ACE (A Compact ENDF) is a format used in nuclear data libraries.
    This implementation uses lazy loading to reduce memory consumption.
    """
    # Original filename
    filename: Optional[str] = None
    
    # Header information - loaded immediately
    header: Optional[Header] = None
    
    # XSS data array - loaded immediately
    xss_data: Optional[List[XssEntry]] = None  # Main data array
    
    # Cache for parsed components
    _cache: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    # Debug flag for parsers
    _debug: bool = False
    
    @property
    def esz_block(self):
        """Lazy-loaded ESZ block (energy tables and cross sections)"""
        if "esz_block" not in self._cache:
            from mcnpy.ace.parsers.parse_esz import read_esz_block
            from mcnpy.ace.classes.esz import EszBlock
            self._cache["esz_block"] = EszBlock()
            read_esz_block(self, debug=self._debug)
        return self._cache["esz_block"]
    
    @property
    def energies(self):
        """Energy grid (backward compatibility) - returns list of float values"""
        if self.esz_block and self.esz_block.energies:
            return [e.value for e in self.esz_block.energies]
        return None
    
    @property
    def total_xs(self):
        """Total cross section (backward compatibility) - returns list of float values"""
        if self.esz_block and self.esz_block.total_xs:
            return [xs.value for xs in self.esz_block.total_xs]
        return None
    
    @property
    def absorption_xs(self):
        """Absorption cross section (backward compatibility) - returns list of float values"""
        if self.esz_block and self.esz_block.absorption_xs:
            return [xs.value for xs in self.esz_block.absorption_xs]
        return None
    
    @property
    def elastic_xs(self):
        """Elastic cross section (backward compatibility) - returns list of float values"""
        if self.esz_block and self.esz_block.elastic_xs:
            return [xs.value for xs in self.esz_block.elastic_xs]
        return None
    
    @property
    def heating_numbers(self):
        """Heating numbers (backward compatibility) - returns list of float values"""
        if self.esz_block and self.esz_block.heating_numbers:
            return [h.value for h in self.esz_block.heating_numbers]
        return None
    
    @property
    def nubar(self):
        """Lazy-loaded nubar data"""
        if "nubar" not in self._cache:
            from mcnpy.ace.parsers.parse_nubar import read_nubar_data
            from mcnpy.ace.classes.nubar.nubar import NuContainer
            self._cache["nubar"] = NuContainer()
            read_nubar_data(self, debug=self._debug)
        return self._cache["nubar"]
    
    @property
    def delayed_neutron_data(self):
        """Lazy-loaded delayed neutron data"""
        if "delayed_neutron_data" not in self._cache:
            from mcnpy.ace.parsers.parse_delayed import read_delayed_neutron_data
            from mcnpy.ace.classes.delayed_neutron import DelayedNeutronData
            self._cache["delayed_neutron_data"] = DelayedNeutronData()
            read_delayed_neutron_data(self, debug=self._debug)
        return self._cache["delayed_neutron_data"]
    
    @property
    def reaction_mt_data(self):
        """Lazy-loaded reaction MT data"""
        if "reaction_mt_data" not in self._cache:
            from mcnpy.ace.parsers.parse_mtr import read_mtr_blocks
            from mcnpy.ace.classes.mtr import ReactionMTData
            self._cache["reaction_mt_data"] = ReactionMTData()
            read_mtr_blocks(self, debug=self._debug)
        return self._cache["reaction_mt_data"]
    
    @property
    def energy_distribution_locators(self):
        """Lazy-loaded energy distribution locators"""
        if "energy_distribution_locators" not in self._cache:
            from mcnpy.ace.parsers.parse_energy_distribution_locators import read_energy_locator_blocks
            from mcnpy.ace.classes.energy_distribution_locators import EnergyDistributionLocators
            self._cache["energy_distribution_locators"] = EnergyDistributionLocators()
            read_energy_locator_blocks(self, debug=self._debug)
        return self._cache["energy_distribution_locators"]
    
    @property
    def q_values(self):
        """Lazy-loaded reaction Q-values"""
        if "q_values" not in self._cache:
            from mcnpy.ace.parsers.parse_lqr import read_lqr_block
            from mcnpy.ace.classes.q_values import QValues
            self._cache["q_values"] = QValues()
            read_lqr_block(self, debug=self._debug)
        return self._cache["q_values"]
    
    @property
    def particle_release(self):
        """Lazy-loaded particle release data"""
        if "particle_release" not in self._cache:
            from mcnpy.ace.parsers.parse_tyr import read_tyr_blocks
            from mcnpy.ace.classes.particle_release import ParticleRelease
            self._cache["particle_release"] = ParticleRelease()
            read_tyr_blocks(self, debug=self._debug, strict_validation=False)
        return self._cache["particle_release"]
    
    @property
    def xs_locators(self):
        """Lazy-loaded cross section locators"""
        if "xs_locators" not in self._cache:
            from mcnpy.ace.parsers.parse_xs_locators import read_xs_locator_blocks
            from mcnpy.ace.classes.xs_locators import CrossSectionLocators
            self._cache["xs_locators"] = CrossSectionLocators()
            read_xs_locator_blocks(self, debug=self._debug)
        return self._cache["xs_locators"]
    
    @property
    def xs_data(self):
        """Lazy-loaded cross section data"""
        if "xs_data" not in self._cache:
            from mcnpy.ace.parsers.parse_xs_data import read_xs_data_block
            from mcnpy.ace.classes.xs_data import CrossSectionData
            self._cache["xs_data"] = CrossSectionData()
            read_xs_data_block(self, debug=self._debug)
        return self._cache["xs_data"]
    
    @property
    def angular_locators(self):
        """Lazy-loaded angular distribution locators"""
        if "angular_locators" not in self._cache:
            from mcnpy.ace.parsers.parse_angular_locators import read_angular_locator_blocks
            from mcnpy.ace.classes.angular_distribution.angular_locators import AngularDistributionLocators
            self._cache["angular_locators"] = AngularDistributionLocators()
            read_angular_locator_blocks(self, debug=self._debug)
        return self._cache["angular_locators"]
    
    @property
    def angular_distributions(self):
        """Lazy-loaded angular distributions"""
        if "angular_distributions" not in self._cache:
            from mcnpy.ace.parsers.parse_angular_distribution import read_angular_distribution_blocks
            from mcnpy.ace.classes.angular_distribution.angular_distribution import AngularDistributionContainer
            self._cache["angular_distributions"] = AngularDistributionContainer()
            try:
                read_angular_distribution_blocks(self, debug=self._debug)
            except ValueError:
                # Skip if we don't have angular locators yet
                pass
        return self._cache["angular_distributions"]
    
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
    def photon_production_data(self):
        """Lazy-loaded photon production data"""
        if "photon_production_data" not in self._cache:
            from mcnpy.ace.parsers.parse_gpd import read_gpd_block
            from mcnpy.ace.classes.gpd import PhotonProductionData
            self._cache["photon_production_data"] = PhotonProductionData()
            read_gpd_block(self, debug=self._debug)
        return self._cache["photon_production_data"]
    
    @property
    def photon_production_xs(self):
        """Lazy-loaded photon production cross sections"""
        if "photon_production_xs" not in self._cache:
            from mcnpy.ace.parsers.parse_photon_production_xs import read_production_xs_blocks
            from mcnpy.ace.classes.photon_production_xs import PhotonProductionCrossSections
            self._cache["photon_production_xs"] = PhotonProductionCrossSections()
            read_production_xs_blocks(self, debug=self._debug)
        return self._cache["photon_production_xs"]
    
    @property
    def particle_production_xs(self):
        """Lazy-loaded particle production cross sections"""
        if "particle_production_xs" not in self._cache:
            from mcnpy.ace.parsers.parse_photon_production_xs import read_production_xs_blocks
            from mcnpy.ace.classes.photon_production_xs import ParticleProductionCrossSections
            self._cache["particle_production_xs"] = ParticleProductionCrossSections()
            read_production_xs_blocks(self, debug=self._debug)
        return self._cache["particle_production_xs"]
    
    @property
    def photon_yield_multipliers(self):
        """Lazy-loaded photon yield multipliers"""
        if "photon_yield_multipliers" not in self._cache:
            from mcnpy.ace.parsers.parse_yield_multipliers import read_yield_multiplier_blocks
            from mcnpy.ace.classes.yield_multipliers import PhotonYieldMultipliers
            self._cache["photon_yield_multipliers"] = PhotonYieldMultipliers()
            read_yield_multiplier_blocks(self, debug=self._debug)
        return self._cache["photon_yield_multipliers"]
    
    @property
    def particle_yield_multipliers(self):
        """Lazy-loaded particle yield multipliers"""
        if "particle_yield_multipliers" not in self._cache:
            from mcnpy.ace.parsers.parse_yield_multipliers import read_yield_multiplier_blocks
            from mcnpy.ace.classes.yield_multipliers import ParticleYieldMultipliers
            self._cache["particle_yield_multipliers"] = ParticleYieldMultipliers()
            read_yield_multiplier_blocks(self, debug=self._debug)
        return self._cache["particle_yield_multipliers"]
    
    @property
    def fission_xs(self):
        """Lazy-loaded fission cross section"""
        if "fission_xs" not in self._cache:
            from mcnpy.ace.parsers.parse_fission_xs import read_fission_xs_block
            from mcnpy.ace.classes.fission_xs import FissionCrossSection
            self._cache["fission_xs"] = FissionCrossSection()
            read_fission_xs_block(self, debug=self._debug)
        return self._cache["fission_xs"]
    
    @property
    def unresolved_resonance(self):
        """Lazy-loaded unresolved resonance data"""
        if "unresolved_resonance" not in self._cache:
            from mcnpy.ace.parsers.parse_unresolved_resonance import read_unresolved_resonance_block
            from mcnpy.ace.classes.unresolved_resonance import UnresolvedResonanceTables
            self._cache["unresolved_resonance"] = UnresolvedResonanceTables()
            read_unresolved_resonance_block(self, debug=self._debug)
        return self._cache["unresolved_resonance"]
    
    @property
    def secondary_particles(self):
        """Lazy-loaded secondary particle types"""
        if "secondary_particles" not in self._cache:
            from mcnpy.ace.parsers.parse_particle_production import read_particle_types_block
            from mcnpy.ace.classes.particle_production import ParticleProductionTypes
            self._cache["secondary_particles"] = ParticleProductionTypes()
            read_particle_types_block(self, debug=self._debug)
        return self._cache["secondary_particles"]
    
    @property
    def particle_types(self):
        """Lazy-loaded particle types (alias for secondary_particles)"""
        return self.secondary_particles
    
    @property
    def particle_reaction_counts(self):
        """Lazy-loaded particle reaction counts"""
        if "particle_reaction_counts" not in self._cache:
            from mcnpy.ace.parsers.parse_particle_reaction_counts import read_particle_reaction_counts_block
            from mcnpy.ace.classes.particle_reaction_counts import ParticleReactionCounts
            self._cache["particle_reaction_counts"] = ParticleReactionCounts()
            read_particle_reaction_counts_block(self, debug=self._debug)
        return self._cache["particle_reaction_counts"]
    
    @property
    def particle_production_locators(self):
        """Lazy-loaded particle production locators"""
        if "particle_production_locators" not in self._cache:
            from mcnpy.ace.parsers.parse_particle_production_locators import read_particle_production_locators_block
            from mcnpy.ace.classes.particle_production_locators import ParticleProductionLocators
            self._cache["particle_production_locators"] = ParticleProductionLocators()
            read_particle_production_locators_block(self, debug=self._debug)
        return self._cache["particle_production_locators"]
    
    @property
    def particle_production_xs_data(self):
        """Lazy-loaded particle production cross section data"""
        if "particle_production_xs_data" not in self._cache:
            from mcnpy.ace.parsers.parse_particle_production_xs_data import read_particle_production_xs_data_blocks
            from mcnpy.ace.classes.particle_production_xs_data import ParticleProductionXSContainer
            self._cache["particle_production_xs_data"] = ParticleProductionXSContainer()
            read_particle_production_xs_data_blocks(self, debug=self._debug)
        return self._cache["particle_production_xs_data"]
    
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
    
    def __repr__(self) -> str:
        """Returns a concise overview of the ACE data available in this object.
        
        This representation shows what data components are already loaded,
        without triggering lazy loading of additional components.
        
        :return: Formatted string representation of the ACE data
        :rtype: str
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'ACE Format Data':^{header_width}}\n"
        
        # Include material info in header if available
        if self.header and self.header.matid:
            header += f"{'Material: ' + str(self.header.matid):^{header_width}}\n"
        if self.header and self.header.zaid is not None:
            header += f"{'ZA: ' + str(self.header.zaid) + ', Temperature: ' + str(self.header.temperature) + ' K':^{header_width}}\n"
        
        header += "=" * header_width + "\n\n"
        
        # Create a summary table of what data is available and how to access it
        method_col_width = 40
        desc_col_width = header_width - method_col_width - 3  # -3 for spacing and formatting
        
        summary = "ACE Data Components:\n"
        summary += "-" * header_width + "\n"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Component", "Status", width1=method_col_width, width2=desc_col_width)
        summary += "-" * header_width + "\n"
        
        # Header information
        header_access = "Loaded (.header)" if self.header else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Header Information", header_access, width1=method_col_width, width2=desc_col_width)
        
        # Energy and Cross Section Data (ESZ block)
        esz_status = "Loaded (.esz_block)" if "esz_block" in self._cache else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Energy Grid & Cross Sections", esz_status, width1=method_col_width, width2=desc_col_width)
        
        # Nubar Data
        nubar_status = "Loaded (.nubar)" if "nubar" in self._cache else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Nubar (ν) Data", nubar_status, width1=method_col_width, width2=desc_col_width)
        
        # Delayed Neutron Data
        delayed_status = "Loaded (.delayed_neutron_data)" if "delayed_neutron_data" in self._cache else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Delayed Neutron Data", delayed_status, width1=method_col_width, width2=desc_col_width)
        
        # Reaction MT Numbers
        mtr_status = "Loaded (.reaction_mt_data)" if "reaction_mt_data" in self._cache else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Reaction MT Numbers", mtr_status, width1=method_col_width, width2=desc_col_width)
        
        # Q-values
        q_values_status = "Loaded (.q_values)" if "q_values" in self._cache else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Reaction Q-values", q_values_status, width1=method_col_width, width2=desc_col_width)
        
        # Particle Release
        particle_release_status = "Loaded (.particle_release)" if "particle_release" in self._cache else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Particle Release Data", particle_release_status, width1=method_col_width, width2=desc_col_width)
        
        # Cross Section Locators
        xs_locators_status = "Loaded (.xs_locators)" if "xs_locators" in self._cache else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Cross Section Locators", xs_locators_status, width1=method_col_width, width2=desc_col_width)
        
        # Cross Section Data
        xs_data_status = "Loaded (.xs_data)" if "xs_data" in self._cache else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Cross Section Data", xs_data_status, width1=method_col_width, width2=desc_col_width)
        
        # Angular Distribution Locators
        angular_locators_status = "Loaded (.angular_locators)" if "angular_locators" in self._cache else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Angular Distribution Locators", angular_locators_status, width1=method_col_width, width2=desc_col_width)
        
        # Angular Distributions
        angular_dist_status = "Loaded (.angular_distributions)" if "angular_distributions" in self._cache else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Angular Distributions", angular_dist_status, width1=method_col_width, width2=desc_col_width)
        
        # Energy Distribution Locators
        energy_locators_status = "Loaded (.energy_distribution_locators)" if "energy_distribution_locators" in self._cache else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Energy Distribution Locators", energy_locators_status, width1=method_col_width, width2=desc_col_width)
        
        # Energy Distributions
        energy_dist_status = "Loaded (.energy_distributions)" if "energy_distributions" in self._cache else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Energy Distributions", energy_dist_status, width1=method_col_width, width2=desc_col_width)
        
        # Photon Production Data
        photon_prod_status = "Loaded (.photon_production_data)" if "photon_production_data" in self._cache else "Not Loaded"
        summary += "{:<{width1}} {:<{width2}}\n".format(
            "Photon Production Data", photon_prod_status, width1=method_col_width, width2=desc_col_width)
        
        # Add the remaining components with the same pattern
        components = [
            ("Photon Production Cross Sections", "photon_production_xs"),
            ("Particle Production Cross Sections", "particle_production_xs"),
            ("Photon Yield Multipliers", "photon_yield_multipliers"),
            ("Particle Yield Multipliers", "particle_yield_multipliers"),
            ("Total Fission Cross Section", "fission_xs"),
            ("Unresolved Resonance Tables", "unresolved_resonance"),
            ("Secondary Particle Types", "secondary_particles"),
            ("Particle Reaction Counts", "particle_reaction_counts"),
            ("Particle Production Locators", "particle_production_locators"),
            ("Particle Production XS Data", "particle_production_xs_data")
        ]
        
        for name, cache_key in components:
            status = f"Loaded (.{cache_key})" if cache_key in self._cache else "Not Loaded"
            summary += "{:<{width1}} {:<{width2}}\n".format(
                name, status, width1=method_col_width, width2=desc_col_width)
        
        # Add note about XSS data
        if self.xss_data and len(self.xss_data) > 0:
            xss_note = f"Raw XSS array available via .xss_data ({len(self.xss_data)} elements)"
            summary += "-" * header_width + "\n"
            summary += xss_note + "\n"
        
        summary += "-" * header_width + "\n\n"
        
        # Add a reminder about how to access data
        reminder = "Note: Components marked as 'Not Loaded' will be loaded when accessed.\n"
        reminder += "Components are loaded on demand to reduce memory usage.\n"
        reminder += "Example: ace.nubar will load the nubar data when first accessed.\n"
        
        return header + summary + reminder
    
        
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
        import matplotlib.pyplot as plt
        
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