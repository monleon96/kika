from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from mcnpy.ace.classes.xss import XssEntry
from mcnpy.ace.classes.angular_distribution.base import AngularDistribution
from mcnpy.ace.classes.angular_distribution.utils import (
    ErrorMessageDict,
    ErrorMessageList,
)
from mcnpy._utils import create_repr_section
from mcnpy.ace.classes.angular_distribution.distributions.isotropic import IsotropicAngularDistribution
from mcnpy.ace.classes.angular_distribution.distributions.kalbach_mann import KalbachMannAngularDistribution
from mcnpy.ace.classes.angular_distribution.distributions.equiprobable import EquiprobableAngularDistribution
from mcnpy.ace.classes.angular_distribution.distributions.tabulated import TabulatedAngularDistribution
from mcnpy.ace.classes.angular_distribution.utils import Law44DataError


@dataclass
class AngularDistributionContainer:
    """Container for all angular distributions."""
    elastic: Optional[AngularDistribution] = None  # Angular distribution for elastic scattering
    incident_neutron: Dict[int, AngularDistribution] = field(default_factory=dict)  # MT -> distribution for neutrons
    photon_production: Dict[int, AngularDistribution] = field(default_factory=dict)  # MT -> distribution for photons
    particle_production: List[Dict[int, AngularDistribution]] = field(default_factory=list)  # Particle index -> (MT -> distribution)
    
    def __post_init__(self):
        """Convert standard dictionaries to ErrorMessageDict and lists to ErrorMessageList."""
        # Convert incident_neutron dictionary to ErrorMessageDict
        if isinstance(self.incident_neutron, dict) and not isinstance(self.incident_neutron, ErrorMessageDict):
            self.incident_neutron = ErrorMessageDict(self.incident_neutron, dict_name="incident_neutron distributions")
        
        # Convert photon_production dictionary to ErrorMessageDict
        if isinstance(self.photon_production, dict) and not isinstance(self.photon_production, ErrorMessageDict):
            self.photon_production = ErrorMessageDict(self.photon_production, dict_name="photon_production distributions")
        
        # Convert particle_production list to ErrorMessageList
        if isinstance(self.particle_production, list) and not isinstance(self.particle_production, ErrorMessageList):
            # First convert the list itself
            particle_list = ErrorMessageList(self.particle_production, list_name="particle_production")
            
            # Then convert each dictionary in the list
            for i in range(len(particle_list)):
                if isinstance(particle_list[i], dict) and not isinstance(particle_list[i], ErrorMessageDict):
                    particle_list[i] = ErrorMessageDict(
                        particle_list[i], 
                        dict_name=f"particle_production[{i}] distributions"
                    )
            
            self.particle_production = particle_list
    
    @property
    def has_elastic_data(self) -> bool:
        """Check if elastic scattering angular distribution data is available."""
        return self.elastic is not None and not self.elastic.is_isotropic
    
    @property
    def has_neutron_data(self) -> bool:
        """Check if neutron reaction angular distribution data is available."""
        return len(self.incident_neutron) > 0
    
    @property
    def has_photon_production_data(self) -> bool:
        """Check if photon production angular distribution data is available."""
        return len(self.photon_production) > 0
    
    @property
    def has_particle_production_data(self) -> bool:
        """Check if particle production angular distribution data is available."""
        return len(self.particle_production) > 0 and any(len(p) > 0 for p in self.particle_production)
    
    def get_neutron_reaction_mt_numbers(self) -> List[int]:
        """Get the list of MT numbers for neutron reactions with angular distributions."""
        if isinstance(self.incident_neutron, ErrorMessageDict):
            return sorted(self.incident_neutron.keys_as_int())
        else:
            return sorted(list(self.incident_neutron.keys()))
    
    def get_photon_production_mt_numbers(self) -> List[int]:
        """Get the list of MT numbers for photon production with angular distributions."""
        if isinstance(self.photon_production, ErrorMessageDict):
            return sorted(self.photon_production.keys_as_int())
        else:
            return sorted(list(self.photon_production.keys()))
    
    def get_particle_production_mt_numbers(self, particle_idx: Optional[int] = None) -> Union[Dict[int, List[int]], List[int]]:
        """
        Get the list of MT numbers for particle production with angular distributions.
        
        Parameters
        ----------
        particle_idx : int, optional
            Index of the particle type. If None, returns a dictionary mapping
            particle indices to their MT numbers
            
        Returns
        -------
        Dict[int, List[int]] or List[int]
            If particle_idx is None: Dictionary mapping particle indices to lists of MT numbers
            If particle_idx is given: List of MT numbers for that particle index
        
        Raises
        ------
        IndexError
            If the specified particle index is out of bounds
        """
        # If no particle_idx specified, return dictionary for all particles
        if particle_idx is None:
            result = {}
            for idx in range(len(self.particle_production)):
                particle_data = self.particle_production[idx]
                if isinstance(particle_data, ErrorMessageDict):
                    result[idx] = sorted(particle_data.keys_as_int())
                else:
                    mt_keys = particle_data.keys()
                    if mt_keys and isinstance(next(iter(mt_keys)), XssEntry):
                        result[idx] = sorted([int(mt.value) for mt in mt_keys])
                    else:
                        result[idx] = sorted(list(mt_keys))
            return result
            
        # If particle_idx is specified, return list for that particle
        if particle_idx < 0 or particle_idx >= len(self.particle_production):
            available_indices = list(range(len(self.particle_production)))
            error_message = f"Particle index {particle_idx} is out of bounds."
            
            if len(self.particle_production) == 0:
                error_message += " No particle production data is available."
            else:
                error_message += f" Available particle indices: {available_indices}"
                
                # Add more information about particle counts for each index
                error_message += "\nParticle counts by index:"
                for idx, particle_data in enumerate(self.particle_production):
                    error_message += f"\n  Index {idx}: {len(particle_data)} reactions"
            
            raise IndexError(error_message)
        
        particle_data = self.particle_production[particle_idx]
        
        # Extract the MT values from XssEntry objects before sorting
        if isinstance(particle_data, ErrorMessageDict):
            return sorted(particle_data.keys_as_int())
        else:
            mt_keys = particle_data.keys()
            
            # Check if the keys are XssEntry objects or integers
            if mt_keys and isinstance(next(iter(mt_keys)), XssEntry):
                # If they are XssEntry objects, get their values first
                return sorted([int(mt.value) for mt in mt_keys])
            else:
                # If they are already integers, sort them directly
                return sorted(list(mt_keys))
    
    def get_particle_production_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Get comprehensive information about particle production angular distributions.
        
        This provides a more detailed and user-friendly version of the data compared to
        get_particle_production_mt_numbers().
        
        Returns
        -------
        Dict[int, Dict[str, Any]]
            Dictionary mapping particle indices to dictionaries containing:
            - 'mt_numbers': List of MT numbers
            - 'count': Total number of reactions
            - 'distribution_types': Dictionary counting each distribution type
            - 'description': Text description
            
        Examples
        --------
        >>> info = container.get_particle_production_info()
        >>> for idx, data in info.items():
        ...     print(f"Particle {idx}: {data['count']} reactions, {data['description']}")
        ...     print(f"MT numbers: {data['mt_numbers']}")
        """
        result = {}
        
        for idx in range(len(self.particle_production)):
            particle_data = self.particle_production[idx]
            
            # Get MT numbers for this particle
            if isinstance(particle_data, ErrorMessageDict):
                mt_numbers = sorted(particle_data.keys_as_int())
            else:
                mt_keys = particle_data.keys()
                if mt_keys and isinstance(next(iter(mt_keys)), XssEntry):
                    mt_numbers = sorted([int(mt.value) for mt in mt_keys])
                else:
                    mt_numbers = sorted(list(mt_keys))
            
            # Count distribution types
            distribution_types = {}
            for mt in mt_numbers:
                dist = particle_data[mt]
                dist_type = dist.distribution_type.name
                distribution_types[dist_type] = distribution_types.get(dist_type, 0) + 1
            
            # Create a description
            description = f"{len(mt_numbers)} reactions"
            if distribution_types:
                type_str = ", ".join(f"{count} {dist_type.lower()}" 
                                   for dist_type, count in distribution_types.items())
                description += f" ({type_str})"
            
            # Store all information
            result[idx] = {
                'mt_numbers': mt_numbers,
                'count': len(mt_numbers),
                'distribution_types': distribution_types,
                'description': description
            }
            
        return result

    def to_dataframe(self, mt: int, energy: float, particle_type: str = 'neutron', 
                    particle_idx: int = 0, ace=None, num_points: int = 100, 
                    interpolate: bool = False) -> Optional[pd.DataFrame]:
        """
        Convert an angular distribution to a pandas DataFrame.
        
        Parameters
        ----------
        mt : int
            MT number for the reaction
        energy : float
            Incident energy to evaluate the distribution at
        particle_type : str, optional
            Type of particle: 'neutron', 'photon', or 'particle'
        particle_idx : int, optional
            Index of the particle type (used only for particle_type='particle')
        ace : Ace, optional
            ACE object containing the distribution data (required for Kalbach-Mann)
        num_points : int, optional
            Number of angular points to generate when interpolating, defaults to 100
        interpolate : bool, optional
            Whether to interpolate onto a regular grid (True) or return original points (False)
            
        Returns
        -------
        pandas.DataFrame or None
            DataFrame with 'energy', 'cosine', and 'pdf' columns
            Returns None if pandas is not available
            
        Raises
        ------
        Law44DataError
            If trying to process a Kalbach-Mann distribution without providing ACE data
        KeyError
            If the MT number is not found in the distribution container
        ValueError
            If the particle type is unknown
        """
        try:
            import pandas as pd
            
            # Special case for elastic scattering (MT=2)
            if particle_type == 'neutron' and mt == 2 and self.elastic:
                return self.elastic.to_dataframe(energy, num_points, interpolate)
            
            # Get the appropriate distribution container
            if particle_type == 'neutron':
                dist_container = self.incident_neutron
            elif particle_type == 'photon':
                dist_container = self.photon_production
            elif particle_type == 'particle':
                if particle_idx < 0 or particle_idx >= len(self.particle_production):
                    raise ValueError(f"Particle index {particle_idx} out of bounds")
                dist_container = self.particle_production[particle_idx]
            else:
                raise ValueError(f"Unknown particle type: {particle_type}")
            
            # Get the angular distribution for this MT number
            if mt not in dist_container:
                raise KeyError(f"MT={mt} not found in {particle_type} angular distributions")
            
            # Add information about the particle type and MT number to the dataframe
            distribution = dist_container[mt]
            df = None
            
            # Special handling for Kalbach-Mann distributions
            if isinstance(distribution, KalbachMannAngularDistribution):
                df = distribution.to_dataframe(energy, ace, num_points, interpolate=True)
            else:
                df = distribution.to_dataframe(energy, num_points, interpolate)
                
            if df is not None:
                # Add columns for particle type and MT
                df['particle_type'] = particle_type
                df['mt'] = mt
                if particle_type == 'particle':
                    df['particle_idx'] = particle_idx
                
                return df
            return None
                
        except ImportError:
            return None
    
    def plot(self, mt: int, energy: float, particle_type: str = 'neutron', 
            particle_idx: int = 0, ace=None, ax=None, title=None, **kwargs) -> Optional[Tuple]:
        """
        Plot an angular distribution for a specific reaction and energy.
        
        Parameters
        ----------
        mt : int
            MT number for the reaction
        energy : float
            Incident energy to evaluate the distribution at
        particle_type : str, optional
            Type of particle: 'neutron', 'photon', or 'particle'
        particle_idx : int, optional
            Index of the particle type (used only for particle_type='particle')
        ace : Ace, optional
            ACE object containing the distribution data (needed for Kalbach-Mann)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created
        title : str, optional
            Title for the plot, if None a default title is used
        **kwargs : dict
            Additional keyword arguments passed to the plot function
            
        Returns
        -------
        tuple or None
            Tuple of (fig, ax) or None if matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get the data to plot
            df = self.to_dataframe(mt, energy, particle_type, particle_idx, ace)
            
            # Create figure and axes if not provided
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            else:
                fig = ax.figure
            
            # Set default parameters if not specified
            if 'linewidth' not in kwargs:
                kwargs['linewidth'] = 2
            if 'color' not in kwargs:
                kwargs['color'] = 'blue'
            
            # Plot the data
            ax.plot(df['cosine'], df['pdf'], **kwargs)
            
            # Set labels and title
            ax.set_xlabel('Cosine (μ)')
            ax.set_ylabel('Probability Density')
            
            if title is None:
                title = f'Angular Distribution for MT={mt} at {energy:.4g} MeV ({particle_type})'
            ax.set_title(title)
            
            # Set axis limits
            ax.set_xlim(-1, 1)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            return fig, ax
        except ImportError:
            return None
    
    def plot_energy_comparison(self, mt: int, energies: List[float], particle_type: str = 'neutron',
                              particle_idx: int = 0, ace=None, ax=None, title=None, 
                              colors=None, labels=None, **kwargs) -> Optional[Tuple]:
        """
        Plot angular distributions for multiple energies for comparison.
        
        Parameters
        ----------
        mt : int
            MT number for the reaction
        energies : List[float]
            List of incident energies to evaluate the distribution at
        particle_type : str, optional
            Type of particle: 'neutron', 'photon', or 'particle'
        particle_idx : int, optional
            Index of the particle type (used only for particle_type='particle')
        ace : Ace, optional
            ACE object containing the distribution data (required for Kalbach-Mann)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created
        title : str, optional
            Title for the plot, if None a default title is used
        colors : List[str], optional
            List of colors for each energy, if None, default colors are used
        labels : List[str], optional
            List of labels for each energy, if None, energies are used
        **kwargs : dict
            Additional keyword arguments passed to the plot function
            
        Returns
        -------
        tuple or None
            Tuple of (fig, ax) or None if matplotlib is not available
            
        Raises
        ------
        Law44DataError
            If trying to process a Kalbach-Mann distribution without providing ACE data
        KeyError
            If the MT number is not found in the distribution container
        ValueError
            If the particle type is unknown
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get the appropriate distribution container
            if particle_type == 'neutron':
                dist_container = self.incident_neutron
            elif particle_type == 'photon':
                dist_container = self.photon_production
            elif particle_type == 'particle':
                if particle_idx < 0 or particle_idx >= len(self.particle_production):
                    raise ValueError(f"Particle index {particle_idx} out of bounds")
                dist_container = self.particle_production[particle_idx]
            else:
                raise ValueError(f"Unknown particle type: {particle_type}")
            
            # Get the angular distribution for this MT number
            if mt not in dist_container:
                raise KeyError(f"MT={mt} not found in {particle_type} angular distributions")
                
            # Check if this is a Kalbach-Mann distribution that requires ACE data
            distribution = dist_container[mt]
            if isinstance(distribution, KalbachMannAngularDistribution) and ace is None:
                raise Law44DataError(
                    f"ACE object must be provided for Kalbach-Mann angular distribution (MT={mt})"
                )
            
            # Create figure and axes if not provided
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            else:
                fig = ax.figure
            
            # Set default colors if not provided
            if colors is None:
                colors = plt.cm.viridis(np.linspace(0, 1, len(energies)))
            
            # Set default labels if not provided
            if labels is None:
                labels = [f"{e:.4g} MeV" for e in energies]
            
            # Plot for each energy
            for i, energy in enumerate(energies):
                # Get color and label for this energy
                color = colors[i] if i < len(colors) else 'blue'
                label = labels[i] if i < len(labels) else f"{energy:.4g} MeV"
                
                try:
                    # Get data for this energy - may raise Law44DataError
                    df = self.to_dataframe(mt, energy, particle_type, particle_idx, ace)
                    
                    # Plot data
                    plot_kwargs = kwargs.copy()
                    plot_kwargs['color'] = color
                    plot_kwargs['label'] = label
                    
                    ax.plot(df['cosine'], df['pdf'], **plot_kwargs)
                except Law44DataError as e:
                    # Skip this energy if Law 44 data is missing and add a note to the label
                    import warnings
                    warnings.warn(f"Energy {energy} MeV skipped: {str(e)}")
                    continue
            
            # Set labels and title
            ax.set_xlabel('Cosine (μ)')
            ax.set_ylabel('Probability Density')
            
            if title is None:
                title = f'Angular Distribution for MT={mt} ({particle_type}) at Multiple Energies'
            ax.set_title(title)
            
            # Set axis limits
            ax.set_xlim(-1, 1)
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            return fig, ax
        except ImportError:
            return None
        except Law44DataError:
            # Re-raise Law44DataError to be handled by the caller
            raise
    
    def __repr__(self) -> str:
        """
        Returns a user-friendly, formatted string representation of the container.
        
        Returns
        -------
        str
            Formatted string representation
        """
        header_width = 90
        header = "=" * header_width + "\n"
        header += f"{'Angular Distribution Container':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        description = (
            "This container holds angular distributions for different reaction types and particles.\n"
            "Angular distributions describe the probability of a particle scattering at a specific angle,\n"
            "represented by the cosine of the scattering angle (μ) ranging from -1 to +1.\n\n"
            "Note: Some distributions (Kalbach-Mann/Law=44) require additional data from the energy\n"
            "distribution section. For these distributions, the ACE object must be provided when\n"
            "calling methods to avoid Law44DataError exceptions.\n\n"
        )
        
        # Create a summary table of available data
        property_col_width = 40
        value_col_width = header_width - property_col_width - 3
        
        info_table = "Available Angular Distribution Data:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Distribution Type", "Status", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        # Elastic scattering
        elastic_status = "Available"
        if not self.has_elastic_data:
            elastic_status = "Not available or isotropic"
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Elastic Scattering (MT=2)", elastic_status,
            width1=property_col_width, width2=value_col_width)
        
        # Neutron reaction distributions
        neutron_status = f"Available ({len(self.incident_neutron)} reactions)"
        if not self.has_neutron_data:
            neutron_status = "Not available"
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Neutron Reactions", neutron_status,
            width1=property_col_width, width2=value_col_width)
        
        # Photon production distributions
        photon_status = f"Available ({len(self.photon_production)} reactions)"
        if not self.has_photon_production_data:
            photon_status = "Not available"
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Photon Production", photon_status,
            width1=property_col_width, width2=value_col_width)
        
        # Particle production distributions
        if self.has_particle_production_data:
            num_particle_types = len(self.particle_production)
            particle_counts = [len(p) for p in self.particle_production]
            particle_status = f"Available ({num_particle_types} types, {sum(particle_counts)} total reactions)"
        else:
            particle_status = "Not available"
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Particle Production", particle_status,
            width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for data access - only include available data
        data_access = {}
        
        # Only add elastic if available
        if self.has_elastic_data:
            data_access[".elastic"] = "Access elastic scattering angular distribution"
        
        # Only add neutron reactions if available
        if self.has_neutron_data:
            data_access[".incident_neutron[MT]"] = "Dictionary of angular distributions for neutron reactions"
        
        # Only add photon production if available
        if self.has_photon_production_data:
            data_access[".photon_production[MT]"] = "Dictionary of angular distributions for photon production"
        
        # Only add particle production if available
        if self.has_particle_production_data:
            data_access[".particle_production[particle_idx][MT]"] = "List of dictionaries for particle production"
        
        data_access_section = create_repr_section(
            "Data Access Properties:", 
            data_access, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Add methods section - only include get methods for available data
        methods = {}
        
        # Only add get methods for data types that are available
        if self.has_neutron_data:
            methods[".get_neutron_reaction_mt_numbers()"] = "Get list of MT numbers for neutron reactions"
        
        if self.has_photon_production_data:
            methods[".get_photon_production_mt_numbers()"] = "Get list of MT numbers for photon production"
        
        if self.has_particle_production_data:
            methods[".get_particle_production_mt_numbers()"] = "Get list of MT numbers for each particle type"
        
        # Always include these general methods
        methods.update({
            ".to_dataframe(...)": "Convert distribution to DataFrame",
            ".plot(...)": "Plot an angular distribution",
            ".plot_energy_comparison(...)": "Compare distributions at different energies"
        })
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Add note about Kalbach-Mann to example section
        example = (
            "Example:\n"
            "--------\n"
            "# Get MT numbers for neutron reactions with angular distributions\n"
            "mt_numbers = container.get_neutron_reaction_mt_numbers()\n\n"
            "# Plot the angular distribution for MT=16 at 14 MeV\n"
            "fig, ax = container.plot(mt=16, energy=14.0, ace=ace_object)  # ACE needed for Kalbach-Mann\n\n"
            "# Compare angular distributions at different energies\n"
            "fig, ax = container.plot_energy_comparison(mt=16, energies=[1.0, 5.0, 14.0], ace=ace_object)\n"
        )
        
        return header + description + info_table + data_access_section + "\n" + methods_section + "\n" + example
