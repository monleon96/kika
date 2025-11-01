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
    
    def to_plot_data(self, mt: int, energy: float, particle_type: str = 'neutron',
                    particle_idx: int = 0, ace=None, **kwargs):
        """
        Extract angular distribution plot data in a format compatible with PlotBuilder.
        
        This method provides direct access to angular distribution data without going through
        the parent Ace object. It's equivalent to calling ace.to_plot_data('ang', mt=mt, energy=energy).
        
        Parameters
        ----------
        mt : int
            MT reaction number
        energy : float
            Incident energy in MeV at which to evaluate the distribution
        particle_type : str, optional
            Type of particle: 'neutron', 'photon', or 'particle' (default: 'neutron')
        particle_idx : int, optional
            Particle index for particle_type='particle' (default: 0)
        ace : Ace, optional
            ACE object (required for Kalbach-Mann distributions)
        **kwargs
            Additional parameters for styling and data extraction:
            - label (str): Custom label (default: auto-generated)
            - color (str): Line color
            - linestyle (str): Line style ('-', '--', '-.', ':')
            - linewidth (float): Line width
            - marker (str): Marker style ('o', 's', '^', etc.)
            - markersize (float): Marker size
            - num_points (int): Number of angular points when interpolating (default: 100)
            - interpolate (bool): Whether to interpolate onto regular grid (default: False)
            
        Returns
        -------
        PlotData
            PlotData object compatible with PlotBuilder
            
        Examples
        --------
        >>> ace = mcnpy.read_ace('fe56.ace')
        >>> 
        >>> # Direct access from angular_distributions object
        >>> ang_data = ace.angular_distributions.to_plot_data(mt=2, energy=5.0, label='5 MeV')
        >>> 
        >>> # Use with PlotBuilder
        >>> from mcnpy.plotting import PlotBuilder
        >>> fig = (PlotBuilder()
        ...        .add_data(ang_data)
        ...        .set_labels(x_label='cos(θ)', y_label='Probability Density')
        ...        .build())
        """
        from mcnpy.plotting import PlotData
        import numpy as np
        
        # Additional parameters for to_dataframe
        num_points = kwargs.pop('num_points', 100)
        interpolate = kwargs.pop('interpolate', False)
        
        # Get the angular distribution data as a DataFrame
        df = self.to_dataframe(
            mt=mt,
            energy=energy,
            particle_type=particle_type,
            particle_idx=particle_idx,
            ace=ace,
            num_points=num_points,
            interpolate=interpolate
        )
        
        if df is None:
            raise ValueError(f"Could not extract angular distribution for MT={mt} at energy={energy} MeV")
        
        # Extract cosine (mu) and pdf from the DataFrame
        mu = df['cosine'].values
        pdf = df['pdf'].values
        
        # Get label
        label = kwargs.get('label', None)
        if label is None:
            # Create default label
            label = f"MT={mt} @ {energy} MeV"
        
        return PlotData(
            x=np.array(mu),
            y=np.array(pdf),
            label=label,
            color=kwargs.get('color', None),
            linestyle=kwargs.get('linestyle', '-'),
            linewidth=kwargs.get('linewidth', None),
            marker=kwargs.get('marker', None),
            markersize=kwargs.get('markersize', None),
            plot_type='line'
        )
    
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
            "This container holds angular distributions read directly from the ACE file.\n"
            "Each distribution preserves the original data structure as found in the ACE format.\n\n"
            "Angular distributions describe the probability of scattering as a function of the\n"
            "cosine of the scattering angle (μ), which ranges from -1 (backward scattering) to\n"
            "+1 (forward scattering).\n\n"
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
        else:
            elastic_type = self.elastic.distribution_type.name
            elastic_status = f"Available ({elastic_type})"
        
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
        
        if self.has_neutron_data:
            # Count distribution types
            dist_types = {}
            for mt, dist in self.incident_neutron.items():
                dist_type = dist.distribution_type.name
                dist_types[dist_type] = dist_types.get(dist_type, 0) + 1
            
            # Add distribution type counts
            for dist_type, count in dist_types.items():
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    f"  {dist_type}", f"{count} reactions",
                    width1=property_col_width, width2=value_col_width)
        
        # Photon production distributions
        photon_status = f"Available ({len(self.photon_production)} reactions)"
        if not self.has_photon_production_data:
            photon_status = "Not available"
        
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Photon Production", photon_status,
            width1=property_col_width, width2=value_col_width)
        
        if self.has_photon_production_data:
            # Count distribution types
            dist_types = {}
            for mt, dist in self.photon_production.items():
                dist_type = dist.distribution_type.name
                dist_types[dist_type] = dist_types.get(dist_type, 0) + 1
            
            # Add distribution type counts
            for dist_type, count in dist_types.items():
                info_table += "{:<{width1}} {:<{width2}}\n".format(
                    f"  {dist_type}", f"{count} reactions",
                    width1=property_col_width, width2=value_col_width)
        
        # Particle production distributions
        if self.has_particle_production_data:
            particle_counts = [len(p) for p in self.particle_production]
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Particle Production", f"{len(self.particle_production)} types, {sum(particle_counts)} total reactions",
                width1=property_col_width, width2=value_col_width)
            
            # Add details for each particle type
            for i, particle_dict in enumerate(self.particle_production):
                if len(particle_dict) > 0:
                    info_table += "{:<{width1}} {:<{width2}}\n".format(
                        f"  Particle {i}", f"{len(particle_dict)} reactions",
                        width1=property_col_width, width2=value_col_width)
        else:
            info_table += "{:<{width1}} {:<{width2}}\n".format(
                "Particle Production", "Not available",
                width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Add property and method info without examples
        properties_section = (
            "Accessing Angular Distributions:\n"
            f"{'-' * header_width}\n"
            ".elastic                           Elastic scattering distribution (if available)\n"
            ".incident_neutron[mt]              Get neutron reaction distribution by MT number\n"
            ".photon_production[mt]             Get photon production distribution by MT number\n"
            ".particle_production[part_idx][mt] Get particle production distribution by index and MT\n"
            ".get_neutron_reaction_mt_numbers() Get list of available neutron reaction MT numbers\n"
            ".get_photon_production_mt_numbers() Get list of available photon production MT numbers\n\n"
        )
        
        return header + description + info_table + properties_section
