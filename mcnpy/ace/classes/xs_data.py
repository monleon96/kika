from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from mcnpy.ace.parsers.xss import XssEntry
import numpy as np

@dataclass
class ReactionCrossSection:
    """Container for a single reaction's cross section data."""
    mt: XssEntry = None  # MT number for this reaction
    energy_idx: int = 0  # Starting energy grid index
    num_energies: int = 0  # Number of consecutive energy points
    xs_values: List[XssEntry] = field(default_factory=list)  # Cross section values
    
    def get_energies(self, energy_grid: List[XssEntry]) -> List[float]:
        """
        Get the energy points for this reaction.
        
        Parameters
        ----------
        energy_grid : List[XssEntry]
            The full energy grid
            
        Returns
        -------
        List[float]
            Energy points for this reaction
        """
        if self.energy_idx < 0 or self.energy_idx >= len(energy_grid):
            return []
        end_idx = min(self.energy_idx + self.num_energies, len(energy_grid))
        return [e.value for e in energy_grid[self.energy_idx:end_idx]]

@dataclass
class CrossSectionData:
    """Container for all reaction cross sections from the SIG block."""
    reactions: Dict[int, ReactionCrossSection] = field(default_factory=dict)  # MT number -> cross section data
    
    @property
    def has_data(self) -> bool:
        """Check if any reaction cross section data is available."""
        return len(self.reactions) > 0
    
    @property
    def mt_numbers(self) -> List[int]:
        """Get a list of available MT numbers."""
        return list(self.reactions.keys())
    
    def get_reaction_xs(self, mt: int) -> Optional[ReactionCrossSection]:
        """
        Get cross section data for a specific reaction by MT number.
        
        Parameters
        ----------
        mt : int
            MT number of the reaction
            
        Returns
        -------
        ReactionCrossSection or None
            Cross section data for the reaction, or None if not available
        """
        return self.reactions.get(mt)
    
    def get_interpolated_xs(self, mt: int, energy: float, energy_grid: List[float]) -> Optional[float]:
        """
        Get an interpolated cross section value for a specific reaction at a given energy.
        
        Parameters
        ----------
        mt : int
            MT number of the reaction
        energy : float
            Energy point (in MeV)
        energy_grid : List[float]
            Full energy grid
            
        Returns
        -------
        float or None
            Interpolated cross section value, or None if not available
        """
        reaction = self.get_reaction_xs(mt)
        if not reaction or not reaction.xs_values:
            return None
        
        # Get the energy points for this reaction
        rx_energies = reaction.get_energies(energy_grid)
        if not rx_energies:
            return None
        
        # Get cross section values
        xs_values = [xs.value for xs in reaction.xs_values]
        if len(rx_energies) != len(xs_values):
            return None
        
        # Check if energy is out of range
        if energy < rx_energies[0] or energy > rx_energies[-1]:
            return None
        
        # Use numpy for efficient interpolation
        return np.interp(energy, rx_energies, xs_values)
    
    def plot_reaction_xs(self, mt: int, energy_grid: List[XssEntry], ax=None, **kwargs):
        """
        Plot cross section for a specific reaction.
        
        Parameters
        ----------
        mt : int
            MT number of the reaction
        energy_grid : List[XssEntry]
            Full energy grid
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created
        **kwargs
            Additional keyword arguments passed to plot function
            
        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        reaction = self.get_reaction_xs(mt)
        if reaction and reaction.xs_values:
            rx_energies = reaction.get_energies(energy_grid)
            xs_values = [xs.value for xs in reaction.xs_values]
            ax.plot(rx_energies, xs_values, label=f"MT={mt}", **kwargs)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Energy (MeV)')
            ax.set_ylabel('Cross Section (barns)')
            ax.legend()
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        return ax
