from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from mcnpy.ace.parsers.xss import XssEntry
from mcnpy.ace.classes.cross_section.xs_repr import reaction_xs_repr, xs_data_repr
import numpy as np
import matplotlib.pyplot as plt

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
    
    def __repr__(self):
        return reaction_xs_repr(self)

@dataclass
class CrossSectionData:
    """Container for all reaction cross sections from the SIG block."""
    reaction: Dict[int, ReactionCrossSection] = field(default_factory=dict)  # MT number -> cross section data
    energy_grid: Optional[List[XssEntry]] = None  # Store energy grid for convenience
    
    def set_energy_grid(self, energy_grid: List[XssEntry]) -> None:
        """
        Set the energy grid for this cross section data.
        
        Parameters
        ----------
        energy_grid : List[XssEntry]
            The energy grid to use for plotting and interpolation
        """
        self.energy_grid = energy_grid
    
    @property
    def has_data(self) -> bool:
        """Check if any reaction cross section data is available."""
        return len(self.reaction) > 0
    
    @property
    def mt_numbers(self) -> List[int]:
        """Get a list of available MT numbers."""
        return list(self.reaction.keys())
    
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
        return self.reaction.get(mt)
    
    def get_interpolated_xs(self, mt: int, energy: float, energy_grid: Optional[List[float]] = None) -> Optional[float]:
        """
        Get an interpolated cross section value for a specific reaction at a given energy.
        
        Parameters
        ----------
        mt : int
            MT number of the reaction
        energy : float
            Energy point (in MeV)
        energy_grid : List[float], optional
            Full energy grid, if None uses the stored energy_grid
            
        Returns
        -------
        float or None
            Interpolated cross section value, or None if not available
        """
        reaction = self.get_reaction_xs(mt)
        if not reaction or not reaction.xs_values:
            return None
        
        # Use stored energy grid if available and no grid was passed
        if energy_grid is None and self.energy_grid is not None:
            energy_grid = self.energy_grid
            
        if energy_grid is None:
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
    
    def plot_reaction_xs(self, mt: int, energy_grid: Optional[List[XssEntry]] = None, ax=None, **kwargs):
        """
        Plot cross section for a specific reaction.
        
        Parameters
        ----------
        mt : int
            MT number of the reaction
        energy_grid : List[XssEntry], optional
            Full energy grid, if None uses the stored energy_grid 
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created
        **kwargs
            Additional keyword arguments passed to plot function
            
        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot
        """
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        # Use stored energy grid if available and none was passed
        if energy_grid is None and self.energy_grid is not None:
            energy_grid = self.energy_grid
            
        if energy_grid is None:
            raise ValueError("Energy grid is required for plotting but none is available")
        
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
    
    def __repr__(self):
        return xs_data_repr(self)
