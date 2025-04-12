from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from mcnpy.ace.classes.xss import XssEntry
from mcnpy.ace.classes.cross_section.cross_section_repr import reaction_xs_repr, xs_data_repr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class ReactionCrossSection:
    """Container for a single reaction's cross section data."""
    mt: int = 0  # MT number for this reaction 
    energy_idx: int = 0  # Starting energy grid index
    num_energies: int = 0  # Number of consecutive energy points
    _xs_entries: List[XssEntry] = field(default_factory=list)  # Original XssEntry objects for cross section values
    _energy_entries: List[XssEntry] = field(default_factory=list)  # Original XssEntry objects for energy values
    
    def __post_init__(self):
        """Initialize after creation, ensuring values are properly stored."""
        # Convert XssEntry to value if needed
        if hasattr(self.mt, 'value'):
            self.mt = int(self.mt.value)
    
    @property
    def xs_values(self) -> List[float]:
        """Get cross section values as floats."""
        return [entry.value for entry in self._xs_entries]
    
    @property
    def energies(self) -> List[float]:
        """Get energy values as floats."""
        return [entry.value for entry in self._energy_entries]
    
    def plot(self, ax=None, **kwargs):
        """
        Plot cross section for this reaction.
        
        Parameters
        ----------
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
        
        if not self._xs_entries or not self._energy_entries:
            raise ValueError("No cross section values or energies available for plotting")
        
        # Get the energy points and cross section values as DataFrame
        df = self.to_dataframe()
        
        # Plot data
        ax.plot(df["Energy"], df["Cross Section"], label=f"MT={self.mt}", **kwargs)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Cross Section (barns)')
        ax.set_title(f"Cross Section for MT={self.mt}")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        return ax
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert this reaction's cross section data to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with energy and cross section values
        """
        if not self._xs_entries or not self._energy_entries:
            return pd.DataFrame({"Energy": [], "Cross Section": []})
        
        # Ensure they have the same length
        if len(self._energy_entries) != len(self._xs_entries):
            num_points = min(len(self._energy_entries), len(self._xs_entries))
            energies = [e.value for e in self._energy_entries[:num_points]]
            xs_values = [xs.value for xs in self._xs_entries[:num_points]]
        else:
            energies = [e.value for e in self._energy_entries]
            xs_values = [xs.value for xs in self._xs_entries]
        
        # Create DataFrame
        return pd.DataFrame({
            "Energy": energies,
            "Cross Section": xs_values
        })
    
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
        
        # Update energy entries for each reaction
        for mt, reaction in self.reaction.items():
            # Calculate the proper energy range for this reaction
            if reaction.energy_idx >= 0 and reaction.energy_idx < len(energy_grid):
                end_idx = min(reaction.energy_idx + reaction.num_energies, len(energy_grid))
                reaction._energy_entries = energy_grid[reaction.energy_idx:end_idx]
    
    def add_standard_xs(self, esz_block):
        """
        Add standard cross sections from ESZ block (MT=1, MT=2, MT=101).
        
        Parameters
        ----------
        esz_block : EszBlock
            The ESZ block containing standard cross sections
        """
        if not esz_block or not esz_block.has_data:
            return
            
        # Add total cross section (MT=1)
        if esz_block.total_xs and len(esz_block.total_xs) > 0:
            # Create a ReactionCrossSection for MT=1 (total)
            total_xs = ReactionCrossSection(
                mt=1,  # Total XS
                energy_idx=0,  # Start from beginning of energy grid
                num_energies=len(esz_block.total_xs),
                _xs_entries=esz_block.total_xs,  # Store original XssEntry objects
                _energy_entries=esz_block.energies  # Store original XssEntry objects
            )
            self.reaction[1] = total_xs
            
        # Add elastic cross section (MT=2)
        if esz_block.elastic_xs and len(esz_block.elastic_xs) > 0:
            # Create a ReactionCrossSection for MT=2 (elastic)
            elastic_xs = ReactionCrossSection(
                mt=2,  # Elastic XS
                energy_idx=0,  # Start from beginning of energy grid
                num_energies=len(esz_block.elastic_xs),
                _xs_entries=esz_block.elastic_xs,  # Store original XssEntry objects
                _energy_entries=esz_block.energies  # Store original XssEntry objects
            )
            self.reaction[2] = elastic_xs
            
        # Add absorption cross section (MT=101)
        if esz_block.absorption_xs and len(esz_block.absorption_xs) > 0:
            # Create a ReactionCrossSection for MT=101 (absorption)
            absorption_xs = ReactionCrossSection(
                mt=101,  # Absorption XS
                energy_idx=0,  # Start from beginning of energy grid
                num_energies=len(esz_block.absorption_xs),
                _xs_entries=esz_block.absorption_xs,  # Store original XssEntry objects
                _energy_entries=esz_block.energies  # Store original XssEntry objects
            )
            self.reaction[101] = absorption_xs
    
    @property
    def has_data(self) -> bool:
        """Check if any reaction cross section data is available."""
        return len(self.reaction) > 0
    
    @property
    def mt_numbers(self) -> List[int]:
        """Get a list of available MT numbers in ascending order."""
        return sorted(self.reaction.keys())
    
    def plot(self, mt: Union[int, List[int]], ax=None, **kwargs):
        """
        Plot cross section for one or more reactions.
        
        Parameters
        ----------
        mt : int or List[int]
            MT number(s) of the reaction(s) to plot
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
        
        # Convert single MT to list for consistent handling
        mt_list = [mt] if isinstance(mt, int) else mt
        
        # Check if we have valid MT numbers
        if not mt_list:
            raise ValueError("No MT numbers provided for plotting")
        
        # Keep track of whether we successfully plotted anything
        plotted = False
        
        # Plot each requested MT
        for mt_num in mt_list:
            reaction = self.reaction.get(mt_num)
            if reaction and reaction._xs_entries and reaction._energy_entries:
                # Get data for this reaction
                energies = reaction.energies
                xs_values = reaction.xs_values
                
                # Plot with appropriate label
                ax.plot(energies, xs_values, label=f"MT={mt_num}", **kwargs)
                plotted = True
        
        if not plotted:
            available_mts = ", ".join(str(mt) for mt in self.mt_numbers[:10])
            if len(self.mt_numbers) > 10:
                available_mts += ", ..."
            raise ValueError(f"No valid cross section data available for requested MT numbers. Available: {available_mts}")
        
        # Set plot properties
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Cross Section (barns)')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        return ax
    
    def to_dataframe(self, mt_list=None) -> pd.DataFrame:
        """
        Convert cross section data to a pandas DataFrame.
        
        Parameters
        ----------
        mt_list : int or List[int], optional
            MT number(s) to include. If None, all available reactions are included.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with energies and cross sections for requested MT numbers
        """
        if self.energy_grid is None:
            raise ValueError("Energy grid is required but none is available")
        
        # Get the energy values
        energy_values = [e.value for e in self.energy_grid]
        
        # Create DataFrame with energy column
        result = {"Energy": energy_values}
        
        # Use all available MT numbers if none specified
        if mt_list is None:
            mt_list = self.mt_numbers  # This is now sorted
        elif isinstance(mt_list, int):
            mt_list = [mt_list]
        
        # Add columns for each MT number, but only if they exist in our data
        for mt in mt_list:
            # Verify this MT exists in our data
            if mt in self.reaction:
                reaction = self.reaction[mt]
                
                if reaction._xs_entries:
                    # Create array of zeros for the full energy grid
                    xs_values = np.zeros(len(energy_values))
                    
                    # Determine where to place the values in the full array
                    start_idx = reaction.energy_idx
                    end_idx = start_idx + reaction.num_energies
                    
                    # Ensure bounds are valid
                    if start_idx >= 0 and start_idx < len(energy_values):
                        # Place the values (clipping if necessary)
                        actual_length = min(len(reaction._xs_entries), min(end_idx, len(energy_values)) - start_idx)
                        if actual_length > 0:
                            xs_values[start_idx:start_idx + actual_length] = [xs.value for xs in reaction._xs_entries[:actual_length]]
                    
                    result[f"MT={mt}"] = xs_values
                else:
                    # If reaction has no values, fill with zeros
                    result[f"MT={mt}"] = np.zeros(len(energy_values))
            else:
                # Skip MTs that don't exist in our data
                continue
        
        return pd.DataFrame(result)
    
    def __repr__(self):
        return xs_data_repr(self)
