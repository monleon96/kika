"""
Utility for plotting cross sections from ACE files.

This module provides a simple function for plotting cross sections from ACE objects.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional, Tuple, Union
from mcnpy.ace.classes.ace import Ace

def plot_cross_sections(
    ace_objects: List[Ace], 
    mt_number: Union[int, List[int]],
    labels: Optional[List[str]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 6),
    log_scale: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    style: str = "default",
    font_family: str = "serif",
    show: bool = True,
    **plot_kwargs
) -> plt.Figure:
    """
    Plot cross sections from multiple ACE objects for a specific MT number or list of MT numbers.
    
    Parameters
    ----------
    ace_objects : List[Ace]
        List of Ace objects
    mt_number : int or List[int]
        MT number to plot for all ACE objects, or list of MT numbers (one per ACE object)
    labels : List[str], optional
        Labels to use in the legend (defaults to ZAID if available)
    energy_range : Tuple[float, float], optional
        Energy range to plot (min, max) in MeV
    figsize : Tuple[float, float], optional
        Figure size (width, height) in inches
    log_scale : bool, optional
        Use logarithmic scales (True by default)
    save_path : str, optional
        Path to save figure (if None, figure is not saved)
    dpi : int, optional
        DPI for saved figure
    style : str, optional
        Plot style: 'default', 'dark', 'paper', 'publication', 'presentation'
    font_family : str, optional
        Font family for text elements
    show : bool, optional
        Whether to display the figure (default: True)
    **plot_kwargs
        Additional keyword arguments passed to plot function
        
    Returns
    -------
    plt.Figure
        Figure object
        
    Examples
    --------
    >>> # Compare total cross section between two ACE objects (same MT for both)
    >>> plot_cross_sections([u235_ace, u238_ace], 1)
    
    >>> # Plot different MT numbers for each ACE object
    >>> plot_cross_sections([u235_ace, u238_ace], [1, 2])  # Total for u235, Elastic for u238
    
    >>> # With custom labels and energy range
    >>> plot_cross_sections(
    ...     [fe56_ace, fe54_ace], 
    ...     2,  # Elastic scattering for both
    ...     labels=['Fe-56', 'Fe-54'],
    ...     energy_range=(1e-5, 20.0),
    ...     style='publication'
    ... )
    """
    # Reset matplotlib to default state to avoid contamination from previous plots
    plt.rcdefaults()
    
    # Setup publication-quality settings
    plt.rcParams.update({
        'font.family': font_family,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': figsize,
        'figure.dpi': dpi,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'xtick.major.size': 5.0,
        'ytick.major.size': 5.0,
        'xtick.minor.size': 3.0,
        'ytick.minor.size': 3.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'figure.constrained_layout.use': True,
    })
    
    # Apply style customizations
    if style == 'dark':
        plt.rcParams.update({
            'axes.facecolor': 'black',
            'figure.facecolor': 'black',
            'savefig.facecolor': 'black',
            'axes.edgecolor': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'text.color': 'white',
        })
    elif style == 'presentation':
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 16,
            'lines.linewidth': 3.0,
        })
    elif style in {'paper', 'publication'}:
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'legend.fontsize': 10,
        })
    
    # Close any existing figures to prevent interference
    plt.close('all')
    
    if not ace_objects:
        print("No ACE objects provided")
        return None
    
    # Handle mt_number parameter - convert to list if single value
    if isinstance(mt_number, int):
        mt_numbers = [mt_number] * len(ace_objects)
    else:
        mt_numbers = mt_number
        if len(mt_numbers) != len(ace_objects):
            print(f"Error: {len(mt_numbers)} MT numbers provided for {len(ace_objects)} ACE objects")
            return None
    
    # Generate default labels if none provided
    if labels is None:
        labels = []
        for i, ace in enumerate(ace_objects):
            if ace.header and ace.header.zaid:
                labels.append(ace.header.zaid)
            elif ace.filename:
                labels.append(os.path.basename(ace.filename))
            else:
                labels.append(f"ACE Object {i+1}")
    elif len(labels) != len(ace_objects):
        print(f"Warning: {len(labels)} labels provided for {len(ace_objects)} objects, using default labels instead")
        labels = [f"ACE Object {i+1}" for i in range(len(ace_objects))]
    
    # Helper function to get reaction description
    def get_reaction_description(mt_num):
        try:
            from mcnpy._constants import MT_TO_REACTION
            if mt_num in MT_TO_REACTION:
                return MT_TO_REACTION[mt_num]
            else:
                return f"MT={mt_num}"
        except ImportError:
            return f"MT={mt_num}"
    
    # Create a figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Determine title based on MT numbers
    if len(set(mt_numbers)) == 1:
        # All same MT number
        mt_num = mt_numbers[0]
        mt_desc = get_reaction_description(mt_num)
        title = f"{mt_desc} (MT={mt_num}) Cross Section"
    else:
        # Multiple different MT numbers
        title = f"Cross Sections (MT={mt_numbers})"
    
    # Colors for plotting - use consistent color cycle
    colors = plt.cm.tab10.colors
    
    # Plot cross section from each ACE object
    for i, (ace, mt_num) in enumerate(zip(ace_objects, mt_numbers)):
        try:
            # Get cross section data
            xs_data = ace.get_cross_section(mt_num)
            
            # Apply energy range filter if specified
            if energy_range is not None:
                mask = (xs_data["Energy"] >= energy_range[0]) & (xs_data["Energy"] <= energy_range[1])
                energy = xs_data["Energy"][mask]
                xs = xs_data[f"MT={mt_num}"][mask]
            else:
                energy = xs_data["Energy"]
                xs = xs_data[f"MT={mt_num}"]
            
            # Create label with MT info and reaction description
            if len(set(mt_numbers)) == 1:
                plot_label = labels[i]
            else:
                reaction_desc = get_reaction_description(mt_num)
                plot_label = f"{labels[i]} {reaction_desc}"
            
            # Plot with color from cycle
            color = colors[i % len(colors)]
            ax.plot(energy, xs, label=plot_label, color=color, **plot_kwargs)
            
        except Exception as e:
            label = labels[i] if i < len(labels) else f"ACE Object {i+1}"
            print(f"Warning: Could not plot MT={mt_num} for {label}: {str(e)}")
    
    # Set up the plot with improved styling
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Set labels and title with consistent styling
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Cross Section (barns)')
    
    # Only add title for non-publication styles
    if style not in {"paper", "publication"}:
        ax.set_title(title)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    
    # Improve layout
    fig.tight_layout()
    
    # Save figure if requested with consistent DPI
    if save_path is not None:
        try:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor=fig.get_facecolor(), edgecolor='none')
            print(f"Saved figure to {save_path}")
        except Exception as e:
            print(f"Warning: Could not save figure to {save_path}: {str(e)}")
    
    # Show figure if requested
    if show:
        plt.show()
    
    return fig

def get_cross_section_dataframe(
    ace_objects: List[Ace], 
    mt_number: int,
    labels: Optional[List[str]] = None,
    energy_range: Optional[Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Get cross section data from multiple ACE objects for a specific MT number as a DataFrame.
    
    Parameters
    ----------
    ace_objects : List[Ace]
        List of Ace objects
    mt_number : int
        MT number for the cross section data
    labels : List[str], optional
        Labels to use for column names (defaults to ZAID if available)
    energy_range : Tuple[float, float], optional
        Energy range to include (min, max) in MeV
        
    Returns
    -------
    pd.DataFrame
        DataFrame with energy as the first column and cross sections as additional columns
        
    Raises
    ------
    ValueError
        If ACE objects have incompatible energy grids
        
    Examples
    --------
    >>> # Get total cross section data for two ACE objects
    >>> df = get_cross_section_dataframe([u235_ace, u238_ace], 1)
    >>> 
    >>> # With custom labels and energy range
    >>> df = get_cross_section_dataframe(
    ...     [fe56_ace, fe54_ace], 
    ...     2,  # Elastic scattering
    ...     labels=['Fe-56', 'Fe-54'],
    ...     energy_range=(1e-5, 20.0)
    ... )
    """
    if not ace_objects:
        raise ValueError("No ACE objects provided")
    
    # Generate default labels if none provided
    if labels is None:
        labels = []
        for i, ace in enumerate(ace_objects):
            if ace.header and ace.header.zaid:
                labels.append(ace.header.zaid)
            elif ace.filename:
                labels.append(os.path.basename(ace.filename))
            else:
                labels.append(f"ACE Object {i+1}")
    elif len(labels) != len(ace_objects):
        raise ValueError(f"{len(labels)} labels provided for {len(ace_objects)} objects")
    
    # Ensure unique labels
    unique_labels = []
    label_counts = {}
    
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
            unique_labels.append(f"{label}_{label_counts[label]}")
        else:
            label_counts[label] = 0
            unique_labels.append(label)
    
    # Use the unique labels instead of the original ones
    labels = unique_labels
    
    # Get cross section data from each ACE object
    all_xs_data = []
    for i, ace in enumerate(ace_objects):
        try:
            xs_data = ace.get_cross_section(mt_number)
            
            # Apply energy range filter if specified
            if energy_range is not None:
                mask = (xs_data["Energy"] >= energy_range[0]) & (xs_data["Energy"] <= energy_range[1])
                energy = xs_data["Energy"][mask]
                xs = xs_data[f"MT={mt_number}"][mask]
                all_xs_data.append((energy, xs))
            else:
                energy = xs_data["Energy"]
                xs = xs_data[f"MT={mt_number}"]
                all_xs_data.append((energy, xs))
        except Exception as e:
            label = labels[i] if i < len(labels) else f"ACE Object {i+1}"
            raise ValueError(f"Could not get cross section data for {label}: {str(e)}")
    
    # Check if all energy grids are compatible
    reference_energy = all_xs_data[0][0]
    for i, (energy, _) in enumerate(all_xs_data[1:], 1):
        if len(energy) != len(reference_energy):
            raise ValueError(f"Incompatible energy grids: {labels[0]} has {len(reference_energy)} points, " 
                             f"{labels[i]} has {len(energy)} points")
        
        if not np.allclose(energy, reference_energy):
            raise ValueError(f"Incompatible energy grids: {labels[0]} and {labels[i]} have different energy values")
    
    # Create DataFrame
    df = pd.DataFrame({"Energy": all_xs_data[0][0]})
    for i, (_, xs) in enumerate(all_xs_data):
        df[labels[i]] = xs
    
    return df