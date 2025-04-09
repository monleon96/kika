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
    mt_number: int,
    labels: Optional[List[str]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 6),
    log_scale: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    **plot_kwargs
) -> plt.Figure:
    """
    Plot cross sections from multiple ACE objects for a specific MT number.
    
    Parameters
    ----------
    ace_objects : List[Ace]
        List of Ace objects
    mt_number : int
        MT number to plot
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
    **plot_kwargs
        Additional keyword arguments passed to plot function
        
    Returns
    -------
    plt.Figure
        Figure object
        
    Examples
    --------
    >>> # Compare total cross section between two ACE objects
    >>> plot_cross_sections([u235_ace, u238_ace], 1)
    
    >>> # With custom labels and energy range
    >>> plot_cross_sections(
    ...     [fe56_ace, fe54_ace], 
    ...     2,  # Elastic scattering
    ...     labels=['Fe-56', 'Fe-54'],
    ...     energy_range=(1e-5, 20.0)
    ... )
    """
    if not ace_objects:
        print("No ACE objects provided")
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
    
    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get MT description if available
    mt_desc = ""
    try:
        from mcnpy._constants import MT_TO_REACTION
        if mt_number == 1:
            mt_desc = "Total"
        elif mt_number == 2:
            mt_desc = "Elastic Scattering"
        elif mt_number == 101:
            mt_desc = "Neutron Disappearance"
        elif mt_number in MT_TO_REACTION:
            mt_desc = MT_TO_REACTION[mt_number]
        else:
            mt_desc = f"MT={mt_number}"
    except ImportError:
        mt_desc = f"MT={mt_number}"
    
    # Colors for plotting
    colors = plt.cm.tab10.colors  # Use a color cycle
    
    # Plot cross section from each ACE object
    for i, ace in enumerate(ace_objects):
        try:
            # Get cross section data
            xs_data = ace.get_cross_section(mt_number)
            
            # Apply energy range filter if specified
            if energy_range is not None:
                mask = (xs_data["Energy"] >= energy_range[0]) & (xs_data["Energy"] <= energy_range[1])
                energy = xs_data["Energy"][mask]
                xs = xs_data[f"MT={mt_number}"][mask]
            else:
                energy = xs_data["Energy"]
                xs = xs_data[f"MT={mt_number}"]
            
            # Plot with color from cycle
            color = colors[i % len(colors)]
            ax.plot(energy, xs, label=labels[i], color=color, **plot_kwargs)
            
        except Exception as e:
            label = labels[i] if i < len(labels) else f"ACE Object {i+1}"
            print(f"Warning: Could not plot {mt_desc} for {label}: {str(e)}")
    
    # Set up the plot
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Set labels and title
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Cross Section (barns)')
    ax.set_title(f"{mt_desc} (MT={mt_number}) Cross Section")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    
    # Make the figure look nicer
    fig.tight_layout()
    
    # Save figure if requested
    if save_path is not None:
        try:
            fig.savefig(save_path, dpi=dpi)
            print(f"Saved figure to {save_path}")
        except Exception as e:
            print(f"Warning: Could not save figure to {save_path}: {str(e)}")
    
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
