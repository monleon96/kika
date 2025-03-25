"""
Utility for plotting cross sections from ACE files.

This module provides a simple function for plotting cross sections from ACE files.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

def plot_cross_sections(
    ace_files: List[str], 
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
    Plot cross sections from multiple ACE files for a specific MT number.
    
    Parameters
    ----------
    ace_files : List[str]
        List of paths to ACE files
    mt_number : int
        MT number to plot
    labels : List[str], optional
        Labels to use in the legend (defaults to filenames)
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
    >>> # Compare total cross section between two ACE files
    >>> plot_cross_sections(['u235.ace', 'u238.ace'], 1)
    
    >>> # With custom labels and energy range
    >>> plot_cross_sections(
    ...     ['fe56.ace', 'fe54.ace'], 
    ...     2,  # Elastic scattering
    ...     labels=['Fe-56', 'Fe-54'],
    ...     energy_range=(1e-5, 20.0)
    ... )
    """
    from mcnpy.ace.parse_ace import read_ace
    
    # Read ACE files
    ace_data = []
    for file_path in ace_files:
        try:
            # Get base filename without directory and extension
            filename = os.path.basename(file_path)
            ace = read_ace(file_path)
            ace_data.append((filename, ace))
        except Exception as e:
            print(f"Warning: Could not read ACE file {file_path}: {str(e)}")
    
    if not ace_data:
        print("No ACE files could be read successfully")
        return None
        
    # Use filenames as default labels if none provided
    if labels is None:
        labels = [filename for filename, _ in ace_data]
    elif len(labels) != len(ace_data):
        print(f"Warning: {len(labels)} labels provided for {len(ace_data)} files, using filenames instead")
        labels = [filename for filename, _ in ace_data]
    
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
    
    # Plot cross section from each ACE file
    for i, (filename, ace) in enumerate(ace_data):
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
            print(f"Warning: Could not plot {mt_desc} for {filename}: {str(e)}")
    
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
