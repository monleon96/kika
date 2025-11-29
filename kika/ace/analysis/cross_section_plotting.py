"""
Utility for plotting cross sections from ACE objects.

This module provides functions for plotting cross sections from ACE objects that have
already been loaded/parsed. The functions accept ACE objects directly, promoting
better separation of concerns and reusability.

Functions
---------
plot_cross_sections : Plot cross sections from ACE objects
get_cross_section_dataframe : Get cross section data as DataFrame from ACE objects

Examples
--------
>>> # Load ACE objects first using kika.ace.read_ace, then plot
>>> from kika.ace import read_ace
>>> ace1 = read_ace(ace_file1)
>>> ace2 = read_ace(ace_file2)
>>> plot_cross_sections([ace1, ace2], 1)  # Total cross section
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional, Tuple, Union
from kika.ace import read_ace
from kika._plot_settings import setup_plot_style, format_axes
from kika._constants import MT_GROUPS
from kika._utils import zaid_to_symbol


def _get_cross_section_with_fallback(ace, mt_number, verbose=False):
    """
    Get cross section data for a given MT number, with automatic fallback to summing
    component reactions if the primary MT is not available but its components are.
    
    Parameters
    ----------
    ace : ACE object
        The ACE data object
    mt_number : int
        The primary MT number requested
    verbose : bool, optional
        If True, print information about component summation
        
    Returns
    -------
    dict or None
        Dictionary with 'Energy' and f'MT={mt_number}' keys, or None if unavailable
    """
    # First, try to get the MT directly
    try:
        if verbose:
            print(f"Found MT={mt_number} directly in ACE file")
        return ace.get_cross_section(mt_number)
    except:
        # If direct MT is not available, check if it's in MT_GROUPS
        pass
    
    # Look for the MT in MT_GROUPS
    component_range = None
    for primary_mt, mt_range in MT_GROUPS:
        if primary_mt == mt_number:
            component_range = mt_range
            break
    
    if component_range is None:
        # MT is not in MT_GROUPS, can't do anything
        if verbose:
            print(f"MT={mt_number} not found directly and not in MT_GROUPS")
        return None
    
    # Check which component MTs are available in the ACE file
    available_components = []
    for component_mt in component_range:
        try:
            ace.get_cross_section(component_mt)
            available_components.append(component_mt)
        except:
            continue
    
    if not available_components:
        # No component MTs are available
        if verbose:
            print(f"MT={mt_number} not found directly and no component MTs available from range {list(component_range)}")
        return None
    
    if verbose:
        print(f"MT={mt_number} not found directly, summing {len(available_components)} component MTs: {available_components}")
    
    # Sum the available component cross sections
    energy = None
    total_xs = None
    
    for component_mt in available_components:
        try:
            xs_data = ace.get_cross_section(component_mt)
            component_energy = xs_data["Energy"]
            component_xs = xs_data[f"MT={component_mt}"]
            
            if energy is None:
                # First component - use its energy grid as reference
                energy = component_energy.copy()
                total_xs = component_xs.copy()
            else:
                # Subsequent components - need to interpolate to common energy grid
                if not np.array_equal(energy, component_energy):
                    # Interpolate component cross section to reference energy grid
                    component_xs_interp = np.interp(energy, component_energy, component_xs)
                    total_xs += component_xs_interp
                else:
                    # Same energy grid, just add
                    total_xs += component_xs
        except Exception as e:
            # Skip this component if there's an error
            if verbose:
                print(f"Warning: Could not process component MT={component_mt}: {e}")
            continue
    
    if energy is None or total_xs is None:
        if verbose:
            print(f"Failed to compute sum for MT={mt_number}")
        return None
    
    # Return in the same format as ace.get_cross_section()
    return {
        "Energy": energy,
        f"MT={mt_number}": total_xs
    }

def plot_cross_sections(
    ace_objects: List,
    mt_number: Union[int, List[int]],
    labels: Optional[List[str]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 5),
    y_range: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    style: str = "default",
    show: bool = True,
    **plot_kwargs
) -> plt.Figure:
    """
    Plot cross sections from multiple ACE objects for a specific MT number or list of MT numbers.
    
    Parameters
    ----------
    ace_objects : List
        List of ACE objects (already loaded/parsed)
    mt_number : int or List[int]
        MT number to plot for all ACE objects, or list of MT numbers (one per ACE object)
    labels : List[str], optional
        Labels to use in the legend (defaults to element symbol like Fe56 if ZAID available)
    energy_range : Tuple[float, float], optional
        Energy range to plot (min, max) in MeV
    figsize : Tuple[float, float], optional
        Figure size (width, height) in inches
    y_range : Tuple[float, float], optional
        Optional y-axis limits to apply to the cross section axis (min, max). If provided,
        these will be used when formatting the axes. Note: values must be positive for log scale.
    save_path : str, optional
        Path to save figure (if None, figure is not saved)
    dpi : int, optional
        DPI for saved figure
    style : str, optional
        Plot style: 'default', 'dark', 'paper', 'publication', 'presentation'
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
    >>> ace1 = read_ace(ace_file1)
    >>> ace2 = read_ace(ace_file2)
    >>> plot_cross_sections([ace1, ace2], 1)
    
    >>> # Plot different MT numbers for each ACE object
    >>> plot_cross_sections([ace1, ace2], [1, 2])  # Total for first, Elastic for second
    
    >>> # With custom labels and energy range
    >>> plot_cross_sections(
    ...     [ace_fe56, ace_fe54], 
    ...     2,  # Elastic scattering for both
    ...     labels=['Fe-56', 'Fe-54'],
    ...     energy_range=(1e-5, 20.0),
    ...     style='publication'
    ... )
    """
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
            if hasattr(ace, 'header') and ace.header and hasattr(ace.header, 'zaid') and ace.header.zaid:
                # Convert ZAID to symbol format (e.g., 26056 -> Fe56)
                symbol = zaid_to_symbol(ace.header.zaid)
                labels.append(symbol)
            elif hasattr(ace, 'filename') and ace.filename:
                labels.append(os.path.basename(ace.filename))
            else:
                labels.append(f"ACE Object {i+1}")
    elif len(labels) != len(ace_objects):
        print(f"Warning: {len(labels)} labels provided for {len(ace_objects)} objects, using default labels instead")
        labels = [f"ACE Object {i+1}" for i in range(len(ace_objects))]
    
    # Helper function to get reaction description
    def get_reaction_description(mt_num):
        try:
            from kika._constants import MT_TO_REACTION
            if mt_num in MT_TO_REACTION:
                return MT_TO_REACTION[mt_num]
            else:
                return f"MT={mt_num}"
        except ImportError:
            return f"MT={mt_num}"
    
    # Setup plot style using common plotting settings
    plot_setup = setup_plot_style(style=style, figsize=figsize, dpi=dpi, **plot_kwargs)
    fig = plot_setup['_fig']
    ax = plot_setup['ax']
    colors = plot_setup['_colors']
    
    # Determine title based on MT numbers
    if len(set(mt_numbers)) == 1:
        # All same MT number
        mt_num = mt_numbers[0]
        mt_desc = get_reaction_description(mt_num)
        title = f"{mt_desc} (MT={mt_num}) Cross Section"
    else:
        # Multiple different MT numbers
        title = f"Cross Sections (MT={mt_numbers})"
    
    # Plot cross section from each ACE object
    for i, (ace, mt_num) in enumerate(zip(ace_objects, mt_numbers)):
        try:
            # Get cross section data with fallback to component summation
            xs_data = _get_cross_section_with_fallback(ace, mt_num)
            
            if xs_data is None:
                label = labels[i] if i < len(labels) else f"ACE Object {i+1}"
                print(f"Warning: MT={mt_num} not available for {label} (neither directly nor through component reactions)")
                continue
            
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
    
    # Format axes using common plotting settings (always log-log for cross sections)
    ax = format_axes(
        ax, 
        style=style, 
        use_log_scale=True,  # Always use log scale for energy axis
        is_energy_axis=True,
        x_label="Energy (MeV)",
        y_label="Cross Section (barns)",
        title=title if style not in {"paper", "publication"} else None,
        legend_loc='best',
        use_y_log_scale=True  # Always use log scale for cross section axis
    )
    
    # Set y-axis to log scale explicitly
    ax.set_yscale('log')
    
    # Apply energy range limits if specified
    if energy_range is not None:
        ax.set_xlim(energy_range)
    
    # Apply y-axis range limits if specified
    if y_range is not None:
        ax.set_ylim(y_range)
    
    # Save figure if requested
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
    ace_objects: List,
    mt_number: int,
    labels: Optional[List[str]] = None,
    energy_range: Optional[Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Get cross section data from multiple ACE objects for a specific MT number as a DataFrame.
    
    Parameters
    ----------
    ace_objects : List
        List of ACE objects (already loaded/parsed)
    mt_number : int
        MT number for the cross section data
    labels : List[str], optional
        Labels to use for column names (defaults to element symbol like Fe56 if ZAID available)
    energy_range : Tuple[float, float], optional
        Energy range to include (min, max) in MeV
        
    Returns
    -------
    pd.DataFrame
        DataFrame with energy as the first column and cross sections as additional columns
        
    Raises
    ------
    ValueError
        If ACE objects are invalid or have incompatible energy grids
        
    Examples
    --------
    >>> # Get total cross section data for two ACE objects
    >>> ace1 = read_ace(ace_file1)
    >>> ace2 = read_ace(ace_file2)
    >>> df = get_cross_section_dataframe([ace1, ace2], 1)
    >>> 
    >>> # With custom labels and energy range
    >>> df = get_cross_section_dataframe(
    ...     [ace_fe56, ace_fe54], 
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
            if hasattr(ace, 'header') and ace.header and hasattr(ace.header, 'zaid') and ace.header.zaid:
                # Convert ZAID to symbol format (e.g., 26056 -> Fe56)
                symbol = zaid_to_symbol(ace.header.zaid)
                labels.append(symbol)
            elif hasattr(ace, 'filename') and ace.filename:
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
            xs_data = _get_cross_section_with_fallback(ace, mt_number)
            
            if xs_data is None:
                label = labels[i] if i < len(labels) else f"ACE Object {i+1}"
                raise ValueError(f"MT={mt_number} not available for {label} (neither directly nor through component reactions)")
            
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