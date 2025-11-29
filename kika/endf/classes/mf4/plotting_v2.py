"""
Refactored plotting functions using the new plotting infrastructure.

These functions provide backward compatibility while using the new
PlotData and PlotBuilder system internally.
"""

from typing import List, Optional, Union, Tuple
import matplotlib.pyplot as plt

from kika.plotting import PlotBuilder
from .plot_utils import (
    create_legendre_coeff_plot_data,
    create_legendre_uncertainty_plot_data,
    create_multiple_legendre_coeff_plot_data,
    create_multiple_legendre_uncertainty_plot_data
)


def plot_legendre_coefficients_from_endf_v2(
    endf: Union[object, List[object]],
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 5),
    legend_loc: str = 'best',
    labels: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot Legendre coefficients using the new plotting infrastructure.
    
    This is a refactored version of plot_legendre_coefficients_from_endf
    that uses PlotData objects and PlotBuilder internally.
    
    Parameters
    ----------
    endf : ENDF or list of ENDF
        ENDF data object(s) containing MF4 files
    mt : int
        MT reaction number to plot
    orders : int or list of int, optional
        Polynomial orders to plot. If None, plots all available orders
    energy_range : tuple of float, optional
        Energy range (min, max) for x-axis
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    legend_loc : str
        Legend location
    labels : str or list of str, optional
        Labels for each ENDF object in the legend
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Convert single ENDF to list
    if not isinstance(endf, list):
        endf_list = [endf]
    else:
        endf_list = endf
    
    # Handle labels
    if labels is None:
        endf_labels = [None] * len(endf_list)
    elif isinstance(labels, str):
        if len(endf_list) == 1:
            endf_labels = [labels]
        else:
            endf_labels = [f"{labels} {i+1}" for i in range(len(endf_list))]
    else:
        endf_labels = list(labels)
    
    # Validate ENDF objects
    for i, endf_obj in enumerate(endf_list):
        if 4 not in endf_obj.mf:
            raise ValueError(f"ENDF object {i+1} does not contain MF4 data")
        if mt not in endf_obj.mf[4].mt:
            available_mts = list(endf_obj.mf[4].mt.keys())
            raise ValueError(f"MT{mt} not found in MF4. Available: {available_mts}")
    
    # Determine orders to plot
    first_mf4 = endf_list[0].mf[4].mt[mt]
    if orders is None:
        coeffs_list = first_mf4.legendre_coefficients
        if coeffs_list:
            max_available = max(len(coeffs) for coeffs in coeffs_list)
            orders = list(range(max_available))
        else:
            raise ValueError("No Legendre coefficient data available")
    elif isinstance(orders, int):
        orders = [orders]
    
    # Create PlotBuilder
    builder = PlotBuilder(style=style, figsize=figsize, **kwargs)
    
    # Create plot data for each ENDF object
    for endf_obj, endf_label in zip(endf_list, endf_labels):
        mf4_data = endf_obj.mf[4].mt[mt]
        
        # Create plot data for all requested orders
        plot_data_list = create_multiple_legendre_coeff_plot_data(
            mf4_data,
            orders=orders,
            label_prefix=endf_label
        )
        
        # Add to builder
        builder.add_multiple(plot_data_list)
    
    # Configure plot
    from kika._constants import MT_TO_REACTION
    reaction_name = MT_TO_REACTION.get(mt, f"MT={mt}")
    
    builder.set_labels(
        title=f"{reaction_name} - Legendre Coefficients",
        x_label="Energy (eV)",
        y_label="Coefficient value"
    )
    builder.set_scales(log_x=True)
    builder.set_legend(loc=legend_loc)
    
    if energy_range is not None:
        builder.set_limits(x_lim=energy_range)
    
    return builder.build()


def plot_legendre_coefficient_uncertainties_from_endf_v2(
    endf: Union[object, List[object]],
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 5),
    legend_loc: str = 'best',
    uncertainty_type: str = 'relative',
    labels: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot Legendre coefficient uncertainties using the new plotting infrastructure.
    
    This is a refactored version that uses PlotData objects and PlotBuilder internally.
    
    Parameters
    ----------
    endf : ENDF or list of ENDF
        ENDF data object(s) containing MF4 and MF34 files
    mt : int
        MT reaction number to plot
    orders : int or list of int, optional
        Polynomial orders to plot uncertainties for
    energy_range : tuple of float, optional
        Energy range (min, max) for x-axis
    style : str
        Plot style
    figsize : tuple
        Figure size
    legend_loc : str
        Legend location
    uncertainty_type : str
        'relative' or 'absolute'
    labels : str or list of str, optional
        Labels for each ENDF object
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Convert single ENDF to list
    if not isinstance(endf, list):
        endf_list = [endf]
    else:
        endf_list = endf
    
    # Handle labels
    if labels is None:
        endf_labels = [None] * len(endf_list)
    elif isinstance(labels, str):
        if len(endf_list) == 1:
            endf_labels = [labels]
        else:
            endf_labels = [f"{labels} {i+1}" for i in range(len(endf_list))]
    else:
        endf_labels = list(labels)
    
    # Validate ENDF objects
    for i, endf_obj in enumerate(endf_list):
        if 4 not in endf_obj.mf:
            raise ValueError(f"ENDF object {i+1} does not contain MF4 data")
        if 34 not in endf_obj.mf:
            raise ValueError(f"ENDF object {i+1} does not contain MF34 data")
        if mt not in endf_obj.mf[4].mt:
            available_mts = list(endf_obj.mf[4].mt.keys())
            raise ValueError(f"MT{mt} not found in MF4. Available: {available_mts}")
        if mt not in endf_obj.mf[34].mt:
            available_mts = list(endf_obj.mf[34].mt.keys())
            raise ValueError(f"MT{mt} not found in MF34. Available: {available_mts}")
    
    # Determine orders to plot
    first_mf4 = endf_list[0].mf[4].mt[mt]
    if orders is None:
        coeffs_list = first_mf4.legendre_coefficients
        if coeffs_list:
            max_available = max(len(coeffs) for coeffs in coeffs_list)
            orders = list(range(max_available))
        else:
            raise ValueError("No Legendre coefficient data available")
    elif isinstance(orders, int):
        orders = [orders]
    
    # Create PlotBuilder
    builder = PlotBuilder(style=style, figsize=figsize, **kwargs)
    
    # Create plot data for each ENDF object
    for endf_obj, endf_label in zip(endf_list, endf_labels):
        mf4_data = endf_obj.mf[4].mt[mt]
        mf34_mt = endf_obj.mf[34].mt[mt]
        mf34_covmat = mf34_mt.to_ang_covmat()
        
        # Create uncertainty plot data for all requested orders
        try:
            plot_data_list = create_multiple_legendre_uncertainty_plot_data(
                mf4_data,
                mf34_covmat,
                orders=orders,
                uncertainty_type=uncertainty_type,
                label_prefix=endf_label
            )
            
            # Add to builder
            builder.add_multiple(plot_data_list)
        except Exception as e:
            print(f"Warning: Could not create uncertainty data for {endf_label}: {e}")
            continue
    
    # Configure plot
    from kika._constants import MT_TO_REACTION
    reaction_name = MT_TO_REACTION.get(mt, f"MT={mt}")
    
    if uncertainty_type == 'relative':
        y_label = "Relative uncertainty (%)"
    else:
        y_label = "Absolute uncertainty"
    
    builder.set_labels(
        title=f"{reaction_name} - Legendre Coefficient Uncertainties",
        x_label="Energy (eV)",
        y_label=y_label
    )
    builder.set_scales(log_x=True)
    builder.set_legend(loc=legend_loc)
    
    if energy_range is not None:
        builder.set_limits(x_lim=energy_range)
    
    return builder.build()


def quick_comparison_plot(
    endf_list: List[object],
    mt: int,
    order: int,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    style: str = 'publication',
    **kwargs
) -> plt.Figure:
    """
    Quick function to compare Legendre coefficients from multiple ENDF files.
    
    This demonstrates how the new system makes custom plots very easy.
    
    Parameters
    ----------
    endf_list : list of ENDF
        ENDF objects to compare
    mt : int
        MT reaction number
    order : int
        Legendre order to compare
    labels : list of str, optional
        Labels for each file
    colors : list of str, optional
        Colors for each file
    style : str
        Plot style
    **kwargs
        Additional kwargs for PlotBuilder
        
    Returns
    -------
    plt.Figure
        Comparison plot
    """
    builder = PlotBuilder(style=style, **kwargs)
    
    for i, endf_obj in enumerate(endf_list):
        mf4_data = endf_obj.mf[4].mt[mt]
        
        label = labels[i] if labels and i < len(labels) else None
        color = colors[i] if colors and i < len(colors) else None
        
        plot_data = create_legendre_coeff_plot_data(
            mf4_data, 
            order=order, 
            label=label
        )
        
        builder.add_data(plot_data, color=color, linewidth=2)
    
    from kika._constants import MT_TO_REACTION
    reaction_name = MT_TO_REACTION.get(mt, f"MT={mt}")
    
    builder.set_labels(
        title=f"{reaction_name} - L={order} Coefficient Comparison",
        x_label="Energy (eV)",
        y_label="Coefficient Value"
    )
    builder.set_scales(log_x=True)
    
    return builder.build()
