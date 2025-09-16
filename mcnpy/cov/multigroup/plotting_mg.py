"""
Plotting functions for multigroup covariance data and Legendre coefficients.

This module provides plotting capabilities for MGMF34CovMat objects,
including visualization of multigroup Legendre coefficients and comparison
with continuous ENDF data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from typing import List, Optional, Union, Tuple, Dict, Any, TYPE_CHECKING
from ..._plot_settings import setup_plot_style, format_axes

if TYPE_CHECKING:
    from .mg_mf34_covmat import MGMF34CovMat


def _generate_unique_labels(mg_list, labels):
    """
    Generate unique labels for multigroup objects, handling duplicates properly.
    
    Parameters
    ----------
    mg_list : list
        List of MGMF34CovMat objects
    labels : None, str, or list of str
        Label specification
        
    Returns
    -------
    list of str
        Unique labels for each object
    """
    if labels is None:
        # Generate default labels based on isotope info if available
        mg_labels = [f"MG Data {i+1}" for i in range(len(mg_list))]
    elif isinstance(labels, str):
        # Single string - create unique labels with indices
        if len(mg_list) == 1:
            mg_labels = [labels]
        else:
            mg_labels = [f"{labels} {i+1}" for i in range(len(mg_list))]
    else:
        # List of labels - pad with defaults if needed
        mg_labels = list(labels)
        while len(mg_labels) < len(mg_list):
            mg_labels.append(f"MG Data {len(mg_labels)+1}")
    
    return mg_labels


def _get_energy_group_centers(energy_grid: np.ndarray) -> np.ndarray:
    """
    Calculate energy group centers from group boundaries.
    
    Parameters
    ----------
    energy_grid : np.ndarray
        Energy group boundaries (length n+1 for n groups)
        
    Returns
    -------
    np.ndarray
        Energy group centers (length n)
    """
    return np.sqrt(energy_grid[:-1] * energy_grid[1:])  # Geometric mean


def plot_mg_legendre_coefficients(
    mg_covmat: Union["MGMF34CovMat", List["MGMF34CovMat"]],
    isotope: int,
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (10, 6),
    legend_loc: str = 'best',
    marker: bool = True,
    include_uncertainties: bool = False,
    uncertainty_sigma: float = 1.0,
    labels: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot multigroup Legendre coefficients from MGMF34CovMat object(s).
    
    This function plots the group-averaged Legendre coefficients as a function
    of energy group centers, optionally including uncertainty bands.
    
    Parameters
    ----------
    mg_covmat : MGMF34CovMat or list of MGMF34CovMat
        Multigroup covariance matrix object(s) containing Legendre coefficients
    isotope : int
        Isotope ID to plot
    mt : int
        Reaction MT number to plot
    orders : int or list of int, optional
        Legendre orders to plot. If None, plots all available orders
    energy_range : tuple(float, float), optional
        If provided, (emin, emax) limits for x-axis. Values are clamped to data range.
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    legend_loc : str
        Legend location
    marker : bool
        Whether to include markers on the plot lines
    include_uncertainties : bool
        Whether to include uncertainty bands if available
    uncertainty_sigma : float
        Number of sigma levels for uncertainty bands
    labels : str or list of str, optional
        Labels for each dataset
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The matplotlib figure containing the plot
    """
    # Convert single object to list for uniform processing
    if not isinstance(mg_covmat, list):
        mg_list = [mg_covmat]
    else:
        mg_list = mg_covmat
    
    # Generate labels
    mg_labels = _generate_unique_labels(mg_list, labels)
    
    # Validate inputs
    for i, mg_obj in enumerate(mg_list):
        if mg_obj.num_groups == 0:
            raise ValueError(f"MGMF34CovMat object {i+1} has no energy groups")
    
    # Setup plot style
    plot_kwargs = setup_plot_style(style=style, figsize=figsize, **kwargs)
    fig = plot_kwargs['_fig']
    ax = plot_kwargs['ax']
    colors = plot_kwargs['_colors']
    
    # Determine which orders to plot
    first_mg = mg_list[0]
    available_orders = []
    for key in first_mg.legendre_coefficients.keys():
        iso_key, mt_key, l_key = key
        if iso_key == isotope and mt_key == mt:
            available_orders.append(l_key)
    
    if not available_orders:
        ax.text(0.5, 0.5, f'No Legendre coefficient data available\nfor isotope {isotope}, MT {mt}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"No Data Found for Isotope {isotope}, MT {mt}")
        return fig
    
    available_orders = sorted(set(available_orders))
    
    if orders is None:
        orders = available_orders
    elif isinstance(orders, int):
        orders = [orders]
    
    # Filter orders to only those available
    orders = [order for order in orders if order in available_orders]
    
    if not orders:
        ax.text(0.5, 0.5, f'Requested orders not available\nAvailable orders: {available_orders}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Requested Orders Not Available")
        return fig
    
    from mcnpy._utils import zaid_to_symbol

    # Track all energy boundaries for applying global limits
    all_energy_bounds: List[float] = []

    # Plot data for each multigroup object
    for mg_idx, (mg_obj, mg_label) in enumerate(zip(mg_list, mg_labels)):
        # Energy group boundaries (length n+1) and coefficients (length n)
        energy_grid = np.asarray(mg_obj.energy_grid, dtype=float)
        if energy_grid.ndim != 1 or energy_grid.size < 2:
            continue
        all_energy_bounds.extend(energy_grid.tolist())

        # Plot each requested order
        for order_idx, order in enumerate(orders):
            key = (isotope, mt, order)
            if key not in mg_obj.legendre_coefficients:
                continue

            coeffs = np.asarray(mg_obj.legendre_coefficients[key], dtype=float)
            if coeffs.size != energy_grid.size - 1:
                # Size mismatch; skip to avoid plotting errors
                continue

            # Color by order (primary) then dataset variation if many datasets
            base_color_idx = order_idx % len(colors)
            # Slight variation for different datasets of same order (cycle linestyle only)
            color = colors[base_color_idx]

            linestyle = '-' if len(mg_list) == 1 else ['-', '--', '-.', ':'][mg_idx % 4]

            # Build isotope symbol using ZAID conversion (isotope is ZAID)
            try:
                isotope_symbol = zaid_to_symbol(isotope)
            except Exception:
                isotope_symbol = f"Isotope {isotope}"

            # Base label (without uncertainty)
            base_label = f"{isotope_symbol} - $a_{{{order}}}$"
            # If uncertainties requested: include sigma text on the LINE label (not on fill)
            if include_uncertainties and getattr(mg_obj, 'relative_matrices', None):
                if uncertainty_sigma == 1.0:
                    base_label += " ±1σ"
                else:
                    base_label += f" ±{uncertainty_sigma}σ"
            line_label = base_label if len(mg_list) == 1 else f"{base_label} ({mg_label})"

            # Prepare step arrays: for where='post', need y length n+1
            coeff_steps = np.append(coeffs, coeffs[-1])

            # Plot step line
            line_kwargs = {
                'color': color,
                'linestyle': linestyle,
                'label': line_label,
                'where': 'post',
                'zorder': 3
            }
            if marker:
                # Plot markers at bin centers separately (optional)
                centers = np.sqrt(energy_grid[:-1] * energy_grid[1:])
                ax.plot(centers, coeffs, 'o', color=color, markersize=4, zorder=4)
            ax.step(energy_grid, coeff_steps, **line_kwargs)

            # Uncertainty bands (no label; legend handled by line label above)
            if include_uncertainties and getattr(mg_obj, 'relative_matrices', None):
                _plot_mg_uncertainty_bands(
                    ax=ax,
                    mg_obj=mg_obj,
                    isotope=isotope,
                    mt=mt,
                    order=order,
                    energy_grid=energy_grid,
                    coeffs=coeffs,
                    uncertainty_sigma=uncertainty_sigma,
                    color=color,
                    label=None
                )

    # Apply energy range limits if specified
    if energy_range is not None and all_energy_bounds:
        data_min = min(all_energy_bounds)
        data_max = max(all_energy_bounds)
        e_min = max(energy_range[0], data_min)
        e_max = min(energy_range[1], data_max)
        if e_min < e_max:
            ax.set_xlim(e_min, e_max)
    
    # Create an improved title with isotope and reaction information
    title_parts = []
    
    # Add reaction information (import constants if available)
    try:
        from mcnpy._constants import MT_TO_REACTION
        if mt in MT_TO_REACTION:
            reaction_name = MT_TO_REACTION[mt]
            title_parts.append(f"MT={mt} {reaction_name}")
        else:
            title_parts.append(f"MT={mt}")
    except ImportError:
        title_parts.append(f"MT={mt}")
    
    # Add isotope information (convert ZAID to symbol)
    try:
        isotope_symbol = zaid_to_symbol(isotope)
    except Exception:
        isotope_symbol = f"Isotope {isotope}"
    title_parts.append(isotope_symbol)
    
    # Add "Multigroup Legendre Coefficients" at the end
    title_parts.append("Multigroup Legendre Coefficients")
    
    title = " - ".join(title_parts)
    
    # Format axes using the same style as continuous plotting
    ax = format_axes(
        ax,
        style=style,
        use_log_scale=True,
        is_energy_axis=True,
        x_label="Energy (eV)",
        y_label="Coefficient value",
        title=title,
        legend_loc=legend_loc
    )

    return fig


def _plot_mg_uncertainty_bands(
    ax,
    mg_obj,
    isotope: int,
    mt: int,
    order: int,
    energy_grid: np.ndarray,
    coeffs: np.ndarray,
    uncertainty_sigma: float,
    color,
    label: Optional[str] = None,
):
    """Plot step-style uncertainty bands for a given order.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes
    mg_obj : MGMF34CovMat
        Multigroup covariance object
    isotope, mt, order : int
        Selection identifiers
    energy_grid : np.ndarray
        Bin boundaries (length n+1)
    coeffs : np.ndarray
        Coefficient values (length n)
    uncertainty_sigma : float
        Sigma multiplier
    color : matplotlib color
        Band color (matches line color)
    label : str, optional
        Legend label for the band
    """
    try:
        # Locate matching diagonal covariance block
        for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c) in enumerate(
            zip(
                mg_obj.isotope_rows,
                mg_obj.reaction_rows,
                mg_obj.l_rows,
                mg_obj.isotope_cols,
                mg_obj.reaction_cols,
                mg_obj.l_cols,
            )
        ):
            if (
                iso_r == isotope
                and mt_r == mt
                and l_r == order
                and iso_c == isotope
                and mt_c == mt
                and l_c == order
            ):
                cov_matrix = mg_obj.relative_matrices[i]
                diag = np.diag(cov_matrix)
                if diag.size != coeffs.size:
                    break
                rel_unc = np.sqrt(diag) * uncertainty_sigma
                abs_unc = rel_unc * np.abs(coeffs)
                upper = coeffs + abs_unc
                lower = coeffs - abs_unc
                # Extend for step representation
                upper_steps = np.append(upper, upper[-1])
                lower_steps = np.append(lower, lower[-1])
                ax.fill_between(
                    energy_grid,
                    lower_steps,
                    upper_steps,
                    step='post',
                    alpha=0.18,
                    color=color,
                    edgecolor='none',
                    label=label,
                    zorder=2,
                )
                break
    except Exception:
        # Silent fail – uncertainty bands are optional
        pass


def plot_mg_vs_endf_comparison(
    mg_covmat: "MGMF34CovMat",
    endf: object,
    isotope: int,
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (12, 8),
    legend_loc: str = 'best',
    mg_marker: bool = True,
    endf_native: bool = False,
    include_uncertainties: bool = False,
    uncertainty_sigma: float = 1.0,
    **kwargs
) -> plt.Figure:
    """
    Compare multigroup Legendre coefficients with ENDF data.
    
    This function plots both the multigroup averaged coefficients and the
    coefficients from the original ENDF data for comparison.
    
    Parameters
    ----------
    mg_covmat : MGMF34CovMat
        Multigroup covariance matrix object
    endf : ENDF object
        Original ENDF data object containing MF4 data
    isotope : int
        Isotope ID to plot
    mt : int
        Reaction MT number to plot
    orders : int or list of int, optional
        Legendre orders to plot. If None, plots all available orders
    energy_range : tuple of float, optional
        Energy range for plotting ENDF data
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    legend_loc : str
        Legend location
    mg_marker : bool
        Whether to include markers for multigroup data
    endf_native : bool
        If True, plot the ENDF data (and its uncertainty band) using the native
        Legendre coefficient energy mesh from the MF4 file instead of a dense interpolated grid.
        This mirrors the appearance of plot_legendre_coefficients_from_endf and ensures that
        the uncertainty band is centered exactly on the plotted line. If False (default), a
        dense log-spaced energy grid is used for the line for smooth appearance; when
        include_uncertainties=True the band will still be computed on native energies and may
        appear slightly coarser or visually offset if interpolation introduces differences.
    include_uncertainties : bool
        If True and covariance data (MF34 / multigroup relative matrices) are available, show ±σ bands.
    uncertainty_sigma : float
        Number of sigma for uncertainty bands (default 1.0)
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The matplotlib figure containing the plot
    """
    # Validate inputs
    if 4 not in endf.mf:
        raise ValueError("ENDF object does not contain MF4 data")
    if mt not in endf.mf[4].mt:
        raise ValueError(f"MT {mt} not found in ENDF MF4 data")
    
    # Setup plot style
    plot_kwargs = setup_plot_style(style=style, figsize=figsize, **kwargs)
    fig = plot_kwargs['_fig']
    ax = plot_kwargs['ax']
    colors = plot_kwargs['_colors']
    
    # Multigroup energy grid (boundaries) and centers
    mg_energy_grid = np.asarray(mg_covmat.energy_grid, dtype=float)
    mg_energy_centers = _get_energy_group_centers(mg_energy_grid)
    
    # Determine which orders to plot
    available_mg_orders = []
    for key in mg_covmat.legendre_coefficients.keys():
        iso_key, mt_key, l_key = key
        if iso_key == isotope and mt_key == mt:
            available_mg_orders.append(l_key)
    
    available_mg_orders = sorted(set(available_mg_orders))
    
    if orders is None:
        orders = available_mg_orders
    elif isinstance(orders, int):
        orders = [orders]
    
    # Filter orders to only those available
    orders = [order for order in orders if order in available_mg_orders]
    
    if not orders:
        ax.text(0.5, 0.5, f'No matching orders available\nMG orders: {available_mg_orders}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Get continuous ENDF data
    mf4_data = endf.mf[4].mt[mt]
    
    # Check if ENDF data has extract_legendre_coefficients method
    if not hasattr(mf4_data, 'extract_legendre_coefficients'):
        ax.text(0.5, 0.5, 'ENDF data does not support coefficient extraction\n'
                          'Object type may not have Legendre coefficients', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Create energy grid for ENDF plotting
    if energy_range is None:
        energy_range = (mg_covmat.energy_grid[0], mg_covmat.energy_grid[-1])
    
    # Prepare dense energy grid (may be overridden by native usage)
    log_e_min = np.log10(energy_range[0])
    log_e_max = np.log10(energy_range[1])
    dense_endf_energies = np.logspace(log_e_min, log_e_max, 1000)
    
    from mcnpy._utils import zaid_to_symbol
    try:
        isotope_symbol = zaid_to_symbol(isotope)
    except Exception:
        isotope_symbol = f"Isotope {isotope}"

    # Attempt to access MF34 covariance for ENDF uncertainties once if needed
    mf34_covmat = None
    if include_uncertainties and 34 in endf.mf and mt in endf.mf[34].mt:
        try:
            mf34_covmat = endf.mf[34].mt[mt].to_ang_covmat()
        except Exception:
            mf34_covmat = None  # Silent fail
    # Import existing ENDF uncertainty plotting helper (works on native ENDF energies)
    if include_uncertainties and mf34_covmat is not None:
        try:
            from mcnpy.endf.classes.mf4.plotting import _plot_uncertainty_bands as _plot_endf_uncertainty_bands
        except Exception:
            _plot_endf_uncertainty_bands = None

    # Plot each order (multigroup as step, ENDF as solid)
    for order_idx, order in enumerate(orders):
        color = colors[order_idx % len(colors)]

        mg_key = (isotope, mt, order)
        if mg_key in mg_covmat.legendre_coefficients:
            mg_coeffs = np.asarray(mg_covmat.legendre_coefficients[mg_key], dtype=float)
            if mg_coeffs.size == mg_energy_grid.size - 1:
                mg_steps = np.append(mg_coeffs, mg_coeffs[-1])
                # Create label with uncertainty notation if uncertainties are enabled
                line_label = f"{isotope_symbol} - $a_{{{order}}}$"
                if include_uncertainties:
                    line_label += f" ±{uncertainty_sigma}σ"
                ax.step(
                    mg_energy_grid,
                    mg_steps,
                    where='post',
                    color=color,
                    linewidth=2,
                    label=line_label,
                    zorder=3
                )
                if mg_marker:
                    centers = mg_energy_centers
                    ax.plot(centers, mg_coeffs, 'o', color=color, markersize=4, zorder=4)
                # MG uncertainties from relative matrices (diagonal) - using dashed outline
                if include_uncertainties:
                    try:
                        # Find matching relative matrix for diagonal uncertainties
                        for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c) in enumerate(zip(
                                mg_covmat.isotope_rows,
                                mg_covmat.reaction_rows,
                                mg_covmat.l_rows,
                                mg_covmat.isotope_cols,
                                mg_covmat.reaction_cols,
                                mg_covmat.l_cols)):
                            if (iso_r == isotope and mt_r == mt and l_r == order and
                                    iso_c == isotope and mt_c == mt and l_c == order):
                                matrix = mg_covmat.relative_matrices[i]
                                diag = np.diag(matrix)
                                if diag.size == mg_coeffs.size:
                                    rel_unc = np.sqrt(diag) * uncertainty_sigma
                                    abs_unc = rel_unc * np.abs(mg_coeffs)
                                    upper = mg_coeffs + abs_unc
                                    lower = mg_coeffs - abs_unc
                                    upper_steps = np.append(upper, upper[-1])
                                    lower_steps = np.append(lower, lower[-1])
                                    # Plot upper and lower bounds as thin dashed lines
                                    ax.step(
                                        mg_energy_grid,
                                        upper_steps,
                                        where='post',
                                        linestyle='--',
                                        linewidth=0.8,
                                        color=color,
                                        alpha=0.7,
                                        zorder=2,
                                    )
                                    ax.step(
                                        mg_energy_grid,
                                        lower_steps,
                                        where='post',
                                        linestyle='--',
                                        linewidth=0.8,
                                        color=color,
                                        alpha=0.7,
                                        zorder=2,
                                    )
                                break
                    except Exception:
                        pass

        # Always plot ENDF line for ENDF data
        # Decide whether to use native energies (recommended when uncertainties shown)
        use_native_for_line = endf_native or (include_uncertainties and mf34_covmat is not None)
        try:
            if use_native_for_line:
                # Build coefficient arrays directly from native MF4 data
                # Note: legendre_coefficients typically store a1..aNL per energy (a0 implicit)
                native_energies = getattr(mf4_data, 'legendre_energies', [])
                native_coeff_lists = getattr(mf4_data, 'legendre_coefficients', [])
                energy_values_native: List[float] = []
                coeff_values_native: List[float] = []
                for j, coeff_list in enumerate(native_coeff_lists):
                    if order == 0:
                        coeff_values_native.append(1.0)
                        energy_values_native.append(native_energies[j])
                    elif (order - 1) < len(coeff_list):
                        coeff_values_native.append(coeff_list[order - 1])
                        energy_values_native.append(native_energies[j])
                if energy_values_native:
                    # Don't add label for ENDF line (same color as MG, would be redundant)
                    ax.plot(
                        energy_values_native,
                        coeff_values_native,
                        linestyle='-',  # Always solid line
                        color=color,
                        linewidth=1.4,
                        alpha=0.85,
                        zorder=2
                    )
                    # Uncertainty band exactly aligned with line
                    if include_uncertainties and mf34_covmat is not None and '_plot_endf_uncertainty_bands' in locals() and _plot_endf_uncertainty_bands is not None:
                        try:
                            isotope_id = int(getattr(mf4_data, 'zaid', isotope))
                            _plot_endf_uncertainty_bands(
                                ax,
                                energy_values_native,
                                coeff_values_native,
                                mf34_covmat,
                                isotope_id,
                                mt,
                                order,
                                uncertainty_sigma,
                                color,
                                alpha=0.15
                            )
                        except Exception:
                            pass
            else:
                # Dense interpolation path (no or deferred uncertainties)
                endf_coeffs_dict = mf4_data.extract_legendre_coefficients(
                    dense_endf_energies,
                    max_legendre_order=max(orders),
                    out_of_range="hold"
                )
                if order in endf_coeffs_dict:
                    endf_coeffs = endf_coeffs_dict[order]
                    # Don't add label for ENDF line (same color as MG, would be redundant)
                    ax.plot(
                        dense_endf_energies,
                        endf_coeffs,
                        linestyle='-',  # Always solid line
                        color=color,
                        linewidth=1.4,
                        alpha=0.75,
                        zorder=2
                    )
                    # If uncertainties requested, still draw band using native energies (may look coarser)
                    if include_uncertainties and mf34_covmat is not None and '_plot_endf_uncertainty_bands' in locals() and _plot_endf_uncertainty_bands is not None:
                        try:
                            native_energies = getattr(mf4_data, 'legendre_energies', [])
                            native_coeff_lists = getattr(mf4_data, 'legendre_coefficients', [])
                            coeff_values_native = []
                            energy_values_native = []
                            for j, coeff_list in enumerate(native_coeff_lists):
                                if order == 0:
                                    coeff_values_native.append(1.0)
                                    energy_values_native.append(native_energies[j])
                                elif (order - 1) < len(coeff_list):
                                    coeff_values_native.append(coeff_list[order - 1])
                                    energy_values_native.append(native_energies[j])
                            if energy_values_native:
                                isotope_id = int(getattr(mf4_data, 'zaid', isotope))
                                _plot_endf_uncertainty_bands(
                                    ax,
                                    energy_values_native,
                                    coeff_values_native,
                                    mf34_covmat,
                                    isotope_id,
                                    mt,
                                    order,
                                    uncertainty_sigma,
                                    color,
                                    alpha=0.15
                                )
                        except Exception:
                            pass
        except Exception as e:
            print(f"Warning: Could not extract ENDF coefficients for order {order}: {e}")
    
    # Format the plot
    # Improved title with reaction name if available
    title_parts = []
    try:
        from mcnpy._constants import MT_TO_REACTION
        if mt in MT_TO_REACTION:
            title_parts.append(f"MT={mt} {MT_TO_REACTION[mt]}")
        else:
            title_parts.append(f"MT={mt}")
    except ImportError:
        title_parts.append(f"MT={mt}")
    title_parts.append(isotope_symbol)
    title_parts.append("MG vs ENDF Legendre Coefficients")
    title = " - ".join(title_parts)

    ax = format_axes(
        ax,
        style=style,
        use_log_scale=True,
        is_energy_axis=True,
        x_label="Energy (eV)",
        y_label="Coefficient value",
        title=title,
        legend_loc=legend_loc
    )
    ax.set_xlim(energy_range)
    return fig


def plot_mg_covariance_heatmap(
    mg_covmat: "MGMF34CovMat",
    isotope: int,
    mt: int,
    orders: Optional[Union[int, List[int]]] = None,
    matrix_type: str = 'cov',
    covariance_type: str = 'rel',
    style: str = 'default',
    figsize: Tuple[float, float] = (6, 6),
    dpi: int = 300,
    font_family: str = "serif",
    colormap: Optional[str] = None,
    show_colorbar: bool = True,
    annotate: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric_scale: bool = True,
    use_log_scale: Optional[bool] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot a heatmap of the covariance or correlation matrix for specific isotope, MT, and Legendre orders.
    
    This function creates a heatmap visualization of either the covariance or correlation matrix
    for the specified isotope and MT reaction, optionally filtered by Legendre orders.
    
    Parameters
    ----------
    mg_covmat : MGMF34CovMat
        Multigroup covariance matrix object
    isotope : int
        Isotope ID to plot
    mt : int
        Reaction MT number to plot
    orders : int or list of int, optional
        Legendre orders to include. If None, includes all available orders
    matrix_type : str, default 'cov'
        Type of matrix to plot: 'cov' for covariance, 'corr' for correlation
    covariance_type : str, default 'rel'
        Type of covariance matrix: 'rel' for relative, 'abs' for absolute
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    colormap : str, optional
        Matplotlib colormap name. If None, uses 'RdYlGn' for correlation and 'viridis' for covariance
    show_colorbar : bool
        Whether to show the colorbar
    annotate : bool
        Whether to annotate matrix values on the heatmap
    vmin : float, optional
        Minimum value for color scaling
    vmax : float, optional
        Maximum value for color scaling  
    symmetric_scale : bool
        Whether to use symmetric color scale around zero (for correlation matrices)
    use_log_scale : bool, optional
        Whether to use logarithmic color scale. If None, uses linear for correlation and log for covariance
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The matplotlib figure containing the heatmap
    """
    # Validate inputs
    if matrix_type not in ['cov', 'corr']:
        raise ValueError("matrix_type must be 'cov' or 'corr'")
    if covariance_type not in ['rel', 'abs']:
        raise ValueError("covariance_type must be 'rel' or 'abs'")
    
    # Reset matplotlib to default state
    plt.rcdefaults()
    
    # Setup publication-quality settings (same as MF34 heatmap)
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
        'figure.constrained_layout.use': False,
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
    
    background_color = "#F0F0F0"
    
    # Filter the data for the specified isotope and MT
    filtered_mg = mg_covmat.filter_by_isotope_reaction(isotope, mt)
    
    if filtered_mg.num_matrices == 0:
        raise ValueError(f"No data found for isotope {isotope}, MT {mt}")
    
    # Determine which orders to include
    available_orders = sorted(set(filtered_mg.l_rows + filtered_mg.l_cols))
    
    if orders is None:
        orders = available_orders
    elif isinstance(orders, int):
        orders = [orders]
    
    # Filter orders to only those available
    orders = [order for order in orders if order in available_orders]
    
    if not orders:
        raise ValueError(f"Requested orders {orders} not available. Available orders: {available_orders}")
    
    # Build the matrix for the specified orders
    G = filtered_mg.num_groups
    N = len(orders)
    matrix_size = N * G
    
    # Create mapping from order to index
    order_to_idx = {order: i for i, order in enumerate(orders)}
    
    # Initialize the matrix
    if matrix_type == 'cov':
        full_matrix = np.zeros((matrix_size, matrix_size))
    else:  # correlation
        full_matrix = np.full((matrix_size, matrix_size), np.nan)
        
    # Fill in the matrix blocks
    for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c, rel_matrix, abs_matrix) in enumerate(zip(
        filtered_mg.isotope_rows, filtered_mg.reaction_rows, filtered_mg.l_rows,
        filtered_mg.isotope_cols, filtered_mg.reaction_cols, filtered_mg.l_cols,
        filtered_mg.relative_matrices, filtered_mg.absolute_matrices
    )):
        # Skip if orders not in our selection
        if l_r not in orders or l_c not in orders:
            continue
            
        # Get the appropriate matrix
        if covariance_type == 'rel':
            matrix = rel_matrix
        else:
            matrix = abs_matrix
            
        # Calculate block positions
        row_idx = order_to_idx[l_r]
        col_idx = order_to_idx[l_c]
        r0, r1 = row_idx * G, (row_idx + 1) * G
        c0, c1 = col_idx * G, (col_idx + 1) * G
        
        # Fill the block
        full_matrix[r0:r1, c0:c1] = matrix
        
        # Fill transpose if not diagonal block
        if row_idx != col_idx:
            full_matrix[c0:c1, r0:r1] = matrix.T
    
    # Convert to correlation matrix if requested
    if matrix_type == 'corr':
        # Compute correlation matrix
        diag_sqrt = np.sqrt(np.diag(full_matrix))
        
        # Handle zero or negative diagonal elements
        valid_diag = diag_sqrt > 0
        corr_matrix = np.full_like(full_matrix, np.nan)
        
        if np.any(valid_diag):
            # Create outer product for normalization
            diag_outer = np.outer(diag_sqrt, diag_sqrt)
            
            # Only compute correlation for valid diagonal elements
            valid_mask = np.outer(valid_diag, valid_diag)
            corr_matrix[valid_mask] = (full_matrix[valid_mask] / diag_outer[valid_mask])
            
            # Set diagonal to 1.0 for valid elements
            np.fill_diagonal(corr_matrix, 1.0)
        
        full_matrix = corr_matrix
    
    # Set default colormap and log scale based on matrix type
    if colormap is None:
        if matrix_type == 'corr':
            colormap = 'RdYlGn'
        else:
            colormap = 'viridis'
    
    if use_log_scale is None:
        if matrix_type == 'corr':
            use_log_scale = False  # Linear scale for correlation
        else:
            use_log_scale = True   # Log scale for covariance
    
    # Setup plot style - create figure manually like MF34
    plt.close('all')
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(background_color)
    ax.grid(False, which="both")
    for axis_obj in (ax.xaxis, ax.yaxis):
        axis_obj.grid(False)
    
    # Handle colormap selection - use user-specified or smart defaults
    if colormap is None:
        if matrix_type == 'corr':
            colormap_obj = plt.get_cmap("RdYlGn").copy()
        else:
            colormap_obj = plt.get_cmap("viridis").copy()
    else:
        # Use user-specified colormap
        if isinstance(colormap, str):
            try:
                colormap_obj = plt.get_cmap(colormap).copy()
            except ValueError:
                raise ValueError(f"Invalid colormap name: '{colormap}'. "
                               f"Please use a valid matplotlib colormap name.")
        elif hasattr(colormap, 'copy'):
            # Assume it's already a matplotlib colormap object
            colormap_obj = colormap.copy()
        else:
            raise ValueError("colormap must be a string name or matplotlib Colormap object")
    
    colormap_obj.set_bad(color=background_color)
    
    # Mask zeros and prepare matrix for display
    M = np.ma.masked_where(full_matrix == 0.0, full_matrix)
    
    # Determine color scale limits
    if matrix_type == 'corr':
        # For correlation matrices, clamp to [-1, 1] and use symmetric scale
        if symmetric_scale:
            if vmin is None and vmax is None:
                # Check for valid (non-NaN) correlation values
                valid_values = full_matrix[~np.isnan(full_matrix)]
                if len(valid_values) > 0:
                    max_abs = min(1.0, np.max(np.abs(valid_values)))
                    vmin, vmax = -max_abs, max_abs
                else:
                    vmin, vmax = -1.0, 1.0
            elif vmin is None:
                vmin = max(-1.0, -vmax)
            elif vmax is None:
                vmax = min(1.0, -vmin)
        else:
            if vmin is None:
                vmin = max(-1.0, np.nanmin(full_matrix[~np.isnan(full_matrix)]))
            if vmax is None:
                vmax = min(1.0, np.nanmax(full_matrix[~np.isnan(full_matrix)]))
                
        # Handle edge cases for normalization
        if np.isclose(vmin, vmax, atol=1e-10):
            if np.isclose(vmin, 0.0, atol=1e-10):
                vmin = -1e-6
                vmax = 1e-6
            else:
                padding = abs(vmax) * 0.01
                vmin = vmax - padding
                vmax = vmax + padding
        elif vmin >= vmax:
            vmin = vmax - 1e-6 if vmax != 0 else -1e-6
            
        # Use centered normalization for correlation
        from matplotlib.colors import TwoSlopeNorm
        vcenter = 0.0
        if vcenter <= vmin:
            vcenter = vmin + (vmax - vmin) * 0.1
        elif vcenter >= vmax:
            vcenter = vmax - (vmax - vmin) * 0.1
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        
    else:
        # For covariance matrices, handle log scale
        valid_values = full_matrix[~np.isnan(full_matrix)]
        if use_log_scale and len(valid_values) > 0:
            # For log scale, ensure all values are positive
            min_positive = np.min(valid_values[valid_values > 0]) if np.any(valid_values > 0) else 1e-10
            if vmin is None:
                vmin = max(min_positive, np.nanmin(valid_values[valid_values > 0])) if np.any(valid_values > 0) else 1e-10
            if vmax is None:
                vmax = np.nanmax(valid_values)
            # Replace non-positive values with vmin for log scale
            full_matrix = np.where(full_matrix <= 0, vmin, full_matrix)
            M = np.ma.masked_where(full_matrix == 0.0, full_matrix)
        else:
            if vmin is None:
                vmin = np.nanmin(valid_values)
            if vmax is None:
                vmax = np.nanmax(valid_values)
        norm = None
    
    # Create the heatmap
    im = ax.imshow(M, cmap=colormap_obj, norm=norm,
                   origin="upper", interpolation="nearest", aspect="auto")
    
    # Add grid lines to separate Legendre order blocks
    for g_val in range(0, matrix_size + 1, G):
        ax.axhline(g_val - 0.5, color="white", lw=1)
        ax.axvline(g_val - 0.5, color="white", lw=1)
    
    # Set up tick labels - show Legendre orders at block centers
    order_centers = [i * G + (G - 1) / 2 for i in range(len(orders))]
    order_labels = [f'L={order}' for order in orders]
    
    ax.set_xticks(order_centers)
    ax.set_yticks(order_centers)
    ax.set_xticklabels(order_labels, rotation=0, ha="center")
    ax.set_yticklabels(order_labels)
    
    ax.set_xlabel("Legendre coefficient")
    ax.set_ylabel("Legendre coefficient")
    
    # Set tick parameters
    ax.tick_params(axis="both", which="major", length=0, pad=5)
    
    # Create title
    try:
        from mcnpy._utils import zaid_to_symbol
        isotope_symbol = zaid_to_symbol(isotope)
    except Exception:
        isotope_symbol = f"Isotope {isotope}"
    
    if style not in {"paper", "publication"}:
        order_labels_str = ', '.join([str(order) for order in orders])
        matrix_name = "Covariance" if matrix_type == 'cov' else "Correlation"
        title_text = f"{isotope_symbol} MT:{mt}   L: {order_labels_str} {matrix_name}"
        ax.set_title(title_text)
    
    ax.tick_params(axis="both", which="major", length=0)
    
    # Add colorbar like MF34
    if show_colorbar:
        fig.canvas.draw()
        heatmap_pos = ax.get_position()
        
        cbar_horizontal_offset = 0.03
        cbar_ax = fig.add_axes([
            heatmap_pos.x1 + cbar_horizontal_offset,
            heatmap_pos.y0,
            0.03,
            heatmap_pos.height
        ])
        
        cbar = fig.colorbar(im, cax=cbar_ax)
        if matrix_type == 'cov':
            matrix_name = "Covariance"
        else:
            matrix_name = "Correlation"
        cbar.set_label(matrix_name)
    
    # Layout configuration (like MF34)
    num_orders = len(orders)
    bottom_margin_config = {
        1: 0.12, 2: 0.15, 3: 0.16, 4: 0.17, 5: 0.18, 6: 0.19, 7: 0.20,
    }
    bottom_margin = bottom_margin_config.get(num_orders, bottom_margin_config[7])
    
    fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=bottom_margin)
    
    # Add annotations if requested
    if annotate and matrix_size <= 20:  # Only annotate for small matrices
        for i in range(matrix_size):
            for j in range(matrix_size):
                value = full_matrix[i, j]
                if not np.isnan(value) and value != 0:
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           fontsize=8, color='white' if abs(value) > 0.5 * max(abs(vmin or 0), abs(vmax or 1)) else 'black')
    
    return fig


