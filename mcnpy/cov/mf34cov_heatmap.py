import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Union, List, Optional, Tuple, Any
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from mcnpy.cov.mf34_covmat import MF34CovMat
from mcnpy._utils import zaid_to_symbol
from mcnpy._plot_settings import setup_plot_style, format_axes


def plot_mf34_uncertainties(
    mf34_covmat: MF34CovMat,
    isotope: int,
    mt: int,
    legendre_coeffs: Union[int, List[int]],
    *,
    ax: plt.Axes | None = None,
    uncertainty_type: str = "relative",
    style: str = "default",
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
    font_family: str = "serif",
    legend_loc: str = "best",
    energy_range: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot uncertainties for MF34 angular distribution data for specific Legendre coefficients.
    
    This function extracts and plots the diagonal uncertainties from the covariance matrix
    for the specified isotope, MT reaction, and Legendre coefficients.
    
    Parameters
    ----------
    mf34_covmat : MF34CovMat
        The MF34 covariance matrix object
    isotope : int
        Isotope ID
    mt : int
        Reaction MT number
    legendre_coeffs : int or list of int
        Legendre coefficient(s) to plot uncertainties for.
        Can be a single int or a list of ints.
    ax : plt.Axes, optional
        Matplotlib axes to draw into. If None, creates new figure.
    uncertainty_type : str, default "relative"
        Type of uncertainty to plot: "relative" (%) or "absolute"
    style : str, default "default"
        Plot style: 'default', 'dark', 'paper', 'publication', 'presentation'
    figsize : tuple, default (8, 5)
        Figure size in inches (width, height)
    dpi : int, default 300
        Dots per inch for figure resolution
    font_family : str, default "serif"
        Font family for text elements
    legend_loc : str, default "best"
        Legend location
    energy_range : tuple of float, optional
        Energy range (min, max) for x-axis. If None, uses the full data range.
        Values are used directly without clamping to data range.
    **kwargs
        Additional arguments passed to matplotlib plot functions
    
    Returns
    -------
    plt.Figure
        The matplotlib figure containing the uncertainty plots
    
    Raises
    ------
    ValueError
        If no matrices found for the specified isotope/MT combination,
        or if requested Legendre coefficients are not available
    
    Examples
    --------
    Plot relative uncertainties for Legendre coefficients L=1,2,3:
    
    >>> fig = plot_mf34_uncertainties(mf34_covmat, isotope=92235, mt=2, 
    ...                              legendre_coeffs=[1, 2, 3])
    >>> fig.show()
    
    Plot absolute uncertainties for a single Legendre coefficient:
    
    >>> fig = plot_mf34_uncertainties(mf34_covmat, isotope=92235, mt=2,
    ...                              legendre_coeffs=1, uncertainty_type="absolute")
    >>> fig.show()
    """
    
    # Validate uncertainty_type parameter
    if uncertainty_type not in ["relative", "absolute"]:
        raise ValueError(f"uncertainty_type must be 'relative' or 'absolute', got '{uncertainty_type}'")
    
    # 1. Filter and locate the sub-matrix for the specified isotope and MT
    filtered_mf34 = mf34_covmat.filter_by_isotope_reaction(isotope, mt)
    
    if filtered_mf34.num_matrices == 0:
        raise ValueError(f"No matrices found for isotope={isotope}, MT={mt}")

    # Get all available Legendre coefficients for this isotope/MT combination
    all_triplets = filtered_mf34._get_param_triplets()
    available_legendre = sorted(list(set(t[2] for t in all_triplets if t[0] == isotope and t[1] == mt)))
    
    if not available_legendre:
        raise ValueError(f"No Legendre coefficients found for isotope={isotope}, MT={mt}")
    
    # Handle Legendre coefficient input format
    if isinstance(legendre_coeffs, int):
        legendre_list = [legendre_coeffs]
    else:
        legendre_list = list(legendre_coeffs)
        if not legendre_list:
            # Use all available Legendre coefficients
            legendre_list = available_legendre
    
    # Validate requested Legendre coefficients
    for l_val in legendre_list:
        if l_val not in available_legendre:
            raise ValueError(f"Legendre coefficient L={l_val} not available for isotope={isotope}, MT={mt}")
    
    legendre_coeffs_sorted = sorted(list(set(legendre_list)))
    
    # Find the energy group size (assume all matrices have the same size)
    G = filtered_mf34.matrices[0].shape[0] if filtered_mf34.matrices else 0
    if G == 0:
        raise ValueError("Number of energy groups (G) cannot be zero.")
    
    # 2. Setup plot style and create figure/axes
    plot_settings = setup_plot_style(
        style=style,
        figsize=figsize,
        dpi=dpi,
        font_family=font_family,
        ax=ax,
        **kwargs
    )
    
    ax = plot_settings['ax']
    fig = plot_settings['_fig']
    colors = plot_settings['_colors']
    
    # 3. Setup energy grid
    energy_grid = None
    if filtered_mf34.energy_grids:
        energy_grid = filtered_mf34.energy_grids[0]
    
    if energy_grid is not None and len(energy_grid) == G + 1:
        energy_grid_mids = [(energy_grid[i] + energy_grid[i+1]) / 2 for i in range(G)]
        xs = energy_grid_mids
        use_log_scale = True
    else:
        xs = list(range(1, G+1))
        use_log_scale = False
    
    # 4. Get diagonal uncertainties from the filtered covariance matrix
    diag_sqrt = np.sqrt(np.diag(filtered_mf34.covariance_matrix))
    
    # 5. Plot uncertainties for each Legendre coefficient
    for i, l_val in enumerate(legendre_coeffs_sorted):
        # Find the index of this Legendre coefficient
        triplet_idx = all_triplets.index((isotope, mt, l_val))
        sigma = diag_sqrt[triplet_idx*G : (triplet_idx+1)*G]
        
        if uncertainty_type == "relative":
            # Convert to percentage
            plot_values = sigma * 100
            y_label = "Relative uncertainty (%)"
        else:  # absolute
            plot_values = sigma
            y_label = "Absolute uncertainty"
        
        # Create label
        isotope_symbol = zaid_to_symbol(isotope)
        label = f"{isotope_symbol} - σ$_{{a_{{{l_val}}}}}$"
        if uncertainty_type == "relative":
            label += " (%)"
        else:
            label += " (absolute)"
        
        # Plot as step function (uncertainties are constant within energy bins)
        color = colors[i % len(colors)]
        
        # Filter out step-related kwargs that should not be passed to step()
        step_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['ax', '_fig', '_colors', '_linestyles', '_style', 
                                    '_notebook_mode', '_interactive', '_backend_info']}
        
        # Get the actual energy bin boundaries from the covariance matrix's energy_grids attribute
        bin_boundaries = None
        
        # Find the energy grid that corresponds to this (isotope, mt, l_val) combination
        for j, (iso_r, mt_r, l_r, iso_c, mt_c, l_c) in enumerate(zip(
            filtered_mf34.isotope_rows, filtered_mf34.reaction_rows, filtered_mf34.l_rows,
            filtered_mf34.isotope_cols, filtered_mf34.reaction_cols, filtered_mf34.l_cols
        )):
            # Look for diagonal variance matrix (L = L) for the specified parameters
            if (iso_r == isotope and iso_c == isotope and 
                mt_r == mt and mt_c == mt and 
                l_r == l_val and l_c == l_val):
                
                bin_boundaries = np.array(filtered_mf34.energy_grids[j])
                break
        
        if len(plot_values) == 1:
            # Single point - plot as horizontal line across entire range
            ax.axhline(y=plot_values[0], color=color, label=label, linewidth=2.0)
        else:
            if bin_boundaries is not None and len(bin_boundaries) == len(plot_values) + 1:
                # We have the actual bin boundaries - use them for proper step plot
                step_energies = bin_boundaries
                step_values = np.append(plot_values, plot_values[-1])  # Extend last value for step plot
                
                # Plot as step function
                ax.step(step_energies[:-1], step_values[:-1], where='post', color=color, 
                       label=label, linewidth=2.0, **step_kwargs)
                
                # Extend the last step to the final boundary
                ax.hlines(step_values[-1], step_energies[-2], step_energies[-1], 
                         colors=color, linewidth=2.0)
                         
            else:
                # Fallback: estimate bin boundaries from bin centers (xs)
                # This approach may not be accurate for actual energy bin structure
                step_energies = []
                step_values = []
                
                # Add first boundary (extrapolated)
                if len(xs) > 1:
                    first_boundary = xs[0] - (xs[1] - xs[0]) / 2
                    step_energies.append(max(first_boundary, 1e-5))  # Don't go below 1e-5 eV
                else:
                    step_energies.append(xs[0] / 2)
                step_values.append(plot_values[0])
                
                # Add boundaries between consecutive points
                for j in range(len(xs) - 1):
                    boundary = (xs[j] + xs[j + 1]) / 2
                    step_energies.append(boundary)
                    step_values.append(plot_values[j])
                
                # Add the last bin
                if len(xs) > 1:
                    step_energies.append(xs[-1] + (xs[-1] - xs[-2]) / 2)
                else:
                    step_energies.append(xs[-1] * 2)
                step_values.append(plot_values[-1])
                
                # Plot as step function with 'post' positioning
                ax.step(step_energies[:-1], step_values[:-1], where='post', color=color, 
                       label=label, linewidth=2.0, **step_kwargs)
                
                # Extend the last step to the final boundary
                ax.hlines(step_values[-1], step_energies[-2], step_energies[-1], 
                         colors=color, linewidth=2.0)
    
    # 6. Set up plot title
    l_labels = [str(l_val) for l_val in legendre_coeffs_sorted]
    title_text = f"{zaid_to_symbol(isotope)} MT:{mt}   L: {', '.join(l_labels)} Uncertainties"
    
    # 7. Calculate y-axis limits
    y_min = None
    y_max = None
    if len(diag_sqrt) > 0:
        all_plot_values = []
        for l_val in legendre_coeffs_sorted:
            triplet_idx = all_triplets.index((isotope, mt, l_val))
            sigma = diag_sqrt[triplet_idx*G : (triplet_idx+1)*G]
            if uncertainty_type == "relative":
                all_plot_values.extend(sigma * 100)
            else:
                all_plot_values.extend(sigma)
        
        if all_plot_values:
            y_min = 0 if uncertainty_type == "relative" else min(all_plot_values)
            y_max = max(all_plot_values)
    
    # 8. Format axes using the centralized function
    format_axes(
        ax=ax,
        style=style,
        use_log_scale=use_log_scale,
        is_energy_axis=use_log_scale,  # True if we have actual energy values
        x_label="Energy (eV)" if use_log_scale else "Energy-group index",
        y_label=y_label,
        title=title_text if style not in {"paper", "publication"} else None,
        legend_loc=legend_loc if len(legendre_coeffs_sorted) > 1 else None,
        y_min=y_min,
        y_max=y_max,
        use_y_log_scale=False,
    )
    
    # Apply energy range limits if specified
    if energy_range is not None:
        # Use user-specified energy range directly, allowing extension beyond data range
        e_min, e_max = energy_range
        ax.set_xlim(e_min, e_max)
    
    return fig


def plot_mf34_covariance_heatmap(
    mf34_covmat: MF34CovMat,
    isotope: int,
    mt: int,
    legendre_coeffs: Union[int, List[int], Tuple[int, int]],
    *,
    ax: plt.Axes | None = None,
    matrix_type: str = "corr",
    style: str = "default",
    figsize: Tuple[float, float] = (6, 6),
    dpi: int = 300,
    font_family: str = "serif",
    vmax: float | None = None,
    vmin: float | None = None,
    show_uncertainties: bool = False,
    cmap: Optional[Union[str, mpl.colors.Colormap]] = None,
    **imshow_kwargs,
) -> plt.Figure:
    """
    Draw a covariance or correlation matrix heat-map for MF34 angular distribution data
    for *one* isotope and MT reaction, with one or more Legendre coefficients.

    This function follows the same structure and formatting as the regular heatmap.py
    but is adapted for Legendre coefficients instead of MT numbers.

    Parameters
    ----------
    mf34_covmat : MF34CovMat
        The MF34 covariance matrix object
    isotope : int
        Isotope ID
    mt : int
        Reaction MT number
    legendre_coeffs : int, list of int, or tuple of (row_l, col_l)
        Legendre coefficient(s). Can be:
        - Single int: diagonal block for that L
        - List of ints: diagonal blocks for those L values  
        - Tuple of (row_l, col_l): off-diagonal block between row and column L
    ax : plt.Axes, optional
        Matplotlib axes to draw into (only used when show_uncertainties=False)
    matrix_type : str
        Type of matrix to plot: "cov" for covariance or "corr" for correlation
    style : str
        Plot style: 'default', 'dark', 'paper', 'publication', 'presentation'
    figsize : tuple
        Figure size in inches (width, height)
    dpi : int
        Dots per inch for figure resolution
    font_family : str
        Font family for text elements
    vmax, vmin : float, optional
        Color scale limits
    show_uncertainties : bool
        Whether to show uncertainty plots above the heatmap
    show : bool
        Whether to display the figure (default: True)
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for the heatmap. Can be a string name of any matplotlib 
        colormap (e.g., 'viridis', 'plasma', 'RdYlBu', 'coolwarm') or a matplotlib 
        Colormap object. If None, defaults to 'RdYlGn' for correlation matrices 
        and 'viridis' for covariance matrices.
    **imshow_kwargs
        Additional arguments passed to imshow

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the heatmap and optional uncertainty plots
    """
    # Validate matrix_type parameter
    if matrix_type not in ["cov", "corr"]:
        raise ValueError(f"matrix_type must be 'cov' or 'corr', got '{matrix_type}'")
    
    # Reset matplotlib to default state
    plt.rcdefaults()
    
    # Setup publication-quality settings (same as heatmap.py)
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

    # 1. Filter and locate the sub-matrix for the specified isotope and MT
    filtered_mf34 = mf34_covmat.filter_by_isotope_reaction(isotope, mt)
    
    if filtered_mf34.num_matrices == 0:
        raise ValueError(f"No matrices found for isotope={isotope}, MT={mt}")

    # Get all available Legendre coefficients for this isotope/MT combination
    all_triplets = filtered_mf34._get_param_triplets()
    available_legendre = sorted(list(set(t[2] for t in all_triplets if t[0] == isotope and t[1] == mt)))
    
    if not available_legendre:
        raise ValueError(f"No Legendre coefficients found for isotope={isotope}, MT={mt}")

    # Find the energy group size (assume all matrices have the same size)
    G = filtered_mf34.matrices[0].shape[0] if filtered_mf34.matrices else 0
    if G == 0:
        raise ValueError("Number of energy groups (G) cannot be zero.")

    # Handle different Legendre coefficient input formats
    if isinstance(legendre_coeffs, tuple) and len(legendre_coeffs) == 2:
        # Off-diagonal block case: (row_l, col_l)
        row_l, col_l = legendre_coeffs
        if not isinstance(row_l, int) or not isinstance(col_l, int):
            raise ValueError("For off-diagonal blocks, legendre_coeffs tuple must contain two integers: (row_l, col_l)")
        
        legendre_coeffs_sorted = [row_l, col_l]
        is_diagonal = False
        
        # Extract the specific off-diagonal block
        l_triplets = [(isotope, mt, row_l), (isotope, mt, col_l)]
        
        # Verify both Legendre coefficients exist
        for l_val in [row_l, col_l]:
            if l_val not in available_legendre:
                raise ValueError(f"Legendre coefficient L={l_val} not available for isotope={isotope}, MT={mt}")
                
    else:
        # Diagonal block logic
        if isinstance(legendre_coeffs, int):
            legendre_list = [legendre_coeffs]
        else:
            legendre_list = list(legendre_coeffs)
            if not legendre_list:
                # Use all available Legendre coefficients
                legendre_list = available_legendre
                
        # Validate requested Legendre coefficients
        for l_val in legendre_list:
            if l_val not in available_legendre:
                raise ValueError(f"Legendre coefficient L={l_val} not available for isotope={isotope}, MT={mt}")
        
        legendre_coeffs_sorted = sorted(list(set(legendre_list)))
        is_diagonal = True
        
        # Create triplets for the diagonal blocks
        l_triplets = [(isotope, mt, l_val) for l_val in legendre_coeffs_sorted]

    if not l_triplets:
        raise ValueError("No valid Legendre coefficient data to plot.")

    # Determine if we're plotting a single block (single Legendre coefficient)
    single_block = (is_diagonal and len(legendre_coeffs_sorted) == 1) or (not is_diagonal)

    # Get the appropriate matrix based on matrix_type
    if matrix_type == "cov":
        full_matrix = filtered_mf34.covariance_matrix
        matrix_name = "Covariance"
    else:  # matrix_type == "corr"
        full_matrix = filtered_mf34.correlation_matrix
        matrix_name = "Correlation"

    # Extract the submatrix for the requested Legendre coefficients
    if is_diagonal:
        # For diagonal blocks, extract blocks for all requested Legendre coefficients
        indices = []
        for l_val in legendre_coeffs_sorted:
            # Find the index of this Legendre coefficient in the full matrix
            triplet_idx = all_triplets.index((isotope, mt, l_val))
            indices.extend(range(triplet_idx * G, (triplet_idx + 1) * G))
        
        M_full = full_matrix[np.ix_(indices, indices)]
    else:
        # For off-diagonal blocks, extract the specific block
        row_triplet_idx = all_triplets.index((isotope, mt, row_l))
        col_triplet_idx = all_triplets.index((isotope, mt, col_l))
        
        row_indices = list(range(row_triplet_idx * G, (row_triplet_idx + 1) * G))
        col_indices = list(range(col_triplet_idx * G, (col_triplet_idx + 1) * G))
        
        M_full = full_matrix[np.ix_(row_indices, col_indices)]

    # 2. Setup figure and layout based on whether we show uncertainties
    if show_uncertainties:
        # Create figure with uncertainty plots
        plt.ioff()
        
        uncertainty_height_ratio = 0.2
        # For off-diagonal blocks, show both row and column uncertainties
        num_legendre = 2 if not is_diagonal else len(legendre_coeffs_sorted)
        fig_height = figsize[1] * (1 + uncertainty_height_ratio)
        
        plt.close('all')
        
        fig = plt.figure(figsize=(figsize[0], fig_height), dpi=dpi)
        
        gs = GridSpec(2, num_legendre, figure=fig,
                     height_ratios=[uncertainty_height_ratio, 1],
                     hspace=0.02, wspace=0.01)
        
        uncertainty_axes = []
        for i in range(num_legendre):
            ax_u = fig.add_subplot(gs[0, i])
            uncertainty_axes.append(ax_u)
            
        ax_heatmap = fig.add_subplot(gs[1, :])
    else:
        plt.close('all')
        if ax is None:
            fig, ax_heatmap = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            ax_heatmap = ax
            fig = ax_heatmap.get_figure()
        uncertainty_axes = None

    # 3. Mask zeros and setup display
    M = np.ma.masked_where(M_full == 0.0, M_full)
    ax_heatmap.set_facecolor(background_color)
    ax_heatmap.grid(False, which="both")
    for axis_obj in (ax_heatmap.xaxis, ax_heatmap.yaxis):
        axis_obj.grid(False)

    # 4. Color normalization and colormap setup
    # Handle colormap selection
    if cmap is None:
        # Use default colormaps based on matrix type
        if matrix_type == "corr":
            colormap = plt.get_cmap("RdYlGn").copy()
        else:
            colormap = plt.get_cmap("viridis").copy()
    else:
        # Use user-specified colormap
        if isinstance(cmap, str):
            try:
                colormap = plt.get_cmap(cmap).copy()
            except ValueError:
                raise ValueError(f"Invalid colormap name: '{cmap}'. "
                               f"Please use a valid matplotlib colormap name.")
        elif hasattr(cmap, 'copy'):
            # Assume it's already a matplotlib colormap object
            colormap = cmap.copy()
        else:
            raise ValueError("cmap must be a string name or matplotlib Colormap object")
    
    colormap.set_bad(color=background_color)

    if vmax is None or vmin is None:
        if matrix_type == "corr":
            # For correlation matrices, use symmetric range around 0
            absmax = np.nanmax(np.abs(M_full)) if M_full.size > 0 else 1.0
            vmax_calc = min(1.0, absmax) if vmax is None else vmax
            vmin_calc = max(-1.0, -absmax) if vmin is None else vmin
        else:
            # For covariance matrices, use 0 to max
            matrix_max = np.nanmax(M_full) if M_full.size > 0 else 1.0
            vmax_calc = matrix_max if vmax is None else vmax
            vmin_calc = 0.0 if vmin is None else vmin
    else:
        vmax_calc = vmax
        vmin_calc = vmin
        
    # Handle edge cases for normalization
    if np.isclose(vmin_calc, vmax_calc, atol=1e-10):
        if np.isclose(vmin_calc, 0.0, atol=1e-10):
            vmin_calc = -1e-6
            vmax_calc = 1e-6
        else:
            padding = abs(vmax_calc) * 0.01
            vmin_calc = vmax_calc - padding
            vmax_calc = vmax_calc + padding
    elif vmin_calc >= vmax_calc:
        vmin_calc = vmax_calc - 1e-6 if vmax_calc != 0 else -1e-6
    
    # Choose normalization based on matrix type
    if matrix_type == "corr":
        vcenter = 0.0
        if vcenter <= vmin_calc:
            vcenter = vmin_calc + (vmax_calc - vmin_calc) * 0.1
        elif vcenter >= vmax_calc:
            vcenter = vmax_calc - (vmax_calc - vmin_calc) * 0.1
        norm = TwoSlopeNorm(vmin=vmin_calc, vcenter=vcenter, vmax=vmax_calc)
    else:
        # For covariance matrices, use linear normalization
        norm = None

    # 5. Draw the heatmap matrix
    # Remove 'cmap' from imshow_kwargs to avoid conflict
    plot_kwargs = {k: v for k, v in imshow_kwargs.items()
                   if not k.startswith("_") and k not in ["ax", "cmap"]}
    
    # Use regular imshow for all cases (single or multiple blocks)
    im = ax_heatmap.imshow(M, cmap=colormap, norm=norm,
                           origin="upper", interpolation="nearest", **plot_kwargs)

    # Draw block boundaries only for multiple blocks
    if not single_block and is_diagonal:
        matrix_size = len(legendre_coeffs_sorted) * G
        # Draw block boundaries
        for g_val in range(0, matrix_size + 1, G):
            ax_heatmap.axhline(g_val - 0.5, color="black", lw=0.2)
            ax_heatmap.axvline(g_val - 0.5, color="black", lw=0.2)

    # 6. Ticks, labels for heatmap
    # Always show Legendre coefficients consistently, regardless of single or multiple blocks
    if single_block:
        # For single blocks, show the single Legendre coefficient in the center
        centres = [G / 2 - 0.5]  # Center of the matrix
        
        if is_diagonal:
            # Single diagonal block
            l_val = legendre_coeffs_sorted[0]
            x_labels = [str(l_val)]
            y_labels = [str(l_val)]
        else:
            # Off-diagonal block between two Legendre coefficients
            x_labels = [str(col_l)]  # X-axis shows second number of tuple
            y_labels = [str(row_l)]  # Y-axis shows first number of tuple
        
        ax_heatmap.set_xticks(centres)
        ax_heatmap.set_yticks(centres)
        ax_heatmap.set_xticklabels(x_labels, rotation=0, ha="center")
        ax_heatmap.set_yticklabels(y_labels)
        
        ax_heatmap.set_xlabel("Legendre coefficient")
        ax_heatmap.set_ylabel("Legendre coefficient")
    else:
        # For multiple blocks, show Legendre coefficients
        centres = [i * G + (G - 1) / 2 for i in range(len(legendre_coeffs_sorted))]
        l_labels = [str(l_val) for l_val in legendre_coeffs_sorted]
        
        ax_heatmap.set_xticks(centres)
        ax_heatmap.set_yticks(centres)
        ax_heatmap.set_xticklabels(l_labels, rotation=0, ha="center")
        ax_heatmap.set_yticklabels(l_labels)
        
        ax_heatmap.set_xlabel("Legendre coefficient")
        ax_heatmap.set_ylabel("Legendre coefficient")
    
    # Set tick parameters for heatmap
    ax_heatmap.tick_params(axis="both", which="major", length=0, pad=5)

    # Set title
    if style not in {"paper", "publication"}:
        if single_block:
            # For single blocks, include Legendre coefficient in title
            if is_diagonal:
                l_val = legendre_coeffs_sorted[0]
                title_text = f"{zaid_to_symbol(isotope)} MT:{mt}   L={l_val} {matrix_name}"
            else:
                title_text = f"{zaid_to_symbol(isotope)} MT:{mt}   L={row_l}-{col_l} {matrix_name}"
        else:
            # For multiple blocks, show coefficient range
            l_labels = [str(l_val) for l_val in legendre_coeffs_sorted]
            title_text = f"{zaid_to_symbol(isotope)} MT:{mt}   L: {', '.join(l_labels)} {matrix_name}"
            
        if show_uncertainties:
            fig.suptitle(title_text)
        else:
            ax_heatmap.set_title(title_text)
    ax_heatmap.tick_params(axis="both", which="major", length=0)

    # 7. Draw uncertainty plots (if requested)
    if show_uncertainties and uncertainty_axes is not None:
        # Setup energy grid - try to get from the first available matrix
        energy_grid = None
        if filtered_mf34.energy_grids:
            energy_grid = filtered_mf34.energy_grids[0]
        
        if energy_grid is not None and len(energy_grid) == G + 1:
            energy_grid_mids = [(energy_grid[i] + energy_grid[i+1]) / 2 for i in range(G)]
            xs = energy_grid_mids
            use_log_scale = True
        else:
            xs = list(range(1, G+1))
            use_log_scale = False
        
        # Get diagonal uncertainties from the filtered covariance matrix
        diag_sqrt = np.sqrt(np.diag(filtered_mf34.covariance_matrix))
        line_props = {'color': 'black', 'linewidth': 0.2}

        if is_diagonal:
            # Process each Legendre coefficient independently for diagonal case
            for i, (ax_uncert, l_val) in enumerate(zip(uncertainty_axes, legendre_coeffs_sorted)):
                # Find the index of this Legendre coefficient
                triplet_idx = all_triplets.index((isotope, mt, l_val))
                sigma = diag_sqrt[triplet_idx*G : (triplet_idx+1)*G]
                sigma_pct = sigma * 100
                
                ax_uncert.set_facecolor(background_color)
                ax_uncert.grid(True, which="major", axis='y', linestyle='--', color='gray', alpha=0.3)
                ax_uncert.step(xs, sigma_pct, where='mid', linewidth=1.5, color=f'C{i}')
                
                if use_log_scale:
                    ax_uncert.set_xscale('log')
                
                y_data_min, y_data_max = 0, max(sigma_pct) if len(sigma_pct) > 0 and max(sigma_pct) > 0 else 5
                padding = 0.1 * (y_data_max - y_data_min)
                ax_uncert.set_ylim(y_data_min, y_data_max + padding)
                
                x_min, x_max = min(xs), max(xs)
                ax_uncert.set_xlim(x_min, x_max)
                
                ax_uncert.set_xticklabels([])
                ax_uncert.set_xticks([])
                # Explicitly hide x-axis ticks and labels
                ax_uncert.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                
                # Custom tick setup for y-axis (same as heatmap.py)
                y_max = y_data_max
                if y_max <= 5:
                    yticks = [0, 1, 2, 3, 4, 5]
                elif y_max <= 10:
                    yticks = [0, 2, 4, 6, 8, 10]
                elif y_max <= 20:
                    yticks = [0, 5, 10, 15, 20]
                elif y_max <= 50:
                    yticks = [0, 10, 20, 30, 40, 50]
                else:
                    step = max(5, int(y_max / 5 / 5) * 5)
                    yticks = list(range(0, int(y_max * 1.1) + step, step))
                    
                yticks = [int(round(tick)) for tick in yticks]
                yticks = [tick for tick in yticks if tick <= y_data_max + padding]
                yticks = sorted(set(yticks))
                
                if len(yticks) < 2:
                    yticks = [0, int(round(y_data_max + padding))]
                
                ax_uncert.set_yticks(yticks)
                ax_uncert.set_yticklabels([])
                ax_uncert.tick_params(axis='y', which='major', direction='in', length=0, color='none')
                
                if i == 0:
                    ax_uncert.set_ylabel('% Unc.', fontsize='small')
                
                ax_uncert.spines['bottom'].set_visible(False)
                ax_uncert.spines['top'].set_visible(True)
                ax_uncert.spines['top'].set(**line_props)

                if i == 0: 
                    ax_uncert.spines['left'].set_visible(True)
                    ax_uncert.spines['left'].set(**line_props)
                else: 
                    ax_uncert.spines['left'].set_visible(False)

                if i == len(uncertainty_axes) - 1:
                    ax_uncert.spines['right'].set_visible(True)
                    ax_uncert.spines['right'].set(**line_props)
                else:
                    ax_uncert.spines['right'].set_visible(False)

                # Add internal ytick values as text labels
                for ytick in yticks:
                    if ytick > 0:
                        ax_uncert.text(
                            xs[0] if not use_log_scale else xs[0] * 1.05,
                            ytick,
                            f"{ytick}",
                            fontsize='x-small',
                            ha='left',
                            va='center',
                            alpha=0.7
                        )
        else:
            # For off-diagonal blocks, show both row and column L uncertainties
            l_vals_for_uncertainties = [row_l, col_l]
            l_labels_for_uncertainties = [str(row_l), str(col_l)]
            
            for i, (ax_uncert, l_val, l_label) in enumerate(zip(uncertainty_axes, l_vals_for_uncertainties, l_labels_for_uncertainties)):
                # Find the index of this Legendre coefficient
                triplet_idx = all_triplets.index((isotope, mt, l_val))
                sigma = diag_sqrt[triplet_idx*G : (triplet_idx+1)*G]
                sigma_pct = sigma * 100
                
                ax_uncert.set_facecolor(background_color)
                ax_uncert.grid(True, which="major", axis='y', linestyle='--', color='gray', alpha=0.3)
                ax_uncert.step(xs, sigma_pct, where='mid', linewidth=1.5, color=f'C{i}')
                
                if use_log_scale:
                    ax_uncert.set_xscale('log')
                
                y_data_min, y_data_max = 0, max(sigma_pct) if len(sigma_pct) > 0 and max(sigma_pct) > 0 else 5
                padding = 0.1 * (y_data_max - y_data_min)
                ax_uncert.set_ylim(y_data_min, y_data_max + padding)
                
                x_min, x_max = min(xs), max(xs)
                ax_uncert.set_xlim(x_min, x_max)
                
                ax_uncert.set_xticklabels([])
                ax_uncert.set_xticks([])
                # Explicitly hide x-axis ticks and labels
                ax_uncert.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                
                # Custom tick setup for y-axis
                y_max = y_data_max
                if y_max <= 5:
                    yticks = [0, 1, 2, 3, 4, 5]
                elif y_max <= 10:
                    yticks = [0, 2, 4, 6, 8, 10]
                elif y_max <= 20:
                    yticks = [0, 5, 10, 15, 20]
                elif y_max <= 50:
                    yticks = [0, 10, 20, 30, 40, 50]
                else:
                    step = max(5, int(y_max / 5 / 5) * 5)
                    yticks = list(range(0, int(y_max * 1.1) + step, step))
                    
                yticks = [int(round(tick)) for tick in yticks]
                yticks = [tick for tick in yticks if tick <= y_data_max + padding]
                yticks = sorted(set(yticks))
                
                if len(yticks) < 2:
                    yticks = [0, int(round(y_data_max + padding))]
                
                ax_uncert.set_yticks(yticks)
                ax_uncert.set_yticklabels([])
                ax_uncert.tick_params(axis='y', which='major', direction='in', length=0, color='none')
                
                # First plot gets y-axis label
                if i == 0:
                    ax_uncert.set_ylabel('% Unc.', fontsize='small')
                
                # Add title to identify which L this uncertainty belongs to
                ax_uncert.set_title(f'L {l_label}', fontsize='small', pad=2)
                
                # Spine setup for two uncertainty plots
                ax_uncert.spines['bottom'].set_visible(False)
                ax_uncert.spines['top'].set_visible(True)
                ax_uncert.spines['top'].set(**line_props)

                if i == 0: 
                    ax_uncert.spines['left'].set_visible(True)
                    ax_uncert.spines['left'].set(**line_props)
                    ax_uncert.spines['right'].set_visible(False)
                else: 
                    ax_uncert.spines['left'].set_visible(False)
                    ax_uncert.spines['right'].set_visible(True)
                    ax_uncert.spines['right'].set(**line_props)

                # Add internal ytick values as text labels
                for ytick in yticks:
                    if ytick > 0:
                        ax_uncert.text(
                            xs[0] if not use_log_scale else xs[0] * 1.05,
                            ytick,
                            f"{ytick}",
                            fontsize='x-small',
                            ha='left',
                            va='center',
                            alpha=0.7
                        )

    # 8. Layout configuration (same as heatmap.py)
    num_legendre = 2 if (not is_diagonal and show_uncertainties) else (1 if not is_diagonal else len(legendre_coeffs_sorted))
    bottom_margin_config = {
        1: 0.12, 2: 0.15, 3: 0.16, 4: 0.17, 5: 0.18, 6: 0.19, 7: 0.20,
    }
    bottom_margin = bottom_margin_config.get(num_legendre, bottom_margin_config[7])
    
    if show_uncertainties:
        plt.ion()
        fig.subplots_adjust(left=0.12, right=0.90, top=0.93, bottom=bottom_margin)
    else:
        fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=bottom_margin)

    # 9. Color-bar (same as heatmap.py)
    fig.canvas.draw()
    heatmap_pos = ax_heatmap.get_position()
    
    cbar_horizontal_offset = 0.03
    
    if show_uncertainties and uncertainty_axes is not None:
        cbar_ax = fig.add_axes([
            heatmap_pos.x1 + cbar_horizontal_offset,
            heatmap_pos.y0,
            0.03,
            heatmap_pos.height
        ])
    else:
        cbar_ax = fig.add_axes([
            heatmap_pos.x1 + cbar_horizontal_offset,
            heatmap_pos.y0,
            0.03,
            heatmap_pos.height
        ])
    
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(matrix_name)

    # Always return the figure object for consistency
    return fig