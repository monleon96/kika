import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Sequence, List, Optional, Tuple
from mcnpy.cov.covmat import CovMat
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from mcnpy._plot_settings import setup_plot_style, format_axes, finalize_plot
from mcnpy._constants import MT_TO_REACTION
from mcnpy._utils import zaid_to_symbol

def plot_uncertainties(
    covmat: CovMat,
    zaid: Union[int, Sequence[int]],
    mt:   Union[int, Sequence[int]],
    ax: plt.Axes = None,
    *,
    energy_range: Optional[Tuple[float, float]] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
    font_family: str = 'serif',
    legend_loc: str = 'best',
    **step_kwargs
) -> plt.Axes:
    """
    Step-plot the 1-sigma uncertainties for one or more (ZAID, MT) pairs
    across all energy groups.

    Parameters
    ----------
    covmat
        your CovMat instance
    zaid
        single ZAID or list of ZAIDs
    mt
        single MT or list of MTs
    ax
        optional matplotlib Axes to draw into
    energy_range
        optional tuple (min_energy, max_energy) to limit the plot range
    style
        plot style: 'default', 'dark', 'paper', 'publication', 'presentation'
    figsize
        figure size in inches (width, height)
    dpi
        dots per inch for figure resolution
    font_family
        font family for text elements ('serif', 'sans-serif', etc.)
    legend_loc
        location for the legend placement
    step_kwargs
        forwarded to Axes.step (e.g. where="mid", color=...)
    """
    # normalize inputs to lists
    zaids = [zaid] if isinstance(zaid, int) else list(zaid)
    mts   = [mt]   if isinstance(mt,   int) else list(mt)
    # if either list is empty, default to all present in covmat
    if not isinstance(zaid, int) and len(zaids) == 0:
        zaids = covmat.isotopes
    if not isinstance(mt,   int) and len(mts)   == 0:
        mts   = covmat.reactions
    
    # Configure plot style and get plotting utilities
    step_kwargs['ax'] = ax
    step_kwargs['_func'] = 'step'  # Hint for setup_plot_style
    plot_settings = setup_plot_style(
        style=style, figsize=figsize, dpi=dpi, font_family=font_family, **step_kwargs
    )
    
    # Extract settings from the returned dict
    ax = plot_settings['ax']
    colors = plot_settings.get('_colors', [])
    linestyles = plot_settings.get('_linestyles', [])
    fig = plot_settings.get('_fig')
    
    # Create dictionaries to map MT to color and ZAID to line style
    mt_to_color = {m: colors[i % len(colors)] for i, m in enumerate(sorted(set(mts)))}
    zaid_to_linestyle = {z: linestyles[i % len(linestyles)] for i, z in enumerate(sorted(set(zaids)))}
    
    # Track if we've determined the x-axis type and values
    energy_grid = None
    xs = None
    use_log_scale = False
    
    # Store min/max y values for axis scaling
    y_min, y_max = float('inf'), float('-inf')
    
    # Plot for each ZAID separately, using filter_by_isotope
    for Z in zaids:
        # Filter covmat to only include this isotope
        try:
            # Use filter_by_isotope to ensure we only have data for this isotope
            iso_covmat = covmat.filter_by_isotope(Z)
            
            # Get the necessary data from the filtered matrix
            G = iso_covmat.num_groups
            diag = np.sqrt(np.diag(iso_covmat.covariance_matrix))
            pairs = iso_covmat._get_param_pairs()
            
            # Set up the x-axis values if not done already
            if xs is None:
                xs = list(range(1, G+1))  # group indices
                
                # Get energy grid if available
                if iso_covmat.energy_grid is not None and len(iso_covmat.energy_grid) == G + 1:
                    # Use mid-point of energy bins for plotting
                    energy_grid = [(iso_covmat.energy_grid[i] + iso_covmat.energy_grid[i+1]) / 2 for i in range(G)]
                    xs = energy_grid
                    use_log_scale = True
            
            # Filter by energy range if specified and energy_grid is available
            energy_mask = slice(None)  # Default: use all points
            if energy_range is not None and energy_grid is not None:
                min_e, max_e = energy_range
                energy_mask = np.where((np.array(energy_grid) >= min_e) & 
                                    (np.array(energy_grid) <= max_e))[0]
                if len(energy_mask) == 0:
                    continue  # Skip this isotope if no points in range
            
            for M in mts:
                if (Z, M) not in pairs:
                    continue
                    
                # Start with original kwargs for this curve
                curve_kwargs = {k: v for k, v in step_kwargs.items() 
                               if not k.startswith('_') and k != 'ax'}
                
                # Assign color based on MT and line style based on ZAID
                if 'color' not in step_kwargs:
                    curve_kwargs['color'] = mt_to_color[M]
                
                if 'linestyle' not in step_kwargs:
                    curve_kwargs['linestyle'] = zaid_to_linestyle[Z]
                
                i = pairs.index((Z, M))
                sigma = diag[i*G : (i+1)*G]
                
                # Apply energy filter
                if isinstance(energy_mask, slice):
                    filtered_xs = xs[energy_mask]
                    filtered_sigma = sigma[energy_mask]
                else:
                    filtered_xs = [xs[i] for i in energy_mask]
                    filtered_sigma = [sigma[i] for i in energy_mask]
                
                # Always plot percentage uncertainty
                y = np.array(filtered_sigma) * 100
                
                # Update min/max for y-axis scaling
                if len(y) > 0:
                    y_min = min(y_min, np.min(y))
                    y_max = max(y_max, np.max(y))
                
                # Format element symbol and MT (with reaction name if available)
                el_symbol = zaid_to_symbol(Z)
                reaction_name = MT_TO_REACTION.get(M, "")
                
                # Format labels consistently
                if style == 'paper' or style == 'publication':
                    label = f"{el_symbol}, MT={M} ({reaction_name})" if reaction_name else f"{el_symbol}, MT={M}"
                else:
                    label = f"{el_symbol}-{M}"
                    if reaction_name:
                        label += f" ({reaction_name})"
                
                # Add the plot
                ax.step(filtered_xs, y, label=label, **curve_kwargs)
                
        except Exception as e:
            print(f"Warning: Could not process ZAID={Z}: {str(e)}")
            continue
    
    # Generate title if needed for non-paper styles
    title = None
    if style != 'paper' and style != 'publication':
        title_parts = []
        if len(zaids) == 1:
            title_parts.append(f"{zaid_to_symbol(zaids[0])}")
        if len(mts) == 1:
            mt_val = mts[0]
            reaction_name = MT_TO_REACTION.get(mt_val, "")
            mt_part = f"MT={mt_val}"
            if reaction_name:
                mt_part += f" ({reaction_name})"
            title_parts.append(mt_part)
        if title_parts:
            title = " ".join(title_parts)
    
    # Format the axes
    format_axes(
        ax=ax,
        style=style,
        use_log_scale=use_log_scale,
        is_energy_axis=(energy_grid is not None),
        y_label="% uncertainty",
        title=title,
        legend_loc=legend_loc,
        y_min=y_min if not np.isinf(y_min) else None,
        y_max=y_max if not np.isinf(y_max) else None
    )
    
    # Ensure figure is displayed
    finalize_plot(fig)
    
    return ax

def compare_uncertainties(
    covmats: List[CovMat],
    zaid: int,
    mt: Union[int, List[int]],
    labels: Optional[List[str]] = None,
    ax: plt.Axes = None,
    *,
    energy_range: Optional[Tuple[float, float]] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
    font_family: str = 'serif',
    legend_loc: str = 'best',
    title: Optional[str] = None,
    **step_kwargs
) -> plt.Axes:
    """
    Compare uncertainty plots for the same ZAID from multiple CovMat objects.

    Parameters
    ----------
    covmats
        List of CovMat instances to compare
    zaid
        Single ZAID to plot from all CovMats
    mt
        Either a single MT reaction to plot from all CovMats,
        or a list of MTs (one per CovMat) to compare
    labels
        Optional list of labels for each CovMat (defaults to "CovMat 1", "CovMat 2", etc.)
    ax
        optional matplotlib Axes to draw into
    energy_range
        optional tuple (min_energy, max_energy) to limit the plot range
    style
        plot style: 'default', 'dark', 'paper', 'publication', 'presentation'
    figsize
        figure size in inches (width, height)
    dpi
        dots per inch for figure resolution
    font_family
        font family for text elements ('serif', 'sans-serif', etc.)
    legend_loc
        location for the legend placement
    title
        optional custom title for the plot
    step_kwargs
        forwarded to Axes.step (e.g. where="mid", color=...)

    Returns
    -------
    plt.Axes
        The matplotlib axes containing the plot

    Raises
    ------
    ValueError
        If any CovMat doesn't contain the specified ZAID and MT
    """
    if not covmats:
        raise ValueError("No CovMat objects provided")
    
    # Handle MT parameter - either single value or list
    if isinstance(mt, int):
        mts = [mt] * len(covmats)
    else:
        if len(mt) != len(covmats):
            raise ValueError(f"If mt is a list, it must have the same length as covmats "
                            f"(got {len(mt)} MT values for {len(covmats)} CovMat objects)")
        mts = mt
    
    # Ensure we have labels for all CovMats
    if labels is None:
        if len(set(mts)) == 1:
            # If all MTs are the same, just use CovMat numbers
            labels = [f"CovMat {i+1}" for i in range(len(covmats))]
        else:
            # If different MTs, include the MT number in the label
            labels = [f"MT={mt_val}" for mt_val in mts]
    elif len(labels) != len(covmats):
        raise ValueError(f"Expected {len(covmats)} labels, got {len(labels)}")
    
    # Convert ZAID to element symbol
    el_symbol = zaid_to_symbol(zaid)
    
    # Configure plot style and get plotting utilities
    step_kwargs['ax'] = ax
    step_kwargs['_func'] = 'step'  # Hint for setup_plot_style
    
    # Always use solid line style for comparison plots
    if 'linestyle' not in step_kwargs:
        step_kwargs['linestyle'] = '-'
        
    plot_settings = setup_plot_style(
        style=style, figsize=figsize, dpi=dpi, font_family=font_family, **step_kwargs
    )
    
    # Extract settings from the returned dict
    ax = plot_settings['ax']
    colors = plot_settings.get('_colors', [])
    fig = plot_settings.get('_fig')
    
    # Track axis properties
    use_log_scale = False
    y_min, y_max = float('inf'), float('-inf')
    
    # Plot uncertainty from each CovMat
    for i, (covmat, mt_val, label) in enumerate(zip(covmats, mts, labels)):
        # First filter the covmat to just include the requested isotope
        try:
            # Use filter_by_isotope to ensure we only have data for this isotope
            iso_covmat = covmat.filter_by_isotope(zaid)
            
            # Check if the specified ZAID, MT exists in this filtered CovMat
            pairs = iso_covmat._get_param_pairs()
            if (zaid, mt_val) not in pairs:
                print(f"Warning: ZAID={zaid}, MT={mt_val} not found in {label}, skipping")
                continue
            
            # Get the uncertainty data
            sigma_index = pairs.index((zaid, mt_val))
            G = iso_covmat.num_groups
            diag = np.sqrt(np.diag(iso_covmat.covariance_matrix))
            sigma = diag[sigma_index*G : (sigma_index+1)*G]
            
            # Set up the x-axis values
            if iso_covmat.energy_grid is not None and len(iso_covmat.energy_grid) == G + 1:
                # Use mid-point of energy bins for plotting
                energy_grid = [(iso_covmat.energy_grid[j] + iso_covmat.energy_grid[j+1]) / 2 for j in range(G)]
                xs = energy_grid
                use_log_scale = True
            else:
                xs = list(range(1, G+1))
            
            # Filter by energy range if specified and energy_grid is available
            if energy_range is not None and 'energy_grid' in locals() and energy_grid is not None:
                min_e, max_e = energy_range
                energy_mask = np.where((np.array(energy_grid) >= min_e) & 
                                    (np.array(energy_grid) <= max_e))[0]
                if len(energy_mask) == 0:
                    print(f"Warning: No energy points found in range {min_e} to {max_e} for {label}")
                    continue
                    
                filtered_xs = [xs[j] for j in energy_mask]
                filtered_sigma = [sigma[j] for j in energy_mask]
            else:
                filtered_xs = xs
                filtered_sigma = sigma
            
            # Convert to percentage uncertainty
            y = np.array(filtered_sigma) * 100
            
            # Track min/max for y-axis scaling
            if len(y) > 0:
                y_min = min(y_min, np.min(y))
                y_max = max(y_max, np.max(y))
            
            # Set color for this curve (always use solid line)
            plot_kwargs = {k: v for k, v in step_kwargs.items() 
                          if not k.startswith('_') and k != 'ax'}
            if 'color' not in step_kwargs:
                plot_kwargs['color'] = colors[i % len(colors)]
            
            # Always use solid line style
            plot_kwargs['linestyle'] = '-'
            
            # Enhance label if using different MTs
            if isinstance(mt, list) and len(set(mt)) > 1:
                reaction_name = MT_TO_REACTION.get(mt_val, "")
                # If MTs differ, include MT in label unless it's already there
                if not label.startswith("MT=") and "MT=" not in label:
                    if reaction_name:
                        plot_label = f"{label} (MT={mt_val}, {reaction_name})"
                    else:
                        plot_label = f"{label} (MT={mt_val})"
                else:
                    plot_label = label
            else:
                plot_label = label
            
            # Plot the uncertainty
            ax.step(filtered_xs, y, label=plot_label, where='mid', **plot_kwargs)
            
        except Exception as e:
            print(f"Warning: Error processing {label}: {str(e)}")
            continue
    
    # Generate default title if none provided and all MTs are the same
    if title is None and len(set(mts)) == 1:
        mt_desc = ""
        try:
            if mts[0] in MT_TO_REACTION:
                mt_desc = f" ({MT_TO_REACTION[mts[0]]})"
        except (ImportError, IndexError):
            pass
        title = f"{el_symbol}, MT={mts[0]}{mt_desc} Uncertainty Comparison"
    elif title is None:
        # If different MTs, just use the element symbol
        title = f"{el_symbol} Uncertainty Comparison"
    
    # Format the axes
    format_axes(
        ax=ax,
        style=style,
        use_log_scale=use_log_scale,
        is_energy_axis=use_log_scale,
        y_label="% uncertainty",
        title=title,
        legend_loc=legend_loc,
        y_min=y_min if not np.isinf(y_min) else None,
        y_max=y_max if not np.isinf(y_max) else None
    )
    
    # Ensure figure is displayed
    finalize_plot(fig)
    
    return ax



def plot_covariance_heatmap(
    covmat: "CovMat",
    zaid: int,
    mt: Union[int, Sequence[int]],
    *,
    ax: plt.Axes = None,
    style: str = "default",
    figsize: Tuple[float, float] = (6, 6),
    dpi: int = 300,
    font_family: str = "serif",
    show_colorbar: bool = True,
    cmap=None,
    vmax: float | None = None,
    vmin: float | None = None,
    cbar_label: str = "Covariance",
    **imshow_kwargs
) -> plt.Axes:
    """
    Heat-map the energy-group covariance for one isotope and one or more
    reactions (MT list).  Cross-correlation blocks between the reactions
    are included automatically.

    * Negative values appear red, zero is white, positive values appear green.
    * If a single MT is supplied the plot is just the G × G block.
    """

    # ------------------------------------------------------------
    # 1. normalise MT input and isolate the isotope
    # ------------------------------------------------------------
    mts: List[int] = [mt] if isinstance(mt, int) else list(mt)
    if len(mts) == 0:
        raise ValueError("The MT list must contain at least one reaction number.")

    iso_cov = covmat.filter_by_isotope(zaid)        # single-isotope view
    pairs   = iso_cov._get_param_pairs()
    G       = iso_cov.num_groups

    # confirm every requested MT exists and collect their indices
    mt_indices = []
    for m in mts:
        if (zaid, m) not in pairs:
            raise ValueError(f"(ZAID, MT)=({zaid}, {m}) not present in CovMat")
        mt_indices.append(pairs.index((zaid, m)))

    # rows/cols to keep in the combined matrix, preserving user order
    rows = []
    for idx in mt_indices:
        rows.extend(range(idx * G, (idx + 1) * G))
    M = iso_cov.covariance_matrix[np.ix_(rows, rows)]   # square sub-matrix

    # ------------------------------------------------------------
    # 2. set up figure / axes
    # ------------------------------------------------------------
    imshow_kwargs["_func"] = "imshow"
    imshow_kwargs["ax"]    = ax
    if "setup_plot_style" in globals():
        ps  = setup_plot_style(style=style, figsize=figsize, dpi=dpi,
                               font_family=font_family, **imshow_kwargs)
        ax  = ps["ax"]
        fig = ps["_fig"]
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ------------------------------------------------------------
    # 3. colour map centred on zero
    # ------------------------------------------------------------
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "red_white_green",
            ["#8b0000", "#ffffff", "#006400"],
            N=256
        )
    if vmax is None or vmin is None:
        absmax = np.nanmax(np.abs(M))
        vmax   = absmax if vmax is None else vmax
        vmin   = -absmax if vmin is None else vmin
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # ------------------------------------------------------------
    # 4. draw the matrix
    # ------------------------------------------------------------
    im = ax.imshow(
        M,
        cmap=cmap,
        norm=norm,
        origin="lower",
        interpolation="nearest",
        **imshow_kwargs
    )

    # thin grid lines at every energy-group boundary
    block_size = G
    matrix_size = len(mts) * G
    for g in range(0, matrix_size + 1, block_size):
        ax.axhline(g - 0.5, color="black", lw=0.2)
        ax.axvline(g - 0.5, color="black", lw=0.2)

    # optional colour-bar
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    # ------------------------------------------------------------
    # 5. tidy labels and layout
    # ------------------------------------------------------------
    if style not in {"paper", "publication"}:
        mt_part = ", ".join(str(m) for m in mts) if len(mts) > 1 else f"{mts[0]}"
        ax.set_title(f"{zaid_to_symbol(zaid)}   MT: {mt_part}")

    ax.set_xticks([])
    ax.set_yticks([])

    if "format_axes" in globals():
        format_axes(ax=ax, style=style, use_log_scale=False,
                    is_energy_axis=False, y_label="", title=None,
                    legend_loc="best")
    if "finalize_plot" in globals():
        finalize_plot(fig)
    else:
        fig.tight_layout()

    return ax