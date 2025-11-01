import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Sequence, List, Optional, Tuple
from mcnpy.cov.covmat import CovMat
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
    show: bool = True,
    **step_kwargs
) -> plt.Figure:
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
        optional tuple (min_energy, max_energy) to limit the plot range (in MeV)
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
    show : bool
        Whether to display the figure (default: True)
    step_kwargs
        forwarded to Axes.step (e.g. where="mid", color=...)
        
    Returns
    -------
    plt.Figure
        The matplotlib figure containing the plot
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
    
    # Helper function to infer edges from bin centers
    def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
        """Infer edges from bin centers for step plotting."""
        centers = np.asarray(centers, dtype=float)
        if centers.size == 1:
            width = max(centers[0] * 0.05, 1e-5)
            return np.array([max(centers[0] - width, 1e-5), centers[0] + width], dtype=float)
        diffs = np.diff(centers)
        left0 = centers[0] - diffs[0] / 2.0
        rightN = centers[-1] + diffs[-1] / 2.0
        mid = (centers[:-1] + centers[1:]) / 2.0
        edges = np.concatenate([[max(left0, 1e-5)], mid, [rightN]])
        return edges
    
    use_log_scale = False
    
    # Plot for each ZAID separately, using filter_by_isotope
    for Z in zaids:
        # Filter covmat to only include this isotope
        try:
            # Use filter_by_isotope to ensure we only have data for this isotope
            iso_covmat = covmat.filter_by_isotope(Z)
            
            # Get the necessary data from the filtered matrix
            G = int(iso_covmat.num_groups)
            diag = np.sqrt(np.diag(iso_covmat.covariance_matrix))
            pairs = iso_covmat._get_param_pairs()
            
            for M in mts:
                if (Z, M) not in pairs:
                    continue
                    
                i = pairs.index((Z, M))
                vals = diag[i*G : (i+1)*G]  # length G
                
                # Build energy boundaries for step x
                boundaries = None
                if getattr(iso_covmat, "energy_grid", None) is not None:
                    eg = np.asarray(iso_covmat.energy_grid, dtype=float)
                    if eg.ndim != 1:
                        print(f"Warning: unexpected energy_grid shape for ZAID={Z}, MT={M}, skipping")
                        continue
                    if eg.size == G + 1:
                        boundaries = eg
                    elif eg.size == G:
                        boundaries = _centers_to_edges(eg)
                    else:
                        print(f"Warning: energy_grid length {eg.size} incompatible with groups {G} for ZAID={Z}, MT={M}")
                # Fallback: fake boundaries on indices
                if boundaries is None:
                    boundaries = np.arange(G + 1, dtype=float)
                
                # Now plot with G+1 x and G+1 y (last value repeated)
                y = np.r_[vals * 100.0, (vals[-1] * 100.0 if len(vals) else np.nan)]  # percent
                x = boundaries
                
                # Start with original kwargs for this curve
                curve_kwargs = {k: v for k, v in step_kwargs.items() 
                               if not k.startswith('_') and k != 'ax'}
                
                # Assign color based on MT and line style based on ZAID
                if 'color' not in step_kwargs:
                    curve_kwargs['color'] = mt_to_color[M]
                
                if 'linestyle' not in step_kwargs:
                    curve_kwargs['linestyle'] = zaid_to_linestyle[Z]
                
                if 'linewidth' not in curve_kwargs:
                    curve_kwargs['linewidth'] = 2.0
                curve_kwargs['where'] = 'post'  # ensures the final bin extends to x[-1]
                
                # Format element symbol and MT (with reaction name if available)
                el_symbol = zaid_to_symbol(Z)
                reaction_name = MT_TO_REACTION.get(M, "")
                
                # Format labels consistently
                if style == 'paper' or style == 'publication':
                    label = f"{el_symbol}, MT={M} {reaction_name}" if reaction_name else f"{el_symbol}, MT={M}"
                else:
                    label = f"{el_symbol}-{M}"
                    if reaction_name:
                        label += f" {reaction_name}"
                
                # Add the plot
                ax.step(x, y, label=label, **curve_kwargs)
                
                # If we have real energy edges, use a log energy axis
                if boundaries is not None and (boundaries > 0).all():
                    use_log_scale = True
                
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
                mt_part += f" {reaction_name}"
            title_parts.append(mt_part)
        if title_parts:
            title = " ".join(title_parts)
    
    # Format the axes
    ax = format_axes(
        ax=ax,
        style=style,
        use_log_scale=use_log_scale,
        is_energy_axis=use_log_scale,
        x_label="Energy (MeV)" if use_log_scale else "Group",
        y_label="% uncertainty",
        title=title,
        legend_loc=legend_loc
    )
    
    # Apply x-range like compare_uncertainties: set limits, don't pre-trim data
    if energy_range is not None:
        e_min, e_max = energy_range
        ax.set_xlim(e_min, e_max)
        
        # Recompute y-limits based on visible data region
        y_values_in_range = []
        for line in ax.get_lines():
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            if xdata.size != ydata.size:
                continue
            mask = (xdata >= e_min) & (xdata <= e_max)
            if np.any(mask):
                y_values_in_range.extend(ydata[mask])
        if y_values_in_range:
            y_vals = np.asarray(y_values_in_range)
            y0, y1 = float(np.min(y_vals)), float(np.max(y_vals))
            if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
                pad = 0.05 * (y1 - y0)
                ax.set_ylim(y0 - pad, y1 + pad)
    else:
        # Remove default x-padding when using energy edges
        # Find min & max x from plotted data and tighten
        all_x = []
        for line in ax.get_lines():
            xd = np.asarray(line.get_xdata())
            if xd.size:
                all_x.append((np.min(xd), np.max(xd)))
        if all_x:
            xmin = min(v[0] for v in all_x)
            xmax = max(v[1] for v in all_x)
            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                ax.set_xlim(xmin, xmax)
    
    # Ensure figure is displayed only if requested
    if show:
        finalize_plot(fig)
    
    return fig

def compare_uncertainties(
    covmats: List[CovMat],
    zaid: int,
    mt: Union[int, tuple, List[Union[int, List[int], tuple]]],
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
    show: bool = True,
    **step_kwargs
) -> plt.Figure:
    """
    Compare uncertainty plots for the same ZAID from multiple CovMat objects.

    Parameters
    ----------
    covmats
        List of CovMat instances to compare
    zaid
        Single ZAID to plot from all CovMats
    mt
        Can be:
        - A single int: plots that MT from all CovMats
        - A tuple of ints: plots all those MTs from all CovMats
        - A list with one entry per CovMat, where each entry is:
          - A single int: plots that MT from the corresponding CovMat
          - A list/tuple of ints: plots all those MTs from the corresponding CovMat
    labels
        Optional list of labels for each CovMat (defaults to "CovMat 1", "CovMat 2", etc.)
    ax
        optional matplotlib Axes to draw into
    energy_range
        optional tuple (min_energy, max_energy) to limit the plot range (in MeV)
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
    show : bool
        Whether to display the figure (default: True)
    step_kwargs
        forwarded to Axes.step (e.g. where="mid", color=...)

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the plot

    Raises
    ------
    ValueError
        If any CovMat doesn't contain the specified ZAID and MT
    """
    if not covmats:
        raise ValueError("No CovMat objects provided")

    # Normalize MTs to a list of lists
    if isinstance(mt, int):
        # Single int → apply to all CovMats
        mts_per_covmat = [[mt] for _ in covmats]
    elif isinstance(mt, tuple):
        # Tuple of ints → apply all those MTs to all CovMats
        mts_per_covmat = [list(mt) for _ in covmats]
    else:
        # mt is a list
        if len(mt) != len(covmats):
            raise ValueError(
                f"If mt is a list, it must have the same length as covmats "
                f"(got {len(mt)} MT values for {len(covmats)} CovMat objects)"
            )
        # Normalize each element to a list
        mts_per_covmat = []
        for m in mt:
            if isinstance(m, int):
                mts_per_covmat.append([m])
            elif isinstance(m, (list, tuple)):
                mts_per_covmat.append(list(m))
            else:
                raise ValueError(f"MT element must be int or list of ints, got {type(m)}")

    # Labels
    if labels is None:
        labels = [f"CovMat {i+1}" for i in range(len(covmats))]
    elif len(labels) != len(covmats):
        raise ValueError(f"Expected {len(covmats)} labels, got {len(labels)}")

    # Helpers
    def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
        """Infer edges from bin centers for step plotting."""
        centers = np.asarray(centers, dtype=float)
        if centers.size == 1:
            # Make an arbitrary small bracket around the single center
            width = max(centers[0] * 0.05, 1e-5)
            return np.array([max(centers[0] - width, 1e-5), centers[0] + width], dtype=float)
        diffs = np.diff(centers)
        left0 = centers[0] - diffs[0] / 2.0
        rightN = centers[-1] + diffs[-1] / 2.0
        mid = (centers[:-1] + centers[1:]) / 2.0
        edges = np.concatenate([[max(left0, 1e-5)], mid, [rightN]])
        return edges

    # Convert ZAID to symbol (same as before)
    el_symbol = zaid_to_symbol(zaid)

    # Configure style
    step_kwargs['ax'] = ax
    step_kwargs['_func'] = 'step'  # hint for setup_plot_style
    # Don't set default linestyle here - we'll assign per-covmat below

    plot_settings = setup_plot_style(
        style=style, figsize=figsize, dpi=dpi, font_family=font_family, **step_kwargs
    )
    ax = plot_settings['ax']
    fig = plot_settings['_fig']
    colors = plot_settings.get('_colors') or plt.rcParams.get('axes.prop_cycle', None)
    if hasattr(colors, 'by_key'):
        colors = colors.by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
    if not colors:
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    
    # Get linestyles for different CovMats
    linestyles = plot_settings.get('_linestyles', ['-', '--', '-.', ':'])
    if not linestyles:
        linestyles = ['-', '--', '-.', ':']

    use_log_scale = False

    # Track all unique MTs for title generation and color mapping
    all_mts = set()
    for mt_list in mts_per_covmat:
        all_mts.update(mt_list)
    
    # Create MT to color mapping (same MT = same color across all CovMats)
    mt_to_color = {mt_val: colors[i % len(colors)] for i, mt_val in enumerate(sorted(all_mts))}
    
    # Create CovMat to linestyle mapping (different CovMat = different linestyle)
    covmat_to_linestyle = {i: linestyles[i % len(linestyles)] for i in range(len(covmats))}

    # Plot each series - now we need to iterate over all (covmat, mt_val) pairs
    for covmat_idx, (covmat, mt_list, label) in enumerate(zip(covmats, mts_per_covmat, labels)):
        for mt_idx, mt_val in enumerate(mt_list):
            try:
                iso_covmat = covmat.filter_by_isotope(zaid)

                # Validate presence of (zaid, mt)
                pairs = iso_covmat._get_param_pairs()
                if (zaid, mt_val) not in pairs:
                    print(f"Warning: ZAID={zaid}, MT={mt_val} not found in {label}, skipping")
                    continue

                idx = pairs.index((zaid, mt_val))
                G = int(iso_covmat.num_groups)

                # Uncertainty (std) per group
                diag = np.sqrt(np.diag(iso_covmat.covariance_matrix))
                if diag.size < (idx + 1) * G:
                    print(f"Warning: covariance diagonal too small for {label}, skipping")
                    continue
                vals = diag[idx * G: (idx + 1) * G]  # length G

                # Build energy boundaries for step x
                boundaries = None
                if getattr(iso_covmat, "energy_grid", None) is not None:
                    eg = np.asarray(iso_covmat.energy_grid, dtype=float)
                    if eg.ndim != 1:
                        print(f"Warning: unexpected energy_grid shape for {label}, skipping")
                        continue
                    if eg.size == G + 1:
                        boundaries = eg
                    elif eg.size == G:
                        boundaries = _centers_to_edges(eg)
                    else:
                        print(f"Warning: energy_grid length {eg.size} incompatible with groups {G} for {label}")
                # Fallback: fake boundaries on indices
                if boundaries is None:
                    boundaries = np.arange(G + 1, dtype=float)

                # Now plot with G+1 x and G+1 y (last value repeated)
                y = np.r_[vals * 100.0, (vals[-1] * 100.0 if len(vals) else np.nan)]  # percent
                x = boundaries

                # Style kwargs: use MT for color, CovMat for linestyle
                plot_kwargs = {k: v for k, v in step_kwargs.items() if not k.startswith('_') and k != 'ax'}
                
                # Assign color based on MT (same MT = same color)
                if 'color' not in plot_kwargs:
                    plot_kwargs['color'] = mt_to_color[mt_val]
                
                # Assign linestyle based on CovMat (different CovMat = different linestyle)
                if 'linestyle' not in plot_kwargs:
                    plot_kwargs['linestyle'] = covmat_to_linestyle[covmat_idx]
                
                if 'linewidth' not in plot_kwargs:
                    plot_kwargs['linewidth'] = 2.0
                plot_kwargs['where'] = 'post'  # ensures the final bin extends to x[-1]

                # Build label for this specific series
                plot_label = label
                
                # If this CovMat has multiple MTs, add MT info to distinguish them
                if len(mt_list) > 1:
                    reaction_name = MT_TO_REACTION.get(mt_val, "")
                    plot_label = f"{label} (MT={mt_val}{', ' + reaction_name if reaction_name else ''})"
                # If all CovMats have the same single MT, labels are fine as is
                elif len(all_mts) > 1:
                    # Multiple different MTs across CovMats - add MT info if not already in label
                    if "MT=" not in label:
                        reaction_name = MT_TO_REACTION.get(mt_val, "")
                        plot_label = f"{label} (MT={mt_val}{', ' + reaction_name if reaction_name else ''})"

                ax.step(x, y, label=plot_label, **plot_kwargs)

                # If we have real energy edges, use a log energy axis
                if boundaries is not None and (boundaries > 0).all():
                    use_log_scale = True

            except Exception as e:
                print(f"Warning: Error processing {label}, MT={mt_val}: {e}")
                continue

    # Title
    if title is None:
        if len(all_mts) == 1:
            try:
                mt_val = list(all_mts)[0]
                mt_desc = f" ({MT_TO_REACTION[mt_val]})" if mt_val in MT_TO_REACTION else ""
            except Exception:
                mt_desc = ""
            title = f"{el_symbol}, MT={mt_val}{mt_desc}"
        else:
            title = f"{el_symbol}"

    # Axes formatting (match Legendre plot UX)
    ax = format_axes(
        ax=ax,
        style=style,
        use_log_scale=use_log_scale,
        is_energy_axis=use_log_scale,
        x_label="Energy (MeV)" if use_log_scale else "Group",
        y_label="% uncertainty",
        title=title,
        legend_loc=legend_loc
    )

    # Apply x-range like the Legendre function: set limits, don't pre-trim data
    if energy_range is not None:
        e_min, e_max = energy_range
        ax.set_xlim(e_min, e_max)

        # Recompute y-limits based on visible data region
        y_values_in_range = []
        for line in ax.get_lines():
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            if xdata.size != ydata.size:
                continue  # should not happen with our step construction
            mask = (xdata >= e_min) & (xdata <= e_max)
            if np.any(mask):
                y_values_in_range.extend(ydata[mask])
        if y_values_in_range:
            y_vals = np.asarray(y_values_in_range)
            y0, y1 = float(np.min(y_vals)), float(np.max(y_vals))
            if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
                pad = 0.05 * (y1 - y0)
                ax.set_ylim(y0 - pad, y1 + pad)
    else:
        # Remove default x-padding when using energy edges
        # Find min & max x from plotted data and tighten
        all_x = []
        for line in ax.get_lines():
            xd = np.asarray(line.get_xdata())
            if xd.size:
                all_x.append((np.min(xd), np.max(xd)))
        if all_x:
            xmin = min(v[0] for v in all_x)
            xmax = max(v[1] for v in all_x)
            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                ax.set_xlim(xmin, xmax)

    if show:
        finalize_plot(fig)
    return fig


def plot_multigroup_xs(
    covmat: CovMat,
    zaid: Union[int, Sequence[int]],
    mt: Union[int, Sequence[int]],
    ax: plt.Axes = None,
    *,
    energy_range: Optional[Tuple[float, float]] = None,
    show_uncertainties: bool = False,
    sigma: float = 1.0,
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
    font_family: str = 'serif',
    legend_loc: str = 'best',
    show: bool = True,
    **step_kwargs
) -> plt.Figure:
    """
    Step-plot the multigroup cross sections for one or more (ZAID, MT) pairs
    across all energy groups, with optional uncertainty shading.

    Parameters
    ----------
    covmat : CovMat
        CovMat instance containing cross section data
    zaid : int or sequence of int
        Single ZAID or list of ZAIDs
    mt : int or sequence of int
        Single MT or list of MTs
    ax : plt.Axes, optional
        Optional matplotlib Axes to draw into
    energy_range : tuple, optional
        Optional tuple (min_energy, max_energy) to limit the plot range (in MeV)
    show_uncertainties : bool
        If True, show uncertainty bands as shaded areas around the cross sections
    sigma : float
        Number of standard deviations for uncertainty bands (default: 1.0)
    style : str
        Plot style: 'default', 'dark', 'paper', 'publication', 'presentation'
    figsize : tuple
        Figure size in inches (width, height)
    dpi : int
        Dots per inch for figure resolution
    font_family : str
        Font family for text elements ('serif', 'sans-serif', etc.)
    legend_loc : str
        Location for the legend placement
    show : bool
        Whether to display the figure (default: True)
    **step_kwargs
        Additional arguments forwarded to Axes.step (e.g. where="mid", color=...)

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the plot

    Raises
    ------
    ValueError
        If requested ZAID/MT combinations are not found in the cross_sections data
    """
    # Normalize inputs to lists
    zaids = [zaid] if isinstance(zaid, int) else list(zaid)
    mts = [mt] if isinstance(mt, int) else list(mt)
    
    # If either list is empty, default to all present in covmat cross sections
    if not isinstance(zaid, int) and len(zaids) == 0:
        zaids = sorted(set(iso for (iso, _) in covmat.cross_sections.keys()))
    if not isinstance(mt, int) and len(mts) == 0:
        mts = sorted(set(rxn for (_, rxn) in covmat.cross_sections.keys()))
    
    # Check if we have any cross section data
    if not covmat.cross_sections:
        raise ValueError("No cross section data found in CovMat object")
    
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
    
    # Helper function to infer edges from bin centers
    def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
        """Infer edges from bin centers for step plotting."""
        centers = np.asarray(centers, dtype=float)
        if centers.size == 1:
            width = max(centers[0] * 0.05, 1e-5)
            return np.array([max(centers[0] - width, 1e-5), centers[0] + width], dtype=float)
        diffs = np.diff(centers)
        left0 = centers[0] - diffs[0] / 2.0
        rightN = centers[-1] + diffs[-1] / 2.0
        mid = (centers[:-1] + centers[1:]) / 2.0
        edges = np.concatenate([[max(left0, 1e-5)], mid, [rightN]])
        return edges
    
    use_log_scale = False
    use_y_log_scale = False
    
    # Get energy grid setup
    G = int(covmat.num_groups)
    if G == 0:
        raise ValueError("Number of energy groups is zero")
    
    # Store uncertainty data for proper alignment
    uncertainty_data = {}
    
    # Track y-axis range for log scale decision
    y_min, y_max = float('inf'), float('-inf')
    
    # Plot for each ZAID/MT combination
    for Z in zaids:
        for M in mts:
            # Check if this combination exists in cross_sections
            if (Z, M) not in covmat.cross_sections:
                print(f"Warning: Cross section for ZAID={Z}, MT={M} not found, skipping")
                continue
            
            # Get cross section data
            xs_data = covmat.cross_sections[(Z, M)]
            if len(xs_data) != G:
                print(f"Warning: Cross section data length ({len(xs_data)}) doesn't match num_groups ({G}) for ZAID={Z}, MT={M}")
                continue
            
            # Build energy boundaries for step x
            boundaries = None
            if getattr(covmat, "energy_grid", None) is not None:
                eg = np.asarray(covmat.energy_grid, dtype=float)
                if eg.ndim != 1:
                    print(f"Warning: unexpected energy_grid shape for ZAID={Z}, MT={M}, skipping")
                    continue
                if eg.size == G + 1:
                    boundaries = eg
                elif eg.size == G:
                    boundaries = _centers_to_edges(eg)
                else:
                    print(f"Warning: energy_grid length {eg.size} incompatible with groups {G} for ZAID={Z}, MT={M}")
            # Fallback: fake boundaries on indices
            if boundaries is None:
                boundaries = np.arange(G + 1, dtype=float)
            
            # Now plot with G+1 x and G+1 y (last value repeated)
            y = np.r_[xs_data, (xs_data[-1] if len(xs_data) else np.nan)]
            x = boundaries
            
            # Update min/max for y-axis scaling
            if len(xs_data) > 0 and np.any(xs_data > 0):
                positive_data = xs_data[xs_data > 0]
                y_min = min(y_min, np.min(positive_data))
                y_max = max(y_max, np.max(positive_data))
            
            # Start with original kwargs for this curve
            curve_kwargs = {k: v for k, v in step_kwargs.items() 
                           if not k.startswith('_') and k != 'ax'}
            
            # Assign color based on MT and line style based on ZAID
            if 'color' not in step_kwargs:
                curve_kwargs['color'] = mt_to_color[M]
            
            if 'linestyle' not in step_kwargs:
                curve_kwargs['linestyle'] = zaid_to_linestyle[Z]
            
            if 'linewidth' not in curve_kwargs:
                curve_kwargs['linewidth'] = 2.0
            curve_kwargs['where'] = 'post'  # ensures the final bin extends to x[-1]
            
            # Format element symbol and MT (with reaction name if available)
            el_symbol = zaid_to_symbol(Z)
            reaction_name = MT_TO_REACTION.get(M, "")
            
            # Format labels consistently
            if style == 'paper' or style == 'publication':
                label = f"{el_symbol}, MT={M} {reaction_name}" if reaction_name else f"{el_symbol}, MT={M}"
            else:
                label = f"{el_symbol}-{M}"
                if reaction_name:
                    label += f" {reaction_name}"
            
            # Add sigma information to label if showing uncertainties
            if show_uncertainties and sigma != 1.0:
                label += f" (±{sigma:.1f}σ)"
            elif show_uncertainties:
                label += " (±1σ)"
            
            # Plot the cross section
            line_color = curve_kwargs.get('color', mt_to_color[M])
            ax.step(x, y, label=label, **curve_kwargs)
            
            # If we have real energy edges, use a log energy axis
            if boundaries is not None and (boundaries > 0).all():
                use_log_scale = True
            
            # Prepare uncertainty data if requested
            if show_uncertainties:
                try:
                    # Filter the covmat to get uncertainty for this isotope
                    iso_covmat = covmat.filter_by_isotope(Z)
                    pairs = iso_covmat._get_param_pairs()
                    
                    if (Z, M) in pairs:
                        # Get the uncertainty data
                        sigma_index = pairs.index((Z, M))
                        diag = np.sqrt(np.diag(iso_covmat.covariance_matrix))
                        rel_sigma = diag[sigma_index*G : (sigma_index+1)*G]
                        
                        # Store for later plotting to ensure proper alignment
                        uncertainty_data[(Z, M)] = {
                            'boundaries': boundaries.copy(),
                            'xs_data': xs_data.copy(),
                            'sigma': rel_sigma.copy(),
                            'color': line_color
                        }
                        
                except Exception as e:
                    print(f"Warning: Could not prepare uncertainties for ZAID={Z}, MT={M}: {str(e)}")
    
    # Now add all uncertainty bands after the main plots for proper layering
    if show_uncertainties:
        for (Z, M), unc_data in uncertainty_data.items():
            try:
                # Calculate upper and lower bounds (n-sigma)
                # For cross sections, uncertainty is relative, so sigma represents relative uncertainty
                upper_bound = unc_data['xs_data'] * (1 + sigma * unc_data['sigma'])
                lower_bound = unc_data['xs_data'] * (1 - sigma * unc_data['sigma'])
                
                # Ensure lower bound doesn't go negative
                lower_bound = np.maximum(lower_bound, np.zeros_like(lower_bound))
                
                # Extend bounds to match boundary length (repeat last value)
                upper_y = np.r_[upper_bound, upper_bound[-1] if len(upper_bound) else np.nan]
                lower_y = np.r_[lower_bound, lower_bound[-1] if len(lower_bound) else np.nan]
                
                # Update y-axis limits to include uncertainty bounds
                if len(upper_bound) > 0 and np.any(upper_bound > 0):
                    positive_upper = upper_bound[upper_bound > 0]
                    y_max = max(y_max, np.max(positive_upper))
                
                # Add shaded uncertainty region with step behavior
                ax.fill_between(unc_data['boundaries'], lower_y, upper_y, 
                               color=unc_data['color'], alpha=0.2, step='post')
                
            except Exception as e:
                print(f"Warning: Could not add uncertainties for ZAID={Z}, MT={M}: {str(e)}")
    
    # Set y-axis to log scale if we have positive data spanning multiple orders of magnitude
    if y_min > 0 and y_max > y_min and (y_max / y_min) > 100:
        ax.set_yscale('log')
        use_y_log_scale = True
    
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
                mt_part += f" {reaction_name}"
            title_parts.append(mt_part)
        if title_parts:
            title = " ".join(title_parts) + " Cross Sections"
        else:
            title = "Multigroup Cross Sections"
    
    # Set appropriate y-label
    y_label = "Cross Section (barns)"
    
    # Format the axes
    ax = format_axes(
        ax=ax,
        style=style,
        use_log_scale=use_log_scale,
        is_energy_axis=use_log_scale,
        x_label="Energy (MeV)" if use_log_scale else "Group",
        y_label=y_label,
        title=title,
        legend_loc=legend_loc
    )
    
    # Apply y-axis log scale if determined
    if use_y_log_scale:
        ax.set_yscale('log')
    
    # Apply x-range like compare_uncertainties: set limits, don't pre-trim data
    if energy_range is not None:
        e_min, e_max = energy_range
        ax.set_xlim(e_min, e_max)
        
        # Recompute y-limits based on visible data region
        y_values_in_range = []
        
        # Get data from lines
        for line in ax.get_lines():
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            if xdata.size != ydata.size:
                continue
            mask = (xdata >= e_min) & (xdata <= e_max)
            if np.any(mask):
                y_values_in_range.extend(ydata[mask])
        
        # Also check PolyCollection objects (from fill_between for uncertainty bands)
        for collection in ax.collections:
            for path in collection.get_paths():
                vertices = path.vertices
                if len(vertices) > 0:
                    x_verts = vertices[:, 0]
                    y_verts = vertices[:, 1]
                    mask = (x_verts >= e_min) & (x_verts <= e_max)
                    if np.any(mask):
                        y_values_in_range.extend(y_verts[mask])
        
        if y_values_in_range:
            # Filter out non-positive values if using log scale
            if use_y_log_scale:
                y_values_in_range = [y for y in y_values_in_range if y > 0]
            
            if y_values_in_range:
                y_vals = np.asarray(y_values_in_range)
                y0, y1 = float(np.min(y_vals)), float(np.max(y_vals))
                
                if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
                    if use_y_log_scale and y0 > 0:
                        # For log scale, use multiplicative margin
                        log_range = np.log10(y1) - np.log10(y0)
                        if log_range > 0:
                            margin_factor = 10 ** (0.05 * log_range)
                            ax.set_ylim(y0 / margin_factor, y1 * margin_factor)
                    else:
                        # For linear scale, use additive margin
                        pad = 0.05 * (y1 - y0)
                        ax.set_ylim(y0 - pad, y1 + pad)
    else:
        # Remove default x-padding when using energy edges
        # Find min & max x from plotted data and tighten
        all_x = []
        for line in ax.get_lines():
            xd = np.asarray(line.get_xdata())
            if xd.size:
                all_x.append((np.min(xd), np.max(xd)))
        if all_x:
            xmin = min(v[0] for v in all_x)
            xmax = max(v[1] for v in all_x)
            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                ax.set_xlim(xmin, xmax)
    
    # Ensure figure is displayed only if requested
    if show:
        finalize_plot(fig)
    
    return fig
