import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Union, List, Optional, Tuple, Any
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from kika.cov.mf34_covmat import MF34CovMat
from kika._utils import zaid_to_symbol
from kika._plot_settings import setup_plot_style, format_axes


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
        label = f"{isotope_symbol} - (l={l_val})"
        if uncertainty_type == "absolute":
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
    scale: str = "log",      # "log" | "linear"
    energy_range: Optional[Tuple[float, float]] = None,
    **imshow_kwargs,
) -> plt.Figure:
    """
    Covariance/correlation heatmap with energy-proportional blocks (MF34) and optional
    uncertainty row. Energy labels are in eV (scientific, no units).  

    Parameters
    ----------
    scale : str, default "log"
        Energy axis scale. Options: "log" (default) or "linear".
    
    Updates:
    - Right-side energy labels respect inverted Y (high at top).
    - Log scale: label at each decade (1e+k); ticks at 2..9 within each decade (unlabeled).
    - Uncertainty panels: x ticks only (no labels), aligned with covariance; no per-panel titles.
    - Covariance: energy ticks on TOP/RIGHT (labels), and BOTTOM/LEFT (ticks only).
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.gridspec import GridSpec

    # ----------------------- helpers -----------------------
    def _fmt_sci_eV_full(v: float) -> str:
        """Scientific notation like '1e+02' (no unit suffix)."""
        return f"{v:.0e}"

    def _normalize_edges(edges, G_expected: int) -> Optional[np.ndarray]:
        e = np.asarray(edges, dtype=float).reshape(-1)
        if e.ndim != 1 or len(e) != G_expected + 1 or not np.all(np.isfinite(e)):
            return None
        e = e.copy()
        dif = np.diff(e)
        for i, d in enumerate(dif):
            if d <= 0:
                e[i+1] = e[i] + max(1e-300, 1e-12 * max(1.0, abs(e[i])))
        return e

    def _choose_scale(edges: np.ndarray, mode: str) -> str:
        """Validate and return the scale mode."""
        if mode not in ("log", "linear"):
            raise ValueError(f"scale must be 'log' or 'linear', got '{mode}'")
        # For log scale, check if edges are positive
        if mode == "log" and edges.min() <= 0:
            raise ValueError(f"scale='log' requires positive energy values, got min={edges.min():.2e}")
        return mode

    def _transform_edges(edges: np.ndarray, scale: str) -> np.ndarray:
        if scale == "log":
            e = np.maximum(edges, 1e-300)
            return np.log10(e)
        return edges.astype(float)

    def _concat_blocks(transformed_base: np.ndarray, n_blocks: int):
        base = transformed_base - transformed_base[0]
        w = base[-1] - base[0]
        per_block = []
        ranges = []
        start = 0.0
        for _ in range(n_blocks):
            b = base + start
            per_block.append(b)
            ranges.append((b[0], b[-1]))
            start += w
        stitched = np.concatenate([b[:-1] for b in per_block] + [per_block[-1][-1:]])
        return stitched, ranges, per_block

    def _log_ticks_with_decades(edges_one_block: np.ndarray):
        """Return (all_tick_local, labels_for_all) for LOG scale.
        - Ticks at mantissa m=1..9 of every decade in [Emin,Emax].
        - Labels when m==1 (1e+k) and also at Emin/Emax if they're not decade boundaries.
        Local coords start at 0 inside the block (log10 energy minus log10 Emin).
        """
        Emin = float(edges_one_block[0]); Emax = float(edges_one_block[-1])
        kmin = int(np.ceil(np.log10(max(Emin, 1e-300))))
        kmax = int(np.floor(np.log10(max(Emax, 1e-300))))
        if kmax < kmin:  # narrow window
            k = int(np.floor(np.log10(max(Emin, 1e-300))))
            ks = np.array([k])
        else:
            ks = np.arange(kmin, kmax + 1, dtype=int)

        vals = []
        labels = []
        
        # Check if Emin is a decade boundary (1e+k)
        is_emin_decade = np.isclose(Emin, 10.0 ** np.round(np.log10(Emin)), rtol=1e-10)
        # Check if Emax is a decade boundary (1e+k)
        is_emax_decade = np.isclose(Emax, 10.0 ** np.round(np.log10(Emax)), rtol=1e-10)
        
        # Add Emin if it's not a decade boundary
        if not is_emin_decade:
            vals.append(Emin)
            labels.append(_fmt_sci_eV_full(Emin))
        
        for k in ks:
            decade_base = 10.0 ** k
            for m in range(1, 10):
                v = m * decade_base
                if Emin <= v <= Emax:
                    vals.append(v)
                    labels.append(_fmt_sci_eV_full(v) if m == 1 else "")  # label only at 1e+k
        
        # Add Emax if it's not a decade boundary and not already added
        if not is_emax_decade and (len(vals) == 0 or not np.isclose(vals[-1], Emax, rtol=1e-10)):
            vals.append(Emax)
            labels.append(_fmt_sci_eV_full(Emax))
        
        vals = np.array(vals, dtype=float)
        if vals.size == 0:
            return np.array([]), []

        base0 = np.log10(max(Emin, 1e-300))
        pos = np.log10(vals) - base0
        return pos, labels

    def _linear_ticks_20pct(edges_one_block: np.ndarray):
        """20% increments on linear scale; label at all ticks (using eV sci)."""
        Emin = float(edges_one_block[0]); Emax = float(edges_one_block[-1])
        if Emax <= Emin:
            return np.array([]), []
        fracs = np.linspace(0.0, 1.0, 6)  # 0,20,40,60,80,100%
        vals = Emin + fracs * (Emax - Emin)
        labels = [_fmt_sci_eV_full(v) for v in vals]
        pos = vals - Emin
        return pos, labels

    def _energy_ticks(edges_one_block: np.ndarray, scale_pref: str):
        """Unified tick generator: returns (pos_local_all, labels_all)."""
        if _choose_scale(edges_one_block, scale_pref) == "log" and edges_one_block[0] > 0:
            return _log_ticks_with_decades(edges_one_block)
        # linear
        return _linear_ticks_20pct(edges_one_block)

    # ----------------------- validate & style -----------------------
    if matrix_type not in ["cov", "corr"]:
        raise ValueError(f"matrix_type must be 'cov' or 'corr', got '{matrix_type}'")

    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': font_family,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'figure.figsize': figsize,
        'figure.dpi': dpi,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.2,
        'lines.markersize': 7,
        'axes.grid': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'figure.constrained_layout.use': False,
    })
    if style == 'dark':
        plt.rcParams.update({
            'axes.facecolor': 'black', 'figure.facecolor': 'black', 'savefig.facecolor': 'black',
            'axes.edgecolor': 'white', 'axes.labelcolor': 'white',
            'xtick.color': 'white', 'ytick.color': 'white', 'text.color': 'white',
        })
    elif style == 'presentation':
        plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 16, 'lines.linewidth': 3.0})

    background_color = "#F5F5F5"
    tick_grey = "#707070"

    # ----------------------- data selection -----------------------
    filtered_mf34 = mf34_covmat.filter_by_isotope_reaction(isotope, mt)
    if filtered_mf34.num_matrices == 0:
        raise ValueError(f"No matrices found for isotope={isotope}, MT={mt}")

    all_triplets = filtered_mf34._get_param_triplets()
    available_legendre = sorted({t[2] for t in all_triplets if t[0] == isotope and t[1] == mt})
    if not available_legendre:
        raise ValueError(f"No Legendre coefficients found for isotope={isotope}, MT={mt}")

    G = filtered_mf34.matrices[0].shape[0] if filtered_mf34.matrices else 0
    if G == 0:
        raise ValueError("Number of energy groups (G) cannot be zero.")

    # parse legendre selection
    if isinstance(legendre_coeffs, tuple) and len(legendre_coeffs) == 2:
        row_l, col_l = legendre_coeffs
        if row_l not in available_legendre or col_l not in available_legendre:
            raise ValueError(f"Requested L not available for isotope={isotope}, MT={mt}")
        legendre_coeffs_sorted = [row_l, col_l]
        is_diagonal = False
    else:
        legendre_list = [legendre_coeffs] if isinstance(legendre_coeffs, int) \
                        else (list(legendre_coeffs) if legendre_coeffs else [])
        if not legendre_list:
            legendre_list = available_legendre
        for l_val in legendre_list:
            if l_val not in available_legendre:
                raise ValueError(f"Legendre coefficient L={l_val} not available for isotope={isotope}, MT={mt}")
        legendre_coeffs_sorted = sorted(set(legendre_list))
        is_diagonal = True

    single_block = (is_diagonal and len(legendre_coeffs_sorted) == 1) or (not is_diagonal)

    # matrix choice
    full_matrix_all = filtered_mf34.covariance_matrix if matrix_type == "cov" else filtered_mf34.correlation_matrix
    matrix_name = "Covariance" if matrix_type == "cov" else "Correlation"

    # energy edges (raw)
    energy_edges_raw = None
    if getattr(filtered_mf34, "energy_grids", None):
        energy_edges_raw = _normalize_edges(filtered_mf34.energy_grids[0], G)
    if energy_edges_raw is None:
        energy_edges_raw = np.linspace(1.0, float(G) + 1.0, G + 1)  # fallback 1..G+1 eV

    # energy cropping by range (include bins that overlap with the range)
    # A bin overlaps if its upper edge > emin AND its lower edge < emax
    if energy_range is not None:
        emin, emax = energy_range
        if not (np.isfinite(emin) and np.isfinite(emax)) or emin >= emax:
            raise ValueError("energy_range must be a tuple (emin, emax) with emin < emax.")
        # For each bin i: lower edge = energy_edges_raw[i], upper edge = energy_edges_raw[i+1]
        keep_mask = (energy_edges_raw[1:] > float(emin)) & (energy_edges_raw[:-1] < float(emax))
    else:
        keep_mask = np.ones(G, dtype=bool)
    keep_idx = np.where(keep_mask)[0]
    if keep_idx.size == 0:
        raise ValueError("energy_range removed all groups; nothing to plot.")

    edges_sel_raw = energy_edges_raw[keep_idx[0]: keep_idx[-1] + 2]

    # ----------------------- extract submatrix with energy crop -----------------------
    def _block_indices_for_L(L):
        t = all_triplets.index((isotope, mt, L))
        base = t * G
        return base + keep_idx

    if is_diagonal:
        gather = []
        for L in legendre_coeffs_sorted:
            gather.extend(_block_indices_for_L(L))
        M_full = full_matrix_all[np.ix_(gather, gather)]
    else:
        r_idx = _block_indices_for_L(row_l)
        c_idx = _block_indices_for_L(col_l)
        M_full = full_matrix_all[np.ix_(r_idx, c_idx)]

    # ----------------------- transformed coordinates -----------------------
    chosen_scale = _choose_scale(edges_sel_raw, scale)
    base_transformed = _transform_edges(edges_sel_raw, chosen_scale)

    if is_diagonal:
        n_blocks = len(legendre_coeffs_sorted)
        xedges, x_ranges, per_block_transformed = _concat_blocks(base_transformed, n_blocks)
        yedges = xedges.copy()
        y_ranges = x_ranges
    else:
        xedges = (base_transformed - base_transformed[0]).copy()
        yedges = (base_transformed - base_transformed[0]).copy()
        x_ranges = [(xedges[0], xedges[-1])]
        y_ranges = [(yedges[0], yedges[-1])]
        per_block_transformed = [xedges]

    # ----------------------- figure & axes -----------------------
    if show_uncertainties:
        if is_diagonal:
            ncols = len(legendre_coeffs_sorted)
            fig = plt.figure(figsize=(figsize[0], figsize[1] * (1 + 0.18)), dpi=dpi)
            gs = GridSpec(2, ncols, figure=fig, height_ratios=[0.16, 1.0], hspace=0.08, wspace=0.02)
            uncertainty_axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
            ax_heatmap = fig.add_subplot(gs[1, :])
        else:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * (1 + 0.16)), dpi=dpi)
            gs = GridSpec(2, 1, figure=fig, height_ratios=[0.16, 1.0], hspace=0.08)
            uncertainty_axes = [fig.add_subplot(gs[0, 0])]
            ax_heatmap = fig.add_subplot(gs[1, 0])
    else:
        plt.close('all')
        if ax is None:
            fig, ax_heatmap = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            ax_heatmap = ax
            fig = ax_heatmap.get_figure()
        uncertainty_axes = None

    # ----------------------- heatmap -----------------------
    M = np.ma.masked_where(M_full == 0.0, M_full)
    ax_heatmap.set_facecolor(background_color)
    ax_heatmap.grid(False, which="both")
    ax_heatmap.set_aspect('auto')

    if cmap is None:
        colormap = plt.get_cmap("RdYlGn").copy() if matrix_type == "corr" else plt.get_cmap("viridis").copy()
    else:
        colormap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    colormap.set_bad(color=background_color)

    if vmax is None or vmin is None:
        if matrix_type == "corr":
            absmax = np.nanmax(np.abs(M_full)) if M_full.size > 0 else 1.0
            vmax_calc = min(1.0, absmax) if vmax is None else vmax
            vmin_calc = max(-1.0, -absmax) if vmin is None else vmin
        else:
            matrix_max = np.nanmax(M_full) if M_full.size > 0 else 1.0
            vmax_calc = matrix_max if vmax is None else vmax
            vmin_calc = 0.0 if vmin is None else vmin
    else:
        vmax_calc, vmin_calc = vmax, vmin

    if np.isclose(vmin_calc, vmax_calc, atol=1e-10):
        if np.isclose(vmin_calc, 0.0, atol=1e-10):
            vmin_calc, vmax_calc = -1e-6, 1e-6
        else:
            pad = abs(vmax_calc) * 0.01
            vmin_calc, vmax_calc = vmax_calc - pad, vmax_calc + pad
    elif vmin_calc >= vmax_calc:
        vmin_calc = vmax_calc - (1e-6 if vmax_calc != 0 else 1e-6)

    norm = TwoSlopeNorm(vmin=vmin_calc, vcenter=0.0, vmax=vmax_calc) if matrix_type == "corr" else None

    plot_kwargs = {k: v for k, v in imshow_kwargs.items() if not k.startswith("_")}
    for bad in ("origin", "interpolation", "extent", "cmap"):
        plot_kwargs.pop(bad, None)

    X, Y = np.meshgrid(xedges, yedges)
    im = ax_heatmap.pcolormesh(X, Y, M, cmap=colormap, norm=norm, shading='flat', **plot_kwargs)

    ax_heatmap.set_xlim(xedges[0], xedges[-1])
    ax_heatmap.set_ylim(yedges[0], yedges[-1])
    ax_heatmap.invert_yaxis()  # image-style orientation

    # block boundaries
    if is_diagonal and len(legendre_coeffs_sorted) > 1:
        for i in range(1, len(legendre_coeffs_sorted)):
            edge = x_ranges[i][0]
            ax_heatmap.axvline(edge, color="black", lw=0.3)
            ax_heatmap.axhline(edge, color="black", lw=0.3)

    # L ticks on BOTH primary axes
    def _centers(ranges):
        return [(a+b)*0.5 for (a,b) in ranges]

    if single_block:
        if is_diagonal:
            centers = _centers([x_ranges[0]])
            ltxt = [str(legendre_coeffs_sorted[0])]
            ax_heatmap.set_xticks(centers); ax_heatmap.set_xticklabels(ltxt)
            ax_heatmap.set_yticks(centers); ax_heatmap.set_yticklabels(ltxt)
        else:
            centers = _centers([x_ranges[0]])
            ax_heatmap.set_xticks(centers); ax_heatmap.set_xticklabels([str(col_l)])
            ax_heatmap.set_yticks(centers); ax_heatmap.set_yticklabels([str(row_l)])
    else:
        centers = _centers(x_ranges)
        l_labels = [str(l) for l in legendre_coeffs_sorted]
        ax_heatmap.set_xticks(centers); ax_heatmap.set_xticklabels(l_labels)
        ax_heatmap.set_yticks(centers); ax_heatmap.set_yticklabels(l_labels)

    ax_heatmap.set_xlabel("Legendre coefficient L")
    ax_heatmap.set_ylabel("Legendre coefficient L")
    ax_heatmap.tick_params(axis="both", which="major", length=0, pad=5)

    # ----------------------- energy ticks (ALL sides) -----------------------
    # Build energy ticks in local coords for each block
    if is_diagonal:
        # one block's tick pattern reused
        pos_local_all, labels_all = _energy_ticks(edges_sel_raw, scale)
        # top/right (with labels); bottom/left (ticks only)
        top_ticks = []
        top_labels = []
        side_ticks = []
        side_labels = []
        for i, xr in enumerate(x_ranges):
            pos_g = pos_local_all + xr[0]
            # For top (x-axis): add ticks but filter labels
            top_ticks.extend(pos_g.tolist())
            # Filter out min and max labels for x-axis (top)
            filtered_top_labels = []
            for lbl in labels_all:
                # Remove first (min) and last (max) non-empty labels
                if lbl and (lbl == labels_all[0] or lbl == labels_all[-1]):
                    filtered_top_labels.append("")
                else:
                    filtered_top_labels.append(lbl)
            top_labels.extend(filtered_top_labels)
            
            # For right (y-axis): show max label only in the last block
            is_last_block = (i == len(x_ranges) - 1)
            side_ticks.extend(pos_g.tolist())
            filtered_side_labels = []
            for j, lbl in enumerate(labels_all):
                # Keep max label only in last block; for other blocks, remove it
                if lbl and lbl == labels_all[-1]:  # max label
                    filtered_side_labels.append(lbl if is_last_block else "")
                else:
                    filtered_side_labels.append(lbl)
            side_labels.extend(filtered_side_labels)
    else:
        pos_local_all, labels_all = _energy_ticks(edges_sel_raw, scale)
        pos_g = pos_local_all + xedges[0]
        top_ticks = pos_g.tolist()
        side_ticks = pos_g.tolist()
        
        # Filter labels for top (x-axis): remove min and max
        top_labels = []
        for lbl in labels_all:
            if lbl and (lbl == labels_all[0] or lbl == labels_all[-1]):
                top_labels.append("")
            else:
                top_labels.append(lbl)
        
        # For single block, keep all labels on right (y-axis)
        side_labels = labels_all

    # Create twin axes
    ax_top = ax_heatmap.twiny()
    ax_right = ax_heatmap.twinx()
    # Extra twins for bottom/left ticks (labels off)
    ax_bottom = ax_heatmap.twiny()
    ax_left = ax_heatmap.twinx()

    # --- TOP (labeled) ---
    ax_top.set_xlim(ax_heatmap.get_xlim())
    ax_top.set_xticks(top_ticks); ax_top.set_xticklabels(top_labels, fontsize=9, color=tick_grey)
    ax_top.tick_params(axis='x', direction='out', length=3, colors=tick_grey, pad=2, top=True, bottom=False)
    ax_top.spines['top'].set_visible(True); ax_top.spines['top'].set_linewidth(0.6); ax_top.spines['top'].set_color(tick_grey)
    ax_top.grid(False)

    # --- BOTTOM (ticks only) ---
    ax_bottom.set_xlim(ax_heatmap.get_xlim())
    ax_bottom.xaxis.set_ticks_position('bottom')
    ax_bottom.spines['bottom'].set_position(('outward', 0))
    ax_bottom.set_xticks(top_ticks); ax_bottom.set_xticklabels(['']*len(top_ticks))
    ax_bottom.tick_params(axis='x', direction='out', length=2, colors=tick_grey, pad=2, top=False, bottom=True)
    ax_bottom.spines['bottom'].set_visible(False)  # just tick marks
    ax_bottom.grid(False)

    # --- RIGHT (labeled, inverted-Y-aware) ---
    ax_right.set_ylim(ax_heatmap.get_ylim())
    # flip tick positions so high energy appears at TOP with inverted y
    y0, y1 = yedges[0], yedges[-1]
    # Mirror the tick positions for inverted y-axis
    right_ticks = [(y0 + y1) - t for t in side_ticks]
    # Mirror the labels: reverse the entire list so first label (low energy) goes to bottom (high y-value)
    right_labels = list(reversed(side_labels))
    ax_right.set_yticks(right_ticks); ax_right.set_yticklabels(right_labels, fontsize=9, color=tick_grey)
    ax_right.tick_params(axis='y', direction='out', length=3, colors=tick_grey, pad=2, right=True, left=False)
    ax_right.spines['right'].set_visible(True); ax_right.spines['right'].set_linewidth(0.6); ax_right.spines['right'].set_color(tick_grey)
    ax_right.grid(False)

    # --- LEFT (ticks only) ---
    ax_left.set_ylim(ax_heatmap.get_ylim())
    ax_left.yaxis.set_ticks_position('left')
    ax_left.spines['left'].set_position(('outward', 0))
    left_ticks = [(y0 + y1) - t for t in side_ticks]  # mirror same as right
    ax_left.set_yticks(left_ticks); ax_left.set_yticklabels(['']*len(left_ticks))
    ax_left.tick_params(axis='y', direction='out', length=2, colors=tick_grey, pad=2, left=True, right=False)
    ax_left.spines['left'].set_visible(False)
    ax_left.grid(False)

    # ----------------------- uncertainties -----------------------
    if show_uncertainties and uncertainty_axes is not None:
        diag_sqrt = np.sqrt(np.diag(filtered_mf34.covariance_matrix))

        def _sigma_pct_for_L(L: int) -> np.ndarray:
            t_idx = all_triplets.index((isotope, mt, L))
            return 100.0 * diag_sqrt[t_idx*G:(t_idx+1)*G][keep_idx]

        def _draw_ygrid_inside(ax_u, xr, ymax):
            top = float(np.ceil(ymax / 10.0) * 10.0)
            grid_vals = np.arange(0.0, top + 1e-9, 10.0)
            for y in grid_vals:
                ax_u.axhline(y, color=tick_grey, lw=0.6, alpha=0.35, zorder=0)
            x_label = xr[1] - 0.01 * (xr[1] - xr[0])
            for y in grid_vals:
                ax_u.text(x_label, y, f"{int(y)}", ha="right", va="center",
                          color=tick_grey, fontsize=8, alpha=0.9, zorder=2)

        # precompute energy ticks for uncertainty X (no labels)
        if is_diagonal:
            pos_local_u, _labels_ignore = _energy_ticks(edges_sel_raw, scale)
            for i, (ax_u, L, block_tx_edges, xr) in enumerate(zip(
                    uncertainty_axes, legendre_coeffs_sorted, per_block_transformed, x_ranges)):
                # Use LOCAL coordinates for this subplot (each subplot has its own coord system)
                # Use bin edges (not midpoints) for proper step plotting that covers full bins
                xs_edges_local = block_tx_edges - xr[0]
                sigma_pct = _sigma_pct_for_L(L)

                ax_u.set_facecolor(background_color)
                ax_u.grid(False)

                # draw line - only if data exists and is not all zeros
                # Use 'post' stepping: each value extends from its edge to the next edge (covers full bin)
                if sigma_pct.size > 0 and np.any(sigma_pct > 0):
                    ax_u.step(xs_edges_local[:-1], sigma_pct, where='post', linewidth=1.4, color=f"C{i}", zorder=3)
                    y_max = float(np.nanmax(sigma_pct))
                else:
                    # If no data or all zeros, set a reasonable default
                    y_max = 5.0
                
                # Ensure minimum y_max for visibility
                y_max = max(y_max, 1.0)
                pad = max(0.08 * y_max, 0.4)
                ax_u.set_ylim(0.0, np.ceil((y_max + pad)/10.0)*10.0)
                # Set xlim using local coordinates (0 to width)
                ax_u.set_xlim(0, xr[1] - xr[0])

                # x ticks aligned to energy ticks, but no labels (local coords)
                ax_u.set_xticks(pos_local_u)
                ax_u.set_xticklabels([])
                ax_u.tick_params(axis='x', direction='in', length=3, colors=tick_grey, pad=2)

                # y-axis inside 10% grid (kept) - use local x-range
                _draw_ygrid_inside(ax_u, (0, xr[1] - xr[0]), ax_u.get_ylim()[1])
                ax_u.set_yticks([])

                if i == 0:
                    ax_u.set_ylabel('Unc. (%)', fontsize=10, color='black')

                # clean spines; remove titles (per request)
                for side in ('left','right','top','bottom'):
                    ax_u.spines[side].set_visible(False)
        else:
            ax_u = uncertainty_axes[0]
            # Use bin edges (not midpoints) for proper step plotting that covers full bins
            sigma_pct = _sigma_pct_for_L(col_l)

            ax_u.set_facecolor(background_color); ax_u.grid(False)
            
            # draw line - only if data exists and is not all zeros
            # Use 'post' stepping: each value extends from its edge to the next edge (covers full bin)
            if sigma_pct.size > 0 and np.any(sigma_pct > 0):
                ax_u.step(xedges[:-1], sigma_pct, where='post', linewidth=1.4, color='C0', zorder=3)
                y_max = float(np.nanmax(sigma_pct))
            else:
                # If no data or all zeros, set a reasonable default
                y_max = 5.0
            
            # Ensure minimum y_max for visibility
            y_max = max(y_max, 1.0)
            pad = max(0.08 * y_max, 0.4)
            ax_u.set_ylim(0.0, np.ceil((y_max + pad)/10.0)*10.0)
            ax_u.set_xlim(xedges[0], xedges[-1])

            pos_local_u, _ = _energy_ticks(edges_sel_raw, scale)
            ax_u.set_xticks(pos_local_u + xedges[0])
            ax_u.set_xticklabels([])
            ax_u.tick_params(axis='x', direction='in', length=3, colors=tick_grey, pad=2)

            _draw_ygrid_inside(ax_u, (xedges[0], xedges[-1]), ax_u.get_ylim()[1])
            ax_u.set_yticks([])
            ax_u.set_ylabel('Unc. (%)', fontsize=10)
            for side in ('left','right','top','bottom'):
                ax_u.spines[side].set_visible(False)

    # ----------------------- title, layout, colorbar -----------------------
    if style not in {"paper", "publication"}:
        if single_block:
            ttl = (f"{zaid_to_symbol(isotope)} MT:{mt}   L={legendre_coeffs_sorted[0]} {matrix_name}"
                   if is_diagonal else f"{zaid_to_symbol(isotope)} MT:{mt}   L={row_l}-{col_l} {matrix_name}")
        else:
            ttl = f"{zaid_to_symbol(isotope)} MT:{mt}   L: {', '.join(str(l) for l in legendre_coeffs_sorted)} {matrix_name}"
        if show_uncertainties:
            fig.suptitle(ttl, y=0.985)
        else:
            ax_heatmap.set_title(ttl)

    num_legendre = len(legendre_coeffs_sorted) if is_diagonal else 1
    bottom_margin_config = {1: 0.12, 2: 0.15, 3: 0.16, 4: 0.17, 5: 0.18, 6: 0.19, 7: 0.20}
    bottom_margin = bottom_margin_config.get(num_legendre, bottom_margin_config[7])

    if show_uncertainties:
        fig.subplots_adjust(left=0.12, right=0.94, top=0.92, bottom=bottom_margin)
    else:
        fig.subplots_adjust(left=0.12, right=0.94, top=0.94, bottom=bottom_margin)

    fig.canvas.draw()
    pos = ax_heatmap.get_position()
    # push colorbar farther right (room for right labels)
    cbar_ax = fig.add_axes([pos.x1 + 0.10, pos.y0, 0.03, pos.height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(matrix_name)

    return fig
