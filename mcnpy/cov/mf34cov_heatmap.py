import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Union, List, Optional, Tuple, Any
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from mcnpy.cov.mf34_covmat import MF34CovMat
from mcnpy._utils import zaid_to_symbol


def plot_mf34_covariance_heatmap(
    mf34_covmat: MF34CovMat,
    isotope: int,
    mt: int,
    legendre_coeffs: Union[int, List[int], Tuple[int, int]],
    *,
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: Tuple[float, float] = (6, 6),
    dpi: int = 300,
    font_family: str = "serif",
    vmax: float | None = None,
    vmin: float | None = None,
    show_uncertainties: bool = True,
    show_energy_ticks: bool = False,
    cmap: any = None,
    **imshow_kwargs,
) -> Union[plt.Axes, Tuple[plt.Axes, List[plt.Axes]]]:
    """
    Draw a correlation-matrix heat-map for MF34 angular distribution covariance data.

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
    show_energy_ticks : bool
        Whether to show subtle energy group tick marks at the heatmap borders
    **imshow_kwargs
        Additional arguments passed to imshow

    Returns
    -------
    plt.Axes or tuple
        If show_uncertainties=False: returns the heatmap axes
        If show_uncertainties=True: returns (heatmap_axes, uncertainty_axes_list)
    """
    # Reset matplotlib to default state
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

    # Filter and get available Legendre coefficients
    filtered_mf34 = mf34_covmat.filter_by_isotope_reaction(isotope, mt)
    available_legendre = filtered_mf34._get_legendre_pairs(isotope, mt)
    
    if not available_legendre:
        raise ValueError(f"No Legendre coefficients found for isotope={isotope}, MT={mt}")

    # Handle different Legendre coefficient input formats
    if isinstance(legendre_coeffs, tuple) and len(legendre_coeffs) == 2:
        # Off-diagonal block case
        row_l, col_l = legendre_coeffs
        if not isinstance(row_l, int) or not isinstance(col_l, int):
            raise ValueError("For off-diagonal blocks, legendre_coeffs tuple must contain two integers: (row_l, col_l)")
        
        legendre_sorted = [row_l, col_l]
        is_diagonal = False
        
        # Get the specific off-diagonal block
        try:
            M_full, energy_grid = filtered_mf34.get_matrix_for_legendre_pair(isotope, mt, row_l, col_l)
            G = M_full.shape[0]
        except ValueError:
            raise ValueError(f"No matrix found for L_row={row_l}, L_col={col_l}")
            
    else:
        # Diagonal block logic
        if isinstance(legendre_coeffs, int):
            legendre_list = [legendre_coeffs]
        else:
            legendre_list = list(legendre_coeffs)
            
        # Validate requested Legendre coefficients
        for l_val in legendre_list:
            if l_val not in available_legendre:
                raise ValueError(f"Legendre coefficient L={l_val} not available for isotope={isotope}, MT={mt}")
        
        legendre_sorted = sorted(legendre_list)
        is_diagonal = True
        
        # Build full correlation matrix with proper energy grid handling
        M_full, energy_grid = filtered_mf34.build_full_correlation_matrix(isotope, mt, legendre_sorted)
        # Get G from the first diagonal block to ensure consistency
        first_matrix, _ = filtered_mf34.get_matrix_for_legendre_pair(isotope, mt, legendre_sorted[0], legendre_sorted[0])
        G = first_matrix.shape[0]

    # Setup figure and layout based on whether we show uncertainties
    if show_uncertainties:
        # Create figure with uncertainty plots
        plt.ioff()
        
        uncertainty_height_ratio = 0.2
        num_reactions = 2 if not is_diagonal else len(legendre_sorted)
        fig_height = figsize[1] * (1 + uncertainty_height_ratio)
        
        plt.close('all')
        
        fig = plt.figure(figsize=(figsize[0], fig_height), dpi=dpi)
        
        gs = GridSpec(2, num_reactions, figure=fig,
                     height_ratios=[uncertainty_height_ratio, 1],
                     hspace=0.05, wspace=0.01)
        
        uncertainty_axes = []
        for i in range(num_reactions):
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

    # Mask zeros and setup display
    M = np.ma.masked_where(M_full == 0.0, M_full)
    ax_heatmap.set_facecolor(background_color)
    ax_heatmap.grid(False, which="both")
    for axis_obj in (ax_heatmap.xaxis, ax_heatmap.yaxis):
        axis_obj.grid(False)

    # Color normalization

    if cmap is None:
        cmap = plt.get_cmap("RdYlGn").copy()
    else:
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap).copy()
        else:
            cmap = cmap
    cmap.set_bad(color=background_color)

    if vmax is None or vmin is None:
        absmax = np.nanmax(np.abs(M_full)) if M_full.size > 0 else 1.0
        vmax_calc = absmax if vmax is None else vmax
        vmin_calc = -absmax if vmin is None else vmin
    else:
        vmax_calc = vmax
        vmin_calc = vmin
        
    # Handle edge cases for TwoSlopeNorm
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
        
    vcenter = 0.0
    if vcenter <= vmin_calc:
        vcenter = vmin_calc + (vmax_calc - vmin_calc) * 0.1
    elif vcenter >= vmax_calc:
        vcenter = vmax_calc - (vmax_calc - vmin_calc) * 0.1
        
    norm = TwoSlopeNorm(vmin=vmin_calc, vcenter=vcenter, vmax=vmax_calc)

    # Draw the heatmap
    plot_kwargs = {k: v for k, v in imshow_kwargs.items()
                   if not k.startswith("_") and k != "ax"}
    im = ax_heatmap.imshow(M, cmap=cmap, norm=norm,
                           origin="upper", interpolation="nearest", **plot_kwargs)

    if is_diagonal:
        matrix_size = len(legendre_sorted) * G
        # Draw block boundaries
        for g_val in range(0, matrix_size + 1, G):
            ax_heatmap.axhline(g_val - 0.5, color="black", lw=0.2)
            ax_heatmap.axvline(g_val - 0.5, color="black", lw=0.2)
    else:
        # For off-diagonal blocks, draw outer boundary
        ax_heatmap.axhline(-0.5, color="black", lw=0.2)
        ax_heatmap.axhline(G - 0.5, color="black", lw=0.2)
        ax_heatmap.axvline(-0.5, color="black", lw=0.2)
        ax_heatmap.axvline(G - 0.5, color="black", lw=0.2)

    # Setup ticks and labels
    if show_energy_ticks:
        # Add subtle energy group tick marks at borders only
        if is_diagonal:
            # For diagonal blocks, add minor ticks for energy groups within each Legendre block
            energy_ticks_minor = []
            for l_idx in range(len(legendre_sorted)):
                for g_idx in range(G):
                    energy_ticks_minor.append(l_idx * G + g_idx)
            
            # Set minor ticks for energy groups (subtle marks)
            ax_heatmap.set_xticks(energy_ticks_minor, minor=True)
            ax_heatmap.set_yticks(energy_ticks_minor, minor=True)
            ax_heatmap.tick_params(axis="both", which="minor", length=2, width=0.5, 
                                 color='gray', alpha=0.6, direction='out')
            
            # Keep main ticks for Legendre coefficients
            centres = [i * G + (G - 1) / 2 for i in range(len(legendre_sorted))]
            l_labels = [str(l_val) for l_val in legendre_sorted]
            
            ax_heatmap.set_xticks(centres)
            ax_heatmap.set_yticks(centres)
            ax_heatmap.set_xticklabels(l_labels, rotation=0, ha="center")
            ax_heatmap.set_yticklabels(l_labels)
        else:
            # For off-diagonal blocks, add minor ticks for energy groups
            energy_ticks_minor = list(range(G))
            ax_heatmap.set_xticks(energy_ticks_minor, minor=True)
            ax_heatmap.set_yticks(energy_ticks_minor, minor=True)
            ax_heatmap.tick_params(axis="both", which="minor", length=2, width=0.5, 
                                 color='gray', alpha=0.6, direction='out')
            
            # Keep main ticks for Legendre coefficients
            center = (G - 1) / 2
            ax_heatmap.set_xticks([center])
            ax_heatmap.set_yticks([center])
            ax_heatmap.set_xticklabels([str(col_l)], rotation=0, ha="center")
            ax_heatmap.set_yticklabels([str(row_l)])
        
        ax_heatmap.tick_params(axis="both", which="major", length=0, pad=5)
        ax_heatmap.set_xlabel("Legendre coefficient")
        ax_heatmap.set_ylabel("Legendre coefficient")
    else:
        # Show only Legendre coefficient labels (current behavior)
        if is_diagonal:
            centres = [i * G + (G - 1) / 2 for i in range(len(legendre_sorted))]
            l_labels = [str(l_val) for l_val in legendre_sorted]
            
            ax_heatmap.set_xticks(centres)
            ax_heatmap.set_yticks(centres)
            ax_heatmap.set_xticklabels(l_labels, rotation=0, ha="center")
            ax_heatmap.set_yticklabels(l_labels)
        else:
            # For off-diagonal blocks
            center = (G - 1) / 2
            ax_heatmap.set_xticks([center])
            ax_heatmap.set_yticks([center])
            ax_heatmap.set_xticklabels([str(col_l)], rotation=0, ha="center")
            ax_heatmap.set_yticklabels([str(row_l)])
        
        ax_heatmap.tick_params(axis="both", which="major", length=0, pad=5)
        ax_heatmap.set_xlabel("Legendre coefficient")
        ax_heatmap.set_ylabel("Legendre coefficient")

    # Set title
    if style not in {"paper", "publication"}:
        if is_diagonal:
            title_text = f"{zaid_to_symbol(isotope)} MT:{mt}   L: {', '.join([str(l) for l in legendre_sorted])}"
        else:
            title_text = f"{zaid_to_symbol(isotope)} MT:{mt}   L: {row_l}-{col_l} block"
        if show_uncertainties:
            fig.suptitle(title_text)
        else:
            ax_heatmap.set_title(title_text)

    # Draw uncertainty plots (if requested)
    if show_uncertainties and uncertainty_axes is not None:
        # Get energy grid - use the one from the first block for x-axis
        first_l = legendre_sorted[0] if is_diagonal else row_l
        _, block_energy_grid = filtered_mf34.get_matrix_for_legendre_pair(isotope, mt, first_l, first_l)
        
        if len(block_energy_grid) == G + 1:
            energy_grid_mids = [(block_energy_grid[i] + block_energy_grid[i+1]) / 2 for i in range(G)]
            xs = energy_grid_mids
            use_log_scale = True
        else:
            xs = list(range(1, G+1))
            use_log_scale = False
        
        line_props = {'color': 'lightgray', 'linewidth': 0.1, 'alpha': 0.5}

        if is_diagonal:
            # Process each Legendre coefficient independently for diagonal case
            for i, (ax_uncert, l_val) in enumerate(zip(uncertainty_axes, legendre_sorted)):
                # Get diagonal covariance for this L coefficient
                try:
                    diag_matrix, _ = filtered_mf34.get_matrix_for_legendre_pair(isotope, mt, l_val, l_val)
                    sigma = np.sqrt(np.diag(diag_matrix))
                    sigma_pct = sigma * 100
                except ValueError:
                    # If no diagonal matrix, create zeros
                    sigma_pct = np.zeros(G)
                
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
                ax_uncert.tick_params(axis='x', which='minor', bottom=False, top=False)

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
                try:
                    diag_matrix, _ = filtered_mf34.get_matrix_for_legendre_pair(isotope, mt, l_val, l_val)
                    sigma = np.sqrt(np.diag(diag_matrix))
                    sigma_pct = sigma * 100
                except ValueError:
                    sigma_pct = np.zeros(G)
                
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
                
                if i == 0:
                    ax_uncert.set_ylabel('% Unc.', fontsize='small')
                
                # Remove individual titles that might cause visual artifacts
                # Instead, we'll rely on the main figure title and positioning
                
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
                
                # Add subtle label for L coefficient in corner
                if not is_diagonal:
                    ax_uncert.text(
                        0.02, 0.85, f'L {l_label}', 
                        transform=ax_uncert.transAxes,
                        fontsize='x-small', 
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
                    )

    # Layout and colorbar
    num_legendre = 2 if not is_diagonal else len(legendre_sorted)
    bottom_margin_config = {
        1: 0.12, 2: 0.15, 3: 0.16, 4: 0.17, 5: 0.18, 6: 0.19, 7: 0.20,
    }
    bottom_margin = bottom_margin_config.get(num_legendre, bottom_margin_config[7])
    
    if show_uncertainties:
        plt.ion()
        fig.subplots_adjust(left=0.12, right=0.90, top=0.93, bottom=bottom_margin)
    else:
        fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=bottom_margin)

    # Add colorbar
    fig.canvas.draw()
    heatmap_pos = ax_heatmap.get_position()
    cbar_ax = fig.add_axes([
        heatmap_pos.x1 + 0.03,
        heatmap_pos.y0,
        0.03,
        heatmap_pos.height
    ])
    
    fig.colorbar(im, cax=cbar_ax)
    
    plt.show()

    # Return appropriate result
    if show_uncertainties:
        return (ax_heatmap, uncertainty_axes)
    else:
        return ax_heatmap
