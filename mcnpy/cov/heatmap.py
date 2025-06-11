import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Union, Sequence, List, Optional, Tuple, Any
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from mcnpy.cov.covmat import CovMat
from mcnpy._constants import MT_TO_REACTION
from mcnpy._utils import zaid_to_symbol


def _setup_energy_group_ticks(ax_heatmap, mts_sorted, G, matrix_size):
    """Helper function to setup energy group ticks and labels.
    - Tick lines are drawn only for 10th bin boundaries.
    - Labels for 10th/20th bins (not G) are at the tick line.
    - Label for G is always shown, centered in G's cell.
    """
    tick_line_positions = []  # For drawing physical tick lines
    label_info_list = []      # For storing {'pos': coordinate, 'label': string, 'mt_idx': int}
    
    num_mts = len(mts_sorted) # Still needed for label_offset_config

    # Determine label frequency based on number of MTs
    label_frequency = 20 if num_mts > 4 else 10

    for i in range(len(mts_sorted)): # Iterate over each MT block
        # Add tick lines for every 10th bin
        for g_line_idx in range(1, G):  # Index for lines between bins (1 to G-1)
            current_line_pos = i * G + g_line_idx - 0.5
            bin_number_after_line = g_line_idx + 1

            # Add tick lines for every 10th bin
            if bin_number_after_line % 10 == 0:
                tick_line_positions.append(current_line_pos)
            
            # Add labels based on label frequency
            if bin_number_after_line % label_frequency == 0:
                # Add label, unless it's the last bin G
                if bin_number_after_line != G:
                    label_info_list.append({'pos': current_line_pos, 'label': str(bin_number_after_line), 'mt_idx': i})
        
        # Always add label for G, at the center of G's cell.
        # If G is also a 10th/20th bin, ensure its preceding tick line is added.
        if G > 0:
            # The G-th bin (1-indexed) is the (G-1)-th cell (0-indexed) in this MT block.
            # Its 0-indexed position in the full matrix is (i*G + G-1). This is its center.
            g_cell_center_coord = i * G + (G - 1)
            label_info_list.append({'pos': g_cell_center_coord, 'label': str(G), 'mt_idx': i})

            # G is already labeled, but we need to ensure its tick line is present
            if G % 10 == 0: # If G is a 10th bin, its preceding line needs a tick
                # Line before G-th bin is at (G-1)-th position, so g_line_idx = G-1
                line_before_G_pos = i * G + (G - 1) - 0.5
                # Ensure line_before_G_pos is valid (i.e., G > 0, so G-1 >= 0)
                # and it's within the current MT block boundaries.
                # The smallest valid line_pos is i*G - 0.5 (before bin 1, if G>=1)
                if (G - 1) >= 0 and line_before_G_pos >= (i * G - 0.5) : 
                    tick_line_positions.append(line_before_G_pos)

    # Ensure uniqueness and sort positions
    tick_line_positions = sorted(list(set(tick_line_positions)))
    
    # label_info_list should not have duplicates for (label, mt_idx) due to the logic above.
    # (e.g. if G=10, it's only added by the G-specific block, centered)

    # Determine all unique positions involved for setting Matplotlib ticks
    all_involved_positions = sorted(list(set(tick_line_positions + [item['pos'] for item in label_info_list])))
    
    if all_involved_positions:
        current_xticks = list(ax_heatmap.get_xticks()) # Preserve MT center ticks
        current_yticks = list(ax_heatmap.get_yticks())
        ax_heatmap.set_xticks(current_xticks + all_involved_positions)
        ax_heatmap.set_yticks(current_yticks + all_involved_positions)
        
    # Custom draw the tick lines
    if tick_line_positions:
        tick_draw_length = 1.5 
        for tick_pos_val in tick_line_positions:
            # Draw custom longer tick marks
            ax_heatmap.plot([tick_pos_val, tick_pos_val], [-0.5, -0.5 + tick_draw_length], 
                           color='gray', linewidth=1.0, clip_on=False)  # Bottom edge
            ax_heatmap.plot([tick_pos_val, tick_pos_val], [matrix_size - 0.5, matrix_size - 0.5 - tick_draw_length], 
                           color='gray', linewidth=1.0, clip_on=False)  # Top edge
            ax_heatmap.plot([-0.5, -0.5 + tick_draw_length], [tick_pos_val, tick_pos_val], 
                           color='gray', linewidth=1.0, clip_on=False)  # Left edge
            ax_heatmap.plot([matrix_size - 0.5, matrix_size - 0.5 - tick_draw_length], [tick_pos_val, tick_pos_val], 
                           color='gray', linewidth=1.0, clip_on=False)  # Right edge
        
    # Label positioning configuration
    label_offset_config = {
        1: 1.5, 2: 3.5, 3: 4.8, 4: 6.5, 5: 8.5, 6: 10.5, 7: 12.5,
    }
    bottom_offset = label_offset_config.get(num_mts, label_offset_config[7])
    
    # Add labels for the selected ticks/positions
    if label_info_list:
        for item in label_info_list:
            text_coord = item['pos']
            bin_label_str = item['label']
            # mt_idx = item['mt_idx'] # Not directly needed for ax.text if text_coord is global

            ax_heatmap.text(text_coord, matrix_size + bottom_offset, bin_label_str, 
                           fontsize=8, color='gray', ha='center', va='bottom')
            ax_heatmap.text(-1.2, text_coord, bin_label_str, 
                           fontsize=8, color='gray', ha='right', va='center')


def plot_covariance_heatmap(
    covmat: "CovMat",
    zaid: int,
    mt: Union[int, Sequence[int], Tuple[int, int]],
    *,
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: Tuple[float, float] = (6, 6),
    dpi: int = 300,
    font_family: str = "serif",
    vmax: float | None = None,
    vmin: float | None = None,
    show_uncertainties: bool = True,
    show_energy_ticks: bool = True,
    show: bool = True,
    **imshow_kwargs,
) -> plt.Figure:
    """
    Draw a correlation-matrix heat-map for *one* isotope (`zaid`) and one
    or more MT reactions, with optional uncertainty plots shown above the heatmap columns.

    This is a self-contained version that doesn't depend on external plot settings modules.

    Parameters
    ----------
    covmat : CovMat
        The covariance matrix object
    zaid : int
        Isotope ID
    mt : int, sequence of int, or tuple of (row_mt, col_mt)
        MT reaction number(s). Can be:
        - Single int: diagonal block for that MT
        - Sequence of ints: diagonal blocks for those MTs  
        - Tuple of (row_mt, col_mt): off-diagonal block between row and column MT
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
        Whether to show energy group ticks and labels on the heatmap axes
    show : bool
        Whether to display the figure (default: True)
    **imshow_kwargs
        Additional arguments passed to imshow

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the heatmap and optional uncertainty plots
    """
    # Reset matplotlib to default state to avoid contamination from previous plots
    plt.rcdefaults()
    
    # Setup publication-quality settings with white background - embedded version
    # Note: Do NOT use constrained_layout when we'll be adding colorbars manually
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
        'figure.constrained_layout.use': False,  # Disable to avoid conflicts with colorbars
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

    # 1. Normalise MT input and locate the sub-matrix
    iso_cov = covmat.filter_by_isotope(zaid)
    pairs = iso_cov._get_param_pairs()
    G = iso_cov.num_groups

    if G == 0:
        raise ValueError("Number of groups (num_groups) cannot be zero.")

    # Handle different MT input formats
    if isinstance(mt, tuple) and len(mt) == 2:
        # Off-diagonal block case: (row_mt, col_mt)
        row_mt, col_mt = mt
        if not isinstance(row_mt, int) or not isinstance(col_mt, int):
            raise ValueError("For off-diagonal blocks, MT tuple must contain two integers: (row_mt, col_mt)")
        
        mts_sorted = [row_mt, col_mt]  # For backward compatibility with existing code
        is_diagonal = False
        
        # Get indices for the specific MTs
        row_pair = (zaid, row_mt)
        col_pair = (zaid, col_mt)
        
        try:
            row_mt_idx = pairs.index(row_pair)
        except ValueError:
            raise ValueError(f"(ZAID, MT)=({zaid}, {row_mt}) not present in CovMat")
            
        try:
            col_mt_idx = pairs.index(col_pair)
        except ValueError:
            raise ValueError(f"(ZAID, MT)=({zaid}, {col_mt}) not present in CovMat")
        
        mt_indices = [row_mt_idx, col_mt_idx]  # For backward compatibility
        
        # Get the specific off-diagonal block
        rows = list(range(row_mt_idx * G, (row_mt_idx + 1) * G))
        cols = list(range(col_mt_idx * G, (col_mt_idx + 1) * G))
        
    else:
        # Existing diagonal block logic
        if isinstance(mt, int):
            mts = [mt]
        else:
            mts = list(mt)
            if not mts:
                unique_mts = sorted(list(set(rxn for iso, rxn in pairs if iso == zaid)))
                if not unique_mts:
                    raise ValueError(f"No MT reactions found for ZAID={zaid}")
                mts = unique_mts
                
        mts_sorted = sorted(list(set(mts)))
        is_diagonal = True

        mt_indices = []
        for m_val in mts_sorted:
            pair = (zaid, m_val)
            try:
                mt_indices.append(pairs.index(pair))
            except ValueError:
                raise ValueError(f"(ZAID, MT)=({zaid}, {m_val}) not present in CovMat")

        if not mt_indices:
            raise ValueError(f"No valid MT data to plot for ZAID={zaid} with specified MTs.")

        rows = [i for idx in mt_indices for i in range(idx * G, (idx + 1) * G)]
        cols = rows  # Same for diagonal blocks
    
    if not rows or not cols:
        raise ValueError("Calculated rows/cols for matrix slicing are empty. Check MT numbers and data.")
    
    M_full = iso_cov.clipped_correlation_matrix[np.ix_(rows, cols)]

    # 2. Setup figure and layout based on whether we show uncertainties
    if show_uncertainties:
        # Create figure with uncertainty plots
        plt.ioff()  # Turn off interactive mode to prevent display issues
        
        # Fixed uncertainty height ratio
        uncertainty_height_ratio = 0.2
        # For off-diagonal blocks, show both row and column uncertainties
        num_reactions = 2 if not is_diagonal else len(mts_sorted)
        fig_height = figsize[1] * (1 + uncertainty_height_ratio)
        
        # Close any existing figures to prevent interference
        plt.close('all')
        
        fig = plt.figure(figsize=(figsize[0], fig_height), dpi=dpi)
        
        # Further reduce vertical spacing between uncertainty plots and heatmap
        gs = GridSpec(2, num_reactions, figure=fig,
                     height_ratios=[uncertainty_height_ratio, 1],
                     hspace=0.02, wspace=0.01)
        
        uncertainty_axes = []
        for i in range(num_reactions):
            ax_u = fig.add_subplot(gs[0, i])
            uncertainty_axes.append(ax_u)
            
        ax_heatmap = fig.add_subplot(gs[1, :])
    else:
        # Single heatmap without uncertainty plots - don't create any extra space or axes
        plt.close('all')  # Close any existing figures to prevent interference
        if ax is None:
            fig, ax_heatmap = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            ax_heatmap = ax
            fig = ax_heatmap.get_figure()
        uncertainty_axes = None

    # 3. Mask zeros
    M = np.ma.masked_where(M_full == 0.0, M_full)
    ax_heatmap.set_facecolor(background_color)
    ax_heatmap.grid(False, which="both")
    for axis_obj in (ax_heatmap.xaxis, ax_heatmap.yaxis):
        axis_obj.grid(False)

    # 4. Colour normalisation
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color=background_color)

    if vmax is None or vmin is None:
        absmax = np.nanmax(np.abs(M_full)) if M_full.size > 0 else 1.0
        vmax_calc = absmax if vmax is None else vmax
        vmin_calc = -absmax if vmin is None else vmin
    else:
        vmax_calc = vmax
        vmin_calc = vmin
        
    # Fix TwoSlopeNorm error by ensuring proper ordering and handling edge cases
    if np.isclose(vmin_calc, vmax_calc, atol=1e-10):
        # Handle case where all values are essentially the same
        if np.isclose(vmin_calc, 0.0, atol=1e-10):
            vmin_calc = -1e-6
            vmax_calc = 1e-6
        else:
            padding = abs(vmax_calc) * 0.01
            vmin_calc = vmax_calc - padding
            vmax_calc = vmax_calc + padding
    elif vmin_calc >= vmax_calc:
        vmin_calc = vmax_calc - 1e-6 if vmax_calc != 0 else -1e-6
        
    # Ensure vcenter is between vmin and vmax
    vcenter = 0.0
    if vcenter <= vmin_calc:
        vcenter = vmin_calc + (vmax_calc - vmin_calc) * 0.1
    elif vcenter >= vmax_calc:
        vcenter = vmax_calc - (vmax_calc - vmin_calc) * 0.1
        
    norm = TwoSlopeNorm(vmin=vmin_calc, vcenter=vcenter, vmax=vmax_calc)

    # 5. Draw the heatmap matrix
    plot_kwargs = {k: v for k, v in imshow_kwargs.items()
                   if not k.startswith("_") and k != "ax"}
    im = ax_heatmap.imshow(M, cmap=cmap, norm=norm,
                           origin="upper", interpolation="nearest", **plot_kwargs)

    if is_diagonal:
        matrix_size = len(mts_sorted) * G
        # Draw block boundaries
        for g_val in range(0, matrix_size + 1, G):
            ax_heatmap.axhline(g_val - 0.5, color="black", lw=0.2)
            ax_heatmap.axvline(g_val - 0.5, color="black", lw=0.2)
    else:
        # For off-diagonal blocks, just draw the outer boundary
        matrix_height = G
        matrix_width = G
        ax_heatmap.axhline(-0.5, color="black", lw=0.2)
        ax_heatmap.axhline(G - 0.5, color="black", lw=0.2)
        ax_heatmap.axvline(-0.5, color="black", lw=0.2)
        ax_heatmap.axvline(G - 0.5, color="black", lw=0.2)

    # 6. Ticks, labels for heatmap
    if is_diagonal:
        centres = [i * G + (G - 1) / 2 for i in range(len(mts_sorted))]
        mt_labels = [str(m_val) for m_val in mts_sorted]
        
        ax_heatmap.set_xticks(centres)
        ax_heatmap.set_yticks(centres)
        ax_heatmap.set_xticklabels(mt_labels, rotation=0, ha="center")
        ax_heatmap.set_yticklabels(mt_labels)
    else:
        # For off-diagonal blocks, show single MT labels centered
        center = (G - 1) / 2
        ax_heatmap.set_xticks([center])
        ax_heatmap.set_yticks([center])
        ax_heatmap.set_xticklabels([str(col_mt)], rotation=0, ha="center")
        ax_heatmap.set_yticklabels([str(row_mt)])
    
    # Move MT labels further from the heatmap to make room for energy bin ticks
    # Adjust padding based on whether energy ticks are shown
    pad_value = 15 if show_energy_ticks else 5 
    ax_heatmap.tick_params(axis="both", which="major", length=0, pad=pad_value)

    # Add energy group ticks using helper function (conditionally)
    if show_energy_ticks:
        if is_diagonal:
            _setup_energy_group_ticks(ax_heatmap, mts_sorted, G, matrix_size)
        else:
            # For off-diagonal blocks, use simplified energy ticks
            _setup_energy_group_ticks_single_block(ax_heatmap, G)

    ax_heatmap.set_xlabel("MT number")
    ax_heatmap.set_ylabel("MT number")

    if style not in {"paper", "publication"}:
        if is_diagonal:
            title_text = f"{zaid_to_symbol(zaid)}   MT: {', '.join(mt_labels)}"
        else:
            title_text = f"{zaid_to_symbol(zaid)}   MT: {row_mt}-{col_mt} block"
        if show_uncertainties:
            fig.suptitle(title_text)
        else:
            ax_heatmap.set_title(title_text)
    ax_heatmap.tick_params(axis="both", which="major", length=0)

    # 7. Draw uncertainty plots (if requested)
    if show_uncertainties and uncertainty_axes is not None:
        if iso_cov.energy_grid is not None and len(iso_cov.energy_grid) == G + 1:
            energy_grid_mids = [(iso_cov.energy_grid[i] + iso_cov.energy_grid[i+1]) / 2 for i in range(G)]
            xs = energy_grid_mids
            use_log_scale = True
        else:
            xs = list(range(1, G+1))
            use_log_scale = False
        
        diag_sqrt = np.sqrt(np.diag(iso_cov.covariance_matrix))
        line_props = {'color': 'black', 'linewidth': 0.2}

        if is_diagonal:
            # Process each subplot independently for diagonal case
            for i, (ax_uncert, mt_idx, mt_val_loop) in enumerate(zip(uncertainty_axes, mt_indices, mts_sorted)):
                sigma = diag_sqrt[mt_idx*G : (mt_idx+1)*G]
                sigma_pct = sigma * 100
                
                ax_uncert.set_facecolor(background_color)
                ax_uncert.grid(True, which="major", axis='y', linestyle='--', color='gray', alpha=0.3)
                ax_uncert.step(xs, sigma_pct, where='mid', linewidth=1.5, color=f'C{i}')
                
                if use_log_scale:
                    ax_uncert.set_xscale('log')
                
                y_data_min, y_data_max = 0, max(sigma_pct) if len(sigma_pct) > 0 else 5
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
            # For off-diagonal blocks, show both row and column MT uncertainties
            mt_indices_for_uncertainties = [row_mt_idx, col_mt_idx]
            mt_labels_for_uncertainties = [str(row_mt), str(col_mt)]
            
            for i, (ax_uncert, mt_idx, mt_label) in enumerate(zip(uncertainty_axes, mt_indices_for_uncertainties, mt_labels_for_uncertainties)):
                sigma = diag_sqrt[mt_idx*G : (mt_idx+1)*G]
                sigma_pct = sigma * 100
                
                ax_uncert.set_facecolor(background_color)
                ax_uncert.grid(True, which="major", axis='y', linestyle='--', color='gray', alpha=0.3)
                ax_uncert.step(xs, sigma_pct, where='mid', linewidth=1.5, color=f'C{i}')
                
                if use_log_scale:
                    ax_uncert.set_xscale('log')
                
                y_data_min, y_data_max = 0, max(sigma_pct) if len(sigma_pct) > 0 else 5
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
                
                # First plot gets y-axis label
                if i == 0:
                    ax_uncert.set_ylabel('% Unc.', fontsize='small')
                
                # Add title to identify which MT this uncertainty belongs to
                ax_uncert.set_title(f'MT {mt_label}', fontsize='small', pad=2)
                
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

    # 8. Layout configuration first - shared between both modes
    num_mts = 2 if (not is_diagonal and show_uncertainties) else (1 if not is_diagonal else len(mts_sorted))
    bottom_margin_config = {
        1: 0.12, 2: 0.15, 3: 0.16, 4: 0.17, 5: 0.18, 6: 0.19, 7: 0.20,
    }
    bottom_margin = bottom_margin_config.get(num_mts, bottom_margin_config[7])
    
    if show_uncertainties:
        plt.ion()
        fig.subplots_adjust(left=0.12, right=0.90, top=0.93, bottom=bottom_margin)
    else:
        fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=bottom_margin)

    # 9. Colour-bar - positioned after layout adjustment to get correct heatmap position
    fig.canvas.draw()
    heatmap_pos = ax_heatmap.get_position()
    
    # Use the same horizontal offset for both modes
    cbar_horizontal_offset = 0.03
    
    if show_uncertainties and uncertainty_axes is not None:
        # Create colorbar with spacing for uncertainty plots
        cbar_ax = fig.add_axes([
            heatmap_pos.x1 + cbar_horizontal_offset,
            heatmap_pos.y0,         # Start at actual heatmap bottom
            0.03,                   # Width
            heatmap_pos.height      # Full heatmap height
        ])
    else:
        # Create colorbar aligned with heatmap - with same horizontal offset
        cbar_ax = fig.add_axes([
            heatmap_pos.x1 + cbar_horizontal_offset,
            heatmap_pos.y0,         # Start at actual heatmap bottom
            0.03,                   # Same width
            heatmap_pos.height      # Full heatmap height
        ])
    
    fig.colorbar(im, cax=cbar_ax)
    
    if show:
        plt.show()

    # Always return the figure object for consistency
    return fig


def _setup_energy_group_ticks_offdiag(ax_heatmap, row_mts_sorted, col_mts_sorted, G, matrix_height, matrix_width):
    """Helper function to setup energy group ticks and labels for off-diagonal blocks."""
    tick_line_positions_x = []  # For column (x-axis) tick lines
    tick_line_positions_y = []  # For row (y-axis) tick lines
    label_info_list_x = []      # For column labels
    label_info_list_y = []      # For row labels
    
    num_row_mts = len(row_mts_sorted)
    num_col_mts = len(col_mts_sorted)

    # Determine label frequency based on number of MTs
    label_frequency_x = 20 if num_col_mts > 4 else 10
    label_frequency_y = 20 if num_row_mts > 4 else 10

    # Setup column (x-axis) ticks and labels
    for i in range(len(col_mts_sorted)):
        for g_line_idx in range(1, G):
            current_line_pos = i * G + g_line_idx - 0.5
            bin_number_after_line = g_line_idx + 1

            if bin_number_after_line % 10 == 0:
                tick_line_positions_x.append(current_line_pos)
            
            if bin_number_after_line % label_frequency_x == 0:
                if bin_number_after_line != G:
                    label_info_list_x.append({'pos': current_line_pos, 'label': str(bin_number_after_line), 'mt_idx': i})
        
        if G > 0:
            g_cell_center_coord = i * G + (G - 1)
            label_info_list_x.append({'pos': g_cell_center_coord, 'label': str(G), 'mt_idx': i})
            
            if G % 10 == 0:
                line_before_G_pos = i * G + (G - 1) - 0.5
                if (G - 1) >= 0 and line_before_G_pos >= (i * G - 0.5):
                    tick_line_positions_x.append(line_before_G_pos)

    # Setup row (y-axis) ticks and labels  
    for i in range(len(row_mts_sorted)):
        for g_line_idx in range(1, G):
            current_line_pos = i * G + g_line_idx - 0.5
            bin_number_after_line = g_line_idx + 1

            if bin_number_after_line % 10 == 0:
                tick_line_positions_y.append(current_line_pos)
            
            if bin_number_after_line % label_frequency_y == 0:
                if bin_number_after_line != G:
                    label_info_list_y.append({'pos': current_line_pos, 'label': str(bin_number_after_line), 'mt_idx': i})
        
        if G > 0:
            g_cell_center_coord = i * G + (G - 1)
            label_info_list_y.append({'pos': g_cell_center_coord, 'label': str(G), 'mt_idx': i})
            
            if G % 10 == 0:
                line_before_G_pos = i * G + (G - 1) - 0.5
                if (G - 1) >= 0 and line_before_G_pos >= (i * G - 0.5):
                    tick_line_positions_y.append(line_before_G_pos)

    # Ensure uniqueness and sort positions
    tick_line_positions_x = sorted(list(set(tick_line_positions_x)))
    tick_line_positions_y = sorted(list(set(tick_line_positions_y)))
    
    # Set up matplotlib ticks
    all_involved_positions_x = sorted(list(set(tick_line_positions_x + [item['pos'] for item in label_info_list_x])))
    all_involved_positions_y = sorted(list(set(tick_line_positions_y + [item['pos'] for item in label_info_list_y])))
    
    if all_involved_positions_x or all_involved_positions_y:
        current_xticks = list(ax_heatmap.get_xticks())
        current_yticks = list(ax_heatmap.get_yticks())
        if all_involved_positions_x:
            ax_heatmap.set_xticks(current_xticks + all_involved_positions_x)
        if all_involved_positions_y:
            ax_heatmap.set_yticks(current_yticks + all_involved_positions_y)
        
    # Draw custom tick lines
    tick_draw_length = 1.5
    
    # X-axis (column) tick lines
    for tick_pos_val in tick_line_positions_x:
        ax_heatmap.plot([tick_pos_val, tick_pos_val], [-0.5, -0.5 + tick_draw_length], 
                       color='gray', linewidth=1.0, clip_on=False)  # Bottom edge
        ax_heatmap.plot([tick_pos_val, tick_pos_val], [matrix_height - 0.5, matrix_height - 0.5 - tick_draw_length], 
                       color='gray', linewidth=1.0, clip_on=False)  # Top edge
    
    # Y-axis (row) tick lines
    for tick_pos_val in tick_line_positions_y:
        ax_heatmap.plot([-0.5, -0.5 + tick_draw_length], [tick_pos_val, tick_pos_val], 
                       color='gray', linewidth=1.0, clip_on=False)  # Left edge
        ax_heatmap.plot([matrix_width - 0.5, matrix_width - 0.5 - tick_draw_length], [tick_pos_val, tick_pos_val], 
                       color='gray', linewidth=1.0, clip_on=False)  # Right edge
        
    # Label positioning configuration
    label_offset_config = {
        1: 1.5, 2: 3.5, 3: 4.8, 4: 6.5, 5: 8.5, 6: 10.5, 7: 12.5,
    }
    bottom_offset = label_offset_config.get(num_col_mts, label_offset_config[7])
    left_offset = label_offset_config.get(num_row_mts, label_offset_config[7])
    
    # Add column (x-axis) labels
    for item in label_info_list_x:
        text_coord = item['pos']
        bin_label_str = item['label']
        ax_heatmap.text(text_coord, matrix_height + bottom_offset, bin_label_str, 
                       fontsize=8, color='gray', ha='center', va='bottom')
    
    # Add row (y-axis) labels  
    for item in label_info_list_y:
        text_coord = item['pos']
        bin_label_str = item['label']
        ax_heatmap.text(-1.2, text_coord, bin_label_str, 
                       fontsize=8, color='gray', ha='right', va='center')


def _setup_energy_group_ticks_single_block(ax_heatmap, G):
    """Helper function to setup energy group ticks and labels for single off-diagonal block."""
    tick_line_positions = []
    label_info_list = []
    
    # Determine label frequency
    label_frequency = 20 if G > 100 else 10

    # Add tick lines and labels for this single block
    for g_line_idx in range(1, G):
        current_line_pos = g_line_idx - 0.5
        bin_number_after_line = g_line_idx + 1

        if bin_number_after_line % 10 == 0:
            tick_line_positions.append(current_line_pos)
        
        if bin_number_after_line % label_frequency == 0:
            if bin_number_after_line != G:
                label_info_list.append({'pos': current_line_pos, 'label': str(bin_number_after_line)})
    
    # Always add label for G, at the center of G's cell
    if G > 0:
        g_cell_center_coord = G - 1
        label_info_list.append({'pos': g_cell_center_coord, 'label': str(G)})

        if G % 10 == 0:
            line_before_G_pos = (G - 1) - 0.5
            if line_before_G_pos >= -0.5:
                tick_line_positions.append(line_before_G_pos)

    # Set up matplotlib ticks
    all_involved_positions = sorted(list(set(tick_line_positions + [item['pos'] for item in label_info_list])))
    
    if all_involved_positions:
        current_xticks = list(ax_heatmap.get_xticks())
        current_yticks = list(ax_heatmap.get_yticks())
        ax_heatmap.set_xticks(current_xticks + all_involved_positions)
        ax_heatmap.set_yticks(current_yticks + all_involved_positions)
        
    # Draw custom tick lines
    if tick_line_positions:
        tick_draw_length = 1.5
        for tick_pos_val in tick_line_positions:
            ax_heatmap.plot([tick_pos_val, tick_pos_val], [-0.5, -0.5 + tick_draw_length], 
                           color='gray', linewidth=1.0, clip_on=False)
            ax_heatmap.plot([tick_pos_val, tick_pos_val], [G - 0.5, G - 0.5 - tick_draw_length], 
                           color='gray', linewidth=1.0, clip_on=False)
            ax_heatmap.plot([-0.5, -0.5 + tick_draw_length], [tick_pos_val, tick_pos_val], 
                           color='gray', linewidth=1.0, clip_on=False)
            ax_heatmap.plot([G - 0.5, G - 0.5 - tick_draw_length], [tick_pos_val, tick_pos_val], 
                           color='gray', linewidth=1.0, clip_on=False)
        
    # Add labels
    bottom_offset = 1.5
    for item in label_info_list:
        text_coord = item['pos']
        bin_label_str = item['label']
        ax_heatmap.text(text_coord, G + bottom_offset, bin_label_str, 
                       fontsize=8, color='gray', ha='center', va='bottom')
        ax_heatmap.text(-1.2, text_coord, bin_label_str, 
                       fontsize=8, color='gray', ha='right', va='center')
