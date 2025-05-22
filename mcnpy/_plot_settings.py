import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Dict, Optional, Tuple, Any

def setup_plot_style(
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 300,
    font_family: str = 'serif',
    **plot_kwargs
) -> Dict[str, Any]:
    """
    Setup matplotlib plotting style with publication-quality defaults.
    
    Parameters
    ----------
    style : str
        Plot style: 'default', 'dark', 'paper', 'publication', 'presentation'
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Dots per inch for figure resolution
    font_family : str
        Font family for text elements
    **plot_kwargs
        Additional kwargs that will be returned in the result
        
    Returns
    -------
    dict
        Dictionary with plot settings and kwargs
    """
    # Reset matplotlib settings to avoid style contamination
    plt.style.use('default')
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    # Set up publication-quality settings with white background
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
        'lines.linewidth': 2.5,  # Thicker lines by default
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
        'figure.constrained_layout.use': True,
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
        if 'alpha' not in plot_kwargs:
            plot_kwargs['alpha'] = 0.9
    elif style == 'paper' or style == 'publication':
        # Specific publication settings (maintaining white background)
        if 'linewidth' not in plot_kwargs:
            plot_kwargs['linewidth'] = 2.0
        if 'where' not in plot_kwargs and 'step' in str(plot_kwargs.get('_func', '')):
            plot_kwargs['where'] = 'mid'
    elif style == 'presentation':
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 16,
            'lines.linewidth': 3.0,  # Even thicker for presentations
        })
        if 'linewidth' not in plot_kwargs:
            plot_kwargs['linewidth'] = 3.0
    
    # Create figure and axes if not provided
    fig, ax = None, None
    if 'ax' not in plot_kwargs or plot_kwargs['ax'] is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        plot_kwargs['ax'] = ax
    
    # Get color palettes based on style
    if 'color' not in plot_kwargs:
        if style == 'paper' or style == 'publication':
            # Color-blind friendly palette that works well in print
            colors = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', 
                      '#CC78BC', '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9']
        else:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
        plot_kwargs['_colors'] = colors
    
    # Add linestyles for cycling
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
    plot_kwargs['_linestyles'] = linestyles
    
    # Store the style and figure for later use
    plot_kwargs['_style'] = style
    plot_kwargs['_fig'] = fig
    
    return plot_kwargs

def format_energy_axis_ticks(ax: plt.Axes) -> None:
    """
    Format the ticks on a log-scale energy axis to ensure readability.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes with log-scale energy x-axis
    """
    # Get current x-axis limits
    xmin, xmax = ax.get_xlim()
    
    # Determine appropriate major tick locations based on range
    if xmax / xmin > 1e6:
        # Very wide energy range - use order of magnitude ticks
        major_locator = mpl.ticker.LogLocator(base=10.0, numticks=10)
    else:
        # Narrower range - use more frequent ticks
        major_locator = mpl.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12)
    
    # Set major ticks
    ax.xaxis.set_major_locator(major_locator)
    
    # Use a formatter that shows enough precision for energy values
    formatter = mpl.ticker.FuncFormatter(lambda x, pos: f"{x:.3g}")
    ax.xaxis.set_major_formatter(formatter)
    
    # Add minor ticks
    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))

def format_axes(
    ax: plt.Axes,
    style: str,
    use_log_scale: bool = False,
    is_energy_axis: bool = False,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    legend_loc: str = 'best',
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> plt.Axes:
    """
    Format matplotlib axes for consistent appearance.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to format
    style : str
        Plot style being used
    use_log_scale : bool
        Whether to use logarithmic scale on x-axis
    is_energy_axis : bool
        Whether the x-axis represents energy (vs. group indices)
    x_label : str, optional
        Custom x-axis label
    y_label : str, optional
        Custom y-axis label
    title : str, optional
        Plot title
    legend_loc : str
        Location for the legend
    y_min : float, optional
        Minimum y value for axis limits
    y_max : float, optional
        Maximum y value for axis limits
        
    Returns
    -------
    plt.Axes
        The formatted axes
    """
    # Set x-axis scale and label
    if use_log_scale:
        ax.set_xscale('log')
        if x_label is None and is_energy_axis:
            ax.set_xlabel("Energy (MeV)")
        else:
            ax.set_xlabel(x_label or "")
        # Format the energy axis ticks for better readability
        format_energy_axis_ticks(ax)
    else:
        if x_label is None and not is_energy_axis:
            ax.set_xlabel("Energy-group index")
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        else:
            ax.set_xlabel(x_label or "")
    
    # Set y-axis label
    if y_label is not None:
        ax.set_ylabel(y_label)
        
    # Set title if provided
    if title is not None:
        ax.set_title(title)
    
    # Set y-axis limits if provided
    if y_min is not None and y_max is not None:
        padding = (y_max - y_min) * 0.05  # 5% padding
        ax.set_ylim(max(0, y_min - padding), y_max + padding)
    
    # Grid and legend formatting
    ax.grid(True, which='major', linestyle='--', alpha=0.4)
    if style == 'paper' or style == 'publication':
        ax.grid(True, which='minor', linestyle=':', alpha=0.25)
        legend = ax.legend(loc=legend_loc, frameon=True, framealpha=0.9, 
                           fancybox=False, edgecolor='black')
    else:
        legend = ax.legend(loc=legend_loc, framealpha=0.9, fancybox=True)
    
    return ax

def finalize_plot(fig: Optional[plt.Figure]) -> None:
    """
    Finalize a plot by ensuring it's displayed or saved as needed.
    
    Parameters
    ----------
    fig : plt.Figure or None
        Figure to finalize
    """
    if fig is not None:
        plt.show()
