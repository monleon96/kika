import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Dict, Optional, Tuple, Any

def _is_notebook():
    """Check if code is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return False
        # Check if we're in a notebook environment
        return hasattr(ipython, 'kernel')
    except ImportError:
        return False

def _detect_interactive_backend():
    """
    Detect if matplotlib is using an interactive backend suitable for widgets.
    
    Returns
    -------
    bool
        True if using widget/interactive backend, False otherwise
    """
    current_backend = plt.get_backend()
    
    # Widget backends that support full interactivity
    widget_backends = [
        'module://ipympl.backend_nbagg',  # ipympl widget backend
        'widget',                         # Alternative ipympl backend name
        'Qt5Agg', 'Qt4Agg', 'TkAgg',     # Desktop interactive backends
        'nbAgg'                           # Notebook backend (limited interactivity)
    ]
    
    # Check if current backend supports interactivity
    is_interactive = current_backend in widget_backends
    
    # Additional check for ipympl specifically
    if current_backend in ['module://ipympl.backend_nbagg', 'widget']:
        try:
            import ipympl
            return True
        except ImportError:
            return False
    
    return is_interactive

def _setup_notebook_backend():
    """Setup appropriate matplotlib backend for notebooks."""
    if not _is_notebook():
        return False, "Not in notebook"
    
    current_backend = plt.get_backend()
    
    # If already using widget backend, don't change it
    if current_backend in ['module://ipympl.backend_nbagg', 'widget']:
        try:
            import ipympl
            return True, f"ipympl widget backend ({current_backend})"
        except ImportError:
            pass
    
    try:
        # Try ipympl first (recommended for interactive plots)
        import ipympl
        if current_backend != 'module://ipympl.backend_nbagg':
            plt.switch_backend('widget')
        return True, "ipympl widget backend"
    except ImportError:
        try:
            # Fallback to notebook backend
            if current_backend != 'nbAgg':
                plt.switch_backend('notebook')
            return True, "notebook backend"
        except Exception:
            # Last resort - inline
            plt.switch_backend('inline')
            return False, "inline backend (no interactivity)"

def _configure_figure_interactivity(fig, interactive_mode: bool):
    """
    Configure figure interactivity based on the mode.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object
    interactive_mode : bool
        Whether to enable interactive features
    """
    if interactive_mode:
        # Enable interactive features for widget backends
        try:
            # Enable toolbar if available
            if hasattr(fig.canvas, 'toolbar_visible'):
                fig.canvas.toolbar_visible = True
        except Exception:
            # Silently fall back if interactive features aren't available
            pass
    else:
        # Disable interactive features for static backends
        try:
            if hasattr(fig.canvas, 'toolbar_visible'):
                fig.canvas.toolbar_visible = False
        except Exception:
            pass

def setup_plot_style(
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    font_family: str = 'serif',
    notebook_mode: bool = None,
    interactive: bool = None,
    projection: Optional[str] = None,  # New parameter for 3D plots
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
        Dots per inch for figure resolution (auto-adjusted for notebooks)
    font_family : str
        Font family for text elements
    notebook_mode : bool, optional
        Force notebook mode (auto-detected if None)
    interactive : bool, optional
        Force interactive mode (auto-detected if None)
    projection : str, optional
        Matplotlib projection type (e.g., '3d' for 3D plots)
    **plot_kwargs
        Additional kwargs that will be returned in the result
        
    Returns
    -------
    dict
        Dictionary with plot settings and kwargs including:
        - 'ax': matplotlib axes object
        - '_fig': matplotlib figure object
        - '_interactive': boolean indicating if interactive mode is active
        - '_backend_info': string describing the backend
        - '_colors': list of colors for plotting
        - '_linestyles': list of linestyles for plotting
    """
    # Auto-detect notebook and interactive mode
    if notebook_mode is None:
        notebook_mode = _is_notebook()
    
    if interactive is None:
        interactive = notebook_mode and _detect_interactive_backend()
    
    # Adjust settings for notebook environment and interactivity
    if notebook_mode:
        # Adjust DPI and figure size for notebooks
        if dpi > 120:
            dpi = 90 if interactive else 120
        
        # More aggressive figure size reduction for notebooks
        if figsize[0] > 10 or figsize[1] > 7:
            scale = min(8/figsize[0], 5/figsize[1])
            figsize = (figsize[0] * scale, figsize[1] * scale)
        elif figsize[0] > 8 or figsize[1] > 6:
            scale = min(7/figsize[0], 5/figsize[1])
            figsize = (figsize[0] * scale, figsize[1] * scale)
        
        # Setup interactive backend if needed
        if interactive:
            backend_success, backend_msg = _setup_notebook_backend()
            plot_kwargs['_backend_info'] = backend_msg
            plot_kwargs['_interactive'] = backend_success
        else:
            plot_kwargs['_backend_info'] = f"Notebook mode, backend: {plt.get_backend()}"
            plot_kwargs['_interactive'] = False
    else:
        plot_kwargs['_backend_info'] = f"Non-notebook environment, backend: {plt.get_backend()}"
        plot_kwargs['_interactive'] = interactive
    
    # Reset matplotlib settings to avoid style contamination
    plt.close('all')
    mpl.rcdefaults()
    plt.style.use('default')
    
    # Set up publication-quality settings
    # For 3D plots and non-interactive backends, avoid constrained_layout
    use_constrained_layout = (projection != '3d' and interactive)
    
    plt.rcParams.update({
        'font.family': font_family,
        'font.size': 11 if notebook_mode else 12,
        'axes.labelsize': 12 if notebook_mode else 14,
        'axes.titlesize': 13 if notebook_mode else 14,
        'xtick.labelsize': 10 if notebook_mode else 12,
        'ytick.labelsize': 10 if notebook_mode else 12,
        'legend.fontsize': 10 if notebook_mode else 12,
        'figure.figsize': figsize,
        'figure.dpi': dpi,
        'axes.linewidth': 1.0 if notebook_mode else 1.2,
        'lines.linewidth': 2.0 if notebook_mode else 2.5,
        'lines.markersize': 6 if notebook_mode else 8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'xtick.major.width': 1.0 if notebook_mode else 1.2,
        'ytick.major.width': 1.0 if notebook_mode else 1.2,
        'xtick.minor.width': 0.8 if notebook_mode else 1.0,
        'ytick.minor.width': 0.8 if notebook_mode else 1.0,
        'xtick.major.size': 4.0 if notebook_mode else 5.0,
        'ytick.major.size': 4.0 if notebook_mode else 5.0,
        'xtick.minor.size': 2.5 if notebook_mode else 3.0,
        'ytick.minor.size': 2.5 if notebook_mode else 3.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'figure.constrained_layout.use': use_constrained_layout,
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
    
    # Create figure and axes
    fig, ax = None, None
    if 'ax' not in plot_kwargs or plot_kwargs['ax'] is None:
        if projection is not None:
            # For 3D plots or other projections
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection=projection)
        else:
            # Standard 2D plots
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Configure figure interactivity
        _configure_figure_interactivity(fig, interactive)
        
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
    
    # Store additional info
    plot_kwargs['_style'] = style
    plot_kwargs['_fig'] = fig
    plot_kwargs['_notebook_mode'] = notebook_mode
    
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
    use_y_log_scale: bool = False,
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
    use_y_log_scale : bool
        Whether the y-axis is using log scale
        
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
    
    # Set y-axis limits if provided - handle log scale properly
    if y_min is not None and y_max is not None:
        padding = (y_max - y_min) * 0.05  # 5% padding
        if use_y_log_scale:
            # For log scale, ensure limits are positive
            lower_limit = max(y_min * 0.5, 1e-10) if y_min > 0 else 1e-10
            upper_limit = y_max * 1.5
            ax.set_ylim(lower_limit, upper_limit)
        else:
            ax.set_ylim(max(0, y_min - padding), y_max + padding)
    
    # Grid and legend formatting
    ax.grid(True, which='major', linestyle='--', alpha=0.4)
    if style in ('paper','publication'):
        ax.grid(True, which='minor', linestyle=':', alpha=0.25)

    # Only add a legend if there are labelled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        if style in ('paper','publication'):
            ax.legend(loc=legend_loc,
                      frameon=True, framealpha=0.9,
                      fancybox=False, edgecolor='black')
        else:
            ax.legend(loc=legend_loc,
                      framealpha=0.9, fancybox=True)

    return ax


def finalize_plot(fig: Optional[plt.Figure], notebook_mode: bool = None) -> None:
    """
    Finalize a plot by ensuring it's displayed properly.
    
    Parameters
    ----------
    fig : plt.Figure or None
        Figure to finalize
    notebook_mode : bool, optional
        Whether we're in notebook mode (auto-detected if None)
    """
    if notebook_mode is None:
        notebook_mode = _is_notebook()
    
    if fig is not None:
        if notebook_mode:
            # In notebooks, the plot should auto-display
            # Only call show() if using inline backend
            if plt.get_backend() == 'module://matplotlib_inline.backend_inline':
                plt.show()
        else:
            plt.show()
