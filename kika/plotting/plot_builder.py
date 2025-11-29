"""
PlotBuilder class for composing and rendering plots.

This module provides the PlotBuilder class that takes PlotData objects
and creates publication-quality plots with consistent styling.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

from .plot_data import PlotData, UncertaintyBand, LegendreUncertaintyPlotData, MultigroupUncertaintyPlotData
from .._plot_settings import setup_plot_style, format_axes, finalize_plot


class PlotBuilder:
    """
    Builder class for creating plots from PlotData objects.
    
    This class handles the composition of multiple plot elements and applies
    consistent styling and formatting.
    
    Examples
    --------
    >>> # Create plot data objects
    >>> data1 = LegendreCoeffPlotData(x=energies1, y=coeffs1, order=1, isotope='U235')
    >>> data2 = LegendreCoeffPlotData(x=energies2, y=coeffs2, order=1, isotope='U238')
    >>> 
    >>> # Build the plot
    >>> builder = PlotBuilder(style='publication', figsize=(10, 6))
    >>> builder.add_data(data1, color='blue')
    >>> builder.add_data(data2, color='red')
    >>> builder.set_labels(
    ...     title='Elastic Scattering Legendre Coefficients',
    ...     x_label='Energy (eV)',
    ...     y_label='Coefficient Value'
    ... )
    >>> fig = builder.build()
    >>> fig.show()
    """
    
    def __init__(
        self,
        style: str = 'default',
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        ax: Optional[plt.Axes] = None,
        projection: Optional[str] = None,
        **style_kwargs
    ):
        """
        Initialize the PlotBuilder.
        
        Parameters
        ----------
        style : str
            Plot style: 'default', 'dark', 'paper', 'publication', 'presentation'
        figsize : tuple
            Figure size (width, height) in inches
        dpi : int
            Dots per inch for figure resolution
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new figure and axes.
        projection : str, optional
            Projection type (e.g., '3d' for 3D plots)
        **style_kwargs
            Additional kwargs passed to setup_plot_style
        """
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        self.projection = projection
        self.style_kwargs = style_kwargs
        
        # Storage for plot elements
        self._data_list: List[PlotData] = []
        self._uncertainty_bands: List[Tuple[UncertaintyBand, int]] = []  # (band, data_index)
        self._custom_styling: List[Dict[str, Any]] = []  # Per-data styling overrides
        
        # Plot configuration
        self._x_label: Optional[str] = None
        self._y_label: Optional[str] = None
        self._title: Optional[str] = None
        self._legend_loc: str = 'best'
        self._use_log_x: bool = False
        self._use_log_y: bool = False
        self._x_lim: Optional[Tuple[float, float]] = None
        self._y_lim: Optional[Tuple[float, float]] = None
        self._grid: bool = True
        self._grid_alpha: float = 0.3  # Alpha (transparency) for major grid
        self._show_minor_grid: bool = False  # Whether to show minor grid
        self._minor_grid_alpha: float = 0.15  # Alpha for minor grid
        
        # Font size configuration
        self._title_fontsize: Optional[float] = None
        self._label_fontsize: Optional[float] = None
        self._tick_labelsize: Optional[float] = None
        self._legend_fontsize: Optional[float] = None
        
        # Setup plot style if ax is provided
        if ax is not None:
            self.fig = ax.figure
            self.ax = ax
            self._plot_kwargs = {'ax': ax, '_fig': self.fig}
        else:
            self.fig = None
            self.ax = None
            self._plot_kwargs = None
    
    def add_data(
        self,
        data: Union[PlotData, Tuple[PlotData, Optional[Union[UncertaintyBand, PlotData]]]],
        uncertainty: Optional[Union[UncertaintyBand, PlotData]] = None,
        **styling_overrides
    ) -> 'PlotBuilder':
        """
        Add a PlotData object to the plot.
        
        Parameters
        ----------
        data : PlotData or tuple
            The data to plot. Can be either:
            - A PlotData object
            - A tuple (PlotData, UncertaintyBand) as returned by to_plot_data with uncertainty=True
            If a tuple is provided and uncertainty is None, the UncertaintyBand from the tuple will be used.
        uncertainty : UncertaintyBand or PlotData, optional
            Uncertainty to plot with this data. Can be either:
            - UncertaintyBand object (will be plotted as shaded region)
            - PlotData object with uncertainty values (will be converted to band)
            If data is a tuple and this parameter is provided, this parameter takes precedence.
        **styling_overrides
            Styling overrides for this specific data (color, linestyle, etc.)
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        # Handle tuple input from to_plot_data(uncertainty=True)
        if isinstance(data, tuple):
            if len(data) == 2:
                plot_data, tuple_uncertainty = data
                # Use the uncertainty from tuple if no explicit uncertainty is provided
                if uncertainty is None:
                    uncertainty = tuple_uncertainty
                data = plot_data
            else:
                raise ValueError(f"Expected tuple of length 2, got {len(data)}")
        
        self._data_list.append(data)
        self._custom_styling.append(styling_overrides)
        
        if uncertainty is not None:
            # Convert PlotData to UncertaintyBand if needed
            if isinstance(uncertainty, PlotData):
                uncertainty = self._convert_plotdata_to_band(uncertainty, data)
            
            data_index = len(self._data_list) - 1
            self._uncertainty_bands.append((uncertainty, data_index))
        
        return self
    
    def add_multiple(
        self,
        data_list: List[PlotData],
        colors: Optional[List[Union[str, Tuple]]] = None,
        linestyles: Optional[List[str]] = None,
        **common_styling
    ) -> 'PlotBuilder':
        """
        Add multiple PlotData objects at once.
        
        Parameters
        ----------
        data_list : list of PlotData
            List of data objects to plot
        colors : list of str/tuple, optional
            Colors for each data object. If None, uses automatic color cycling.
        linestyles : list of str, optional
            Line styles for each data object. If None, uses solid lines.
        **common_styling
            Styling applied to all data objects
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        for i, data in enumerate(data_list):
            styling = common_styling.copy()
            
            if colors is not None and i < len(colors):
                styling['color'] = colors[i]
            if linestyles is not None and i < len(linestyles):
                styling['linestyle'] = linestyles[i]
            
            self.add_data(data, **styling)
        
        return self
    
    def set_labels(
        self,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None
    ) -> 'PlotBuilder':
        """
        Set plot labels.
        
        Parameters
        ----------
        title : str, optional
            Plot title
        x_label : str, optional
            X-axis label
        y_label : str, optional
            Y-axis label
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        if title is not None:
            self._title = title
        if x_label is not None:
            self._x_label = x_label
        if y_label is not None:
            self._y_label = y_label
        
        return self
    
    def set_scales(
        self,
        log_x: bool = False,
        log_y: bool = False
    ) -> 'PlotBuilder':
        """
        Set axis scales.
        
        Parameters
        ----------
        log_x : bool
            Use logarithmic scale for x-axis
        log_y : bool
            Use logarithmic scale for y-axis
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._use_log_x = log_x
        self._use_log_y = log_y
        return self
    
    def set_limits(
        self,
        x_lim: Optional[Tuple[float, float]] = None,
        y_lim: Optional[Tuple[float, float]] = None
    ) -> 'PlotBuilder':
        """
        Set axis limits.
        
        Parameters
        ----------
        x_lim : tuple, optional
            (min, max) for x-axis
        y_lim : tuple, optional
            (min, max) for y-axis
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._x_lim = x_lim
        self._y_lim = y_lim
        return self
    
    def set_legend(self, loc: str = 'best') -> 'PlotBuilder':
        """
        Set legend location.
        
        Parameters
        ----------
        loc : str
            Legend location
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._legend_loc = loc
        return self
    
    def set_grid(
        self, 
        grid: bool = True,
        alpha: float = 0.3,
        show_minor: bool = False,
        minor_alpha: float = 0.15
    ) -> 'PlotBuilder':
        """
        Configure grid display settings.
        
        Parameters
        ----------
        grid : bool
            Whether to show major grid
        alpha : float
            Alpha (transparency) for major grid lines. Range: 0.0-1.0
        show_minor : bool
            Whether to show minor grid lines
        minor_alpha : float
            Alpha (transparency) for minor grid lines. Range: 0.0-1.0
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._grid = grid
        self._grid_alpha = alpha
        self._show_minor_grid = show_minor
        self._minor_grid_alpha = minor_alpha
        return self
    
    def set_tick_params(
        self,
        max_ticks_x: Optional[int] = None,
        max_ticks_y: Optional[int] = None,
        rotate_x: Optional[float] = None,
        rotate_y: Optional[float] = None,
        auto_rotate: bool = False
    ) -> 'PlotBuilder':
        """
        Configure tick parameters to avoid overlapping labels.
        
        This method helps prevent tick label overlap, especially useful for
        small figures or dense data.
        
        Parameters
        ----------
        max_ticks_x : int, optional
            Maximum number of ticks on x-axis. If None, matplotlib default.
        max_ticks_y : int, optional
            Maximum number of ticks on y-axis. If None, matplotlib default.
        rotate_x : float, optional
            Rotation angle for x-axis tick labels (degrees). 
            Common values: 45, 90 for rotated text.
        rotate_y : float, optional
            Rotation angle for y-axis tick labels (degrees).
        auto_rotate : bool, optional
            If True, automatically rotate x-axis labels by 45° for better spacing.
            Default is False.
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
            
        Examples
        --------
        >>> # Limit to 8 ticks on x-axis and rotate labels
        >>> builder.set_tick_params(max_ticks_x=8, rotate_x=45)
        >>> 
        >>> # Auto-rotate x-axis labels for better spacing
        >>> builder.set_tick_params(auto_rotate=True)
        >>> 
        >>> # Control both axes
        >>> builder.set_tick_params(max_ticks_x=6, max_ticks_y=8)
        """
        # Store tick parameters
        if not hasattr(self, '_tick_params'):
            self._tick_params = {}
        
        if max_ticks_x is not None:
            self._tick_params['max_ticks_x'] = max_ticks_x
        if max_ticks_y is not None:
            self._tick_params['max_ticks_y'] = max_ticks_y
        if rotate_x is not None:
            self._tick_params['rotate_x'] = rotate_x
        if rotate_y is not None:
            self._tick_params['rotate_y'] = rotate_y
        if auto_rotate:
            self._tick_params['rotate_x'] = 45
        
        return self
    
    def set_font_sizes(
        self,
        title: Optional[float] = None,
        labels: Optional[float] = None,
        ticks: Optional[float] = None,
        legend: Optional[float] = None
    ) -> 'PlotBuilder':
        """
        Set font sizes for plot elements.
        
        This method allows fine-grained control over font sizes, overriding
        the defaults set by the style parameter.
        
        Parameters
        ----------
        title : float, optional
            Font size for the plot title
        labels : float, optional
            Font size for axis labels (x and y)
        ticks : float, optional
            Font size for tick labels
        legend : float, optional
            Font size for legend text
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
            
        Examples
        --------
        >>> builder = PlotBuilder(style='paper')
        >>> builder.add_data(data)
        >>> builder.set_labels(title='My Plot', x_label='Energy', y_label='Value')
        >>> builder.set_font_sizes(title=18, labels=14, ticks=12, legend=11)
        >>> fig = builder.build()
        """
        if title is not None:
            self._title_fontsize = title
        if labels is not None:
            self._label_fontsize = labels
        if ticks is not None:
            self._tick_labelsize = ticks
        if legend is not None:
            self._legend_fontsize = legend
        
        return self
    
    def build(self, show: bool = False) -> plt.Figure:
        """
        Build and return the figure.
        
        Parameters
        ----------
        show : bool
            Whether to display the figure immediately
            
        Returns
        -------
        matplotlib.figure.Figure
            The completed figure
        """
        # Setup plot style if not already done
        if self._plot_kwargs is None:
            self._plot_kwargs = setup_plot_style(
                style=self.style,
                figsize=self.figsize,
                dpi=self.dpi,
                projection=self.projection,
                **self.style_kwargs
            )
            self.fig = self._plot_kwargs['_fig']
            self.ax = self._plot_kwargs['ax']
        
        # Get default colors from plot settings
        default_colors = self._plot_kwargs.get('_colors', None)
        
        # Plot uncertainty bands first (so they appear behind the lines)
        for band, data_idx in self._uncertainty_bands:
            # Use the same color as the associated data if not specified
            fill_kwargs = band.get_fill_kwargs()
            
            if band.color is None and data_idx < len(self._data_list):
                # Get color from associated data or custom styling
                if 'color' in self._custom_styling[data_idx]:
                    fill_kwargs['color'] = self._custom_styling[data_idx]['color']
                elif self._data_list[data_idx].color is not None:
                    fill_kwargs['color'] = self._data_list[data_idx].color
            
            # Check if the associated data is a step plot (for multigroup/uncertainty data)
            data = self._data_list[data_idx] if data_idx < len(self._data_list) else None
            
            # Convert relative uncertainties to absolute if needed
            if band.is_relative():
                if data is None:
                    raise ValueError("Cannot convert relative uncertainty to absolute without associated data")
                y_lower, y_upper = band.to_absolute(data.y)
            else:
                y_lower, y_upper = band.y_lower, band.y_upper
            
            if data and data.plot_type == 'step':
                # For step plots, use step='post' to make uncertainty bands follow the steps
                self.ax.fill_between(band.x, y_lower, y_upper, step='post', **fill_kwargs)
            else:
                # Regular smooth fill for line plots
                self.ax.fill_between(band.x, y_lower, y_upper, **fill_kwargs)
        
        # Plot each data object
        for i, (data, styling_overrides) in enumerate(zip(self._data_list, self._custom_styling)):
            # Merge styling: data defaults < custom overrides
            plot_kwargs = data.get_plot_kwargs()
            plot_kwargs.update(styling_overrides)
            
            # Auto-assign color if not specified
            if 'color' not in plot_kwargs and default_colors is not None:
                plot_kwargs['color'] = default_colors[i % len(default_colors)]
            
            # Plot based on plot_type
            if data.plot_type == 'line':
                self.ax.plot(data.x, data.y, **plot_kwargs)
            
            elif data.plot_type == 'step':
                # Handle step plots (common for uncertainties and multigroup data)
                # Use 'post' where parameter to match the old plotting method
                where = getattr(data, 'step_where', 'post')
                
                # Check if markers are requested
                marker = plot_kwargs.get('marker', None)
                if marker is not None and marker != '':
                    # For step plots with markers, we need to:
                    # 1. Plot the step line without markers
                    # 2. Plot markers separately at segment midpoints
                    
                    # Extract marker properties
                    markersize = plot_kwargs.pop('markersize', None)
                    markerfacecolor = plot_kwargs.get('color', None)
                    markeredgecolor = plot_kwargs.get('color', None)
                    
                    # Plot step line without marker
                    marker_backup = plot_kwargs.pop('marker')
                    self.ax.step(data.x, data.y, where=where, **plot_kwargs)
                    
                    # Calculate midpoints for markers
                    # For 'post' step: each segment spans from x[i] to x[i+1] at height y[i]
                    # For 'pre' step: each segment spans from x[i-1] to x[i] at height y[i]
                    # For 'mid' step: segments are centered at x[i]
                    
                    if where == 'post' and len(data.x) > 1:
                        # Midpoints between consecutive x values
                        x_mid = (data.x[:-1] + data.x[1:]) / 2
                        y_mid = data.y[:-1]  # Heights correspond to the left point
                    elif where == 'pre' and len(data.x) > 1:
                        x_mid = (data.x[:-1] + data.x[1:]) / 2
                        y_mid = data.y[1:]  # Heights correspond to the right point
                    elif where == 'mid':
                        x_mid = data.x
                        y_mid = data.y
                    else:
                        # Fallback: use original points
                        x_mid = data.x
                        y_mid = data.y
                    
                    # Plot markers at midpoints
                    marker_kwargs = {
                        'marker': marker_backup,
                        'linestyle': 'none',
                        'color': markerfacecolor,
                        'markeredgecolor': markeredgecolor,
                    }
                    if markersize is not None:
                        marker_kwargs['markersize'] = markersize
                    if 'alpha' in plot_kwargs:
                        marker_kwargs['alpha'] = plot_kwargs['alpha']
                    
                    self.ax.plot(x_mid, y_mid, **marker_kwargs)
                else:
                    # No markers, just plot the step normally
                    self.ax.step(data.x, data.y, where=where, **plot_kwargs)
            
            elif data.plot_type == 'scatter':
                self.ax.scatter(data.x, data.y, **plot_kwargs)
            
            elif data.plot_type == 'errorbar':
                # Extract error bar specific kwargs
                yerr = plot_kwargs.pop('yerr', None)
                xerr = plot_kwargs.pop('xerr', None)
                self.ax.errorbar(data.x, data.y, yerr=yerr, xerr=xerr, **plot_kwargs)
            
            else:
                raise ValueError(f"Unknown plot_type: {data.plot_type}")
        
        # Apply scales
        if self._use_log_x:
            self.ax.set_xscale('log')
        if self._use_log_y:
            self.ax.set_yscale('log')
        
        # Set tight limits by default (no margins) if limits are not specified
        if self._x_lim is None and self._data_list:
            # Find the data range across all datasets
            x_min = min(np.min(data.x) for data in self._data_list if len(data.x) > 0)
            x_max = max(np.max(data.x) for data in self._data_list if len(data.x) > 0)
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                self.ax.set_xlim(x_min, x_max)
        else:
            # Apply user-specified limits
            if self._x_lim is not None:
                self.ax.set_xlim(self._x_lim)
        
        if self._y_lim is None and self._data_list:
            # Find the data range across all datasets (including uncertainty bands)
            y_values = []
            for data in self._data_list:
                if len(data.y) > 0:
                    y_values.extend(data.y)
            # Also include uncertainty band values
            for band, data_idx in self._uncertainty_bands:
                if band.is_relative():
                    # Need to convert to absolute first
                    if data_idx < len(self._data_list):
                        data = self._data_list[data_idx]
                        y_lower, y_upper = band.to_absolute(data.y)
                        if len(y_lower) > 0:
                            y_values.extend(y_lower)
                        if len(y_upper) > 0:
                            y_values.extend(y_upper)
                else:
                    # Already absolute
                    if band.y_lower is not None and len(band.y_lower) > 0:
                        y_values.extend(band.y_lower)
                    if band.y_upper is not None and len(band.y_upper) > 0:
                        y_values.extend(band.y_upper)
            
            if y_values:
                y_arr = np.array(y_values)
                y_arr = y_arr[np.isfinite(y_arr)]  # Remove inf/nan
                if len(y_arr) > 0:
                    y_min, y_max = np.min(y_arr), np.max(y_arr)
                    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                        # Add small padding for y-axis (5% on each side)
                        if self._use_log_y and y_min > 0:
                            # For log scale, use multiplicative padding
                            log_range = np.log10(y_max) - np.log10(y_min)
                            padding = 0.05 * log_range
                            y_min = 10 ** (np.log10(y_min) - padding)
                            y_max = 10 ** (np.log10(y_max) + padding)
                        else:
                            # For linear scale, use additive padding
                            padding = 0.05 * (y_max - y_min)
                            y_min = y_min - padding
                            y_max = y_max + padding
                        self.ax.set_ylim(y_min, y_max)
        else:
            # Apply user-specified limits
            if self._y_lim is not None:
                self.ax.set_ylim(self._y_lim)
        
        # Format axes using existing utility
        format_axes(
            self.ax,
            style=self.style,
            use_log_scale=self._use_log_x,
            is_energy_axis=True if self._x_label and 'energy' in self._x_label.lower() else False,
            x_label=self._x_label,
            y_label=self._y_label,
            title=self._title,
            legend_loc=self._legend_loc,
            use_y_log_scale=self._use_log_y,
        )
        
        # Apply custom font sizes if specified (overrides style defaults)
        if self._title is not None and self._title_fontsize is not None:
            self.ax.set_title(self._title, fontsize=self._title_fontsize)
        
        if self._x_label is not None and self._label_fontsize is not None:
            self.ax.set_xlabel(self._x_label, fontsize=self._label_fontsize)
        
        if self._y_label is not None and self._label_fontsize is not None:
            self.ax.set_ylabel(self._y_label, fontsize=self._label_fontsize)
        
        if self._tick_labelsize is not None:
            self.ax.tick_params(axis='both', which='major', labelsize=self._tick_labelsize)
        
        if self._legend_fontsize is not None:
            # Update legend if it exists
            legend = self.ax.get_legend()
            if legend is not None:
                plt.setp(legend.texts, fontsize=self._legend_fontsize)
        
        # Apply tick parameters if specified
        if hasattr(self, '_tick_params') and self._tick_params:
            from matplotlib.ticker import MaxNLocator
            
            # Limit number of ticks
            if 'max_ticks_x' in self._tick_params:
                max_x = self._tick_params['max_ticks_x']
                if self._use_log_x:
                    # For log scale, use LogLocator with numticks parameter
                    from matplotlib.ticker import LogLocator
                    self.ax.xaxis.set_major_locator(LogLocator(numticks=max_x))
                else:
                    # For linear scale, use MaxNLocator
                    self.ax.xaxis.set_major_locator(MaxNLocator(nbins=max_x, integer=False))
            
            if 'max_ticks_y' in self._tick_params:
                max_y = self._tick_params['max_ticks_y']
                if self._use_log_y:
                    from matplotlib.ticker import LogLocator
                    self.ax.yaxis.set_major_locator(LogLocator(numticks=max_y))
                else:
                    self.ax.yaxis.set_major_locator(MaxNLocator(nbins=max_y, integer=False))
            
            # Rotate tick labels
            if 'rotate_x' in self._tick_params:
                rotation = self._tick_params['rotate_x']
                # Also adjust horizontal alignment for better appearance
                ha = 'right' if rotation > 0 else 'center'
                self.ax.tick_params(axis='x', rotation=rotation)
                for label in self.ax.get_xticklabels():
                    label.set_rotation(rotation)
                    label.set_ha(ha)
            
            if 'rotate_y' in self._tick_params:
                rotation = self._tick_params['rotate_y']
                self.ax.tick_params(axis='y', rotation=rotation)
                for label in self.ax.get_yticklabels():
                    label.set_rotation(rotation)
        
        # Apply tight layout to prevent label cutoff (especially after rotation)
        # Only if constrained_layout is not already being used
        if hasattr(self, '_tick_params') and self._tick_params:
            try:
                # Check if constrained_layout is active
                if not self.fig.get_constrained_layout():
                    self.fig.tight_layout()
            except:
                # Fallback if constrained_layout check fails
                try:
                    self.fig.tight_layout()
                except:
                    pass  # If tight_layout fails, just continue
        
        # Apply grid configuration
        if self._grid:
            self.ax.grid(True, which='major', alpha=self._grid_alpha)
            if self._show_minor_grid:
                self.ax.minorticks_on()
                self.ax.grid(True, which='minor', linestyle=':', alpha=self._minor_grid_alpha, linewidth=0.5)
        else:
            self.ax.grid(False)
        
        # Finalize
        if show:
            finalize_plot(self.fig)
        
        return self.fig
    
    def _convert_plotdata_to_band(
        self,
        uncertainty_data: PlotData,
        nominal_data: PlotData
    ) -> UncertaintyBand:
        """
        Convert uncertainty PlotData to UncertaintyBand.
        
        Parameters
        ----------
        uncertainty_data : PlotData
            PlotData containing uncertainty values (MultigroupUncertaintyPlotData or LegendreUncertaintyPlotData)
        nominal_data : PlotData
            The nominal data that these uncertainties correspond to
            
        Returns
        -------
        UncertaintyBand
            Converted uncertainty band object
        """
        # Check if this is a recognized uncertainty PlotData type
        if isinstance(uncertainty_data, (MultigroupUncertaintyPlotData, LegendreUncertaintyPlotData)):
            # These PlotData types now store uncertainties at bin edges (n+1 points)
            # to properly support step plots with where='post'
            
            uncertainty_type = getattr(uncertainty_data, 'uncertainty_type', 'relative')
            
            if uncertainty_type == 'relative':
                # y values are in percentage - convert to fractional (0.05 for 5%)
                rel_unc = np.array(uncertainty_data.y) / 100.0
                unc_x = np.array(uncertainty_data.x)
                nom_x = np.array(nominal_data.x)
                
                # Check if grids match
                if len(unc_x) != len(nom_x) or not np.allclose(unc_x, nom_x):
                    # Different grids - need to interpolate uncertainty to match nominal grid
                    # This happens when ENDF MF34 (sparse) is combined with MF4 (dense)
                    rel_unc = np.interp(nom_x, unc_x, rel_unc)
                elif len(nominal_data.y) == len(rel_unc) + 1:
                    # Same grid but uncertainty is shorter (legacy case - G vs G+1 points)
                    # Extend uncertainties to match nominal data
                    rel_unc = np.append(rel_unc, rel_unc[-1])
                
                # Use nominal data's x values (energy bin boundaries)
                return UncertaintyBand(
                    x=nominal_data.x,
                    relative_uncertainty=rel_unc,
                    sigma=1.0,  # Uncertainties are already at the desired sigma level
                    color=getattr(uncertainty_data, 'color', None),
                    alpha=0.2
                )
            else:  # absolute
                # y values are absolute uncertainties
                abs_unc = np.array(uncertainty_data.y)
                unc_x = np.array(uncertainty_data.x)
                nominal_y = np.array(nominal_data.y)
                nom_x = np.array(nominal_data.x)
                
                # Check if grids match
                if len(unc_x) != len(nom_x) or not np.allclose(unc_x, nom_x):
                    # Different grids - need to interpolate
                    abs_unc = np.interp(nom_x, unc_x, abs_unc)
                elif len(nominal_y) == len(abs_unc) + 1:
                    # Same grid but uncertainty is shorter (legacy case)
                    abs_unc = np.append(abs_unc, abs_unc[-1])
                
                # Convert to bounds
                y_lower = nominal_y - abs_unc
                y_upper = nominal_y + abs_unc
                
                return UncertaintyBand(
                    x=nominal_data.x,
                    y_lower=y_lower,
                    y_upper=y_upper,
                    color=getattr(uncertainty_data, 'color', None),
                    alpha=0.2
                )
        else:
            # For other PlotData types, assume y values are relative uncertainties in fractional form
            rel_unc = np.array(uncertainty_data.y)
            
            # Check if we need to extend
            if len(nominal_data.y) == len(rel_unc) + 1:
                rel_unc = np.append(rel_unc, rel_unc[-1])
            
            return UncertaintyBand(
                x=nominal_data.x,
                relative_uncertainty=rel_unc,
                sigma=1.0,
                color=getattr(uncertainty_data, 'color', None),
                alpha=0.2
            )
    
    def clear(self) -> 'PlotBuilder':
        """
        Clear all data and reset to initial state.
        
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._data_list.clear()
        self._uncertainty_bands.clear()
        self._custom_styling.clear()
        
        if self.ax is not None:
            self.ax.clear()
        
        return self
