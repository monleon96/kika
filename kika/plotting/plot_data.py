"""
Base classes for representing plottable data.

These classes encapsulate the data and basic styling information for plot elements,
separating data representation from the actual plotting logic.
"""

from typing import Optional, Dict, Any, Union, List, Tuple
import numpy as np
from dataclasses import dataclass, field


@dataclass
class PlotData:
    """
    Base class for plottable data.
    
    This encapsulates the data to plot along with styling information,
    allowing flexible composition of plots without recreating plotting logic.
    
    Attributes
    ----------
    x : array-like
        X-axis data
    y : array-like
        Y-axis data
    label : str, optional
        Label for legend
    color : str or tuple, optional
        Line/marker color
    linestyle : str, optional
        Line style ('-', '--', ':', '-.', etc.)
    linewidth : float, optional
        Line width
    marker : str, optional
        Marker style ('o', 's', '^', etc.)
    markersize : float, optional
        Marker size
    alpha : float, optional
        Transparency (0-1)
    plot_type : str
        Type of plot: 'line', 'step', 'scatter', 'errorbar'
    metadata : dict
        Additional metadata about the data (e.g., isotope, MT, order)
    """
    x: np.ndarray
    y: np.ndarray
    label: Optional[str] = None
    color: Optional[Union[str, Tuple]] = None
    linestyle: Optional[str] = '-'
    linewidth: Optional[float] = None
    marker: Optional[str] = None
    markersize: Optional[float] = None
    alpha: Optional[float] = None
    plot_type: str = 'line'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and convert data to numpy arrays."""
        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)
        
        if len(self.x) != len(self.y):
            raise ValueError(f"x and y must have the same length. Got x: {len(self.x)}, y: {len(self.y)}")
    
    def apply_styling(self, **kwargs) -> 'PlotData':
        """
        Create a copy with updated styling.
        
        Parameters
        ----------
        **kwargs : dict
            Styling attributes to update (color, linestyle, linewidth, etc.)
            
        Returns
        -------
        PlotData
            New PlotData object with updated styling
        """
        # Create a copy of the current object
        import copy
        new_data = copy.copy(self)
        
        # Update attributes
        for key, value in kwargs.items():
            if hasattr(new_data, key):
                setattr(new_data, key, value)
            else:
                raise ValueError(f"Unknown styling attribute: {key}")
        
        return new_data
    
    def get_plot_kwargs(self) -> Dict[str, Any]:
        """
        Get keyword arguments for matplotlib plotting functions.
        
        Returns
        -------
        dict
            Dictionary of kwargs suitable for ax.plot(), ax.step(), etc.
        """
        kwargs = {}
        
        if self.label is not None:
            kwargs['label'] = self.label
        if self.color is not None:
            kwargs['color'] = self.color
        if self.linestyle is not None:
            kwargs['linestyle'] = self.linestyle
        if self.linewidth is not None:
            kwargs['linewidth'] = self.linewidth
        if self.marker is not None:
            kwargs['marker'] = self.marker
        if self.markersize is not None:
            kwargs['markersize'] = self.markersize
        if self.alpha is not None:
            kwargs['alpha'] = self.alpha
            
        return kwargs


@dataclass
class LegendreCoeffPlotData(PlotData):
    """
    Plottable data for Legendre coefficients.
    
    Additional Attributes
    ---------------------
    order : int
        Legendre polynomial order
    isotope : str, optional
        Isotope identifier
    mt : int, optional
        MT reaction number
    energy_range : tuple, optional
        (min, max) energy range for this data
    """
    order: int = 0
    isotope: Optional[str] = None
    mt: Optional[int] = None
    energy_range: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        super().__post_init__()
        # Store metadata
        self.metadata['order'] = self.order
        self.metadata['isotope'] = self.isotope
        self.metadata['mt'] = self.mt
        self.metadata['energy_range'] = self.energy_range
        
        # Default label if not provided
        if self.label is None and self.isotope and self.order is not None:
            self.label = f"{self.isotope} - L={self.order}"
        elif self.label is None and self.order is not None:
            self.label = f"L={self.order}"


@dataclass
class LegendreUncertaintyPlotData(PlotData):
    """
    Plottable data for Legendre coefficient uncertainties.
    
    Additional Attributes
    ---------------------
    order : int
        Legendre polynomial order
    isotope : str, optional
        Isotope identifier
    mt : int, optional
        MT reaction number
    uncertainty_type : str
        'relative' or 'absolute'
    energy_bins : array-like, optional
        Energy bin boundaries for step plots
    step_where : str
        Where to place steps: 'pre', 'post', 'mid'
    """
    order: int = 0
    isotope: Optional[str] = None
    mt: Optional[int] = None
    uncertainty_type: str = 'relative'
    energy_bins: Optional[np.ndarray] = None
    step_where: str = 'post'
    
    def __post_init__(self):
        super().__post_init__()
        # Default to step plot for uncertainties
        if self.plot_type == 'line':
            self.plot_type = 'step'
        
        # Store metadata
        self.metadata['order'] = self.order
        self.metadata['isotope'] = self.isotope
        self.metadata['mt'] = self.mt
        self.metadata['uncertainty_type'] = self.uncertainty_type
        
        # Convert energy_bins to numpy array if provided
        if self.energy_bins is not None:
            self.energy_bins = np.asarray(self.energy_bins)
        
        # Default label if not provided
        if self.label is None and self.isotope and self.order is not None:
            self.label = f"{self.isotope} - L={self.order}"
        elif self.label is None and self.order is not None:
            self.label = f"L={self.order}"


@dataclass
class AngularDistributionPlotData(PlotData):
    """
    Plottable data for angular distributions.
    
    Additional Attributes
    ---------------------
    energy : float
        Energy at which this distribution is defined
    isotope : str, optional
        Isotope identifier
    mt : int, optional
        MT reaction number
    distribution_type : str, optional
        Type of distribution: 'legendre', 'tabulated', 'mixed'
    """
    energy: float = 0.0
    isotope: Optional[str] = None
    mt: Optional[int] = None
    distribution_type: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        # Store metadata
        self.metadata['energy'] = self.energy
        self.metadata['isotope'] = self.isotope
        self.metadata['mt'] = self.mt
        self.metadata['distribution_type'] = self.distribution_type
        
        # Default label if not provided
        if self.label is None and self.energy is not None:
            self.label = f"E = {self.energy:.2e} MeV"


class UncertaintyBand:
    """
    Represents an uncertainty band for a plot.
    
    This can store either absolute bounds or relative uncertainties.
    When relative uncertainties are provided, they will be converted to
    absolute bounds by PlotBuilder using the nominal y values.
    
    Attributes
    ----------
    x : array-like
        X-axis data (same as the main plot data)
    y_lower : array-like, optional
        Lower bound of uncertainty (absolute values)
    y_upper : array-like, optional
        Upper bound of uncertainty (absolute values)
    relative_uncertainty : array-like, optional
        Relative uncertainties (fractional, e.g., 0.05 for 5%)
        If provided, y_lower and y_upper should be None
    sigma : float
        Number of sigma levels (default: 1.0)
    color : str or tuple, optional
        Fill color
    alpha : float
        Transparency (default: 0.2)
    label : str, optional
        Label for legend
    """
    
    def __init__(
        self,
        x: np.ndarray,
        y_lower: Optional[np.ndarray] = None,
        y_upper: Optional[np.ndarray] = None,
        relative_uncertainty: Optional[np.ndarray] = None,
        sigma: float = 1.0,
        color: Optional[Union[str, Tuple]] = None,
        alpha: float = 0.2,
        label: Optional[str] = None,
    ):
        self.x = np.asarray(x)
        self.sigma = sigma
        self.color = color
        self.alpha = alpha
        self.label = label
        
        # Store either absolute or relative uncertainties
        if relative_uncertainty is not None:
            if y_lower is not None or y_upper is not None:
                raise ValueError("Cannot specify both relative_uncertainty and y_lower/y_upper")
            self.relative_uncertainty = np.asarray(relative_uncertainty)
            self.y_lower = None
            self.y_upper = None
            # Validate
            if len(self.x) != len(self.relative_uncertainty):
                raise ValueError("x and relative_uncertainty must have the same length")
        elif y_lower is not None and y_upper is not None:
            self.y_lower = np.asarray(y_lower)
            self.y_upper = np.asarray(y_upper)
            self.relative_uncertainty = None
            # Validate
            if not (len(self.x) == len(self.y_lower) == len(self.y_upper)):
                raise ValueError("x, y_lower, and y_upper must have the same length")
        else:
            raise ValueError("Must provide either (y_lower, y_upper) or relative_uncertainty")
    
    def is_relative(self) -> bool:
        """Check if this band stores relative uncertainties."""
        return self.relative_uncertainty is not None
    
    def to_absolute(self, y_nominal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert relative uncertainties to absolute bounds.
        
        Parameters
        ----------
        y_nominal : array-like
            Nominal y values to apply uncertainties to
            
        Returns
        -------
        tuple of (y_lower, y_upper)
            Absolute uncertainty bounds
        """
        if not self.is_relative():
            return self.y_lower, self.y_upper
        
        y_nominal = np.asarray(y_nominal)
        if len(y_nominal) != len(self.relative_uncertainty):
            raise ValueError("y_nominal must have same length as relative_uncertainty")
        
        y_lower = y_nominal * (1.0 - self.sigma * self.relative_uncertainty)
        y_upper = y_nominal * (1.0 + self.sigma * self.relative_uncertainty)
        
        return y_lower, y_upper
    
    def get_fill_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for ax.fill_between()."""
        kwargs = {'alpha': self.alpha}
        
        if self.color is not None:
            kwargs['color'] = self.color
        if self.label is not None:
            kwargs['label'] = self.label
            
        return kwargs


@dataclass
class MultigroupXSPlotData(PlotData):
    """
    Plottable data for multigroup cross sections.
    
    Additional Attributes
    ---------------------
    zaid : int, optional
        Isotope identifier (ZAID)
    mt : int, optional
        MT reaction number
    energy_bins : array-like, optional
        Energy bin boundaries for step plots
    step_where : str
        Where to place steps: 'pre', 'post', 'mid'
    """
    zaid: Optional[int] = None
    mt: Optional[int] = None
    energy_bins: Optional[np.ndarray] = None
    step_where: str = 'post'
    
    def __post_init__(self):
        super().__post_init__()
        # Default to step plot for cross sections
        if self.plot_type == 'line':
            self.plot_type = 'step'
        
        # Store metadata
        self.metadata['zaid'] = self.zaid
        self.metadata['mt'] = self.mt
        
        # Convert energy_bins to numpy array if provided
        if self.energy_bins is not None:
            self.energy_bins = np.asarray(self.energy_bins)
        
        # Default label if not provided
        if self.label is None and self.zaid and self.mt is not None:
            from kika._utils import zaid_to_symbol
            isotope_symbol = zaid_to_symbol(self.zaid)
            self.label = f"{isotope_symbol} MT={self.mt}"
        elif self.label is None and self.mt is not None:
            self.label = f"MT={self.mt}"


@dataclass
class MultigroupUncertaintyPlotData(PlotData):
    """
    Plottable data for multigroup cross section uncertainties.
    
    Additional Attributes
    ---------------------
    zaid : int, optional
        Isotope identifier (ZAID)
    mt : int, optional
        MT reaction number
    uncertainty_type : str
        'relative' or 'absolute'
    energy_bins : array-like, optional
        Energy bin boundaries for step plots
    step_where : str
        Where to place steps: 'pre', 'post', 'mid'
    """
    zaid: Optional[int] = None
    mt: Optional[int] = None
    uncertainty_type: str = 'relative'
    energy_bins: Optional[np.ndarray] = None
    step_where: str = 'post'
    
    def __post_init__(self):
        super().__post_init__()
        # Default to step plot for uncertainties
        if self.plot_type == 'line':
            self.plot_type = 'step'
        
        # Store metadata
        self.metadata['zaid'] = self.zaid
        self.metadata['mt'] = self.mt
        self.metadata['uncertainty_type'] = self.uncertainty_type
        
        # Convert energy_bins to numpy array if provided
        if self.energy_bins is not None:
            self.energy_bins = np.asarray(self.energy_bins)
        
        # Default label if not provided
        if self.label is None and self.zaid and self.mt is not None:
            from kika._utils import zaid_to_symbol
            isotope_symbol = zaid_to_symbol(self.zaid)
            self.label = f"{isotope_symbol} MT={self.mt}"
        elif self.label is None and self.mt is not None:
            self.label = f"MT={self.mt}"
