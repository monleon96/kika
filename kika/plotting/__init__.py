"""
Plotting infrastructure for KIKA.

This module provides a flexible, object-oriented approach to creating plots
by separating data representation from visual styling and plot composition.
"""

from .plot_data import (
    PlotData,
    LegendreCoeffPlotData,
    LegendreUncertaintyPlotData,
    AngularDistributionPlotData,
    MultigroupXSPlotData,
    MultigroupUncertaintyPlotData,
    UncertaintyBand,
)
from .plot_builder import PlotBuilder

__all__ = [
    'PlotData',
    'LegendreCoeffPlotData',
    'LegendreUncertaintyPlotData',
    'AngularDistributionPlotData',
    'MultigroupXSPlotData',
    'MultigroupUncertaintyPlotData',
    'UncertaintyBand',
    'PlotBuilder',
]
