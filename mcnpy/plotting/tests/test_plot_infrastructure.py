"""
Tests for the new plotting infrastructure.

These tests verify that PlotData and PlotBuilder work correctly.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from mcnpy.plotting import (
    PlotData,
    LegendreCoeffPlotData,
    LegendreUncertaintyPlotData,
    PlotBuilder
)


class TestPlotData:
    """Test PlotData base class."""
    
    def test_basic_creation(self):
        """Test basic PlotData creation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 4, 9, 16, 25])
        
        data = PlotData(x=x, y=y, label='Test')
        
        assert len(data.x) == 5
        assert len(data.y) == 5
        assert data.label == 'Test'
        assert data.plot_type == 'line'
    
    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched x and y lengths raise an error."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="same length"):
            PlotData(x=x, y=y)
    
    def test_apply_styling(self):
        """Test applying styling to PlotData."""
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 9])
        
        data = PlotData(x=x, y=y, color='blue')
        styled_data = data.apply_styling(color='red', linewidth=3)
        
        # Original unchanged
        assert data.color == 'blue'
        assert data.linewidth is None
        
        # New data has updated styling
        assert styled_data.color == 'red'
        assert styled_data.linewidth == 3
    
    def test_get_plot_kwargs(self):
        """Test getting plot kwargs."""
        data = PlotData(
            x=[1, 2, 3],
            y=[1, 4, 9],
            label='Test',
            color='blue',
            linestyle='--',
            linewidth=2
        )
        
        kwargs = data.get_plot_kwargs()
        
        assert kwargs['label'] == 'Test'
        assert kwargs['color'] == 'blue'
        assert kwargs['linestyle'] == '--'
        assert kwargs['linewidth'] == 2


class TestLegendreCoeffPlotData:
    """Test LegendreCoeffPlotData specialized class."""
    
    def test_creation_with_metadata(self):
        """Test creation with Legendre-specific metadata."""
        x = np.logspace(1, 7, 100)
        y = np.random.random(100)
        
        data = LegendreCoeffPlotData(
            x=x,
            y=y,
            order=2,
            isotope='U235',
            mt=2
        )
        
        assert data.order == 2
        assert data.isotope == 'U235'
        assert data.mt == 2
        assert data.metadata['order'] == 2
        assert data.metadata['isotope'] == 'U235'
    
    def test_auto_label_generation(self):
        """Test automatic label generation."""
        data = LegendreCoeffPlotData(
            x=[1, 2, 3],
            y=[1, 2, 3],
            order=1,
            isotope='U235'
        )
        
        assert data.label == 'U235 - L=1'
        
        # Without isotope
        data2 = LegendreCoeffPlotData(
            x=[1, 2, 3],
            y=[1, 2, 3],
            order=2
        )
        
        assert data2.label == 'L=2'


class TestLegendreUncertaintyPlotData:
    """Test LegendreUncertaintyPlotData specialized class."""
    
    def test_defaults_to_step_plot(self):
        """Test that uncertainty data defaults to step plot type."""
        data = LegendreUncertaintyPlotData(
            x=[1, 2, 3],
            y=[0.1, 0.2, 0.15],
            order=1
        )
        
        assert data.plot_type == 'step'
    
    def test_with_energy_bins(self):
        """Test with explicit energy bin boundaries."""
        x = np.array([1.5, 2.5, 3.5])  # Bin centers
        y = np.array([0.1, 0.2, 0.15])
        bins = np.array([1, 2, 3, 4])  # Bin boundaries
        
        data = LegendreUncertaintyPlotData(
            x=x,
            y=y,
            order=1,
            energy_bins=bins
        )
        
        assert len(data.energy_bins) == 4
        assert data.step_where == 'post'


class TestPlotBuilder:
    """Test PlotBuilder class."""
    
    def test_basic_build(self):
        """Test basic plot building."""
        x = np.logspace(1, 6, 50)
        y = 1 / x
        
        data = PlotData(x=x, y=y, label='Test Data')
        
        builder = PlotBuilder(style='default', figsize=(8, 6))
        builder.add_data(data, color='blue')
        builder.set_labels(title='Test Plot', x_label='X', y_label='Y')
        fig = builder.build()
        
        assert fig is not None
        assert len(builder.ax.lines) == 1
        
        plt.close(fig)
    
    def test_add_multiple_data(self):
        """Test adding multiple data objects."""
        x = np.linspace(0, 10, 100)
        
        data1 = PlotData(x=x, y=np.sin(x), label='sin')
        data2 = PlotData(x=x, y=np.cos(x), label='cos')
        data3 = PlotData(x=x, y=np.tan(x), label='tan')
        
        builder = PlotBuilder()
        builder.add_multiple([data1, data2, data3])
        fig = builder.build()
        
        assert len(builder.ax.lines) == 3
        
        plt.close(fig)
    
    def test_method_chaining(self):
        """Test method chaining."""
        data = PlotData(x=[1, 2, 3], y=[1, 4, 9])
        
        fig = (PlotBuilder()
               .add_data(data, color='blue')
               .set_labels(title='Chain Test')
               .set_scales(log_x=False, log_y=False)
               .build())
        
        assert fig is not None
        
        plt.close(fig)
    
    def test_clear(self):
        """Test clearing builder."""
        data1 = PlotData(x=[1, 2, 3], y=[1, 4, 9])
        data2 = PlotData(x=[1, 2, 3], y=[2, 8, 18])
        
        builder = PlotBuilder()
        builder.add_data(data1)
        fig1 = builder.build()
        
        assert len(builder.ax.lines) == 1
        
        builder.clear()
        builder.add_data(data2)
        fig2 = builder.build()
        
        assert len(builder.ax.lines) == 1
        
        plt.close(fig1)
        plt.close(fig2)
    
    def test_log_scales(self):
        """Test logarithmic scales."""
        x = np.logspace(0, 3, 50)
        y = x**2
        
        data = PlotData(x=x, y=y)
        
        builder = PlotBuilder()
        builder.add_data(data)
        builder.set_scales(log_x=True, log_y=True)
        fig = builder.build()
        
        assert builder.ax.get_xscale() == 'log'
        assert builder.ax.get_yscale() == 'log'
        
        plt.close(fig)
    
    def test_axis_limits(self):
        """Test setting axis limits."""
        data = PlotData(x=[1, 2, 3, 4, 5], y=[1, 4, 9, 16, 25])
        
        builder = PlotBuilder()
        builder.add_data(data)
        builder.set_limits(x_lim=(1, 4), y_lim=(0, 20))
        fig = builder.build()
        
        x_lim = builder.ax.get_xlim()
        y_lim = builder.ax.get_ylim()
        
        assert x_lim == (1, 4)
        assert y_lim == (0, 20)
        
        plt.close(fig)
    
    def test_step_plot(self):
        """Test step plot type."""
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2, 1.5, 2.5])
        
        data = PlotData(x=x, y=y, plot_type='step')
        
        builder = PlotBuilder()
        builder.add_data(data)
        fig = builder.build()
        
        # Step plots create Line2D objects
        assert len(builder.ax.lines) > 0
        
        plt.close(fig)
    
    def test_with_existing_axes(self):
        """Test using an existing axes object."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data = PlotData(x=[1, 2, 3], y=[1, 4, 9])
        
        builder = PlotBuilder(ax=ax)
        builder.add_data(data, color='red')
        builder.build()
        
        assert len(ax.lines) == 1
        
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
