"""
Uncertainty Quantification (UQ) module for MCNPy.

This module provides tools for uncertainty propagation and analysis
in Monte Carlo neutron transport calculations.
"""

from .fastTMC import fastTMC, create_summary_table

__all__ = ['fastTMC', 'create_summary_table']
