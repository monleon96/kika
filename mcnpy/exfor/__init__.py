"""
EXFOR (Experimental Nuclear Reaction Data) utilities for MCNPy.

This module provides functions for loading, processing, and plotting
experimental angular distribution data from EXFOR with ACE nuclear data.
"""

from .AD_utils import (
    # ACE Data Processing
    extract_angular_distribution,
    calculate_differential_cross_section,
    cosine_to_angle_degrees,
    angle_degrees_to_cosine,
    
    # EXFOR Data Processing
    cos_cm_from_cos_lab,
    jacobian_cm_to_lab,
    transform_lab_to_cm,
    load_exfor_data,
    extract_experiment_info,
    load_all_exfor_data,
    
    # Plotting Functions
    plot_angular_distribution,
    plot_combined_angular_distribution,
    plot_individual_energy_comparisons,
    plot_combined_angular_distribution_multi_ace,
    plot_all_energies_comparison,
)

__all__ = [
    # ACE Data Processing
    'extract_angular_distribution',
    'calculate_differential_cross_section',
    'cosine_to_angle_degrees',
    'angle_degrees_to_cosine',
    
    # EXFOR Data Processing
    'cos_cm_from_cos_lab',
    'jacobian_cm_to_lab',
    'transform_lab_to_cm',
    'load_exfor_data',
    'extract_experiment_info',
    'load_all_exfor_data',
    
    # Plotting Functions
    'plot_angular_distribution',
    'plot_combined_angular_distribution',
    'plot_individual_energy_comparisons',
    'plot_combined_angular_distribution_multi_ace',
    'plot_all_energies_comparison',
]
