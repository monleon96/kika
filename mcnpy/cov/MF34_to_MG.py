"""
Convert MF34 angular distribution covariance data to multigroup format.

This module implements the conversion algorithm from continuous-energy MF34 
covariance matrices to multigroup angular distribution covariance matrices,
following the detailed mathematical procedure outlined in the user requirements.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional, Callable, Any
from scipy import integrate, interpolate
import warnings

from .mf34_covmat import MF34CovMat
from .mg_mf34_covmat import MGMF34CovMat
from ..energy_grids import grids


class WeightingFunction:
    """
    Class to represent various weighting functions phi(E) for multigroup collapse.
    """
    
    @staticmethod
    def constant(energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Constant weighting function phi(E) = 1."""
        if isinstance(energy, np.ndarray):
            return np.ones_like(energy)
        return 1.0
    
    @staticmethod
    def constant_antiderivative(energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Antiderivative of constant weighting: Phi(E) = E."""
        return energy
    
    @staticmethod
    def maxwellian(energy: Union[float, np.ndarray], temperature: float = 2.53e-2) -> Union[float, np.ndarray]:
        """Maxwellian spectrum phi(E) = sqrt(E) * exp(-E/kT)."""
        kT = temperature  # in eV, default = 0.0253 eV (room temperature)
        return np.sqrt(energy) * np.exp(-energy / kT)
    
    @staticmethod
    def fission_spectrum(energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Simplified fission spectrum phi(E) = sqrt(E) * exp(-E/1.29e6)."""
        return np.sqrt(energy) * np.exp(-energy / 1.29e6)  # 1.29 MeV average


def validate_frame_consistency(mf4_frame: str, mf34_frame: str) -> None:
    """
    Validate that MF4 and MF34 data are in consistent reference frames.
    
    Parameters
    ----------
    mf4_frame : str
        Reference frame from MF4 data ("LAB", "CM", or "same-as-MF4")
    mf34_frame : str
        Reference frame from MF34 data ("LAB", "CM", or "same-as-MF4")
        
    Raises
    ------
    ValueError
        If frames are inconsistent
    """
    # Handle "same-as-MF4" case
    if mf34_frame == "same-as-MF4":
        return  # Always consistent
    
    if mf4_frame != mf34_frame:
        raise ValueError(
            f"Frame mismatch between MF4 ({mf4_frame}) and MF34 ({mf34_frame}). "
            f"Angular distribution covariance processing requires consistent reference frames. "
            f"Please transform one dataset to match the other before proceeding."
        )


def compute_overlap_weights(base_energy_grid: np.ndarray, 
                          mg_energy_grid: np.ndarray,
                          phi_func: Callable = WeightingFunction.constant,
                          phi_antiderivative: Callable = WeightingFunction.constant_antiderivative) -> np.ndarray:
    """
    Compute base→group overlap weights w_{i,g}.
    
    w_{i,g} = [Phi(min(E_{i+1}, G_{g+1})) - Phi(max(E_i, G_g))]_+
    
    Parameters
    ----------
    base_energy_grid : np.ndarray
        Base energy grid edges [E_0, E_1, ..., E_N]
    mg_energy_grid : np.ndarray
        Multigroup energy grid edges [G_0, G_1, ..., G_n]
    phi_func : Callable, optional
        Weighting function phi(E)
    phi_antiderivative : Callable, optional
        Antiderivative of weighting function Phi(E)
        
    Returns
    -------
    np.ndarray
        Overlap weights matrix of shape (N_base_cells, N_mg_groups)
    """
    N_base = len(base_energy_grid) - 1  # Number of base cells
    N_mg = len(mg_energy_grid) - 1      # Number of MG groups
    
    weights = np.zeros((N_base, N_mg))
    
    for i in range(N_base):
        E_i = base_energy_grid[i]
        E_i_plus_1 = base_energy_grid[i + 1]
        
        for g in range(N_mg):
            G_g = mg_energy_grid[g]
            G_g_plus_1 = mg_energy_grid[g + 1]
            
            # Compute overlap interval
            lower_bound = max(E_i, G_g)
            upper_bound = min(E_i_plus_1, G_g_plus_1)
            
            # Only proceed if there's actual overlap
            if upper_bound > lower_bound:
                # Compute weight using antiderivative
                weight = phi_antiderivative(upper_bound) - phi_antiderivative(lower_bound)
                weights[i, g] = max(0.0, weight)  # Ensure non-negative
    
    return weights


def compute_base_cell_means(base_energy_grid: np.ndarray,
                          mf4_data,
                          legendre_orders: List[int],
                          phi_func: Callable = WeightingFunction.constant,
                          phi_antiderivative: Callable = WeightingFunction.constant_antiderivative) -> Dict[int, np.ndarray]:
    """
    Compute base-cell means A_{l,i} for each Legendre order.
    
    A_{l,i} = integral(phi(E) * a_l(E) dE) / integral(phi(E) dE) over [E_i, E_{i+1}]
    
    This optimized version pre-computes coefficient interpolation functions to avoid
    repeated evaluations during numerical integration.
    
    Parameters
    ----------
    base_energy_grid : np.ndarray
        Base energy grid edges
    mf4_data : MF4MT object
        MF4 angular distribution data
    legendre_orders : List[int]
        List of Legendre orders to compute
    phi_func : Callable, optional
        Weighting function phi(E)
    phi_antiderivative : Callable, optional
        Antiderivative of weighting function
        
    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping Legendre orders to base-cell means arrays
    """
    N_base = len(base_energy_grid) - 1
    base_means = {}
    
    # Pre-compute coefficient interpolation functions for efficiency
    # Sample energies across the energy range to build interpolation functions
    E_min = base_energy_grid[0]
    E_max = base_energy_grid[-1]
    
    # Use a reasonable number of sample points (adaptive based on data complexity)
    n_sample_points = 100  # Can be adjusted based on needs
    sample_energies = np.logspace(np.log10(max(E_min, 1e-10)), np.log10(E_max), n_sample_points)
    
    # Pre-compute coefficients at sample points using MF4 object methods
    max_order = max(legendre_orders) if legendre_orders else 0
    sample_coeffs = mf4_data.extract_legendre_coefficients(sample_energies, max_order)
    
    # Build interpolation functions for each required Legendre order
    coeff_interp_funcs = {}
    for l in legendre_orders:
        if l in sample_coeffs and len(sample_coeffs[l]) > 1:
            # Create interpolation function
            coeff_interp_funcs[l] = interpolate.interp1d(
                sample_energies, sample_coeffs[l],
                kind='linear', bounds_error=False, fill_value=0.0
            )
        else:
            # Constant or no data available
            const_val = sample_coeffs.get(l, [0.0])[0] if l in sample_coeffs else 0.0
            coeff_interp_funcs[l] = lambda E, val=const_val: np.full_like(E, val, dtype=float)
    
    # Compute base-cell means for each Legendre order
    for l in legendre_orders:
        means = np.zeros(N_base)
        coeff_func = coeff_interp_funcs[l]
        
        for i in range(N_base):
            E_i = base_energy_grid[i]
            E_i_plus_1 = base_energy_grid[i + 1]
            
            # Skip zero-width cells
            if E_i_plus_1 <= E_i:
                means[i] = 0.0
                continue
            
            # Compute denominator (normalization)
            denominator = phi_antiderivative(E_i_plus_1) - phi_antiderivative(E_i)
            
            if abs(denominator) < 1e-15:
                means[i] = 0.0
                continue
            
            # For constant weighting function, can use analytical approach
            if phi_func == WeightingFunction.constant:
                # phi(E) = 1, so integral becomes: integral(a_l(E) dE) / (E_{i+1} - E_i)
                # Use simple numerical integration with pre-computed function
                
                # Sample points within the interval for integration
                n_quad_points = 20  # Reasonable for most cases
                E_quad = np.linspace(E_i, E_i_plus_1, n_quad_points)
                a_l_values = coeff_func(E_quad)
                
                # Simple trapezoidal integration
                numerator = np.trapz(a_l_values, E_quad)
                means[i] = numerator / (E_i_plus_1 - E_i)
            else:
                # General case: use numerical integration with optimized integrand
                def integrand(E):
                    # E can be scalar or array
                    a_l_E = coeff_func(E)
                    if isinstance(a_l_E, np.ndarray):
                        a_l_E = a_l_E.item() if a_l_E.size == 1 else a_l_E[0]
                    return phi_func(E) * a_l_E
                
                # Use numerical integration with moderate precision
                numerator, _ = integrate.quad(integrand, E_i, E_i_plus_1, limit=50)
                means[i] = numerator / denominator
        
        base_means[l] = means
    
    return base_means


def compute_mg_means(base_means: Dict[int, np.ndarray], 
                    overlap_weights: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Compute multigroup means bar{a}_{l,g} from base-cell means and overlap weights.
    
    bar{a}_{l,g} = sum_i(w_{i,g} * A_{l,i}) / W_g
    where W_g = sum_i(w_{i,g})
    
    Parameters
    ----------
    base_means : Dict[int, np.ndarray]
        Base-cell means for each Legendre order
    overlap_weights : np.ndarray
        Overlap weights matrix (N_base, N_mg)
        
    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping Legendre orders to MG means arrays
    """
    N_mg = overlap_weights.shape[1]
    mg_means = {}
    
    # Compute group totals W_g
    W_g = np.sum(overlap_weights, axis=0)
    
    for l, A_l_i in base_means.items():
        bar_a_l_g = np.zeros(N_mg)
        
        for g in range(N_mg):
            if W_g[g] > 1e-15:  # Avoid division by zero
                weighted_sum = np.sum(overlap_weights[:, g] * A_l_i)
                bar_a_l_g[g] = weighted_sum / W_g[g]
            else:
                bar_a_l_g[g] = 0.0
        
        mg_means[l] = bar_a_l_g
    
    return mg_means


def collapse_relative_covariance(relative_base_matrix: np.ndarray,
                                base_means_row: np.ndarray,
                                base_means_col: np.ndarray,
                                overlap_weights: np.ndarray,
                                mg_means_row: np.ndarray,
                                mg_means_col: np.ndarray) -> np.ndarray:
    """
    Collapse MF34 relative covariance to multigroup relative covariance.
    
    R^MG_{gg'} = sum_{ij}(w_{ig} * A_{l,i} * R_{ij} * A_{l',j} * w_{jg'}) / 
                 (W_g * W_g' * bar{a}_{l,g} * bar{a}_{l',g'})
    
    Parameters
    ----------
    relative_base_matrix : np.ndarray
        Base-grid relative covariance matrix R_{ij}
    base_means_row : np.ndarray
        Base-cell means for row Legendre order A_{l,i}
    base_means_col : np.ndarray
        Base-cell means for column Legendre order A_{l',j}
    overlap_weights : np.ndarray
        Overlap weights matrix w_{i,g}
    mg_means_row : np.ndarray
        MG means for row Legendre order bar{a}_{l,g}
    mg_means_col : np.ndarray
        MG means for column Legendre order bar{a}_{l',g'}
        
    Returns
    -------
    np.ndarray
        Multigroup relative covariance matrix
    """
    N_mg = overlap_weights.shape[1]
    mg_relative = np.zeros((N_mg, N_mg))
    
    # Compute group totals W_g
    W_g = np.sum(overlap_weights, axis=0)
    
    for g in range(N_mg):
        for g_prime in range(N_mg):
            # Check for near-isotropy (denominator near zero)
            denominator = W_g[g] * W_g[g_prime] * mg_means_row[g] * mg_means_col[g_prime]
            
            if abs(denominator) < 1e-15:
                mg_relative[g, g_prime] = np.nan  # Undefined due to near-isotropy
                continue
            
            # Compute weighted sum
            weighted_sum = 0.0
            for i in range(len(base_means_row)):
                for j in range(len(base_means_col)):
                    weighted_sum += (overlap_weights[i, g] * base_means_row[i] * 
                                   relative_base_matrix[i, j] * base_means_col[j] * 
                                   overlap_weights[j, g_prime])
            
            mg_relative[g, g_prime] = weighted_sum / denominator
    
    return mg_relative


def collapse_absolute_covariance(absolute_base_matrix: np.ndarray,
                                overlap_weights: np.ndarray) -> np.ndarray:
    """
    Collapse MF34 absolute covariance to multigroup absolute covariance.
    
    C^MG_{gg'} = sum_{ij}(w_{ig} * C^abs_{ij} * w_{jg'}) / (W_g * W_g')
    
    Parameters
    ----------
    absolute_base_matrix : np.ndarray
        Base-grid absolute covariance matrix C^abs_{ij}
    overlap_weights : np.ndarray
        Overlap weights matrix w_{i,g}
        
    Returns
    -------
    np.ndarray
        Multigroup absolute covariance matrix
    """
    N_mg = overlap_weights.shape[1]
    mg_absolute = np.zeros((N_mg, N_mg))
    
    # Compute group totals W_g
    W_g = np.sum(overlap_weights, axis=0)
    
    for g in range(N_mg):
        for g_prime in range(N_mg):
            denominator = W_g[g] * W_g[g_prime]
            
            if abs(denominator) < 1e-15:
                mg_absolute[g, g_prime] = 0.0
                continue
            
            # Compute weighted sum
            weighted_sum = 0.0
            for i in range(absolute_base_matrix.shape[0]):
                for j in range(absolute_base_matrix.shape[1]):
                    weighted_sum += (overlap_weights[i, g] * absolute_base_matrix[i, j] * 
                                   overlap_weights[j, g_prime])
            
            mg_absolute[g, g_prime] = weighted_sum / denominator
    
    return mg_absolute


def enforce_matrix_quality(matrix: np.ndarray, 
                          matrix_type: str = "relative",
                          clip_small_negatives: bool = False,
                          symmetrize: bool = True) -> np.ndarray:
    """
    Apply quality checks and corrections to covariance matrices.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input covariance matrix
    matrix_type : str, optional
        Type of matrix ("relative" or "absolute")
    clip_small_negatives : bool, optional
        Whether to clip small negative diagonal elements to zero
    symmetrize : bool, optional
        Whether to enforce symmetry
        
    Returns
    -------
    np.ndarray
        Quality-corrected matrix
    """
    result = matrix.copy()
    
    # Enforce symmetry
    if symmetrize:
        result = (result + result.T) / 2.0
    
    # Clip small negative variances on diagonal
    if clip_small_negatives:
        diagonal = np.diag(result)
        small_negatives = (diagonal < 0) & (abs(diagonal) < 1e-10)
        if np.any(small_negatives):
            warnings.warn(f"Clipped {np.sum(small_negatives)} small negative diagonal elements to zero")
            np.fill_diagonal(result, np.where(small_negatives, 0.0, diagonal))
    
    return result


def MF34_to_MG(endf_object,
               energy_grid: Union[str, List[float], np.ndarray],
               weighting_function: Union[str, Callable] = "constant",
               isotope: Optional[int] = None,
               mt: Optional[int] = None,
               max_legendre_order: int = 10) -> MGMF34CovMat:
    """
    Convert MF34 angular distribution covariance data to multigroup format.
    
    This function implements the complete algorithm for converting continuous-energy
    MF34 covariance matrices to multigroup format, following the mathematical
    procedure outlined in the specifications.
    
    Parameters
    ----------
    endf_object : ENDF object
        ENDF object containing both MF4 and MF34 data
    energy_grid : str, list, or np.ndarray
        Multigroup energy grid specification:
        - str: Name of predefined grid (e.g., "VITAMINJ174")
        - list/array: Custom energy group boundaries
    weighting_function : str or Callable, optional
        Weighting function for multigroup collapse:
        - "constant": phi(E) = 1 (default)
        - "maxwellian": Maxwellian spectrum
        - "fission": Fission spectrum
        - Callable: Custom function
    isotope : int, optional
        Specific isotope to process (if None, process all)
    mt : int, optional
        Specific MT reaction to process (if None, process all)
    max_legendre_order : int, optional
        Maximum Legendre order to consider (default: 10)
        
    Returns
    -------
    MGMF34CovMat
        Multigroup covariance matrix object
        
    Raises
    ------
    ValueError
        If input parameters are invalid or data is inconsistent
    """
    # Validate inputs and extract data
    if not hasattr(endf_object, 'mf') or 4 not in endf_object.mf or 34 not in endf_object.mf:
        raise ValueError("ENDF object must contain both MF4 and MF34 data")
    
    # Get MF34 covariance data
    mf34_covmat = endf_object.mf[34].to_ang_covmat()
    mf4_data = endf_object.mf[4]
    
    # Process energy grid
    if isinstance(energy_grid, str):
        if hasattr(grids, energy_grid):
            mg_energy_edges = np.array(getattr(grids, energy_grid))
        else:
            raise ValueError(f"Unknown predefined energy grid: {energy_grid}")
    else:
        mg_energy_edges = np.array(energy_grid)
    
    # Validate energy grid
    if len(mg_energy_edges) < 2:
        raise ValueError("Energy grid must have at least 2 points (1 group)")
    if not np.all(np.diff(mg_energy_edges) > 0):
        raise ValueError("Energy grid must be strictly increasing")
    
    # Process weighting function
    if isinstance(weighting_function, str):
        if weighting_function == "constant":
            phi_func = WeightingFunction.constant
            phi_antiderivative = WeightingFunction.constant_antiderivative
            weight_desc = "constant"
        elif weighting_function == "maxwellian":
            phi_func = WeightingFunction.maxwellian
            phi_antiderivative = None  # Would need custom implementation
            weight_desc = "maxwellian"
        elif weighting_function == "fission":
            phi_func = WeightingFunction.fission_spectrum
            phi_antiderivative = None  # Would need custom implementation
            weight_desc = "fission spectrum"
        else:
            raise ValueError(f"Unknown weighting function: {weighting_function}")
    else:
        phi_func = weighting_function
        phi_antiderivative = None  # User must provide if needed
        weight_desc = "custom"
    
    # For now, only support constant weighting (others need antiderivative implementation)
    if phi_antiderivative is None and weight_desc != "constant":
        raise NotImplementedError(f"Antiderivative for {weight_desc} weighting not implemented")
    
    # Create result object
    result = MGMF34CovMat()
    result.energy_grid = mg_energy_edges
    result.weighting_function = weight_desc
    
    # Process each matrix in the MF34 covariance data
    for i in range(mf34_covmat.num_matrices):
        # Get matrix information
        isotope_row = mf34_covmat.isotope_rows[i]
        reaction_row = mf34_covmat.reaction_rows[i]
        l_row = mf34_covmat.l_rows[i]
        isotope_col = mf34_covmat.isotope_cols[i]
        reaction_col = mf34_covmat.reaction_cols[i]
        l_col = mf34_covmat.l_cols[i]
        
        # Filter by isotope/MT if specified
        if isotope is not None and isotope_row != isotope:
            continue
        if mt is not None and reaction_row != mt:
            continue
        
        # Get base energy grid and matrix
        base_energy_grid = np.array(mf34_covmat.energy_grids[i])
        base_matrix = mf34_covmat.matrices[i]
        is_relative = mf34_covmat.is_relative[i]
        frame = mf34_covmat.frame[i]
        
        # Get corresponding MF4 data for row and column channels
        if reaction_row not in mf4_data.mt:
            warnings.warn(f"No MF4 data found for MT{reaction_row} (row), skipping")
            continue
        if reaction_col not in mf4_data.mt:
            warnings.warn(f"No MF4 data found for MT{reaction_col} (col), skipping")
            continue

        mf4_mt_data_row = mf4_data.mt[reaction_row]
        mf4_mt_data_col = mf4_data.mt[reaction_col]
        
        # Validate frame consistency for both row and column
        mf4_frame_row = mf4_mt_data_row.frame if hasattr(mf4_mt_data_row, 'frame') else "unknown"
        mf4_frame_col = mf4_mt_data_col.frame if hasattr(mf4_mt_data_col, 'frame') else "unknown"
        validate_frame_consistency(mf4_frame_row, frame)
        validate_frame_consistency(mf4_frame_col, frame)
        
        # Compute overlap weights
        overlap_weights = compute_overlap_weights(
            base_energy_grid, mg_energy_edges, phi_func, phi_antiderivative
        )
        
        # Compute base-cell means for row and column channels separately
        legendre_orders_row = [l_row]
        legendre_orders_col = [l_col] if l_col != l_row else []
        
        # Compute base means for row channel
        base_means_row = compute_base_cell_means(
            base_energy_grid, mf4_mt_data_row, legendre_orders_row, phi_func, phi_antiderivative
        )
        
        # Compute base means for column channel (if different)
        if l_col != l_row or reaction_col != reaction_row:
            base_means_col = compute_base_cell_means(
                base_energy_grid, mf4_mt_data_col, legendre_orders_col or [l_col], phi_func, phi_antiderivative
            )
        else:
            # Same channel, reuse row data
            base_means_col = base_means_row
        
        # Combine base means for MG calculation
        all_base_means = base_means_row.copy()
        all_base_means.update(base_means_col)
        
        # Compute MG means
        mg_means = compute_mg_means(all_base_means, overlap_weights)
        
        # Collapse covariance matrix
        if is_relative:
            # Input is relative, collapse directly to relative
            mg_relative = collapse_relative_covariance(
                base_matrix,
                all_base_means[l_row],
                all_base_means[l_col],
                overlap_weights,
                mg_means[l_row],
                mg_means[l_col]
            )
            
            # Convert to absolute for completeness
            mg_absolute = np.zeros_like(mg_relative)
            for g in range(len(mg_means[l_row])):
                for g_prime in range(len(mg_means[l_col])):
                    if not np.isnan(mg_relative[g, g_prime]):
                        mg_absolute[g, g_prime] = (mg_relative[g, g_prime] * 
                                                 mg_means[l_row][g] * mg_means[l_col][g_prime])
        else:
            # Input is absolute, collapse to absolute then convert to relative
            mg_absolute = collapse_absolute_covariance(base_matrix, overlap_weights)
            
            # Convert to relative
            mg_relative = np.zeros_like(mg_absolute)
            for g in range(len(mg_means[l_row])):
                for g_prime in range(len(mg_means[l_col])):
                    denominator = mg_means[l_row][g] * mg_means[l_col][g_prime]
                    if abs(denominator) > 1e-15:
                        mg_relative[g, g_prime] = mg_absolute[g, g_prime] / denominator
                    else:
                        mg_relative[g, g_prime] = np.nan
        
        # Apply quality checks
        mg_relative = enforce_matrix_quality(mg_relative, "relative")
        mg_absolute = enforce_matrix_quality(mg_absolute, "absolute")
        
        # Add to result
        result.add_matrix(
            isotope_row, reaction_row, l_row,
            isotope_col, reaction_col, l_col,
            mg_relative, mg_absolute,
            mg_means[l_row], mg_means[l_col],
            frame
        )
    
    return result
