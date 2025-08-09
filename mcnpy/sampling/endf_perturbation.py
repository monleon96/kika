"""
ENDF file perturbation module for angular distribution covariance data.

This module provides functionality to perturb ENDF MF4 angular distribution data
using MF34 covariance matrices. It is designed to be scalable and maintainable
for future extension to other MF sections.
"""
from typing import List, Union, Optional, Dict, Tuple, Any
from multiprocessing import Pool
from datetime import datetime
import os
import numpy as np
import pandas as pd

from mcnpy.sampling.generators import generate_endf_samples
from mcnpy.cov.mf34_covmat import MF34CovMat
from mcnpy.endf.parsers.parse_endf import parse_endf_file
from mcnpy.endf.writers.endf_writer import ENDFWriter
from mcnpy.endf.classes.mf4.polynomial import MF4MTLegendre
from mcnpy.endf.classes.mf4.mixed import MF4MTMixed
from mcnpy.endf.classes.mf import MF
from mcnpy._utils import zaid_to_symbol

# Reuse the DualLogger from ace_perturbation
from mcnpy.sampling.ace_perturbation import DualLogger

# Global logger instance
_logger = None

def _get_logger():
    """Get the global logger instance."""
    return _logger


def _process_sample(
    endf_file: str,
    sample: np.ndarray,
    sample_index: int,
    energy_grids: Dict[Tuple[int, int, int], List[float]],  # (isotope, mt, l) -> energy_grid
    param_mapping: List[Tuple[int, int, int, int]],  # List of (isotope, mt, l, energy_bin) tuples
    output_dir: str,
    dry_run: bool = False,
):
    """
    Process a single perturbation sample for ENDF files.
    
    Parameters
    ----------
    endf_file : str
        Path to the original ENDF file
    sample : np.ndarray
        Perturbation factors for this sample
    sample_index : int
        Index of this sample (0-based)
    energy_grids : Dict[Tuple[int, int, int], List[float]]
        Energy grids for each (isotope, mt, l) combination
    param_mapping : List[Tuple[int, int, int]]
        Mapping of sample indices to (isotope, mt, l) parameters
    output_dir : str
        Output directory for results
    dry_run : bool
        If True, only generate factors without creating ENDF files
    """
    if dry_run:
        # Parse the ENDF file to get ZAID for proper directory structure
        endf = parse_endf_file(endf_file)
        base = os.path.splitext(os.path.basename(endf_file))[0]
        sample_str = f"{sample_index+1:04d}"
        sample_dir = os.path.join(output_dir, str(endf.zaid or "unknown"), sample_str)
        os.makedirs(sample_dir, exist_ok=True)
        
        _write_sample_summary(
            sample=sample,
            sample_index=sample_index,
            energy_grids=energy_grids,
            param_mapping=param_mapping,
            sample_dir=sample_dir,
            base=base,
        )
        return
    
    # Parse the ENDF file
    endf = parse_endf_file(endf_file)
    
    # Apply perturbations to MF4 data
    perturbed_params = apply_perturbation_factors_to_endf(
        endf, sample, sample_index, energy_grids, param_mapping, verbose=False
    )
    
    # Write perturbed ENDF file
    base, ext = os.path.splitext(os.path.basename(endf_file))
    sample_str = f"{sample_index+1:04d}"
    sample_dir = os.path.join(output_dir, str(endf.zaid or "unknown"), sample_str)
    os.makedirs(sample_dir, exist_ok=True)
    out_endf = os.path.join(sample_dir, f"{base}_{sample_str}{ext}")
    
    # Use ENDF writer to save the modified file
    writer = ENDFWriter(endf_file)
    if 4 in endf.files:
        success = writer.replace_mf_section(endf.files[4], out_endf)
        if not success:
            if _get_logger():
                _get_logger().error(f"Failed to write perturbed ENDF file: {out_endf}")
            return
    
    # Write sample summary
    _write_sample_summary(
        sample=sample,
        sample_index=sample_index,
        energy_grids=energy_grids,
        param_mapping=param_mapping,
        sample_dir=sample_dir,
        base=base,
    )


def load_mf34_covariance(path: str) -> Optional[MF34CovMat]:
    """
    Load MF34 covariance matrix from ENDF file containing MF34 section.
    
    Parameters
    ----------
    path : str
        Path to the ENDF file containing MF34 covariance data
        
    Returns
    -------
    Optional[MF34CovMat]
        Loaded MF34CovMat object or None if loading failed
    """
    if not os.path.exists(path):
        if _get_logger():
            _get_logger().error(f"ENDF file not found: {path}")
        return None
    
    try:
        logger = _get_logger()
        if logger:
            logger.info(f"[ENDF] [MF34] Loading covariance from file: {path}")
        
        # Parse the ENDF file
        endf = parse_endf_file(path)
        
        # Get MF34 section
        mf34 = endf.get_file(34)
        if mf34 is None:
            if logger:
                logger.error(f"[ENDF] [MF34] No MF34 section found in file: {path}")
            return None
        
        # Convert all MT sections to MF34CovMat and combine them
        combined_mf34_cov = MF34CovMat()
        
        for mt_number, mt_data in mf34.sections.items():
            if hasattr(mt_data, 'to_ang_covmat'):
                if logger:
                    logger.info(f"[ENDF] [MF34] Converting MT{mt_number} to angular covariance matrix")
                
                mt_covmat = mt_data.to_ang_covmat()
                
                # Add all matrices from this MT to the combined object
                for i in range(mt_covmat.num_matrices):
                    combined_mf34_cov.add_matrix(
                        isotope_row=mt_covmat.isotope_rows[i],
                        reaction_row=mt_covmat.reaction_rows[i],
                        l_row=mt_covmat.l_rows[i],
                        isotope_col=mt_covmat.isotope_cols[i],
                        reaction_col=mt_covmat.reaction_cols[i],
                        l_col=mt_covmat.l_cols[i],
                        matrix=mt_covmat.matrices[i],
                        energy_grid=mt_covmat.energy_grids[i]
                    )
        
        if logger:
            logger.info(f"[ENDF] [MF34] Successfully loaded covariance with {combined_mf34_cov.num_matrices} matrices")
            logger.info(f"Available isotopes: {sorted(combined_mf34_cov.isotopes)}")
            logger.info(f"Available reactions: {sorted(combined_mf34_cov.reactions)}")
            logger.info(f"Available Legendre indices: {sorted(combined_mf34_cov.legendre_indices)}")
        
        return combined_mf34_cov
        
    except Exception as e:
        if _get_logger():
            _get_logger().error(f"Failed to load MF34 covariance: {e}")
        return None


def perturb_ENDF_files(
    endf_files: Union[str, List[str]],
    mf34_cov_files: Union[str, List[str]],
    mt_list: List[int],
    legendre_coeffs: List[int],
    num_samples: int,
    space: str = "log",
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    output_dir: str = '.',
    seed: Optional[int] = None,
    nprocs: int = 1,
    dry_run: bool = False,
    verbose: bool = True,
):
    """
    Perturb ENDF nuclear data files using MF34 angular covariance matrices.
    
    This function generates perturbed ENDF files by sampling perturbation factors
    from multivariate normal distributions derived from MF34 covariance matrices.
    
    Parameters
    ----------
    endf_files : Union[str, List[str]]
        Path(s) to ENDF file(s) to be perturbed
    mf34_cov_files : Union[str, List[str]]
        Path(s) to MF34 covariance matrix file(s).
        Can be a single file (used for all ENDF files) or one file per ENDF file.
    mt_list : List[int]
        List of MT reaction numbers to perturb. Empty list means all available MTs
    legendre_coeffs : List[int]
        List of Legendre coefficient indices to perturb (e.g., [0, 1, 2])
    num_samples : int
        Number of perturbed ENDF files to generate
    space : str, default "log"
        Sampling space: "linear" (factors = 1 + X) or "log" (factors = exp(Y))
    decomposition_method : str, default "svd"
        Matrix decomposition method: "svd", "cholesky", "eigen", or "pca"
    sampling_method : str, default "sobol"
        Sampling method: "sobol", "lhs", or "random"
    output_dir : str, default "."
        Output directory for perturbed files
    seed : Optional[int], default None
        Random seed for reproducible sampling
    nprocs : int, default 1
        Number of parallel processes (currently unused)
    dry_run : bool, default False
        If True, only generate perturbation factors without creating ENDF files
    verbose : bool, default True
        Enable verbose logging output
        
    Notes
    -----
    This function does not apply any autofix to the covariance matrices,
    using them exactly as provided in the MF34 data.
    """
    global _logger
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'endf_perturbation_{timestamp}.log')
    _logger = DualLogger(log_file)
    
    # Console: Basic start message
    print(f"[INFO] Starting ENDF perturbation job")
    print(f"[INFO] Log file: {log_file}")
    print(f"[INFO] Output directory: {os.path.abspath(output_dir)}")
    
    # Print run parameters as metadata TO LOG FILE
    separator = "=" * 80
    _logger.info(f"\n{separator}")
    _logger.info(f"[ENDF] [PARAMETERS] Run Configuration")
    _logger.info(f"{separator}")
    
    # Format and print input files
    if isinstance(endf_files, str):
        endf_files = [endf_files]
        _logger.info(f"ENDF files: {endf_files[0]}")
    else:
        _logger.info(f"ENDF files ({len(endf_files)}):")
        for i, f in enumerate(endf_files):
            _logger.info(f"  [{i+1}] {f}")
    
    if isinstance(mf34_cov_files, str):
        mf34_cov_files = [mf34_cov_files]
        _logger.info(f"MF34 covariance files: {mf34_cov_files[0]}")
    else:
        _logger.info(f"MF34 covariance files ({len(mf34_cov_files)}):")
        for i, f in enumerate(mf34_cov_files):
            _logger.info(f"  [{i+1}] {f}")
    
    _logger.info(f"MT reactions: {mt_list}")
    _logger.info(f"Legendre coefficients: {legendre_coeffs}")
    _logger.info(f"Number of samples: {num_samples}")
    _logger.info(f"Sampling space: {space}")
    _logger.info(f"Decomposition method: {decomposition_method}")
    _logger.info(f"Sampling method: {sampling_method}")
    _logger.info(f"Random seed: {seed}")
    _logger.info(f"Dry run: {dry_run}")
    _logger.info(f"{separator}")
    
    # Validate inputs
    if len(mf34_cov_files) != 1 and len(mf34_cov_files) != len(endf_files):
        raise ValueError("Number of MF34 covariance files must be 1 or equal to number of ENDF files")
    
    # Process each ENDF file
    all_factors = []
    all_param_mappings = []
    all_energy_grids = []
    
    for i, endf_file in enumerate(endf_files):
        _logger.info(f"\n[ENDF] Processing file {i+1}/{len(endf_files)}: {os.path.basename(endf_file)}")
        
        # Determine which covariance file to use
        cov_file = mf34_cov_files[0] if len(mf34_cov_files) == 1 else mf34_cov_files[i]
        
        # Load covariance matrix
        mf34_cov = load_mf34_covariance(cov_file)
        if mf34_cov is None:
            _logger.error(f"[ENDF] [COV] Failed to load covariance matrix from {cov_file}")
            continue
        
        # Filter covariance data by requested MTs and Legendre coefficients
        filtered_cov = _filter_mf34_covariance(mf34_cov, mt_list, legendre_coeffs)
        
        if filtered_cov.num_matrices == 0:
            _logger.warning(f"No covariance data found for requested MTs {mt_list} and L coefficients {legendre_coeffs}")
            continue
        
        # Create parameter mapping and energy grids
        param_mapping, energy_grids = _create_parameter_mapping(filtered_cov)
        
        # Generate perturbation factors
        _logger.info(f"Generating {num_samples} perturbation samples...")
        
        try:
            factors, _ = generate_endf_samples(
                filtered_cov,
                num_samples,
                space=space,
                decomposition_method=decomposition_method,
                sampling_method=sampling_method,
                seed=seed,
                mt_numbers=mt_list,
                verbose=verbose,
            )
            
            _logger.info(f"Successfully generated perturbation factors: shape {factors.shape}")
            
        except Exception as e:
            _logger.error(f"[ENDF] [SAMPLE] Failed to generate perturbation factors: {e}")
            continue
        
        # Store factors and mapping for master file
        all_factors.append(factors)
        all_param_mappings.append(param_mapping)
        all_energy_grids.append(energy_grids)
        
        # Process samples
        if not dry_run:
            _logger.info(f"Applying perturbations to {num_samples} samples...")
        else:
            _logger.info(f"Dry run: generating factor summaries for {num_samples} samples...")
        
        for sample_idx in range(num_samples):
            _process_sample(
                endf_file=endf_file,
                sample=factors[sample_idx],
                sample_index=sample_idx,
                energy_grids=energy_grids,
                param_mapping=param_mapping,
                output_dir=output_dir,
                dry_run=dry_run,
            )
    
    # Write master perturbation factors file
    master_file = _write_master_perturbation_file(
        all_factors, all_param_mappings, all_energy_grids, output_dir, timestamp, verbose
    )
    
    _logger.info(f"\n[ENDF] Perturbation job completed successfully")
    if master_file:
        print(f"[INFO] ENDF perturbation job completed")
        print(f"[INFO] Master matrix file: {os.path.basename(master_file)}")
    else:
        print(f"[INFO] ENDF perturbation job completed")


def apply_perturbation_factors_to_endf(
    endf, 
    sample: np.ndarray, 
    sample_index: int, 
    energy_grids: Dict[Tuple[int, int, int], List[float]], 
    param_mapping: List[Tuple[int, int, int, int]], 
    verbose: bool = True
):
    """
    Apply perturbation factors to ENDF MF4 angular distribution data.
    
    Parameters
    ----------
    endf : ENDF
        ENDF object with parsed data
    sample : np.ndarray
        Array of perturbation factors
    sample_index : int
        Index of this sample
    energy_grids : Dict[Tuple[int, int, int], List[float]]
        Energy grids for each parameter combination
    param_mapping : List[Tuple[int, int, int, int]]
        Mapping of sample indices to (isotope, mt, l, energy_bin) parameters
    verbose : bool
        Whether to log perturbation details
        
    Returns
    -------
    List[Tuple[int, int, int, int]]
        List of perturbed parameter combinations
    """
    if verbose and _get_logger():
        _get_logger().info(f"[ENDF] [SAMPLE] Applying perturbations to sample {sample_index + 1}")
    
    perturbed_params = []
    
    # Get MF4 data
    mf4 = endf.get_file(4)
    if mf4 is None:
        if _get_logger():
            _get_logger().warning("No MF4 data found in ENDF file")
        return perturbed_params
    
    # Apply perturbations to each MT section
    for mt_number, mt_data in mf4.sections.items():
        # Check for both MF4MTLegendre and MF4MTMixed (both have Legendre coefficients)
        if isinstance(mt_data, (MF4MTLegendre, MF4MTMixed)):
            # Apply perturbations to Legendre coefficients
            _apply_factors_to_mf4_legendre(
                mt_data, sample, param_mapping, energy_grids, verbose
            )
            # Add all perturbed parameters for this MT
            for isotope, mt, l_coeff, energy_bin in param_mapping:
                if mt == mt_number:
                    perturbed_params.append((isotope, mt, l_coeff, energy_bin))
    
    return perturbed_params


def _apply_factors_to_mf4_legendre(
    mt_data: MF4MTLegendre,
    factors: np.ndarray,
    param_mapping: List[Tuple[int, int, int, int]],
    energy_grids: Dict[Tuple[int, int, int], List[float]],
    verbose: bool = True
):
    """
    Apply perturbation factors to MF4 Legendre coefficient data.
    
    Parameters
    ----------
    mt_data : MF4MTLegendre
        MF4 MT section with Legendre coefficients
    factors : np.ndarray
        Perturbation factors
    param_mapping : List[Tuple[int, int, int, int]]
        Mapping of factor indices to parameters: (isotope, mt, l_coeff, energy_bin)
    energy_grids : Dict[Tuple[int, int, int], List[float]]
        Energy grids for each parameter combination
    verbose : bool
        Whether to log details
    """
    # Get the current coefficients
    current_coeffs = mt_data.legendre_coefficients
    energies = mt_data.legendre_energies
    
    if not current_coeffs or not energies:
        if verbose and _get_logger():
            _get_logger().warning(f"[ENDF] [MT{mt_data.number}] No Legendre coefficients found")
        return
    
    # Apply factors to each energy point and Legendre coefficient
    applied_count = 0
    for factor_idx, (isotope, mt, l_coeff, energy_bin) in enumerate(param_mapping):
        if mt != mt_data.number:
            continue
        
        if factor_idx >= len(factors):
            continue
        
        factor = factors[factor_idx]
        
        # Get the energy grid for this parameter triplet
        triplet = (isotope, mt, l_coeff)
        energy_grid = energy_grids.get(triplet, [])
        
        if len(energy_grid) < 2:  # Need at least 2 points to define bins
            if verbose and _get_logger():
                _get_logger().debug(f"Skipping {triplet} - insufficient energy grid points ({len(energy_grid)})")
            continue
        
        # Get energy bounds for this bin
        if energy_bin >= len(energy_grid) - 1:
            # This is a padding bin - skip it
            if verbose and _get_logger():
                _get_logger().debug(f"Skipping padding bin {energy_bin} for {triplet}")
            continue
            
        energy_low = energy_grid[energy_bin]
        energy_high = energy_grid[energy_bin + 1]
        
        # Apply factor to coefficients in this energy range
        applied_this_param = 0
        for energy_idx, energy in enumerate(energies):
            if energy_low <= energy < energy_high:
                # Check if this energy point has enough Legendre coefficients
                if energy_idx >= len(current_coeffs):
                    continue
                    
                # Check if this L coefficient exists at this energy
                # Note: MF34 uses L=1,2,3... but coefficient arrays are 0-indexed [L=0,L=1,L=2...]
                # So we need to convert: L=1 -> index 0, L=2 -> index 1, etc.
                coeff_index = l_coeff - 1
                if coeff_index < 0 or coeff_index >= len(current_coeffs[energy_idx]):
                    continue
                
                old_value = current_coeffs[energy_idx][coeff_index]
                current_coeffs[energy_idx][coeff_index] *= factor
                applied_this_param += 1
                applied_count += 1
                
                if verbose and _get_logger():
                    _get_logger().debug(
                        f"[ENDF] [APPLY] MT{mt} L{l_coeff} at {energy:.3e} MeV: "
                        f"factor {factor:.6f}, {old_value:.3e} -> {current_coeffs[energy_idx][coeff_index]:.3e}"
                    )
        

    if verbose and _get_logger():
        _get_logger().info(f"[ENDF] [MT{mt_data.number}] Applied {applied_count} perturbation factors")


def _find_energy_group(energy: float, energy_grid: List[float]) -> int:
    """
    Find the energy group index for a given energy.
    
    Parameters
    ----------
    energy : float
        Energy value
    energy_grid : List[float]
        Energy group boundaries
        
    Returns
    -------
    int
        Energy group index, or -1 if not found
    """
    for i in range(len(energy_grid) - 1):
        if energy_grid[i] <= energy < energy_grid[i + 1]:
            return i
    return -1


def _filter_mf34_covariance(
    mf34_cov: MF34CovMat, 
    mt_list: List[int], 
    legendre_coeffs: List[int]
) -> MF34CovMat:
    """
    Filter MF34 covariance data by MT reactions and Legendre coefficients.
    
    Parameters
    ----------
    mf34_cov : MF34CovMat
        Original MF34 covariance data
    mt_list : List[int]
        List of MT numbers to include (empty means all)
    legendre_coeffs : List[int]
        List of Legendre coefficient indices to include
        
    Returns
    -------
    MF34CovMat
        Filtered covariance data
    """
    if _get_logger():
        _get_logger().info(f"Filtering MF34 covariance data: MT={mt_list}, L={legendre_coeffs}")
        if not legendre_coeffs or legendre_coeffs == [-1]:
            _get_logger().info(f"Using all available Legendre coefficients: {sorted(mf34_cov.legendre_indices)}")
    
    filtered_cov = MF34CovMat()
    
    # If mt_list is empty, use all available MTs
    available_mts = mt_list if mt_list else list(mf34_cov.reactions)
    
    # If legendre_coeffs is empty or contains only -1, use all available L coefficients
    if not legendre_coeffs or legendre_coeffs == [-1]:
        available_ls = list(mf34_cov.legendre_indices)
        if _get_logger():
            _get_logger().info(f"Available L coefficients: {available_ls}")
    else:
        available_ls = legendre_coeffs
    
    for i in range(mf34_cov.num_matrices):
        mt_row = mf34_cov.reaction_rows[i]
        mt_col = mf34_cov.reaction_cols[i]
        l_row = mf34_cov.l_rows[i]
        l_col = mf34_cov.l_cols[i]
        
        # Check if this matrix should be included
        include_mt = (mt_row in available_mts and mt_col in available_mts)
        include_l = (l_row in available_ls and l_col in available_ls)
        
        if include_mt and include_l:
            filtered_cov.add_matrix(
                isotope_row=mf34_cov.isotope_rows[i],
                reaction_row=mt_row,
                l_row=l_row,
                isotope_col=mf34_cov.isotope_cols[i],
                reaction_col=mt_col,
                l_col=l_col,
                matrix=mf34_cov.matrices[i],
                energy_grid=mf34_cov.energy_grids[i]
            )
    
    if _get_logger():
        _get_logger().info(f"Filtered to {filtered_cov.num_matrices} matrices")
    
    return filtered_cov


def _create_parameter_mapping(mf34_cov: MF34CovMat) -> Tuple[List[Tuple[int, int, int, int]], Dict[Tuple[int, int, int], List[float]]]:
    """
    Create parameter mapping and energy grids from MF34 covariance data.
    Each parameter now includes energy bin index: (isotope, MT, l, energy_bin_index).
    
    The parameter mapping must match exactly how MF34CovMat.covariance_matrix builds
    the full covariance matrix: each (isotope, MT, l) triplet gets max_G parameters,
    one for each energy bin, arranged in the same order as param_triplets.
    
    However, we mark which parameters correspond to actual vs padding bins.
    
    Parameters
    ----------
    mf34_cov : MF34CovMat
        MF34 covariance data
        
    Returns
    -------
    Tuple[List[Tuple[int, int, int, int]], Dict[Tuple[int, int, int], List[float]]]
        Parameter mapping with energy bins and energy grids
    """
    # Get unique parameter triplets in the same order as covariance matrix
    param_triplets = mf34_cov._get_param_triplets()
    
    # Create energy grids dictionary
    energy_grids = {}
    
    for i in range(mf34_cov.num_matrices):
        # Add row parameters
        row_param = (mf34_cov.isotope_rows[i], mf34_cov.reaction_rows[i], mf34_cov.l_rows[i])
        energy_grids[row_param] = mf34_cov.energy_grids[i]
        
        # Add column parameters if different
        col_param = (mf34_cov.isotope_cols[i], mf34_cov.reaction_cols[i], mf34_cov.l_cols[i])
        if col_param != row_param:
            energy_grids[col_param] = mf34_cov.energy_grids[i]
    
    # Find maximum energy grid size across all matrices
    max_G = max(matrix.shape[0] for matrix in mf34_cov.matrices) if mf34_cov.matrices else 0
    
    # Create parameter mapping that matches covariance matrix structure exactly
    param_mapping = []
    
    for triplet in param_triplets:
        isotope, mt, l = triplet
        # Add max_G parameters for this triplet, one for each energy bin
        for energy_bin in range(max_G):
            param_mapping.append((isotope, mt, l, energy_bin))
    
    if _get_logger():
        _get_logger().info(f"Created parameter mapping with {len(param_mapping)} parameters (including energy bins)")
        _get_logger().info(f"Parameter triplets: {len(param_triplets)} unique (isotope, MT, l) combinations")
        _get_logger().info(f"Max energy groups: {max_G}")
        _get_logger().info(f"Total parameters: {len(param_triplets)} × {max_G} = {len(param_mapping)}")
        if param_mapping:
            _get_logger().info(f"Sample parameters: {param_mapping[:3]}{'...' if len(param_mapping) > 3 else ''}")
    
    return param_mapping, energy_grids


def _write_sample_summary(
    sample: np.ndarray,
    sample_index: int,
    energy_grids: Dict[Tuple[int, int, int], List[float]],
    param_mapping: List[Tuple[int, int, int, int]],
    sample_dir: str,
    base: str,
):
    """
    Write a summary file for a single sample.
    
    Parameters
    ----------
    sample : np.ndarray
        Perturbation factors for this sample
    sample_index : int
        Sample index
    energy_grids : Dict[Tuple[int, int, int], List[float]]
        Energy grids for each parameter
    param_mapping : List[Tuple[int, int, int, int]]
        Parameter mapping with energy bins
    sample_dir : str
        Sample output directory
    base : str
        Base filename
    """
    sample_str = f"{sample_index+1:04d}"
    summary_file = os.path.join(sample_dir, f"{base}_{sample_str}_summary.txt")
    
    with open(summary_file, 'w') as f:        
        for i, factor in enumerate(sample):
            if i < len(param_mapping):
                isotope, mt, l_coeff, energy_bin = param_mapping[i]
                
                # Get energy bounds for this parameter
                triplet = (isotope, mt, l_coeff)
                energy_grid = energy_grids.get(triplet, [])
                
                # Skip padding bins - only write actual energy bins
                if energy_bin < len(energy_grid) - 1:
                    energy_low = energy_grid[energy_bin]
                    energy_high = energy_grid[energy_bin + 1]
                    
                    # Write with proper alignment for negative numbers
                    # Format: MT(6) L_coeff(6) Energy_Low(15) Energy_High(15) Factor(15)
                    f.write(f"{mt:6d}{l_coeff:6d}{energy_low:15.6e}{energy_high:15.6e}{factor:15.6e}\n")


def _write_master_perturbation_file(
    all_factors: List[np.ndarray],
    all_param_mappings: List[List[Tuple[int, int, int, int]]],
    all_energy_grids: List[Dict[Tuple[int, int, int], List[float]]],
    output_dir: str,
    timestamp: str,
    verbose: bool = True
):
    """
    Write master perturbation factors file in Parquet format.
    
    The format matches ACE perturbation: each row is a sample, each column is a parameter.
    Column names follow the format: Fe56_MT2_L1_E0-E1
    
    Parameters
    ----------
    all_factors : List[np.ndarray]
        List of factor arrays for each ENDF file
    all_param_mappings : List[List[Tuple[int, int, int, int]]]
        List of parameter mappings for each ENDF file with energy bins
    all_energy_grids : List[Dict[Tuple[int, int, int], List[float]]]
        List of energy grids for each ENDF file
    output_dir : str
        Output directory
    timestamp : str
        Timestamp for filename
    verbose : bool
        Whether to log progress
    """
    if not all_factors:
        if verbose and _get_logger():
            _get_logger().info("[ENDF] [OUTPUT] No factors to write to master file")
        return
    
    # Determine the maximum number of samples across all files
    max_samples = max(factors.shape[0] for factors in all_factors)
    
    # Initialize the master DataFrame with Sample_ID column
    master_data = {'Sample_ID': np.arange(1, max_samples + 1, dtype='int32')}
    
    # Process each ENDF file
    for file_idx, (factors, param_mapping, energy_grids) in enumerate(zip(all_factors, all_param_mappings, all_energy_grids)):
        # Convert factors to float32 for consistency
        factors = factors.astype(np.float32)
        
        # Create columns for each parameter in this file
        for param_idx, (isotope, mt, l_coeff, energy_bin) in enumerate(param_mapping):
            # Get element symbol from ZAID
            symbol = zaid_to_symbol(isotope)
            
            # Column name format: Fe56_MT2_L1_E0-E1
            col_name = f"{symbol}_MT{mt}_L{l_coeff}_E{energy_bin}-E{energy_bin+1}"
            
            # Create column data, padding with NaN if this file has fewer samples
            column_data = np.full(max_samples, np.nan, dtype=np.float32)
            if factors.shape[0] > 0:
                column_data[:factors.shape[0]] = factors[:, param_idx]
            
            master_data[col_name] = column_data
    
    # Create DataFrame and save as Parquet
    df = pd.DataFrame(master_data)
    master_file = os.path.join(output_dir, f'endf_perturbation_factors_{timestamp}.parquet')
    df.to_parquet(master_file, index=False, compression='zstd', compression_level=3)
    
    if verbose and _get_logger():
        _get_logger().info(f"[ENDF] [OUTPUT] Master perturbation factors saved to: {master_file}")
        _get_logger().info(f"[ENDF] [OUTPUT] DataFrame shape: {df.shape}")
        _get_logger().info(f"[ENDF] [OUTPUT] Columns: {len(df.columns)} (including Sample_ID)")
        _get_logger().info(f"[ENDF] [OUTPUT] Samples: {max_samples}")
    
    return master_file
