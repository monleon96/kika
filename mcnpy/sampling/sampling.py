import numpy as np
import os
import time
from typing import List, Optional, Tuple, Dict, Union
from mcnpy.ace.classes.ace import Ace
from mcnpy.cov.covmat import CovMat
from mcnpy.sampling.generators import generate_samples
from mcnpy.ace.writers.write_ace import write_ace
from mcnpy.ace.parsers.parse_ace import read_ace
import datetime

def process_sample(
    i: int,
    sample_factors: np.ndarray,
    ace_file_path: str,
    mt_numbers: List[int],
    original_mt_numbers: List[int],
    mt_to_original_map: Dict[int, int],
    energy_grid: np.ndarray,
    base_name: str,
    extension: str,
    output_dir: Optional[str],
    precomputed_mappings: Dict[int, np.ndarray]
) -> Tuple[int, float, bool, Optional[str], float, float, float]:
    """
    Process a single sample and create a perturbed ACE file.
    
    Parameters
    ----------
    i : int
        Sample index
    sample_factors : np.ndarray
        Perturbation factors for this sample
    ace_file_path : str
        Path to the original ACE file to read
    mt_numbers : List[int]
        List of MT numbers to perturb (may include expanded MTs like 51-91)
    original_mt_numbers : List[int]
        Original MT numbers requested (for indexing perturbation factors)
    mt_to_original_map : Dict[int, int]
        Mapping from expanded MT numbers to original MT numbers
    energy_grid : np.ndarray
        Energy grid for perturbation
    base_name : str
        Base filename for output
    extension : str
        File extension for output
    output_dir : str, optional
        Directory to save output file
    precomputed_mappings : Dict[int, np.ndarray]
        Precomputed bin indices for each MT
        
    Returns
    -------
    Tuple[int, float, bool, Optional[str], float, float, float]
        Sample index, total elapsed time, success flag, error message (if any), 
        read time, perturbation time, write time
    """
    start_time = time.time()
    
    # Create file name with 4-digit sample number
    file_name = f"{base_name}_{i+1:04d}{extension}"
    
    # Create full path
    if output_dir:
        file_path = os.path.join(output_dir, file_name)
    else:
        file_path = file_name
    
    read_time = 0.0
    perturb_time = 0.0
    write_time = 0.0
    
    try:
        # Read the ACE file with timing
        read_start = time.time()
        perturbed_ace = read_ace(ace_file_path)
        read_time = time.time() - read_start
        
        # Apply perturbation factors to each MT with timing
        perturb_start = time.time()
        for mt in mt_numbers:
            # Get the original MT to use for perturbation factors
            orig_mt = mt_to_original_map.get(mt, mt)
            orig_mt_idx = original_mt_numbers.index(orig_mt)
            
            # Calculate the start and end indices for this MT in the perturbation factors
            start_idx = orig_mt_idx * (len(energy_grid) - 1)  # Adjust for bins vs grid points
            end_idx = start_idx + (len(energy_grid) - 1)
            
            # Get perturbation factors for this MT
            mt_factors = sample_factors[start_idx:end_idx]
            
            # Use optimized apply perturbation with precomputed mappings
            apply_perturbation_to_mt(
                perturbed_ace, 
                mt, 
                mt_factors, 
                energy_grid, 
                precomputed_mappings.get(mt, None)
            )
        perturb_time = time.time() - perturb_start
        
        # Write the perturbed ACE file with timing
        write_start = time.time()
        write_ace(perturbed_ace, file_path, overwrite=True)
        write_time = time.time() - write_start
        
        elapsed = time.time() - start_time
        
        # Clean up to free memory
        del perturbed_ace
        
        return (i, elapsed, True, None, read_time, perturb_time, write_time)
    except Exception as e:
        elapsed = time.time() - start_time
        return (i, elapsed, False, str(e), read_time, perturb_time, write_time)

def write_perturbation_data(
    output_dir: str,
    base_name: str,
    isotope_id: int,
    mt_numbers: List[int],
    expanded_mt_numbers: List[int],
    mt_to_original_map: Dict[int, int],
    energy_grid: np.ndarray,
    seed: Optional[int],
    decomposition_method: str,
    sampling_method: str,
    perturbation_factors: np.ndarray,
    num_samples: int
) -> str:
    """
    Write perturbation data to a text file for reproducibility.
    
    Parameters
    ----------
    output_dir : str
        Directory to save the file
    base_name : str
        Base name for the file
    isotope_id : int
        Isotope ID
    mt_numbers : List[int]
        Original list of MT numbers requested for perturbation
    expanded_mt_numbers : List[int]
        Expanded list of MT numbers actually perturbed (may include MT=51-91 instead of MT=4)
    mt_to_original_map : Dict[int, int]
        Mapping from expanded MT numbers to original MT numbers
    energy_grid : np.ndarray
        Energy grid used for perturbation
    seed : Optional[int]
        Random seed used
    decomposition_method : str
        Method used to decompose the covariance matrix
    sampling_method : str
        Method used to generate samples
    perturbation_factors : np.ndarray
        Perturbation factors for each sample
    num_samples : int
        Number of samples
        
    Returns
    -------
    str
        Path to the created file
    """
    # Create filename without timestamp
    file_name = f"{base_name}_perturbation_data.txt"
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, 'w') as f:
        # Write header and parameters
        f.write("=" * 80 + "\n")
        f.write("PERTURBATION DATA\n")
        f.write("=" * 80 + "\n\n")
        
        # Include creation timestamp in the file content but not in the filename
        f.write("PARAMETERS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Isotope ID: {isotope_id}\n")
        f.write(f"Original MT numbers requested: {mt_numbers}\n")
        
        # Add information about MT expansion (especially MT=4 to MT=51-91)
        if set(mt_numbers) != set(expanded_mt_numbers):
            f.write("\nMT NUMBER EXPANSIONS:\n")
            # Group expanded MTs by original MT
            original_to_expanded = {}
            for exp_mt, orig_mt in mt_to_original_map.items():
                if orig_mt not in original_to_expanded:
                    original_to_expanded[orig_mt] = []
                original_to_expanded[orig_mt].append(exp_mt)
            
            # Write expansion information
            for orig_mt, exp_mts in original_to_expanded.items():
                if len(exp_mts) > 1:  # Only show if one MT was expanded to multiple MTs
                    exp_mts.sort()
                    f.write(f"MT={orig_mt} expanded to: {exp_mts}\n")
            
            # Write special note for MT=4 expansion
            if 4 in mt_numbers and 4 not in expanded_mt_numbers:
                inelastic_mts = sorted([mt for mt in expanded_mt_numbers if mt_to_original_map.get(mt) == 4])
                if inelastic_mts:
                    f.write(f"\nNOTE: MT=4 (inelastic) not found in ACE file.\n")
                    f.write(f"      MT=4 perturbation was applied to {len(inelastic_mts)} discrete inelastic levels:\n")
                    f.write(f"      MT={min(inelastic_mts)}-{max(inelastic_mts)}: {inelastic_mts}\n")
        
        f.write(f"\nActual MT numbers perturbed: {expanded_mt_numbers}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Sampling method: {sampling_method}\n")
        f.write(f"Decomposition method: {decomposition_method}\n")
        f.write(f"Random seed: {seed if seed is not None else 'None (random)'}\n")
        f.write(f"Energy grid points: {len(energy_grid)}\n")
        f.write(f"Energy range: {energy_grid[0]:.2e} to {energy_grid[-1]:.2e} MeV\n\n")
        
        # Write energy grid for reference
        f.write("ENERGY GRID (MeV):\n")
        f.write("-" * 80 + "\n")
        for i, energy in enumerate(energy_grid):
            f.write(f"{energy:.6e}")
            if (i+1) % 8 == 0 or i == len(energy_grid) - 1:
                f.write("\n")
            else:
                f.write("  ")
        f.write("\n")
        
        # Write perturbation factors for each MT and energy bin
        f.write("\nPERTURBATION FACTORS:\n")
        f.write("-" * 80 + "\n")
        
        # Calculate header width based on number of samples
        header = "MT  | Lower E (MeV) | Upper E (MeV) |"
        for s in range(num_samples):
            header += f" Sample_{s+1:04d} |"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        # Write data for each MT and energy bin with increased precision
        for mt_idx, mt in enumerate(mt_numbers):
            num_bins = len(energy_grid) - 1
            start_idx = mt_idx * num_bins
            
            for bin_idx in range(num_bins):
                # Get perturbation factors for this MT and bin across all samples
                bin_factors = [perturbation_factors[s][start_idx + bin_idx] for s in range(num_samples)]
                
                # Format row with 12 decimal places instead of 6
                row = f"{mt:4d} | {energy_grid[bin_idx]:.6e} | {energy_grid[bin_idx+1]:.6e} |"
                for factor in bin_factors:
                    row += f" {factor:.12f} |"
                f.write(row + "\n")
            
            # Add a separator between MTs
            f.write("-" * len(header) + "\n")
    
    return file_path

def create_perturbed_ace_files(
    ace_file_path: str,
    mt_numbers: List[int],
    energy_grid: np.ndarray,
    covmat: CovMat,
    num_samples: int,
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    output_dir: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: bool = False
) -> None:
    """
    Create perturbed ACE files based on covariance data.
    Memory-optimized version that processes and writes files one by one.
    
    Parameters
    ----------
    ace_file_path : str
        Path to the ACE file to be perturbed
    mt_numbers : List[int]
        List of MT numbers to perturb
    energy_grid : np.ndarray
        Energy grid to be used for perturbation
    covmat : CovMat
        Covariance matrix object containing covariance data
    num_samples : int
        Number of perturbed samples to generate
    decomposition_method : str, default="svd"
        Method to decompose the covariance matrix ("svd", "cholesky")
    sampling_method : str, default="sobol"
        Method to generate the samples ("random", "lhs", or "sobol")
    output_dir : str, optional
        Directory to save the perturbed ACE files. If None, uses the current directory.
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to print detailed information during the execution
    """
    # Start timing the entire process
    overall_start_time = time.time()
    
    if verbose:
        print(f"Starting perturbation of {num_samples} ACE files...")
        print(f"Perturbing MT numbers: {mt_numbers}")
        print(f"Using {sampling_method} sampling with {decomposition_method} decomposition")
    
    # Input validation
    if not mt_numbers:
        raise ValueError("At least one MT number must be provided")
    
    # Read the ACE file once to extract necessary information
    if verbose:
        print(f"Reading ACE file: {ace_file_path}")
    
    ace_object = read_ace(ace_file_path)
    
    if verbose:
        print("ACE file read successfully for initial information extraction")
    
    # Extract isotope from ACE object
    isotope_id = ace_object.header.zaid
    
    if verbose:
        print(f"Extracted isotope ID: {isotope_id}")
    
    # Check if MT=4 is requested but not available, and expand to MT=51-91 if needed
    original_mt_numbers = mt_numbers.copy()  # Keep original MTs for covariance matrix
    expanded_mt_numbers = []
    mt_to_original_map = {}  # Maps expanded MTs back to original MTs
    
    # Get available MTs in the ACE file
    available_mts = ace_object.mt_numbers if hasattr(ace_object, 'mt_numbers') else []
    if hasattr(ace_object, 'cross_section') and ace_object.cross_section:
        available_mts.extend(list(ace_object.cross_section.reaction.keys()))
    available_mts = list(set(available_mts))  # Remove duplicates
    
    for mt in mt_numbers:
        if mt == 4 and mt not in available_mts:
            # MT=4 requested but not available, check for MTs 51-91
            inelastic_mts = [mt_num for mt_num in range(51, 92) if mt_num in available_mts]
            
            if inelastic_mts:
                if verbose:
                    print(f"WARNING: MT=4 not found in ACE file but was requested for perturbation.")
                    print(f"         Applying MT=4 perturbation to discrete inelastic levels (MT={min(inelastic_mts)}-{max(inelastic_mts)})")
                    print(f"         Found {len(inelastic_mts)} discrete inelastic levels: {inelastic_mts}")
                
                expanded_mt_numbers.extend(inelastic_mts)
                # Map each inelastic MT back to MT=4 for perturbation factors
                for inelastic_mt in inelastic_mts:
                    mt_to_original_map[inelastic_mt] = 4
            else:
                if verbose:
                    print(f"WARNING: MT=4 not found in ACE file and no discrete inelastic levels (MT=51-91) found.")
                    print(f"         Skipping perturbation of MT=4.")
        else:
            expanded_mt_numbers.append(mt)
            mt_to_original_map[mt] = mt  # Map to itself
    
    # Use original MTs for covariance matrix and perturbation generation
    if verbose:
        print("Extracting covariance matrix...")
    
    cov_start_time = time.time()
    combined_cov_matrix = extract_covariance_matrix(covmat, isotope_id, original_mt_numbers, energy_grid)
    cov_time = time.time() - cov_start_time
    
    if verbose:
        print(f"Covariance matrix extracted in {cov_time:.2f} seconds")
    
    # Generate perturbation factors using the specified sampling method
    if verbose:
        print(f"Generating {num_samples} perturbation factors...")
    
    generation_start_time = time.time()
    perturbation_factors = generate_samples(
        combined_cov_matrix, 
        num_samples, 
        decomposition_method,
        sampling_method,
        seed
    )
    generation_time = time.time() - generation_start_time
    
    if verbose:
        print(f"Perturbation factors generated in {generation_time:.2f} seconds")
    
    # Prepare output directory if needed
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            if verbose:
                print(f"Created output directory: {output_dir}")
        elif verbose:
            print(f"Using existing output directory: {output_dir}")
    else:
        output_dir = "."  # Use current directory if none specified
    
    # Set up base file name
    base_name = "perturbed_ace"
    if ace_object.filename:
        base_name, extension = os.path.splitext(os.path.basename(ace_object.filename))
    elif ace_object.header and ace_object.header.zaid:
        base_name = f"{ace_object.header.zaid}"
        extension = ".ace"
    else:
        extension = ".ace"
    
    # Write perturbation data to file for reproducibility (always write this regardless of verbose)
    if verbose:
        print("Writing perturbation data to file...")
    
    data_file_path = write_perturbation_data(
        output_dir,
        base_name,
        isotope_id,
        original_mt_numbers,  # Original MT numbers requested
        expanded_mt_numbers,  # Expanded MT numbers (may include MT=51-91 instead of MT=4)
        mt_to_original_map,   # Mapping from expanded MTs to original MTs
        energy_grid,
        seed,
        decomposition_method,
        sampling_method,
        perturbation_factors,
        num_samples
    )
    
    if verbose:
        print(f"Perturbation data written to: {data_file_path}")
    
    # OPTIMIZATION: Precompute mappings from ace energies to perturbation grid bins
    precomp_start_time = time.time()
    precomputed_mappings = {}
    
    for mt in expanded_mt_numbers:
        try:
            if ace_object.cross_section and mt in ace_object.cross_section.reaction:
                reaction_xs = ace_object.cross_section.reaction[mt]
                
                if reaction_xs._energy_entries:
                    ace_energies = np.array([e.value for e in reaction_xs._energy_entries])
                    
                    # Precompute bin indices for this reaction
                    bin_indices = np.searchsorted(energy_grid, ace_energies) - 1
                    bin_indices = np.clip(bin_indices, 0, len(energy_grid) - 2)  # len-2 because we're dealing with bins
                    
                    precomputed_mappings[mt] = bin_indices
        except Exception as e:
            if verbose:
                print(f"Warning: Could not precompute mappings for MT={mt}: {str(e)}")
    
    precomp_time = time.time() - precomp_start_time
    
    if verbose:
        print(f"Precomputed mappings in {precomp_time:.2f} seconds")
        print("Cleaning up initial ACE object to free memory")
    
    # Clean up the initial ACE object to free memory
    del ace_object
    
    # Process samples sequentially
    if verbose:
        print("\nStarting ACE file perturbations:")
        print("-" * 60)
    
    results = []
    read_times = []
    perturb_times = []
    write_times = []
    
    for i in range(num_samples):
        if verbose:
            print(f"Processing sample {i+1}/{num_samples}... ", end="", flush=True)
        
        # Process the sample
        result = process_sample(
            i, 
            perturbation_factors[i], 
            ace_file_path,
            expanded_mt_numbers,  # Use expanded list that may include MT=51-91 instead of MT=4
            original_mt_numbers,  # Original list for indexing perturbation factors
            mt_to_original_map,   # Mapping to track which original MT to use for each expanded MT
            energy_grid,
            base_name, 
            extension, 
            output_dir, 
            precomputed_mappings
        )
        i, elapsed, success, error, read_time, perturb_time, write_time = result
        
        if success:
            if verbose:
                print(f"done in {elapsed:.2f} seconds")
            results.append((i, elapsed))
            read_times.append(read_time)
            perturb_times.append(perturb_time)
            write_times.append(write_time)
        else:
            # Always print errors regardless of verbose setting
            print(f"ERROR processing sample {i+1}: {error}")

    # Calculate and print summary statistics if we have results
    if results and verbose:
        elapsed_times = [t for _, t in results]
        overall_time = time.time() - overall_start_time
        avg_time = sum(elapsed_times) / len(elapsed_times)
        
        avg_read = sum(read_times) / len(read_times) if read_times else 0
        avg_perturb = sum(perturb_times) / len(perturb_times) if perturb_times else 0
        avg_write = sum(write_times) / len(write_times) if write_times else 0
        
        print("-" * 60)
        print(f"Perturbation complete!")
        print(f"Total files created: {len(results)}/{num_samples}")
        print(f"Total time: {overall_time:.2f} seconds")
        print(f"Average time per file: {avg_time:.2f} seconds")
        print("-" * 40)
        print("Performance breakdown (average per file):")
        print(f"  Read ACE file:       {avg_read:.4f} seconds ({100*avg_read/avg_time:.1f}%)")
        print(f"  Apply perturbation:  {avg_perturb:.4f} seconds ({100*avg_perturb/avg_time:.1f}%)")
        print(f"  Write to file:       {avg_write:.4f} seconds ({100*avg_write/avg_time:.1f}%)")
        print(f"  Other:               {avg_time-avg_read-avg_perturb-avg_write:.4f} seconds ({100*(avg_time-avg_read-avg_perturb-avg_write)/avg_time:.1f}%)")
        
        # Also show the initial stages
        print("-" * 40)
        print("Initial stages:")
        print(f"  Covariance extraction: {cov_time:.2f} seconds")
        print(f"  Generating samples:    {generation_time:.2f} seconds")
        print(f"  Precomputing mappings: {precomp_time:.2f} seconds")
        
        if output_dir:
            print(f"Files saved to: {output_dir}")
        print("-" * 60)

def apply_perturbation_to_mt(
    ace_object: Ace, 
    mt: int, 
    perturbation_factors: np.ndarray,
    perturbation_energy_grid: np.ndarray,
    precomputed_bin_indices: Optional[np.ndarray] = None
) -> None:
    """
    Optimized version of apply_perturbation_to_mt that can use precomputed bin indices.
    
    Parameters
    ----------
    ace_object : Ace
        ACE object to modify
    mt : int
        MT number to perturb
    perturbation_factors : np.ndarray
        Perturbation factors for this MT
    perturbation_energy_grid : np.ndarray
        Energy grid used for perturbation
    precomputed_bin_indices : np.ndarray, optional
        Precomputed mapping from ACE energies to perturbation grid bins
    """
    # Check if MT exists in cross section data
    if ace_object.cross_section is None or mt not in ace_object.cross_section.reaction:
        raise ValueError(f"MT={mt} not found in ACE cross section data")
    
    # Get the reaction cross section object
    reaction_xs = ace_object.cross_section.reaction[mt]
    
    # Ensure we have XS entries
    if not reaction_xs._xs_entries:
        raise ValueError(f"No cross section values found for MT={mt}")
    
    if precomputed_bin_indices is not None:
        # Use precomputed bin indices - much faster!
        bin_indices = precomputed_bin_indices
    else:
        # Get ACE energy points for this reaction
        if not reaction_xs._energy_entries:
            raise ValueError(f"No energy grid found for MT={mt}")
        
        # Compute mappings on the fly - slower but works if precomputed mappings not available
        ace_energies = np.array([e.value for e in reaction_xs._energy_entries])
        bin_indices = np.searchsorted(perturbation_energy_grid, ace_energies) - 1
        bin_indices = np.clip(bin_indices, 0, len(perturbation_factors) - 1)
    
    # OPTIMIZATION: Use vectorized operations where possible
    if len(bin_indices) <= len(perturbation_factors):
        # Map perturbation factors to each ACE energy point using numpy indexing
        mapped_factors = perturbation_factors[bin_indices]
        
        # Apply perturbation factors to each XS value
        for i, xs_entry in enumerate(reaction_xs._xs_entries):
            if i < len(mapped_factors):
                xs_entry.value *= mapped_factors[i]
    else:
        # Handle case where bin_indices is longer than perturbation_factors
        # This shouldn't happen with proper precomputation but added as a safeguard
        for i, xs_entry in enumerate(reaction_xs._xs_entries):
            if i < len(bin_indices) and bin_indices[i] < len(perturbation_factors):
                xs_entry.value *= perturbation_factors[bin_indices[i]]


def extract_covariance_matrix(
    covmat: CovMat, 
    isotope_id: int, 
    mt_numbers: List[int],
    energy_grid: np.ndarray
) -> np.ndarray:
    """
    Extract combined covariance matrix for specified isotope and MT numbers.
    
    Parameters
    ----------
    covmat : CovMat
        Covariance matrix object
    isotope_id : int
        Isotope ID
    mt_numbers : List[int]
        List of MT numbers
    energy_grid : np.ndarray
        Energy grid for perturbation
        
    Returns
    -------
    np.ndarray
        Combined covariance matrix
    
    Raises
    ------
    ValueError
        If covariance data is not available for the isotope or MT numbers,
        or if dimensions don't match
    """
    try:
        # Use the existing method from CovMat
        combined_cov = covmat.get_isotope_covariance_matrix(isotope_id, mt_numbers)
        
        # Verify dimensions - energy grid contains n+1 points for n bins
        # The covariance matrix is organized by bins, not grid points
        num_bins = len(energy_grid) - 1
        expected_dim = num_bins * len(mt_numbers)
        actual_dim = combined_cov.shape[0]
        
        if actual_dim != expected_dim:
            raise ValueError(
                f"Covariance matrix dimension mismatch: "
                f"Expected {expected_dim} (energy bins {num_bins} × "
                f"{len(mt_numbers)} reactions), but got {actual_dim}"
            )
        
        return combined_cov
    except Exception as e:
        raise ValueError(f"Failed to extract covariance matrix: {str(e)}")