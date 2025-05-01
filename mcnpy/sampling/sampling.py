import numpy as np
import os
import time
from typing import List, Optional, Tuple, Dict, Union, Set
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

def create_perturbed_ace_files(
    ace_file_path: Union[str, List[str]],
    mt_numbers: List[int],
    energy_grid: np.ndarray,
    covmat: Union[CovMat, List[CovMat]],
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
    ace_file_path : Union[str, List[str]]
        Path to the ACE file to be perturbed
    mt_numbers : List[int]
        List of MT numbers to perturb (e.g., [1, 4, 18])
    energy_grid : np.ndarray
        Energy grid to be used for perturbation
    covmat : Union[CovMat, List[CovMat]]
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
    # Validate list‐type mismatch: covmat list only allowed if ace_file_path is a list
    if isinstance(covmat, (list, tuple)) and not isinstance(ace_file_path, (list, tuple)):
        raise ValueError("Provided multiple CovMat objects but a single ACE path; lengths must match or both be singular.")

    # Handle multiple ACE files at once
    if isinstance(ace_file_path, (list, tuple)):
        ace_paths = list(ace_file_path)
        covmats = covmat if isinstance(covmat, (list, tuple)) else [covmat] * len(ace_paths)
        if len(covmats) != len(ace_paths):
            raise ValueError("Number of covariance matrices must match number of ACE file paths")
        for path, cov in zip(ace_paths, covmats):
            create_perturbed_ace_files(
                path,
                mt_numbers,
                energy_grid,
                cov,
                num_samples,
                decomposition_method,
                sampling_method,
                output_dir,
                seed,
                verbose
            )
        return

    # Start timing the entire process
    overall_start_time = time.time()
    
    if verbose:
        print("\n=== Perturbation Setup ===")
        print(f"Number of samples       : {num_samples}")
        print(f"Requested MT numbers    : {mt_numbers}")
        print(f"Sampling method         : {sampling_method}")
        print(f"Decomposition method    : {decomposition_method}\n")
    
    if verbose:
        print("=== Reading Input Files ===")
        print(f"ACE file path           : {ace_file_path}\n")
    
    # Input validation
    if not mt_numbers:
        raise ValueError("At least one MT number must be provided")
    
    # Read the ACE file once to extract necessary information
    if verbose:
        print(f"Reading ACE file: {ace_file_path}")
    
    ace_object = read_ace(ace_file_path)
    
    if verbose:
        print("Initial ACE load complete.\n")
    
    # Extract isotope from ACE object
    isotope_id = ace_object.header.zaid

    # Get available MTs in the covariance matrix for this isotope
    isotope_reactions = covmat.get_isotope_reactions().get(isotope_id, set())

    if verbose:
        print(f"Isotope ZAID            : {isotope_id}")
        print(f"Covariance MTs available: {sorted(list(isotope_reactions))}\n")
    
    # Get available MTs in the ACE file
    available_mts = set()
    if hasattr(ace_object, 'mt_numbers'):
        available_mts.update(ace_object.mt_numbers)
    if hasattr(ace_object, 'cross_section') and ace_object.cross_section:
        available_mts.update(ace_object.cross_section.reaction.keys())
    
    if verbose:
        print(f"MTs available in ACE file: {sorted(list(available_mts))}\n")
    
    # Determine which MTs to actually perturb based on request and availability
    original_mt_numbers = sorted(set(mt_numbers))
    expanded_mt_numbers = []
    mt_to_original_map = {}
    skipped_mts = []
    mt_warnings = {}

    # Simplified inelastic handling: if user requests MT=4 or any MT in 51-91,
    # perturb MT=4 (if present) and all available MTs 51-91 using MT=4 covariance
    if any(mt == 4 or 51 <= mt <= 91 for mt in original_mt_numbers):
        # Warn for individual inelastic requests
        for mt in original_mt_numbers:
            if 51 <= mt <= 91:
                warning_msg = (
                    f"MT={mt} requested; treating as MT=4. "
                    "Perturbing all MT=4 and available MTs 51-91 with MT=4 covariance."
                )
                print(f"WARNING: {warning_msg}")
                mt_warnings[mt] = warning_msg

        # Ensure MT=4 covariance exists
        if not covmat.has_isotope_mt(isotope_id, 4):
            warning_msg = (
                f"MT=4 covariance data required but not found for isotope {isotope_id}. "
                "Skipping MT=4 and inelastic levels."
            )
            if verbose:
                print(f"WARNING: {warning_msg}")
            mt_warnings[4] = warning_msg
            skipped_mts.extend([m for m in original_mt_numbers if m == 4 or 51 <= m <= 91])
        else:
            # Identify what's in the ACE file
            mt4_present = 4 in available_mts
            inelastic_mts_present = [m for m in range(51, 92) if m in available_mts]
            # Map MT=4
            if mt4_present:
                expanded_mt_numbers.append(4)
                mt_to_original_map[4] = 4
            # Map each available inelastic level back to MT=4
            for m in inelastic_mts_present:
                expanded_mt_numbers.append(m)
                mt_to_original_map[m] = 4
            if verbose:
                detail = (
                    f"MT=4{' present' if mt4_present else ' not present'}, "
                    f"inelastic levels found: {inelastic_mts_present}"
                )
                print(f"  - Inelastic handling: {detail}. Using MT=4 covariance for all.\n")

        # Handle all other requested MTs normally
        for mt in original_mt_numbers:
            if mt == 4 or 51 <= mt <= 91:
                continue
            mt_in_covmat = covmat.has_isotope_mt(isotope_id, mt)
            if mt in available_mts and mt_in_covmat:
                expanded_mt_numbers.append(mt)
                mt_to_original_map[mt] = mt
                if verbose:
                    print(f"  - MT={mt} requested, found in ACE file and covariance data.")
            elif mt in available_mts:
                warning_msg = f"MT={mt} requested, found in ACE file but not in covariance data. Skipping."
                print(f"WARNING: {warning_msg}")
                mt_warnings[mt] = warning_msg
            else:
                missing_msg = f"MT={mt} requested but not found in ACE file. Skipping."
                print(f"WARNING: {missing_msg}")
                mt_warnings[mt] = missing_msg
    else:
        # ...existing per‑MT availability check and mapping logic here...
        # Check each requested MT for presence in covmat
        for mt in original_mt_numbers:
            mt_in_covmat = covmat.has_isotope_mt(isotope_id, mt)
            if mt == 4:
                # Special handling for MT=4: perturb MT=4 and/or MT=51-91 if they exist
                mt4_present = 4 in available_mts
                inelastic_mts_present = sorted([m for m in range(51, 92) if m in available_mts])
                
                if not mt_in_covmat:
                    warning_msg = f"MT=4 requested but not found in covariance matrix for isotope {isotope_id}."
                    print(f"WARNING: {warning_msg}")
                    mt_warnings[4] = warning_msg
                    skipped_mts.append(4)
                    continue
                
                perturbed_mts_for_mt4 = []
                if mt4_present:
                    expanded_mt_numbers.append(4)
                    mt_to_original_map[4] = 4
                    perturbed_mts_for_mt4.append(4)
                
                if inelastic_mts_present:
                    for inelastic_mt in inelastic_mts_present:
                        # First check if this specific MT has its own covariance data
                        if covmat.has_isotope_mt(isotope_id, inelastic_mt):
                            # If user specifically requested this MT, don't handle it here - it will be handled in the other branch
                            continue
                        
                        # Otherwise apply the MT=4 covariance
                        expanded_mt_numbers.append(inelastic_mt)
                        mt_to_original_map[inelastic_mt] = 4  # Map back to original MT=4
                        perturbed_mts_for_mt4.append(inelastic_mt)
                
                if verbose:
                    if perturbed_mts_for_mt4:
                        perturbed_mts_for_mt4.sort()
                        print(f"  - MT=4 requested and found in covariance data; applying to existing ACE MTs: {perturbed_mts_for_mt4}")
                        if not mt4_present and inelastic_mts_present:
                            print(f"    (Note: MT=4 itself not found in ACE, applied only to MT={min(inelastic_mts_present)}-{max(inelastic_mts_present)})")
                        elif mt4_present and not inelastic_mts_present:
                            print(f"    (Note: MTs 51-91 not found in ACE, applied only to MT=4)")
                    else:
                        print(f"  - WARNING: MT=4 requested and found in covariance data, but neither MT=4 nor MTs 51-91 found in ACE. Skipping MT=4 perturbation.")
            
            # Handle individual MT numbers (including any 51-91 if specifically requested)
            elif mt in available_mts:
                if mt_in_covmat:
                    # append MTs that are present in both ACE and covariance
                    expanded_mt_numbers.append(mt)
                    mt_to_original_map[mt] = mt
                    if verbose:
                        print(f"  - MT={mt} requested, found in ACE file and covariance data.")
                else:
                    # special case for MT 51-91
                    if 51 <= mt <= 91 and covmat.has_isotope_mt(isotope_id, 4):
                        expanded_mt_numbers.append(mt)
                        mt_to_original_map[mt] = 4
                        warning_msg = (
                            f"MT={mt} requested, found in ACE file but not in covariance data. "
                            "Using MT=4 covariance data instead."
                        )
                        print(f"WARNING: {warning_msg}")
                        mt_warnings[mt] = warning_msg
                    else:
                        skipped_msg = f"MT={mt} requested, found in ACE file but not in covariance data. Skipping."
                        print(f"WARNING: {skipped_msg}")
                        mt_warnings[mt] = skipped_msg
                        skipped_mts.append(mt)
            else:
                missing_msg = f"MT={mt} requested but not found in ACE file. Skipping."
                print(f"WARNING: {missing_msg}")
                mt_warnings[mt] = missing_msg
                skipped_mts.append(mt)

    if verbose:
        print()   # blank line after availability loop

    # Ensure expanded_mt_numbers is sorted for consistency
    expanded_mt_numbers.sort()

    # Check if we have any MTs to perturb after filtering
    if not expanded_mt_numbers:
        raise ValueError("No valid MT numbers remain after checking availability in ACE file and covariance data. Cannot proceed.")

    # Filter original_mt_numbers to only include those that resulted in some perturbation
    # and are available in covariance data
    effective_original_mts = sorted(list(set(mt_to_original_map.values())))

    if not effective_original_mts:
        raise ValueError("None of the requested MT numbers were found in the ACE file or resulted in valid perturbations. Cannot proceed.")

    if verbose:
        print("=== Covariance Matrix Preparation ===")
        print(f"Covariance MT blocks     : {effective_original_mts}")
        print(f"MTs to perturb in ACE    : {expanded_mt_numbers}\n")
        print("Extracting combined covariance matrix...")
    
    cov_start_time = time.time()
    # Use the effective original MTs to extract the correct covariance blocks
    combined_cov_matrix = extract_covariance_matrix(covmat, isotope_id, effective_original_mts, energy_grid)
    cov_time = time.time() - cov_start_time
    
    if verbose:
        print(f"Covariance matrix extracted in {cov_time:.2f} seconds\n")
        print("Generating perturbation factors...\n")
    
    # Generate perturbation factors using the specified sampling method
    if verbose:
        print(f"Generating {num_samples} perturbation factors...")
    
    generation_start_time = time.time()
    # Perturbation factors are generated based on the effective original MTs
    perturbation_factors = generate_samples(
        combined_cov_matrix, 
        num_samples, 
        decomposition_method,
        sampling_method,
        seed
    )
    generation_time = time.time() - generation_start_time
    
    if verbose:
        print(f"Perturbation factors generated in {generation_time:.2f} seconds\n")
    
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
    
    if output_dir is not None and verbose:
        print(f"Output directory exists  : {output_dir}\n")
    
    # Set up base file name
    base_name = "perturbed_ace"
    if ace_object.filename:
        base_name, extension = os.path.splitext(os.path.basename(ace_object.filename))
    elif ace_object.header and ace_object.header.zaid:
        base_name = f"{ace_object.header.zaid}"
        extension = ".ace"
    else:
        extension = ".ace"
    
    # prepare streaming perturbation‐data file
    data_file_path = os.path.join(output_dir or ".", f"{base_name}_perturbation_data.txt")
    initialize_perturbation_data_file(
        data_file_path,
        isotope_id,
        requested_mt_numbers=original_mt_numbers,
        perturbed_mt_numbers=expanded_mt_numbers,
        available_covmat_mts=isotope_reactions,
        skipped_mts=skipped_mts,
        mt_warnings=mt_warnings,
        energy_grid=energy_grid,
        seed=seed,
        decomposition_method=decomposition_method,
        sampling_method=sampling_method,
        num_samples=num_samples
    )
    if verbose:
        print(f"Initialized perturbation data file: {data_file_path}\n")

    # OPTIMIZATION: Precompute mappings from ace energies to perturbation grid bins
    precomp_start_time = time.time()
    precomputed_mappings = {}
    
    # Precompute mappings only for the MTs that will actually be perturbed
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
        print(f"Precomputed mappings in  {precomp_time:.2f} seconds\n")
        print("Cleaning up ACE object to free memory\n")
    
    # Clean up the initial ACE object to free memory
    del ace_object
    
    if verbose:
        print("=== Starting Sample Perturbations ===\n")
    
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
            perturbation_factors[i], # Factors for this sample, ordered by effective_original_mts
            ace_file_path,
            expanded_mt_numbers,  # List of MTs to actually modify in the ACE file
            effective_original_mts, # List of original MTs corresponding to factor blocks
            mt_to_original_map,   # Map expanded MT -> original MT to find correct factor block
            energy_grid,
            base_name, 
            extension, 
            output_dir, 
            precomputed_mappings
        )
        # Use different names for unpacked results to avoid shadowing loop variable 'i'
        res_idx, res_elapsed, res_success, res_error, res_read_time, res_perturb_time, res_write_time = result
        
        if res_success:
            if verbose:
                # Use res_elapsed for the time taken
                print(f"done in {res_elapsed:.2f} seconds") 
            # Append the original index 'i' and the elapsed time
            results.append((i, res_elapsed)) 
            read_times.append(res_read_time)
            perturb_times.append(res_perturb_time)
            write_times.append(res_write_time)
            append_sample_perturbation_data(
                data_file_path,
                i,
                perturbation_factors[i],
                effective_original_mts,
                energy_grid
            )
            if verbose:
                print(f"Appended sample {i+1:04d} to data file\n")
        else:
            # Always print errors regardless of verbose setting
            # Use i+1 for the sample number in the error message
            print(f"\nERROR processing sample {i+1}: {res_error}") # Add newline for clarity

    # Calculate and print summary statistics if we have results
    if results and verbose:
        elapsed_times = [t for _, t in results]
        overall_time = time.time() - overall_start_time
        avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
        
        avg_read = sum(read_times) / len(read_times) if read_times else 0
        avg_perturb = sum(perturb_times) / len(perturb_times) if perturb_times else 0
        avg_write = sum(write_times) / len(write_times) if write_times else 0
        
        print("\n=== Perturbation Summary ===")
        print(f"Files created           : {len(results)}/{num_samples}")
        print(f"Total run time          : {overall_time:.2f} seconds")
        if avg_time > 0:
            print(f"Average time per file   : {avg_time:.2f} seconds")
            print("\nPerformance breakdown:")
            print(f"  Read ACE file         : {avg_read:.4f} s ({100*avg_read/avg_time:.1f}%)")
            print(f"  Apply perturbation    : {avg_perturb:.4f} s ({100*avg_perturb/avg_time:.1f}%)")
            print(f"  Write to file         : {avg_write:.4f} s ({100*avg_write/avg_time:.1f}%)")
            print(f"  Other                 : {max(0, avg_time-avg_read-avg_perturb-avg_write):.4f} s ({100*(avg_time-avg_read-avg_perturb-avg_write)/avg_time:.1f}%)\n")
        
        print("=== Initial Computation Times ===")
        print(f"Covariance extraction   : {cov_time:.2f} s")
        print(f"Sample factor generation: {generation_time:.2f} s")
        print(f"Mapping precomputation  : {precomp_time:.2f} s\n")
        print(f"Results directory       : {output_dir}\n")

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

def initialize_perturbation_data_file(
    file_path: str,
    isotope_id: int,
    requested_mt_numbers: List[int],
    perturbed_mt_numbers: List[int],
    available_covmat_mts: Set[int],
    skipped_mts: List[int],
    mt_warnings: Dict[int, str],
    energy_grid: np.ndarray,
    seed: Optional[int],
    decomposition_method: str,
    sampling_method: str,
    num_samples: int
) -> None:
    """Write header, parameters, energy grid & table header."""
    with open(file_path, 'w') as f:
        f.write("="*80 + "\nPERTURBATION DATA\n" + "="*80 + "\n\n")
        # basic parameters
        f.write(f"Timestamp                     : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Isotope ZAID                  : {isotope_id}\n")
        f.write(f"Requested MT numbers          : {requested_mt_numbers}\n")
        f.write(f"MTs available in covariance   : {sorted(available_covmat_mts)}\n")
        f.write(f"MTs to be perturbed           : {perturbed_mt_numbers}\n")
        if skipped_mts:
            f.write(f"Skipped MTs                   : {sorted(skipped_mts)}\n")
        if mt_warnings:
            f.write("MT-specific warnings:\n")
            for mt, msg in mt_warnings.items():
                f.write(f"  - MT={mt}: {msg}\n")
        f.write(f"Number of samples             : {num_samples}\n")
        f.write(f"Sampling method               : {sampling_method}\n")
        f.write(f"Decomposition method          : {decomposition_method}\n")
        f.write(f"Random seed                   : {seed if seed is not None else 'None'}\n\n")
        # energy grid
        f.write("ENERGY GRID (MeV):\n" + "-"*80 + "\n")
        for i,e in enumerate(energy_grid):
            f.write(f"{e:.6e}" + ("\n" if (i+1)%8==0 or i==len(energy_grid)-1 else "  "))
        f.write("\n\n")
        # table header
        header = "Orig MT | Lower E (MeV) | Upper E (MeV) | Factor\n"
        f.write(header + "-"*len(header) + "\n")

def append_sample_perturbation_data(
    file_path: str,
    sample_idx: int,
    factors: np.ndarray,
    effective_original_mts: List[int],
    energy_grid: np.ndarray
) -> None:
    """Append one sample's factors row by row."""
    with open(file_path, 'a') as f:
        f.write(f"\n# Sample {sample_idx+1:04d}\n")
        num_bins = len(energy_grid)-1
        for mt_idx, mt in enumerate(effective_original_mts):
            start = mt_idx * num_bins
            for b in range(num_bins):
                f.write(f"{mt:7d} | {energy_grid[b]:.6e} | {energy_grid[b+1]:.6e} | {factors[start+b]:.12f}\n")