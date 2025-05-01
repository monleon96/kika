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
    num_samples: int,
    effective_original_mts: List[int], # Added parameter
    available_covmat_mts: Set[int] = None,
    skipped_mts: List[int] = None,
    mt_warnings: Dict[int, str] = None
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
        Expanded list of MT numbers actually perturbed (may include MT=4 and MT=51-91 for an MT=4 request)
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
    effective_original_mts : List[int]
        List of original MT numbers for which covariance data was found and factors were generated
    available_covmat_mts : Set[int], optional
        Set of MT numbers available in the covariance matrix
    skipped_mts : List[int], optional
        List of MT numbers that were skipped due to missing covariance data
    mt_warnings : Dict[int, str], optional
        Dictionary mapping MT numbers to warning messages
        
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
        
        # Report on available MTs in covariance matrix
        if available_covmat_mts is not None:
            avail_mts_sorted = sorted(list(available_covmat_mts))
            f.write(f"\nMT numbers available in covariance matrix: {avail_mts_sorted}\n")
        
        # Report on skipped MTs
        if skipped_mts and len(skipped_mts) > 0:
            f.write(f"\nWARNING: The following requested MT numbers were skipped due to missing covariance data: {sorted(skipped_mts)}\n")
            
        # Check if MT=4 was requested and describe how it was handled
        mt4_requested = 4 in mt_numbers
        if mt4_requested:
            mt4_perturbed = 4 in expanded_mt_numbers and mt_to_original_map.get(4) == 4
            inelastic_mts_perturbed = sorted([mt for mt in expanded_mt_numbers if mt_to_original_map.get(mt) == 4 and mt >= 51 and mt <= 91])
            
            f.write("\nMT=4 (INELASTIC) PERTURBATION DETAILS:\n")
            if mt4_perturbed and inelastic_mts_perturbed:
                f.write(f"  - Perturbation factors for MT=4 were applied to both MT=4 and the discrete inelastic levels found:\n")
                f.write(f"    MT={min(inelastic_mts_perturbed)}-{max(inelastic_mts_perturbed)} ({len(inelastic_mts_perturbed)} levels: {inelastic_mts_perturbed}).\n")
            elif mt4_perturbed:
                f.write(f"  - Perturbation factors for MT=4 were applied to MT=4.\n")
                f.write(f"  - No discrete inelastic levels (MT=51-91) were found in the ACE file.\n")
            elif inelastic_mts_perturbed:
                f.write(f"  - MT=4 itself was not found in the ACE file.\n")
                f.write(f"  - Perturbation factors for MT=4 were applied to the discrete inelastic levels found:\n")
                f.write(f"    MT={min(inelastic_mts_perturbed)}-{max(inelastic_mts_perturbed)} ({len(inelastic_mts_perturbed)} levels: {inelastic_mts_perturbed}).\n")
            else:
                f.write(f"  - WARNING: MT=4 was requested, but neither MT=4 nor any discrete inelastic levels (MT=51-91) were found in the ACE file.\n")
                f.write(f"  - No perturbation was applied based on the MT=4 request.\n")
        
        # Add information about specific MT warnings if any
        if mt_warnings and len(mt_warnings) > 0:
            f.write("\nMT NUMBER SPECIFIC WARNINGS:\n")
            for mt, warning in mt_warnings.items():
                f.write(f"  - MT={mt}: {warning}\n")

        # Add information about other MT expansions if any (less common)
        other_expansions = False
        original_to_expanded_other = {}
        for exp_mt, orig_mt in mt_to_original_map.items():
            # Exclude MT=4 case handled above and self-mappings
            if orig_mt != 4 and exp_mt != orig_mt:
                if orig_mt not in original_to_expanded_other:
                    original_to_expanded_other[orig_mt] = []
                original_to_expanded_other[orig_mt].append(exp_mt)
                other_expansions = True

        if other_expansions:
            f.write("\nOTHER MT NUMBER EXPANSIONS (if any):\n")
            for orig_mt, exp_mts in original_to_expanded_other.items():
                if exp_mts: # Only show if there was an expansion
                    exp_mts.sort()
                    f.write(f"  - MT={orig_mt} expanded to: {exp_mts}\n")
        
        # List all MT numbers that were actually perturbed
        unique_expanded_mts = sorted(list(set(expanded_mt_numbers)))
        f.write(f"\nActual MT numbers perturbed in ACE files: {unique_expanded_mts}\n")
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
        f.write("\nPERTURBATION FACTORS (Applied based on original requested MTs):\n")
        f.write("-" * 80 + "\n")
        
        # Calculate header width based on number of samples
        header = "Orig MT | Lower E (MeV) | Upper E (MeV) |"
        for s in range(num_samples):
            header += f" Sample_{s+1:04d} |"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        # Write data for each EFFECTIVE ORIGINAL MT and energy bin with increased precision
        # Iterate through MTs for which factors were actually generated
        for mt_idx, eff_orig_mt in enumerate(effective_original_mts): 
            num_bins = len(energy_grid) - 1
            # Index into perturbation_factors is based on the order in effective_original_mts
            start_idx = mt_idx * num_bins 
            
            for bin_idx in range(num_bins):
                # Get perturbation factors for this effective original MT and bin across all samples
                # Access based on the index derived from effective_original_mts
                bin_factors = [perturbation_factors[s][start_idx + bin_idx] for s in range(num_samples)]
                
                # Format row with 12 decimal places instead of 6
                row = f"{eff_orig_mt:7d} | {energy_grid[bin_idx]:.6e} | {energy_grid[bin_idx+1]:.6e} |"
                for factor in bin_factors:
                    row += f" {factor:.12f} |"
                f.write(row + "\n")
            
            # Add a separator between original MTs
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
        List of MT numbers to perturb (e.g., [1, 4, 18])
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
    
    if verbose:
        print(f"Isotope ZAID            : {isotope_id}")
        print(f"Covariance MTs available: {sorted(list(isotope_reactions))}\n")
    
    # Get available MTs in the covariance matrix for this isotope
    isotope_reactions = covmat.get_isotope_reactions().get(isotope_id, set())
    if verbose:
        print(f"MTs available in covariance matrix for isotope {isotope_id}: {sorted(list(isotope_reactions))}")
    
    # Determine which MTs to actually perturb based on request and availability
    original_mt_numbers = sorted(list(set(mt_numbers))) # Use unique sorted list
    expanded_mt_numbers = []
    mt_to_original_map = {}  # Maps actually perturbed MTs back to original requested MTs for factor lookup
    
    # Get available MTs in the ACE file
    available_mts = set()
    if hasattr(ace_object, 'mt_numbers'):
        available_mts.update(ace_object.mt_numbers)
    if hasattr(ace_object, 'cross_section') and ace_object.cross_section:
        available_mts.update(ace_object.cross_section.reaction.keys())
    
    # Track skipped MTs and MT-specific warnings
    skipped_mts = []
    mt_warnings = {}
    
    if verbose:
        print("=== MT Availability Check ===\n")

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
                        if inelastic_mt in original_mt_numbers:
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
    
    # Write perturbation data to file for reproducibility (always write this regardless of verbose)
    if verbose:
        print("Writing perturbation data file...\n")
    
    # Pass the original requested MTs and the expanded list to the writer function
    data_file_path = write_perturbation_data(
        output_dir,
        base_name,
        isotope_id,
        original_mt_numbers,  # Original MT numbers requested by user
        expanded_mt_numbers,  # Actual MT numbers being perturbed in ACE
        mt_to_original_map,   # Mapping from expanded MTs to original MTs
        energy_grid,
        seed,
        decomposition_method,
        sampling_method,
        perturbation_factors, # Factors correspond to effective_original_mts order
        num_samples,
        effective_original_mts, # Pass the effective list here
        isotope_reactions,    # Available MTs in covariance matrix
        skipped_mts,          # MTs that were skipped
        mt_warnings           # MT-specific warnings
    )
    
    if verbose:
        print(f"Data recorded at         : {data_file_path}\n")
    
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