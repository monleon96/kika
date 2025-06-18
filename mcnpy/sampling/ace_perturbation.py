import numpy as np
import os
import shutil
import logging
from typing import List, Union, Optional, Dict, Tuple
from multiprocessing import Pool
from datetime import datetime

from mcnpy.sampling.generators import generate_samples
from mcnpy.ace.parsers import read_ace
from mcnpy.ace.writers.write_ace import write_ace
from mcnpy.cov.parse_covmat import read_scale_covmat, read_njoy_covmat


class DualLogger:
    """Logger that writes detailed info to file and basic info to console."""
    
    def __init__(self, log_file: str, console_level: str = 'INFO'):
        self.log_file = log_file
        
        # Create logger
        self.logger = logging.getLogger('ace_perturbation')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler - gets everything
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler - only basic info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, console: bool = False):
        """Log info message. If console=True, also show in console."""
        if console:
            self.logger.info(message)
        else:
            # Only to file
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.emit(logging.LogRecord(
                        name=self.logger.name, level=logging.INFO, pathname='', lineno=0,
                        msg=message, args=(), exc_info=None
                    ))
    
    def warning(self, message: str, console: bool = True):
        """Log warning message. Shows in console by default."""
        if console:
            self.logger.warning(message)
        else:
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.emit(logging.LogRecord(
                        name=self.logger.name, level=logging.WARNING, pathname='', lineno=0,
                        msg=message, args=(), exc_info=None
                    ))
    
    def error(self, message: str, console: bool = True):
        """Log error message. Shows in console by default."""
        if console:
            self.logger.error(message)
        else:
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.emit(logging.LogRecord(
                        name=self.logger.name, level=logging.ERROR, pathname='', lineno=0,
                        msg=message, args=(), exc_info=None
                    ))

# Global logger instance
_logger = None

def _get_logger():
    """Get the global logger instance."""
    return _logger

def _process_sample(
    ace_file: str,
    sample: np.ndarray,
    sample_index: int,
    energy_grid: List[float],
    mt_numbers: List[int],
    output_dir: str,
    xsdir_file: Optional[str],
):
    # — read & perturb ACE —
    ace = read_ace(ace_file)
    perturbed_mts = apply_perturbation_factor_to_ace(ace, sample, sample_index, energy_grid, mt_numbers, False)  # Set verbose=False to reduce output

    # — write perturbed ACE —
    base, ext = os.path.splitext(os.path.basename(ace_file))
    sample_str = f"{sample_index+1:04d}"
    sample_dir = os.path.join(output_dir, str(ace.zaid), sample_str)
    os.makedirs(sample_dir, exist_ok=True)
    out_ace = os.path.join(sample_dir, f"{base}_{sample_str}{ext}")
    
    # Recalculate cross sections but don't print
    ace.update_cross_sections()
    
    # Write ACE file without verbose output
    write_ace(ace, out_ace, overwrite=True)

    # — write per-sample .xsdir —
    rel = os.path.relpath(out_ace, output_dir).replace(os.sep, "/")
    rel_path = f"../{rel}"
    hdr = ace.header
    ptable = 'ptable' if getattr(ace.unresolved_resonance, 'has_data', False) else ''
    line = (
        f"{hdr.zaid}{ext} "
        f"{hdr.atomic_weight_ratio:.6f} "
        f"{rel_path} 0 1 1 "
        f"{hdr.nxs_array[1]} 0 0 "
        f"{hdr.temperature:.3E} {ptable}"
    )
    xsdir_path = os.path.join(sample_dir, f"{base}_{sample_str}.xsdir")
    with open(xsdir_path, 'w') as fx:
        fx.write(line + "\n")

    if xsdir_file:
        # Create xsdir directory if it doesn't exist
        xsdir_dir = os.path.join(output_dir, "xsdir")
        os.makedirs(xsdir_dir, exist_ok=True)
        
        # 1) build the sample suffix and master-filename (no extra extension)
        sample_tag       = f"_{sample_index+1:04d}"
        orig_xs_base     = os.path.splitext(os.path.basename(xsdir_file))[0]
        master_xs_name   = orig_xs_base + sample_tag
        master_xs_path   = os.path.join(xsdir_dir, master_xs_name)  # Create directly in xsdir directory

        # 2) copy original master once
        if not os.path.exists(master_xs_path):
            shutil.copy(xsdir_file, master_xs_path)

        # 3) read & patch only the ZAID line
        with open(master_xs_path, 'r') as f:
            xs_lines = f.readlines()

        ace_ext      = os.path.splitext(ace_file)[1]   # e.g. ".54c"
        zaid_prefix  = f"{ace.header.zaid}{ace_ext}"
        new_line     = line + "\n"

        with open(master_xs_path, 'w') as f:
            for ln in xs_lines:
                if ln.startswith(zaid_prefix):
                    f.write(new_line)
                else:
                    f.write(ln)

    # — write this sample's small summary file —
    n_groups    = len(energy_grid) - 1
    boundaries  = np.asarray(energy_grid)
    summary_tmp = os.path.join(sample_dir, f"{base}_{sample_str}_summary.txt")

    with open(summary_tmp, 'w') as sf:

        for mt_idx, mt in enumerate(mt_numbers):
            start    = mt_idx * n_groups
            end      = start + n_groups
            grp_facs = sample[start:end]

            for grp in range(n_groups):
                low, high = boundaries[grp], boundaries[grp+1]
                fac       = grp_facs[grp]
                sf.write(f"{mt:<3}\t{low:.12e}\t{high:.12e}\t{fac:.12e}\n")

        sf.write("\n")


def load_covariance(path):
    if not os.path.exists(path):
        return None
    
    for reader in (read_njoy_covmat, read_scale_covmat):
        try:
            cov = reader(path)
        except ValueError:
            continue
        return cov
    return None


def perturb_ACE_files(
    ace_files: Union[str, List[str]],
    cov_files: Union[str, List[str]],
    mt_list: List[int],
    num_samples: int,
    space: str = "linear",
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    output_dir: str = '.',
    xsdir_file: Optional[str] = None,
    seed: Optional[int] = None,
    nprocs: int = 1,
    dry_run: bool = False,
    autofix: Optional[str] = 'soft', 
    high_val_thresh: float = 1.0,
    accept_tol: float = -1.0e-4,
    remove_blocks: Optional[Dict[int, Union[Tuple[int, int], List[Tuple[int, int]]]]] = None,  # Add remove_blocks parameter
    verbose: bool = True,
):
    global _logger
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'ace_perturbation_{timestamp}.log')
    _logger = DualLogger(log_file)
    
    # Console: Basic start message
    print(f"[INFO] Starting ACE perturbation job")
    print(f"[INFO] Log file: {log_file}")
    print(f"[INFO] Output directory: {os.path.abspath(output_dir)}")
    
    # Print run parameters as metadata TO LOG FILE
    separator = "=" * 80
    _logger.info(f"\n{separator}")
    _logger.info(f"[ACE] [PARAMETERS] Run Configuration")
    _logger.info(f"{separator}")
    
    # Format and print input files
    if isinstance(ace_files, str):
        ace_files = [ace_files]
        formatted_ace = ace_files[0]
    else:
        formatted_ace = f"{len(ace_files)} files"
        if len(ace_files) <= 3:
            formatted_ace = ", ".join(os.path.basename(f) for f in ace_files)
    
    if isinstance(cov_files, str):
        cov_files = [cov_files]
        formatted_cov = cov_files[0]
    else:
        formatted_cov = f"{len(cov_files)} files"
        if len(cov_files) <= 3:
            formatted_cov = ", ".join(os.path.basename(f) for f in cov_files)
    
    # Format MT list
    if len(mt_list) == 0:
        formatted_mt = "All available MTs"
    elif len(mt_list) <= 10:
        formatted_mt = ", ".join(str(mt) for mt in mt_list)
    else:
        formatted_mt = f"{len(mt_list)} MTs: {', '.join(str(mt) for mt in mt_list[:5])}..."
    
    # Print all parameters TO LOG FILE
    _logger.info(f"  ACE files:             {formatted_ace}")
    _logger.info(f"  Covariance files:      {formatted_cov}")
    _logger.info(f"  MT numbers:            {formatted_mt}")
    _logger.info(f"  Number of samples:     {num_samples}")
    _logger.info(f"  Sampling space:        {space}")
    _logger.info(f"  Decomposition method:  {decomposition_method}")
    _logger.info(f"  Sampling method:       {sampling_method}")
    _logger.info(f"  Output directory:      {os.path.abspath(output_dir)}")
    _logger.info(f"  XSDIR file:            {xsdir_file if xsdir_file else 'None'}")
    _logger.info(f"  Random seed:           {seed if seed is not None else 'Random'}")
    _logger.info(f"  Parallel processes:    {nprocs}")
    _logger.info(f"  Mode:                  {'Dry run (factors only)' if dry_run else 'Full ACE generation'}")
    _logger.info(f"  Autofix covariance:    {autofix}")
    _logger.info(f"  High value threshold:  {high_val_thresh}")
    _logger.info(f"  Accept tolerance:      {accept_tol}")
    _logger.info(f"  Remove blocks:         {remove_blocks if remove_blocks else 'None'}")
    _logger.info(f"  Verbose output:        {verbose}")
    
    # Print timestamp
    _logger.info(f"  Timestamp:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _logger.info(f"{separator}\n")

    # normalize inputs (already done in the prints, but needed for the rest of the function)
    if isinstance(ace_files, str):
        ace_files = [ace_files]
    if isinstance(cov_files, str):
        cov_files = [cov_files]

    # Initialize a dictionary to collect summary information for each isotope
    summary_data = {}

    # Track skipped isotopes due to missing or invalid covariance
    skipped_isotopes = {}

    # Console: Show progress
    print(f"[INFO] Processing {len(ace_files)} isotope(s)")

    for i, (ace_file, cov_file) in enumerate(zip(ace_files, cov_files)):

        # ====== Start of ACE file processing ======
        separator = "=" * 80
        _logger.info(f"\n{separator}")
        _logger.info(f"[ACE] [PROCESSING] {ace_file}")
        _logger.info(f"{separator}\n")

        # Console: Basic progress
        print(f"\n[INFO] Processing isotope {i+1}/{len(ace_files)}: {os.path.basename(ace_file)}")

        # Check if ACE file exists
        if not os.path.exists(ace_file):
            _logger.error(f"[ACE] [ERROR] ACE file not found: {ace_file}")
            zaid = os.path.splitext(os.path.basename(ace_file))[0]  # Use filename without extension as ZAID
            summary_data[zaid] = {
                "ace_file": os.path.basename(ace_file),
                "cov_file": os.path.basename(cov_file),
                "mt_in_ace": [],
                "mt_in_cov": [],
                "mt_perturbed": [],
                "removed_mts": {},
                "autofix_info": {
                    "level": autofix,
                    "removed_pairs": [],
                    "removed_mts": [],
                    "removed_correlations": [],
                    "converged": False,
                    "soft_threshold_warning": False
                },
                "warnings": [f"ACE file not found: {os.path.basename(ace_file)}"]
            }
            skipped_isotopes[zaid] = "ACE file not found"
            _logger.info(f"\n{separator}\n")
            print(f"[ERROR] ACE file not found for {os.path.basename(ace_file)}")
            continue

        # Read ACE file first to get ZAID
        ace = read_ace(ace_file)
        zaid = ace.zaid
        base, _ = os.path.splitext(os.path.basename(ace_file))

        # Initialize summary information for this isotope
        summary_data[zaid] = {
            "ace_file": os.path.basename(ace_file),
            "cov_file": os.path.basename(cov_file),
            "mt_in_ace": sorted(set(ace.mt_numbers)),
            "mt_in_cov": [],
            "mt_perturbed": [],
            "removed_mts": {},  # Will store MT: reason pairs
            "autofix_info": {  # Track autofix-specific information
                "level": autofix,
                "removed_pairs": [],
                "removed_mts": [],
                "removed_correlations": [],  # Track off-diagonal block removals
                "converged": True,
                "soft_threshold_warning": False  # Track if soft autofix didn't meet threshold
            },
            "warnings": []
        }
        
        cov = load_covariance(cov_file)
        if cov is None:
            _logger.error(f"[ACE] [ERROR] Unable to load a valid covariance matrix for {cov_file}")
            summary_data[zaid]["warnings"].append(f"No valid covariance file found: {os.path.basename(cov_file)}")
            skipped_isotopes[zaid] = "No valid covariance file found"
            _logger.info(f"\n{separator}\n")
            print(f"[ERROR] Failed to load covariance for {os.path.basename(ace_file)}")
            continue

        _logger.info(f"  Covariance file: {cov_file}")
        _logger.info(f"  Isotope: {zaid}")
        
        subseparator = "-" * 60
        _logger.info(f"\n{subseparator}")

        # Check if the covariance matrix is empty (no data)
        if cov.num_matrices == 0:
            _logger.info(f"[ACE] [SKIP] No covariance found in {zaid}. Skipping.")
            summary_data[zaid]["warnings"].append("No covariance data found in matrix")
            skipped_isotopes[zaid] = "No covariance data found in matrix"
            _logger.info(f"\n{separator}\n")
            print(f"[WARNING] No covariance data for {os.path.basename(ace_file)}")
            continue

        # Apply user-specified block removals before clean_cov
        if remove_blocks and zaid in remove_blocks:
            _logger.info(f"[ACE] [BLOCK REMOVAL] Applying user-specified block removals for isotope {zaid}")
            
            # Normalize the removal specification to a list of tuples
            removal_spec = remove_blocks[zaid]
            if isinstance(removal_spec, tuple):
                # Single tuple: convert to list
                blocks_to_remove = [removal_spec]
            else:
                # Should be a list of tuples
                blocks_to_remove = list(removal_spec)
            
            _logger.info(f"  Blocks to remove: {blocks_to_remove}")
            
            # Get reactions before removal for comparison
            reactions_before = cov.reactions_by_isotope(zaid)
            _logger.info(f"  Reactions before removal: {sorted(reactions_before)}")
            
            # Apply the removal
            try:
                cov = cov.remove_matrix(isotope=zaid, reaction_pairs=blocks_to_remove)
                
                # Get reactions after removal
                reactions_after = cov.reactions_by_isotope(zaid)
                _logger.info(f"  Reactions after removal:  {sorted(reactions_after)}")
                
                # Track removed reactions for the summary
                removed_reactions = set(reactions_before) - set(reactions_after)
                for mt in removed_reactions:
                    summary_data[zaid]["removed_mts"][mt] = f"User-specified block removal: {blocks_to_remove}"
                
                if removed_reactions:
                    _logger.info(f"  Successfully removed reactions: {sorted(removed_reactions)}")
                else:
                    _logger.info(f"  No reactions were removed (blocks may not have existed)")
                    
            except Exception as e:
                _logger.error(f"  [ERROR] Failed to remove blocks {blocks_to_remove}: {str(e)}")
                summary_data[zaid]["warnings"].append(f"Failed to remove user-specified blocks: {str(e)}")
            
            _logger.info(f"\n{subseparator}")

        cov = cov.clean_cov(zaid)

        mt_in_cov = cov.reactions_by_isotope(zaid)
        summary_data[zaid]["mt_in_cov"] = sorted(mt_in_cov)
        
        mt_in_ace = sorted(set(ace.mt_numbers))
        if len(mt_list) == 0:
            mt_request = mt_in_ace
        else:
            mt_request = sorted(mt_list)

        # Save original MT list before processing 
        original_mt_perturb = set(mt_in_cov) & set(mt_request) & set(mt_in_ace)
        
        mt_perturb = set(mt_in_cov) & set(mt_request) & set(mt_in_ace)
        groups = [
            (4,   range(51,  92)),
            (103, range(600, 650)),
            (104, range(650, 700)),
            (105, range(700, 750)),
            (106, range(750, 800)),
            (107, range(800, 850)),
        ]

        # Track removed MTs during group processing
        group_removed_mts = {}

        for single, rng in groups:
            cov_has_single  = single in mt_in_cov
            cov_in_range    = [mt for mt in mt_in_cov if mt in rng]
            list_has_single = single in mt_request
            list_in_range   = [mt for mt in mt_request if mt in rng]
            ace_in_range    = [mt for mt in mt_in_ace if mt in rng]

            # --- Logic for single element (e.g., 4, 103, 104, etc.) ---
            if cov_has_single and (list_has_single or (list_in_range and ace_in_range)):
                mt_perturb.add(single)
            else:
                if single in mt_perturb:
                    if not (cov_has_single and (list_has_single or (list_in_range and ace_in_range))):
                        group_removed_mts[single] = f"Missing data in covariance or ACE file for MT={single} or its associated range"
                mt_perturb.discard(single)
                if cov_has_single:
                    cov = cov.remove_matrix(zaid, [(single, 0)])

            # --- Logic for associated range (51-91, 600-649, etc.) ---
            if cov_in_range:
                if (list_has_single or list_in_range) and ace_in_range:
                    # Keep only the triple intersection in this range
                    triple = set(cov_in_range) & set(mt_request) & set(mt_in_ace)
                    removed_range = set(cov_in_range) - triple
                    for mt in removed_range:
                        if mt in original_mt_perturb:
                            group_removed_mts[mt] = f"MT not in triple intersection for range associated with MT={single}"
                    
                    mt_perturb.update(triple)
                    # Remove those outside the triple intersection from cov
                    to_remove = [(mt, 0) for mt in cov_in_range if mt not in triple]
                    if to_remove:
                        cov = cov.remove_matrix(zaid, to_remove)
                else:
                    # Remove the entire range from cov and perturb
                    to_remove = [(mt, 0) for mt in cov_in_range]
                    for mt in cov_in_range:
                        if mt in original_mt_perturb:
                            group_removed_mts[mt] = f"MT range removed because required data missing in ACE or request list"
                    
                    cov = cov.remove_matrix(zaid, to_remove)
                    for mt in cov_in_range:
                        mt_perturb.discard(mt)

        # Update summary with group processing results
        for mt, reason in group_removed_mts.items():
            summary_data[zaid]["removed_mts"][mt] = reason

        # Final sorted list
        mt_perturb = sorted(mt_perturb)

        remaining = cov.reactions_by_isotope(zaid)
        to_remove = [(mt, 0) for mt in remaining if mt not in mt_perturb]
        if to_remove:
            cov = cov.remove_matrix(zaid, to_remove)

        # Print available MT numbers TO LOG FILE
        _logger.info(f"[ACE] [MT SELECTION]")
        _logger.info(f"  MTs in ACE file:          {mt_in_ace}")
        _logger.info(f"  MTs in covariance matrix: {mt_in_cov}")
        _logger.info(f"  MTs to be perturbed:      {mt_perturb}")
        
        _logger.info(f"\n{subseparator}\n")

        energy_grid = cov.energy_grid

        # Save pre-autofix MT list
        pre_autofix_mts = list(mt_perturb)
        
        # Console: Show sample generation start
        print(f"[INFO] Generating {num_samples} samples for ZAID {zaid}")
        
        try:
            factors, mt_perturb_final, fix_info = generate_samples(
                cov                  = cov,
                space                = space,
                n_samples            = num_samples,
                decomposition_method = decomposition_method,
                sampling_method      = sampling_method,
                seed                 = None if seed is None else seed + zaid,
                mt_numbers           = mt_perturb,
                energy_grid          = energy_grid,
                autofix              = autofix, 
                high_val_thresh      = high_val_thresh,
                accept_tol           = accept_tol,
                verbose              = verbose,
            )
        except Exception as e:
            # Import the exception classes to check for them
            from mcnpy.sampling.generators import CovarianceFixError, SoftAutofixWarning
            
            if isinstance(e, SoftAutofixWarning):
                # Soft autofix failed threshold but decomposition also failed
                _logger.error(f"[ACE] [ERROR] Soft autofix warning for isotope {zaid}")
                _logger.error(f"  Error details: {str(e)}")
                
                summary_data[zaid]["warnings"].append(f"Soft autofix failed to meet eigenvalue threshold and decomposition failed")
                summary_data[zaid]["autofix_info"]["converged"] = False
                summary_data[zaid]["autofix_info"]["soft_threshold_warning"] = True
                skipped_isotopes[zaid] = str(e)
                
                _logger.info(f"\n{separator}\n")
                print(f"[ERROR] Soft autofix and decomposition failed for {os.path.basename(ace_file)}")
                continue
            elif isinstance(e, CovarianceFixError):
                _logger.error(f"[ACE] [ERROR] Covariance matrix fixing failed for isotope {zaid}")
                _logger.error(f"  Error details: {str(e)}")
                _logger.error(f"  Suggestion: Try processing isotope {zaid} separately with autofix='medium' or 'hard'")
                
                summary_data[zaid]["warnings"].append(f"Covariance matrix could not be fixed to meet eigenvalue threshold")
                summary_data[zaid]["autofix_info"]["converged"] = False
                skipped_isotopes[zaid] = str(e)
                
                _logger.info(f"\n{separator}\n")
                print(f"[ERROR] Covariance fix failed for {os.path.basename(ace_file)}")
                continue
            else:
                # Re-raise other exceptions
                raise e
        
        # Update summary with autofix results - now get detailed removal information
        if autofix is not None and fix_info is not None:
            # Check for soft autofix threshold warning
            if fix_info.get("soft_autofix_failed", False):
                summary_data[zaid]["autofix_info"]["soft_threshold_warning"] = True
                min_eigenvalue = fix_info.get("min_eigenvalue", float('nan'))
                summary_data[zaid]["warnings"].append(
                    f"Soft autofix failed to meet eigenvalue threshold (λ_min={min_eigenvalue:.4e} < {accept_tol:.4e}) but decomposition succeeded"
                )
                
            removed_in_autofix = set(pre_autofix_mts) - set(mt_perturb_final if mt_perturb_final else [])
            for mt in removed_in_autofix:
                summary_data[zaid]["removed_mts"][mt] = f"Removed during covariance autofix (level='{autofix}')"
                summary_data[zaid]["autofix_info"]["removed_mts"].append(mt)
            
            # Extract correlation removals from fix_info
            if fix_info.get("removed_correlations"):
                summary_data[zaid]["autofix_info"]["removed_correlations"] = fix_info["removed_correlations"]
            
            # Also store the removed pairs for completeness
            if fix_info.get("removed_pairs"):
                summary_data[zaid]["autofix_info"]["removed_pairs"] = fix_info["removed_pairs"]

        # Store final perturbed MTs
        summary_data[zaid]["mt_perturbed"] = mt_perturb_final if mt_perturb_final else []

        # -- prepare output dirs ----------------------------------------------
        iso_dir = os.path.join(output_dir, str(zaid))
        os.makedirs(iso_dir, exist_ok=True)

        # =====================================================================
        #  DRY‑RUN
        # =====================================================================
        if dry_run:
            _logger.info(f"\n[ACE] [DRY-RUN] Generating only perturbation factors (no ACE files will be written)")
            for j in range(num_samples):
                _write_sample_summary(factors[j], j, energy_grid, mt_perturb_final, iso_dir, base)
            _logger.info(f"\n{separator}\n")
            print(f"[INFO] Completed dry run for ZAID {zaid}")
            continue

        # =====================================================================
        #  FULL processing (ACE rewrite)
        # =====================================================================
        _logger.info(f"\n[ACE] [PROCESSING] Creating {num_samples} perturbed ACE files")
        _logger.info(f"  Output directory: {os.path.abspath(output_dir)}")
        if nprocs > 1:
            _logger.info(f"  Using {nprocs} parallel processes")
            
        # Create progress tracking variables
        report_interval = max(1, min(100, num_samples // 10))  # Report at most 10 times
        
        tasks = [(ace_file, factors[j], j, energy_grid, mt_perturb_final, output_dir, xsdir_file) for j in range(num_samples)]

        if nprocs > 1:
            # For parallel processing, just show start and end messages
            _logger.info(f"  Starting parallel sample generation... (progress updates disabled in parallel mode)")
            
            with Pool(processes=nprocs) as pool:
                for args in tasks:
                    pool.apply_async(_process_sample, args=args)
                pool.close()
                pool.join()
                
            _logger.info(f"  Completed generating {num_samples} samples")
            print(f"[INFO] Completed ACE generation for ZAID {zaid}")
        else:
            # For sequential processing, show periodic progress updates
            _logger.info(f"  Generating samples (progress updates every {report_interval} samples)")
            
            for i, args in enumerate(tasks):
                _process_sample(*args)
                
                # Report progress periodically TO LOG FILE
                if (i + 1) % report_interval == 0 or i + 1 == num_samples:
                    progress = (i + 1) / num_samples * 100
                    _logger.info(f"  Progress: {i + 1}/{num_samples} samples ({progress:.1f}%)")
            
            print(f"[INFO] Completed ACE generation for ZAID {zaid}")

        # ====== End of ACE file processing ======
        _logger.info(f"\n{separator}")
        _logger.info(f"[ACE] [COMPLETED] {ace_file} ({num_samples} samples generated)")
        _logger.info(f"{separator}\n")

    # =====================================================================
    #  Print final summary for all isotopes TO LOG FILE
    # =====================================================================
    separator = "=" * 80
    _logger.info(f"\n{separator}")
    _logger.info(f"[ACE] [SUMMARY] Processing Results")
    _logger.info(f"{separator}")
    
    if not summary_data:
        _logger.info("  No isotopes were processed.")
    else:
        # Custom sorting function: numeric ZAIDs first (ascending), then string ZAIDs (alphabetically)
        def sort_key(item):
            zaid = item[0]
            try:
                # Try to convert to int - if successful, return (0, numeric_value) for primary sort
                return (0, int(zaid))
            except (ValueError, TypeError):
                # If not numeric, return (1, string) to sort after numeric ones
                return (1, str(zaid))
        
        # Report isotopes that were successfully processed (had some MTs perturbed)
        successfully_processed = {zaid: data for zaid, data in sorted(summary_data.items(), key=sort_key) 
                                if zaid not in skipped_isotopes and data['mt_perturbed']}
        
        # Report isotopes that were processed but had no MTs perturbed
        processed_no_perturbation = {zaid: data for zaid, data in sorted(summary_data.items(), key=sort_key) 
                                   if zaid not in skipped_isotopes and not data['mt_perturbed']}
        
        if successfully_processed:
            _logger.info("\n  SUCCESSFULLY PROCESSED ISOTOPES:")
            _logger.info(f"  {'-' * 50}")
            
            for zaid, data in successfully_processed.items():
                # Only show filename in parentheses if ZAID is not numeric (i.e., it's a filename)
                try:
                    int(zaid)  # Test if ZAID is numeric
                    isotope_display = f"{zaid}"
                except (ValueError, TypeError):
                    isotope_display = f"{zaid} ({data['ace_file']})"
                
                _logger.info(f"\n  Isotope: {isotope_display}")
                _logger.info(f"  {'-' * 50}")
                
                # MT numbers that were perturbed
                _logger.info(f"  ► Perturbed MT numbers: {', '.join(map(str, data['mt_perturbed']))}")
                
                # Autofix information (for medium/hard levels)
                autofix_info = data.get('autofix_info', {})
                if autofix_info.get('level') in ['medium', 'hard']:
                    autofix_lines = []
                    if autofix_info.get('removed_mts'):
                        autofix_lines.append(f"removed MTs {', '.join(map(str, sorted(autofix_info['removed_mts'])))}")
                    if autofix_info.get('removed_correlations'):
                        corr_pairs = autofix_info['removed_correlations']
                        if len(corr_pairs) <= 3:
                            corr_str = ', '.join(f"({a},{b})" for a, b in corr_pairs)
                        else:
                            corr_str = f"({corr_pairs[0][0]},{corr_pairs[0][1]}), ({corr_pairs[1][0]},{corr_pairs[1][1]}), ... (+{len(corr_pairs)-2} more)"
                        autofix_lines.append(f"removed correlations {corr_str}")
                    
                    if autofix_lines:
                        _logger.info(f"  ► Autofix (level='{autofix_info['level']}'): {'; '.join(autofix_lines)}")

                # MT numbers that were removed and why (excluding autofix info already shown above)
                other_removed = {mt: reason for mt, reason in data['removed_mts'].items() 
                               if not reason.startswith("Removed during covariance autofix")}
                if other_removed:
                    _logger.info(f"  ► Other removed MT numbers:")
                    for mt, reason in sorted(other_removed.items()):
                        _logger.info(f"    • MT={mt}: {reason}")
                
                # Warnings if any
                if data['warnings']:
                    _logger.info(f"  ► Warnings:")
                    for warning in data['warnings']:
                        _logger.info(f"    • {warning}")

        # Report isotopes that were processed but had no perturbation
        if processed_no_perturbation or skipped_isotopes:
            _logger.info("\n  ISOTOPES WITH NO PERTURBATION:")
            _logger.info(f"  {'-' * 50}")
            
            # First report explicitly skipped isotopes - use same sorting and display logic
            for zaid, reason in sorted(skipped_isotopes.items(), key=lambda x: sort_key((x[0], None))):
                if zaid in summary_data:
                    try:
                        int(zaid)  # Test if ZAID is numeric
                        isotope_display = f"{zaid}"
                    except (ValueError, TypeError):
                        isotope_display = f"{zaid} ({summary_data[zaid]['ace_file']})"
                    
                    _logger.info(f"  Isotope: {isotope_display}")
                    _logger.info(f"    • Reason: {reason}")
            
            # Then report isotopes that were processed but had no MTs perturbed
            for zaid, data in processed_no_perturbation.items():
                try:
                    int(zaid)  # Test if ZAID is numeric
                    isotope_display = f"{zaid}"
                except (ValueError, TypeError):
                    isotope_display = f"{zaid} ({data['ace_file']})"
                
                _logger.info(f"  Isotope: {isotope_display}")
                if data['warnings']:
                    for warning in data['warnings']:
                        _logger.info(f"    • Reason: {warning}")
                else:
                    _logger.info(f"    • Reason: No eligible MT numbers to perturb")

    _logger.info(f"\n{separator}")
    
    # Console: Final summary
    processed_count = len([zaid for zaid in summary_data.keys() if zaid not in skipped_isotopes])
    skipped_count = len(skipped_isotopes)
    
    print(f"\n[INFO] Job completed!")
    print(f"[INFO] Processed: {processed_count} isotope(s)")
    print(f"[INFO] Skipped: {skipped_count} isotope(s)")
    print(f"[INFO] Detailed log saved to: {log_file}")


def apply_perturbation_factor_to_ace(ace, sample, sample_index, energy_grid, mt_numbers, verbose=True):
    """Apply per-group perturbation factors to ACE, and list which MTs were actually perturbed."""
    logger = _get_logger()
    
    n_groups = len(energy_grid) - 1
    if sample.shape[0] != len(mt_numbers) * n_groups:
        raise ValueError(f"sample length {sample.shape[0]} ≠ {len(mt_numbers)}×{n_groups}")

    boundaries = np.asarray(energy_grid)
    perturbed_mts = []  # collect the actual MT numbers

    for mt_idx, mt in enumerate(mt_numbers):
        start = mt_idx * n_groups
        end   = start + n_groups
        factors = sample[start:end]

        if mt == 4:
            for mt_inelastic in range(51, 92):
                if mt_inelastic in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_inelastic, factors, boundaries, verbose)
                    perturbed_mts.append(mt_inelastic)

        elif mt == 103:
            for mt_proton in range(600, 650):
                if mt_proton in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_proton, factors, boundaries, verbose)
                    perturbed_mts.append(mt_proton)

        elif mt == 104:
            for mt_H2 in range(650, 700):
                if mt_H2 in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_H2, factors, boundaries, verbose)
                    perturbed_mts.append(mt_H2)

        elif mt == 105:
            for mt_H3 in range(700, 750):
                if mt_H3 in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_H3, factors, boundaries, verbose)
                    perturbed_mts.append(mt_H3)

        elif mt == 106:
            for mt_He3 in range(750, 800):
                if mt_He3 in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_He3, factors, boundaries, verbose)
                    perturbed_mts.append(mt_He3)

        elif mt == 107:
            for mt_He4 in range(800, 850):
                if mt_He4 in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_He4, factors, boundaries, verbose)
                    perturbed_mts.append(mt_He4)

        else:
            # direct mt
            if mt in ace.mt_numbers:
                _apply_factors_to_mt(ace, mt, factors, boundaries, verbose)
                perturbed_mts.append(mt)

    # remove duplicates and sort for neatness
    perturbed_mts = sorted(set(perturbed_mts))

    if verbose and logger:
        logger.info(f"[ACE] [PERTURB] Applying factors for sample #{sample_index+1:04d}")
        logger.info(f"  Perturbed MT numbers: {perturbed_mts}")
        
    return perturbed_mts


def _apply_factors_to_mt(ace, mt, factors, boundaries, verbose=True):
    """Multiply each entry.value by factors[group] for this mt,
    **unless any factor is non-positive**.
    """
    logger = _get_logger()
    
    if (factors <= 0).any():
        bad = ", ".join(f"{f:+.3e}" for f in factors if f <= 0)
        if verbose and logger:
            logger.warning(f"  [WARNING] MT={mt}: negative or zero factors detected ({bad}). Reaction not perturbed.")
        return  # leave this reaction unperturbed

    reac       = ace.cross_section.reaction[mt]
    energies   = np.asarray(reac.energies)
    xs_entries = reac._xs_entries
    bin_idx    = np.digitize(energies, boundaries) - 1

    for i, entry in enumerate(xs_entries):
        grp = bin_idx[i]
        if 0 <= grp < len(factors):
            entry.value *= factors[grp]


def _write_sample_summary(
    sample: np.ndarray,
    sample_index: int,
    energy_grid: List[float],
    mt_numbers: List[int],
    iso_dir: str,
    base: str,
):
    """Serialise a single sample's perturbation factors."""
    n_groups = len(energy_grid) - 1
    boundaries = np.asarray(energy_grid)
    tag = f"{sample_index + 1:04d}"
    sdir = os.path.join(iso_dir, tag)
    os.makedirs(sdir, exist_ok=True)
    path = os.path.join(sdir, f"{base}_{tag}_pert_factors.txt")

    with open(path, "w") as f:
        for mt_idx, mt in enumerate(mt_numbers):
            start = mt_idx * n_groups
            end = start + n_groups
            slice_ = sample[start:end]
            for grp in range(n_groups):
                lo, hi = boundaries[grp], boundaries[grp + 1]
                fac = slice_[grp]
                f.write(f"{mt:<3}\t{lo:.12e}\t{hi:.12e}\t{fac:.12e}\n")
        f.write("\n")

