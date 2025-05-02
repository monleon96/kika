# File: mcnpy/sampling/perturb_parallel.py

import os
import shutil
import re
import fcntl
from multiprocessing import Pool
from typing import List, Optional, Union

import numpy as np

from mcnpy.ace.parsers.parse_ace import read_ace
from mcnpy.sampling.generators import generate_samples
from mcnpy.cov.covmat import CovMat

from mcnpy.sampling.perturb_ace import (
    extract_covariance_matrix,
    initialize_perturbation_data_file,
    process_sample,
)

# ————————————————————————————————————————————
# 1) module‐level globals for the Pool workers
_ACE_FILE_PATH           = None
_EXPANDED_MT_NUMBERS     = None
_EFFECTIVE_ORIG_MTS      = None
_MT_TO_ORIG_MAP          = None
_ENERGY_GRID             = None
_BASE_NAME               = None
_EXTENSION               = None
_OUTPUT_DIR              = None
_PRECOMP_MAPPINGS        = None
_XSDIR                   = None
_PERTURBATION_FACTORS    = None


def _move_xsdir_files(output_dir: str, xsdir: str, num_samples: int):
    """
    Move every file named <base>_0001 … <base>_<num_samples:04d>
    (no extension) into output_dir/<base>/ in O(num_samples) renames.
    """
    xs_base = os.path.splitext(os.path.basename(xsdir))[0]
    dest    = os.path.join(output_dir, xs_base)
    os.makedirs(dest, exist_ok=True)

    for i in range(1, num_samples+1):
        name = f"{xs_base}_{i:04d}"
        src  = os.path.join(output_dir, name)
        if os.path.exists(src):
            os.rename(src, os.path.join(dest, name))


def _append_log_block(path: str, block: str):
    """Safely append under a file lock."""
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(block)
        fcntl.flock(f, fcntl.LOCK_UN)

# ————————————————————————————————————————————
def _init_worker(
    ace_file_path,
    expanded_mt_numbers,
    effective_original_mts,
    mt_to_original_map,
    energy_grid,
    base_name,
    extension,
    output_dir,
    precomputed_mappings,
    xsdir,
    perturbation_factors
):
    """
    This initializer runs once in each child process.
    It fills the module‐level globals that _worker_top will use.
    """
    global _ACE_FILE_PATH, _EXPANDED_MT_NUMBERS, _EFFECTIVE_ORIG_MTS
    global _MT_TO_ORIG_MAP, _ENERGY_GRID, _BASE_NAME, _EXTENSION
    global _OUTPUT_DIR, _PRECOMP_MAPPINGS, _XSDIR, _PERTURBATION_FACTORS

    _ACE_FILE_PATH        = ace_file_path
    _EXPANDED_MT_NUMBERS  = expanded_mt_numbers
    _EFFECTIVE_ORIG_MTS   = effective_original_mts
    _MT_TO_ORIG_MAP       = mt_to_original_map
    _ENERGY_GRID          = energy_grid
    _BASE_NAME            = base_name
    _EXTENSION            = extension
    _OUTPUT_DIR           = output_dir
    _PRECOMP_MAPPINGS     = precomputed_mappings
    _XSDIR                = xsdir
    _PERTURBATION_FACTORS = perturbation_factors

def _worker_top(i: int) -> str:
    """
    Called in each Pool worker.  Uses module‐level globals
    to call your existing process_sample(), then builds the
    same log‐block your append_sample_perturbation_data would.
    """
    idx, elapsed, success, error, rt, pt, wt = process_sample(
        i,
        _PERTURBATION_FACTORS[i],
        _ACE_FILE_PATH,
        _EXPANDED_MT_NUMBERS,
        _EFFECTIVE_ORIG_MTS,
        _MT_TO_ORIG_MAP,
        _ENERGY_GRID,
        _BASE_NAME,
        _EXTENSION,
        _OUTPUT_DIR,
        _PRECOMP_MAPPINGS,
        _XSDIR,
    )

    if success:
        # include elapsed time in header
        lines = [f"\n# Sample {idx+1:04d} | Elapsed: {elapsed:.2f} s\n"]
        nbins = len(_ENERGY_GRID) - 1
        factors = _PERTURBATION_FACTORS[idx]
        for mti, mt in enumerate(_EFFECTIVE_ORIG_MTS):
            start = mti * nbins
            for b in range(nbins):
                lines.append(
                    f"{mt:7d} | {_ENERGY_GRID[b]:.6e} | "
                    f"{_ENERGY_GRID[b+1]:.6e} | "
                    f"{factors[start + b]:.12f}\n"
                )
        return "".join(lines)
    else:
        return f"\nERROR processing sample {idx+1:04d}: {error}\n"

# ————————————————————————————————————————————
def parallel_perturb(
    ace_file_path: Union[str, List[str]],
    mt_numbers: List[int],
    energy_grid: np.ndarray,
    covmat: Union[CovMat, List[CovMat]],
    num_samples: int,
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    output_dir: Optional[str] = None,
    xsdir: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    nprocs: int = 1,
):
    """
    Exactly the same behavior as create_perturbed_ace_files when nprocs=1.
    When nprocs>1, farms out each sample to a multiprocessing.Pool(nprocs)
    and uses a single per-isotope _log.txt safely under a file lock.
    """

    # — if they want purely sequential, just call your existing function
    if nprocs == 1:
        from mcnpy.sampling.perturb_ace import create_perturbed_ace_files
        result = create_perturbed_ace_files(
            ace_file_path,
            mt_numbers,
            energy_grid,
            covmat,
            num_samples,
            decomposition_method,
            sampling_method,
            output_dir,
            xsdir,
            seed,
            verbose,
        )
        if xsdir and output_dir:
            _move_xsdir_files(output_dir, xsdir, num_samples)
        return result

    # — support list of ACEs by recursion
    if isinstance(ace_file_path, (list, tuple)):
        ace_paths = list(ace_file_path)
        covmats   = covmat if isinstance(covmat, (list, tuple)) else [covmat]*len(ace_paths)
        if len(covmats) != len(ace_paths):
            raise ValueError("Number of covariance objects must match number of ACE file paths")
        for path, cm in zip(ace_paths, covmats):
            parallel_perturb(
                path,
                mt_numbers,
                energy_grid,
                cm,
                num_samples,
                decomposition_method,
                sampling_method,
                output_dir,
                xsdir,
                seed,
                verbose,
                nprocs,
            )
        if xsdir and output_dir:
            _move_xsdir_files(output_dir, xsdir, num_samples)
        return

    # ————— single ACE path now —————
    ace_obj    = read_ace(ace_file_path)
    isotope_id = ace_obj.header.zaid

    # Get available MTs in the covariance matrix for this isotope
    isotope_reactions = covmat.get_isotope_reactions().get(isotope_id, set())

    if verbose:
        print(f"Isotope ZAID            : {isotope_id}")
        print(f"Covariance MTs available: {sorted(list(isotope_reactions))}\n")
    
    # Get available MTs in the ACE file
    available_mts = set()
    if hasattr(ace_obj, 'mt_numbers'):
        available_mts.update(ace_obj.mt_numbers)
    if hasattr(ace_obj, 'cross_section') and ace_obj.cross_section:
        available_mts.update(ace_obj.cross_section.reaction.keys())
    
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

    expanded_mt_numbers     = sorted(set(expanded_mt_numbers))
    effective_original_mts  = sorted(set(mt_to_original_map.values()))

    # 2) build covariance & sample factors once
    combined_cov = extract_covariance_matrix(
        covmat, isotope_id, effective_original_mts, energy_grid
    )
    perturbation_factors = generate_samples(
        combined_cov,
        num_samples,
        decomposition_method,
        sampling_method,
        seed
    )

    # 3) precompute bin mappings (your existing loop)
    precomputed_mappings = {}
    # … <YOUR EXISTING PRECOMPUTE MAPPINGS LOOP> …

    # 4) prepare output dir & master log
    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    zaid_dir = os.path.join(output_dir, str(isotope_id))
    os.makedirs(zaid_dir, exist_ok=True)

    base_name, extension = os.path.splitext(os.path.basename(ace_file_path))
    log_path = os.path.join(zaid_dir, f"{base_name}_log.txt")
    initialize_perturbation_data_file(
        log_path,
        isotope_id,
        sorted(set(mt_numbers)),
        expanded_mt_numbers,
        covmat.get_isotope_reactions().get(isotope_id, set()),
        skipped_mts,
        mt_warnings,
        energy_grid,
        seed,
        decomposition_method,
        sampling_method,
        num_samples,
    )

    # 5) now fire up the Pool, passing **all** the context via initargs
    initargs = (
        ace_file_path,
        expanded_mt_numbers,
        effective_original_mts,
        mt_to_original_map,
        energy_grid,
        base_name,
        extension,
        output_dir,
        precomputed_mappings,
        xsdir,
        perturbation_factors,
    )

    with Pool(nprocs, initializer=_init_worker, initargs=initargs) as pool:
        # chunksize=1 gives best dynamic load‐balancing
        for block in pool.imap_unordered(_worker_top, range(num_samples), chunksize=1):
            _append_log_block(log_path, block)

    # --- append sampling verification just like in perturb_core ---
    centered = perturbation_factors - 1.0
    emp_cov = np.cov(centered, rowvar=False, bias=True)
    fro_diff = np.linalg.norm(combined_cov - emp_cov)
    fro_orig = np.linalg.norm(combined_cov)
    rel_pct = 100.0 * fro_diff / fro_orig if fro_orig != 0 else float('nan')
    ver_block = (
        "\n" + "="*40 + "\n"
        "SAMPLING VERIFICATION\n"
        + "="*40 + "\n"
        f"Relative error through Frobenius norm : {rel_pct:.4f}%\n"
    )
    _append_log_block(log_path, ver_block)
    if verbose:
        print(ver_block)

    if verbose:
        print(f"[parallel_perturb] done for isotope {isotope_id}, log → {log_path}\n\n")

    # once *all* samples (and recursive calls) are finished, move xsdir files
    if xsdir and output_dir:
        _move_xsdir_files(output_dir, xsdir, num_samples)