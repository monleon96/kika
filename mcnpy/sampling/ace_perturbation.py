import numpy as np
import os
import shutil
import glob
from typing import List, Union, Optional
from multiprocessing import Pool

from mcnpy.sampling.generators import generate_samples
from mcnpy.ace.parsers import read_ace
from mcnpy.ace.writers.write_ace import write_ace
from mcnpy.cov.parse_covmat import read_scale_covmat, read_njoy_covmat


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
        # 1) build the sample suffix and master-filename (no extra extension)
        sample_tag       = f"_{sample_index+1:04d}"
        orig_xs_base     = os.path.splitext(os.path.basename(xsdir_file))[0]
        master_xs_name   = orig_xs_base + sample_tag
        master_xs_path   = os.path.join(output_dir, master_xs_name)

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

    # — write this sample’s small summary file —
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
    verbose: bool = True,
):
    # Print run parameters as metadata
    separator = "=" * 80
    print(f"\n{separator}")
    print(f"[ACE] [PARAMETERS] Run Configuration")
    print(f"{separator}")
    
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
    
    # Print all parameters
    print(f"  ACE files:             {formatted_ace}")
    print(f"  Covariance files:      {formatted_cov}")
    print(f"  MT numbers:            {formatted_mt}")
    print(f"  Number of samples:     {num_samples}")
    print(f"  Sampling space:        {space}")
    print(f"  Decomposition method:  {decomposition_method}")
    print(f"  Sampling method:       {sampling_method}")
    print(f"  Output directory:      {os.path.abspath(output_dir)}")
    print(f"  XSDIR file:            {xsdir_file if xsdir_file else 'None'}")
    print(f"  Random seed:           {seed if seed is not None else 'Random'}")
    print(f"  Parallel processes:    {nprocs}")
    print(f"  Mode:                  {'Dry run (factors only)' if dry_run else 'Full ACE generation'}")
    print(f"  Autofix covariance:    {autofix}")
    
    # Print timestamp
    from datetime import datetime
    print(f"  Timestamp:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{separator}\n")

    # normalize inputs (already done in the prints, but needed for the rest of the function)
    if isinstance(ace_files, str):
        ace_files = [ace_files]
    if isinstance(cov_files, str):
        cov_files = [cov_files]

    # Initialize a dictionary to collect summary information for each isotope
    summary_data = {}

    # Track skipped isotopes due to missing or invalid covariance
    skipped_isotopes = {}

    for ace_file, cov_file in zip(ace_files, cov_files):

        # ====== Start of ACE file processing ======
        separator = "=" * 80
        print(f"\n{separator}")
        print(f"[ACE] [PROCESSING] {ace_file}")
        print(f"{separator}\n")

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
            "warnings": []
        }
        
        cov = load_covariance(cov_file)
        if cov is None:
            print(f"[ACE] [ERROR] Unable to load a valid covariance matrix for {cov_file}")
            summary_data[zaid]["warnings"].append(f"No valid covariance file found: {os.path.basename(cov_file)}")
            skipped_isotopes[zaid] = "No valid covariance file found"
            print(f"\n{separator}\n")
            continue

        print(f"  Covariance file: {cov_file}")
        print(f"  Isotope: {zaid}")
        
        subseparator = "-" * 60
        print(f"\n{subseparator}")

        # Check if the covariance matrix is empty (no data)
        if cov.num_matrices == 0:
            print(f"[ACE] [SKIP] No covariance found in {zaid}. Skipping.")
            summary_data[zaid]["warnings"].append("No covariance data found in matrix")
            skipped_isotopes[zaid] = "No covariance data found in matrix"
            print(f"\n{separator}\n")
            continue

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

        # Print available MT numbers
        print(f"[ACE] [MT SELECTION]")
        print(f"  MTs in ACE file:          {mt_in_ace}")
        print(f"  MTs in covariance matrix: {mt_in_cov}")
        print(f"  MTs to be perturbed:      {mt_perturb}")
        
        print(f"\n{subseparator}\n")

        energy_grid = cov.energy_grid

        # Save pre-autofix MT list
        pre_autofix_mts = list(mt_perturb)
        
        factors, mt_perturb_final = generate_samples(
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
            verbose              = verbose,
        )
        
        # Update summary with autofix results
        if autofix is not None:
            removed_in_autofix = set(pre_autofix_mts) - set(mt_perturb_final if mt_perturb_final else [])
            for mt in removed_in_autofix:
                summary_data[zaid]["removed_mts"][mt] = "Removed during covariance autofix (non-positive definite matrix)"
        
        # Store final perturbed MTs
        summary_data[zaid]["mt_perturbed"] = mt_perturb_final if mt_perturb_final else []
        
        # -- prepare output dirs ----------------------------------------------
        iso_dir = os.path.join(output_dir, str(zaid))
        os.makedirs(iso_dir, exist_ok=True)

        # =====================================================================
        #  DRY‑RUN
        # =====================================================================
        if dry_run:
            if verbose:
                print(f"\n[ACE] [DRY-RUN] Generating only perturbation factors (no ACE files will be written)")
            for j in range(num_samples):
                _write_sample_summary(factors[j], j, energy_grid, mt_perturb_final, iso_dir, base)
            print(f"\n{separator}\n")
            continue

        # =====================================================================
        #  FULL processing (ACE rewrite)
        # =====================================================================
        print(f"\n[ACE] [PROCESSING] Creating {num_samples} perturbed ACE files")
        print(f"  Output directory: {os.path.abspath(output_dir)}")
        if nprocs > 1:
            print(f"  Using {nprocs} parallel processes")
            
        # Create progress tracking variables
        report_interval = max(1, min(100, num_samples // 10))  # Report at most 10 times
        
        tasks = [(ace_file, factors[j], j, energy_grid, mt_perturb_final, output_dir, xsdir_file) for j in range(num_samples)]

        if nprocs > 1:
            # For parallel processing, just show start and end messages
            print(f"  Starting parallel sample generation... (progress updates disabled in parallel mode)")
            
            with Pool(processes=nprocs) as pool:
                for args in tasks:
                    pool.apply_async(_process_sample, args=args)
                pool.close()
                pool.join()
                
            print(f"  Completed generating {num_samples} samples")
        else:
            # For sequential processing, show periodic progress updates
            print(f"  Generating samples (progress updates every {report_interval} samples)")
            
            for i, args in enumerate(tasks):
                _process_sample(*args)
                
                # Report progress periodically
                if (i + 1) % report_interval == 0 or i + 1 == num_samples:
                    progress = (i + 1) / num_samples * 100
                    print(f"  Progress: {i + 1}/{num_samples} samples ({progress:.1f}%)")

        # ====== End of ACE file processing ======
        print(f"\n{separator}")
        print(f"[ACE] [COMPLETED] {ace_file} ({num_samples} samples generated)")
        print(f"{separator}\n")

    # -- gather all master xsdir files ---------------------------------------
    if xsdir_file and not dry_run:
        _collect_master_xsdir(xsdir_file, output_dir)
    
    # =====================================================================
    #  Print final summary for all isotopes
    # =====================================================================
    separator = "=" * 80
    print(f"\n{separator}")
    print(f"[ACE] [SUMMARY] Processing Results")
    print(f"{separator}")
    
    if not summary_data and not skipped_isotopes:
        print("  No isotopes were processed.")
    else:
        # First report isotopes that were successfully processed
        processed_isotopes = {zaid: data for zaid, data in sorted(summary_data.items()) 
                             if zaid not in skipped_isotopes}
        
        if processed_isotopes:
            print("\n  SUCCESSFULLY PROCESSED ISOTOPES:")
            print(f"  {'-' * 50}")
            
            for zaid, data in processed_isotopes.items():
                print(f"\n  Isotope: {zaid} ({data['ace_file']})")
                print(f"  {'-' * 50}")
                
                # MT numbers that were perturbed
                if data['mt_perturbed']:
                    print(f"  ► Perturbed MT numbers: {', '.join(map(str, data['mt_perturbed']))}")
                else:
                    print(f"  ► No MT numbers were perturbed")
                
                # MT numbers that were removed and why
                if data['removed_mts']:
                    print(f"  ► Removed MT numbers:")
                    for mt, reason in sorted(data['removed_mts'].items()):
                        print(f"    • MT={mt}: {reason}")
                
                # Warnings if any
                if data['warnings']:
                    print(f"  ► Warnings:")
                    for warning in data['warnings']:
                        print(f"    • {warning}")
        
        # Then report isotopes that were skipped completely
        if skipped_isotopes or any(not data['mt_perturbed'] for data in summary_data.values()):
            print("\n  ISOTOPES WITH NO PERTURBATION:")
            print(f"  {'-' * 50}")
            
            # First report explicitly skipped isotopes
            for zaid, reason in sorted(skipped_isotopes.items()):
                if zaid in summary_data:
                    print(f"  Isotope: {zaid} ({summary_data[zaid]['ace_file']})")
                    print(f"    • Reason: {reason}")
            
            # Then report isotopes that were processed but had no MTs perturbed
            for zaid, data in sorted(summary_data.items()):
                if zaid not in skipped_isotopes and not data['mt_perturbed']:
                    print(f"  Isotope: {zaid} ({data['ace_file']})")
                    if data['warnings']:
                        for warning in data['warnings']:
                            print(f"    • Reason: {warning}")
                    else:
                        print(f"    • Reason: No eligible MT numbers to perturb")
    
    print(f"\n{separator}")


def apply_perturbation_factor_to_ace(ace, sample, sample_index, energy_grid, mt_numbers, verbose=True):
    """Apply per-group perturbation factors to ACE, and list which MTs were actually perturbed."""
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

    if verbose:
        print(f"[ACE] [PERTURB] Applying factors for sample #{sample_index+1:04d}")
        print(f"  Perturbed MT numbers: {perturbed_mts}")
        
    return perturbed_mts


def _apply_factors_to_mt(ace, mt, factors, boundaries, verbose=True):
    """Multiply each entry.value by factors[group] for this mt,
    **unless any factor is non-positive**.
    """
    if (factors <= 0).any():
        bad = ", ".join(f"{f:+.3e}" for f in factors if f <= 0)
        if verbose:
            print(f"  [WARNING] MT={mt}: negative or zero factors detected ({bad}). Reaction not perturbed.")
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
    path = os.path.join(sdir, f"{base}_{tag}_summary.txt")

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


def _collect_master_xsdir(xsdir_file: str, output_dir: str):
    """Gather all per-sample xsdir files into *output_dir/xsdir/*."""
    orig = os.path.splitext(os.path.basename(xsdir_file))[0]
    dest = os.path.join(output_dir, "xsdir")
    os.makedirs(dest, exist_ok=True)

    pattern = os.path.join(output_dir, f"{orig}_*")
    for xf in glob.glob(pattern):
        shutil.move(xf, os.path.join(dest, os.path.basename(xf)))
