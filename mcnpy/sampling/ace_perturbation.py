import numpy as np
import os
import shutil
import glob
from typing import List, Union, Optional
from multiprocessing import Pool

from mcnpy.sampling.generators import generate_samples_blocked, mt_blocks
from mcnpy.ace.parsers import read_ace
from mcnpy.ace.writers.write_ace import write_ace
from mcnpy.cov.covmat import CovMat


def calculate_common_mt_for_ranges(mt_in_ace, mt_in_cov, codes_to_check):
    """
    Devuelve, para cada código pedido:
        - 'common'     : MTs presentes en la covarianza (para generar factores)
        - 'targets'    : MTs del ACE que se van a perturbar
        - 'cov_source' : MT(s) que proporcionan los datos
    """
    code_ranges = {
        4:   (51,  91),   
        103: (600, 649),
        104: (650, 699),
        105: (700, 749),
        106: (750, 799),
        107: (800, 849),
    }

    ace_set = set(mt_in_ace)
    cov_set = set(mt_in_cov)
    results = {}

    for code in codes_to_check:

        # ---------- 1) Summary MTs ----------
        if code in code_ranges:
            low, high = code_ranges[code]
            ace_range = {mt for mt in ace_set if low <= mt <= high}
            cov_range = {mt for mt in cov_set if low <= mt <= high}

            if cov_range:
                # La covarianza contiene sub-MT concretos (p. ej. 600-649)
                targets    = sorted(ace_range & cov_range)
                common     = targets                      # mismos códigos
                cov_source = common
            elif code in cov_set and ace_range:
                # Solo está el summary (4, 103, …) en la covarianza
                targets    = sorted(ace_range)            # sub-MT a tocar
                common     = [code]                       # 4 ó 103…
                cov_source = [code]
            elif code in cov_set and code in ace_set:
                # Caso raro: el summary también existe como reacción
                targets    = [code]
                common     = [code]
                cov_source = [code]
            else:
                targets = common = cov_source = []
        # ---------- 2) MT “normales” ----------
        else:
            if code in ace_set and code in cov_set:
                targets = common = cov_source = [code]
            else:
                targets = common = cov_source = []

        results[code] = {
            'common'    : common,
            'targets'   : targets,
            'cov_source': cov_source
        }

    return results


def _process_sample(
    ace_file: str,
    sample: np.ndarray,
    sample_index: int,
    energy_grid: List[float],
    mt_numbers: List[int],
    output_dir: str,
    xsdir_file: Optional[str],
    verbose: bool = True,
):
    # — read & perturb ACE —
    ace = read_ace(ace_file)
    apply_perturbation_factor_to_ace(ace, sample, sample_index, energy_grid, mt_numbers, verbose)

    # — write perturbed ACE —
    base, ext = os.path.splitext(os.path.basename(ace_file))
    sample_str = f"{sample_index+1:04d}"
    sample_dir = os.path.join(output_dir, str(ace.zaid), sample_str)
    os.makedirs(sample_dir, exist_ok=True)
    out_ace = os.path.join(sample_dir, f"{base}_{sample_str}{ext}")
    if verbose:
        print(f"\nRecalculating possible total cross sections [1, 3, 4, 18, 103, 104, 105, 106, 107]")
    ace.update_cross_sections()
    if verbose:
        print(f"\nWriting perturbed ACE file: {out_ace}")
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


def perturb_ACE_files(
    ace_files: Union[str, List[str]],
    covmat: Union[CovMat, List[CovMat]],
    mt_list: List[int],
    num_samples: int,
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    output_dir: str = '.',
    xsdir_file: Optional[str] = None,
    seed: Optional[int] = None,
    nprocs: int = 1,
    dry_run: bool = False,
    verbose: bool = True,
):
    # normalize inputs
    if isinstance(ace_files, str):
        ace_files = [ace_files]
    if isinstance(covmat, CovMat):
        covmat = [covmat]

    for ace_file, cov in zip(ace_files, covmat):
        ace = read_ace(ace_file)
        zaid = ace.zaid
        base, _ = os.path.splitext(os.path.basename(ace_file))
        
        if verbose:
            print(f"\n"+"-"*60)
            print(f"\nProcessing ACE file: {ace_file}")
            print(f"Isotope: {zaid}")

        # Check if the covariance matrix is empty (no data)
        if cov.num_matrices == 0:
            if verbose:
                print(f"[{zaid}] No covariance. Skipping.")
            continue

        # Read and restrict the dataframe for the given isotope
        df = cov.to_dataframe()
        iso_df = df[df["ISO_H"] == zaid]
        mt_in_ace = set(ace.mt_numbers)
        mt_in_cov = set(iso_df["REAC_H"]) - {0, 1, 3}

        # Print available MT numbers
        print("\nMTs in ACE file:            ", sorted(mt_in_ace))
        print("MTs in covariance matrix:   ", sorted(mt_in_cov), "\n")

        summary_codes = {4, 103, 104, 105, 106, 107}
        if mt_list:
            requested = set(mt_list)
        else:
            requested = set(mt_in_ace) | (summary_codes & mt_in_cov)

        common_info = calculate_common_mt_for_ranges(mt_in_ace, mt_in_cov, sorted(requested))
        common_mt = sorted({m for info in common_info.values() for m in info["common"]})
        if not common_mt:
            if verbose:
                print(f"[{zaid}] No MT overlap. Skipping.")
            continue

        # ——— Perturbation summary —————————————————————————
        print("=== Perturbations to be applied ===")
        printed_mts = set()          
        # Pasada de MT individuales (len(targets) == 1)
        for code, info in common_info.items():
            targets = info['targets']
            if len(targets) != 1:
                continue
            mt = targets[0]
            if mt in printed_mts:
                continue
            cov_src   = info['cov_source']
            src_label = f"MT={cov_src[0]}"
            print(f"[{mt}]   with {src_label} covariance data")
            printed_mts.add(mt)

        # Pasada de grupos (len(targets) > 1)
        for code, info in common_info.items():
            targets = info['targets']
            if len(targets) <= 1:
                continue
            # Saltar si TODOS los MT del grupo ya se han mostrado
            if all(t in printed_mts for t in targets):
                continue
            cov_src   = info['cov_source']
            src_label = f"MT={cov_src[0]}" if len(cov_src) == 1 else f"MTs {cov_src}"
            print(f"{targets}   with {src_label} covariance data")
            printed_mts.update(targets)

        print("===================================\n")


        # Build the final list of MTs to generate factors for
        common_mt = sorted({mt for info in common_info.values() for mt in info['common']})
        
        print(f"DEBUG: Final list of MTs to perturb: {common_mt}\n")

        # -- draw factors ------------------------------------------------------
        cov_nd = cov.get_isotope_covariance_matrix(zaid, common_mt)
        energy_grid = cov.energy_grid
        blocks = mt_blocks(cov_nd, common_mt, n_groups=len(energy_grid)-1)
        factors = generate_samples_blocked(
            cov                  = cov_nd,
            blocks               = blocks,
            n_samples            = num_samples,
            decomposition_method = decomposition_method,
            sampling_method      = sampling_method,
            seed                 = None if seed is None else seed + zaid,
            mt_numbers           = common_mt,
            energy_grid          = energy_grid,
            verbose              = verbose,
        )

        # -- prepare output dirs ----------------------------------------------
        iso_dir = os.path.join(output_dir, str(zaid))
        os.makedirs(iso_dir, exist_ok=True)
        #master_summary = os.path.join(iso_dir, f"{base}_perturb_summary.txt")
        #open(master_summary, "w").close()


        # =====================================================================
        #  DRY‑RUN
        # =====================================================================
        if dry_run:
            if verbose:
                print(f"\ndry-run = ON → generating only perturbation factors")
            for j in range(num_samples):
                _write_sample_summary(factors[j], j, energy_grid, common_mt, iso_dir, base)
            #_merge_summaries(num_samples, iso_dir, base, master_summary)
            continue


        # =====================================================================
        #  FULL processing (ACE rewrite)
        # =====================================================================
        tasks = [(ace_file, factors[j], j, energy_grid, common_mt, output_dir, xsdir_file, verbose) for j in range(num_samples)]

        if nprocs > 1:
            with Pool(processes=nprocs) as pool:
                for args in tasks:
                    pool.apply_async(_process_sample, args=args)
                pool.close()
                pool.join()
        else:
            for args in tasks:
                _process_sample(*args)

        #_merge_summaries(num_samples, iso_dir, base, master_summary)

    # -- gather all master xsdir files ---------------------------------------
    if xsdir_file and not dry_run:
        _collect_master_xsdir(xsdir_file, output_dir)


def apply_perturbation_factor_to_ace(ace, sample, sample_index, energy_grid, mt_numbers, verbose=True):
    """Apply per-group perturbation factors to ACE.
    """
    n_groups = len(energy_grid) - 1
    if sample.shape[0] != len(mt_numbers) * n_groups:
        raise ValueError(f"sample length {sample.shape[0]} ≠ {len(mt_numbers)}x{n_groups}")

    boundaries = np.asarray(energy_grid)

    print(f"\nSAMPLE #{sample_index+1}:\n")
    
    for mt_idx, mt in enumerate(mt_numbers):
        start = mt_idx * n_groups
        end   = start + n_groups
        factors = sample[start:end]

        if mt == 4:
            for mt_inelastic in range(51, 92):
                if mt_inelastic in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_inelastic, factors, boundaries, verbose)
        elif mt == 103:
            for mt_proton_production in range(600, 650):
                if mt_proton_production in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_proton_production, factors, boundaries, verbose)
        elif mt == 104:
            for mt_H2_production in range(650, 700):
                if mt_H2_production in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_H2_production, factors, boundaries, verbose)
        elif mt == 105:
            for mt_H3_production in range(700, 750):
                if mt_H3_production in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_H3_production, factors, boundaries, verbose)
        elif mt == 106:
            for mt_He3_production in range(750, 800):
                if mt_He3_production in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_He3_production, factors, boundaries, verbose)
        elif mt == 107:
            for mt_He4_production in range(800, 850):
                if mt_He4_production in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_He4_production, factors, boundaries, verbose)
        else:
            _apply_factors_to_mt(ace, mt, factors, boundaries, verbose)


# =============================================================================
# 3. Helper utilities
# =============================================================================

def _apply_factors_to_mt(ace, mt, factors, boundaries, verbose=True):
    """Multiply each entry.value by factors[group] for this mt,
    **unless any factor is non-positive**.
    """
    if (factors <= 0).any():
        if verbose:
            bad = ", ".join(f"{f:+.3e}" for f in factors if f <= 0)
            print(f"[SKIPPED] MT={mt}: negative or zero factors detected ({bad})")
        return  # leave this reaction unperturbed

    reac       = ace.cross_section.reaction[mt]
    energies   = np.asarray(reac.energies)
    xs_entries = reac._xs_entries
    bin_idx    = np.digitize(energies, boundaries) - 1

    for i, entry in enumerate(xs_entries):
        grp = bin_idx[i]
        if 0 <= grp < len(factors):
            entry.value *= factors[grp]

    if verbose:
        print(f"Applying perturbation to MT={mt}")

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


def _merge_summaries(n_samples: int, iso_dir: str, base: str, master_summary: str):
    """Concatenate all per-sample summaries into a master file."""
    with open(master_summary, "a") as out:
        for j in range(n_samples):
            tag = f"{j + 1:04d}"
            p = os.path.join(iso_dir, tag, f"{base}_{tag}_summary.txt")
            with open(p) as inp:
                out.write(inp.read())


def _collect_master_xsdir(xsdir_file: str, output_dir: str):
    """Gather all per-sample xsdir files into *output_dir/xsdir/*."""
    orig = os.path.splitext(os.path.basename(xsdir_file))[0]
    dest = os.path.join(output_dir, "xsdir")
    os.makedirs(dest, exist_ok=True)

    pattern = os.path.join(output_dir, f"{orig}_*")
    for xf in glob.glob(pattern):
        shutil.move(xf, os.path.join(dest, os.path.basename(xf)))
