import numpy as np
import os
import shutil
import glob
from typing import List, Union, Optional
from multiprocessing import Pool

from mcnpy.sampling.generators import generate_samples
from mcnpy.ace.parsers import read_ace
from mcnpy.ace.writers.write_ace import write_ace
from mcnpy.cov.covmat import CovMat


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
    apply_perturbation_factor_to_ace(ace, sample, sample_index, energy_grid, mt_numbers)

    # — write perturbed ACE —
    base, ext = os.path.splitext(os.path.basename(ace_file))
    sample_str = f"{sample_index+1:04d}"
    sample_dir = os.path.join(output_dir, str(ace.zaid), sample_str)
    os.makedirs(sample_dir, exist_ok=True)
    out_ace = os.path.join(sample_dir, f"{base}_{sample_str}{ext}")
    ace.update_cross_sections()
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
        sf.write(f"Sample {sample_index+1}\n")

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
    mt_numbers: List[int],
    num_samples: int,
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    output_dir: str = '.',
    xsdir_file: Optional[str] = None,
    seed: Optional[int] = None,
    nprocs: int = 1,
):
    # normalize inputs
    if isinstance(ace_files, str):
        ace_files = [ace_files]
    if isinstance(covmat, CovMat):
        covmat = [covmat]

    for i, ace_file in enumerate(ace_files):
        ace = read_ace(ace_file)
        zaid = ace.zaid
        base, _ = os.path.splitext(os.path.basename(ace_file))

        # pick out only the MTs that exist in both ACE and covariance
        df = covmat[i].to_dataframe()
        iso_df = df[df['ISO_H'] == zaid]
        mt_in_ace = set(ace.mt_numbers)
        mt_in_cov = set(iso_df['REAC_H'])
        common_mt = mt_in_ace & mt_in_cov & set(mt_numbers)
        if (4 in mt_in_cov) and (4 in mt_numbers) and any(mt in mt_in_ace for mt in range(51, 92)):
            common_mt = common_mt | {4}
        common_mt = sorted(common_mt)

        # generate all samples (matrix of shape [num_samples, n_groups * n_mts])
        cov_nd = covmat[i].get_isotope_covariance_matrix(zaid, common_mt)
        factors = generate_samples(
            cov_nd,
            num_samples,
            decomposition_method,
            sampling_method,
            seed + zaid if seed is not None else None
        )
        energy_grid = covmat[i].energy_grid

        # prepare output dirs & blank master summary
        iso_dir = os.path.join(output_dir, str(zaid))
        os.makedirs(iso_dir, exist_ok=True)
        master_summary = os.path.join(iso_dir, f"{base}_perturb_summary.txt")
        open(master_summary, 'w').close()

        # set up tasks
        tasks = [(ace_file, factors[j], j, energy_grid, common_mt, output_dir, xsdir_file) for j in range(num_samples)]

        if nprocs > 1:
            with Pool(processes=nprocs) as pool:
                for args in tasks:
                    pool.apply_async(_process_sample, args=args)
                pool.close()
                pool.join()
        else:
            for args in tasks:
                _process_sample(*args)

        # — after all samples: merge the tiny summaries into one file —
        with open(master_summary, 'a') as out:
            for j in range(num_samples):
                sample_str = f"{j+1:04d}"
                tmp = os.path.join(iso_dir, sample_str, f"{base}_{sample_str}_summary.txt")
                with open(tmp) as inp:
                    out.write(inp.read())

    # gather all master xsdir files if requested
    if xsdir_file:
        orig = os.path.splitext(os.path.basename(xsdir_file))[0]
        dest = os.path.join(output_dir, 'xsdir')
        os.makedirs(dest, exist_ok=True)

        # match e.g. output_dir/orig_0001, orig_0002, …
        pattern = os.path.join(output_dir, f"{orig}_*")
        for xf in glob.glob(pattern):
            dst_path = os.path.join(dest, os.path.basename(xf))
            shutil.move(xf, dst_path)


def apply_perturbation_factor_to_ace(ace, sample, sample_index, energy_grid, mt_numbers):
    """Apply per-group perturbation factors to ACE.
    """
    n_groups = len(energy_grid) - 1
    if sample.shape[0] != len(mt_numbers) * n_groups:
        raise ValueError(f"sample length {sample.shape[0]} ≠ {len(mt_numbers)}×{n_groups}")

    boundaries = np.asarray(energy_grid)

    print(f"\n\nSample {sample_index+1} of isotope {ace.zaid}:\n")
    
    for mt_idx, mt in enumerate(mt_numbers):
        start = mt_idx * n_groups
        end   = start + n_groups
        factors = sample[start:end]

        if mt == 4:
            for mt_inelastic in range(51, 92):
                if mt_inelastic in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_inelastic, factors, boundaries)
                    print(f"Applying perturbation to MT={mt_inelastic}")
        elif mt == 103:
            for mt_proton_production in range(600, 650):
                if mt_proton_production in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_proton_production, factors, boundaries)
                    print(f"Applying perturbation to MT={mt_proton_production}")
        elif mt == 104:
            for mt_H2_production in range(650, 700):
                if mt_H2_production in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_H2_production, factors, boundaries)
                    print(f"Applying perturbation to MT={mt_H2_production}")
        elif mt == 105:
            for mt_H3_production in range(700, 750):
                if mt_H3_production in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_H3_production, factors, boundaries)
                    print(f"Applying perturbation to MT={mt_H3_production}")
        elif mt == 106:
            for mt_He3_production in range(750, 800):
                if mt_He3_production in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_He3_production, factors, boundaries)
                    print(f"Applying perturbation to MT={mt_He3_production}")
        elif mt == 107:
            for mt_He4_production in range(800, 850):
                if mt_He4_production in ace.mt_numbers:
                    _apply_factors_to_mt(ace, mt_He4_production, factors, boundaries)
                    print(f"Applying perturbation to MT={mt_He4_production}")
        else:
            _apply_factors_to_mt(ace, mt, factors, boundaries)
            print(f"Applying perturbation to MT={mt}")

def _apply_factors_to_mt(ace, mt, factors, boundaries):
    """Helper: multiply each entry.value by factors[group] for this mt."""
    reac = ace.cross_section.reaction[mt]
    energies = np.asarray(reac.energies)
    xs_entries = reac._xs_entries
    # digitize into group bins
    bin_indices = np.digitize(energies, boundaries) - 1

    for i, entry in enumerate(xs_entries):
        grp = bin_indices[i]
        if 0 <= grp < len(factors):
            entry.value *= factors[grp]
