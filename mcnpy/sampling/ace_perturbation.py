import numpy as np
import pandas as pd
import os
import shutil
import glob
from typing import List, Union, Optional
from mcnpy.sampling.generators import generate_samples
from mcnpy.ace.parsers import read_ace
from mcnpy.ace.writers.write_ace import write_ace
from mcnpy.cov.covmat import CovMat


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
):
    
    for i, ace_file in enumerate(ace_files):
        ace = read_ace(ace_file)
        
        isotope = ace.zaid
        isotope_covmat = covmat[i].to_dataframe()[covmat[i].to_dataframe()['ISO_H'] == isotope]

        mt_in_ace = ace.mt_numbers
        mt_in_cov = isotope_covmat['REAC_H'].unique().tolist()

        common_mt = sorted(set(mt_numbers) & set(mt_in_ace) & set(mt_in_cov))

        # Convert to ndarray the covariance matrix with the common mt numbers
        cov_ndarray = covmat[i].get_isotope_covariance_matrix(isotope, common_mt)

        perturbation_factors = generate_samples(
            cov_ndarray, 
            num_samples, 
            decomposition_method,
            sampling_method,
            seed+isotope
        )

        grouped_energies = covmat[i].energy_grid
        base_name, extension = os.path.splitext(os.path.basename(ace_file))

        for j, sample in enumerate(perturbation_factors):
            
            ace = read_ace(ace_file)  
            apply_perturbation_factor_to_ace(ace, sample, grouped_energies, common_mt)

            # Write perturbed ACE file
            sample_str = f"{j+1:04d}"
            new_filename  = f"{base_name}_{sample_str}{extension}"
            sample_dir = os.path.join(output_dir, str(isotope), sample_str)
            os.makedirs(sample_dir, exist_ok=True)

            out_path = os.path.join(sample_dir, new_filename)
            write_ace(ace, out_path)

            # Write .xsdir file
            rel_path = os.path.relpath(out_path, output_dir).replace(os.sep, "/")
            rel_path = f"../{rel_path}"

            hdr   = ace.header
            zaid  = hdr.zaid
            awr   = hdr.atomic_weight_ratio
            nxs   = hdr.nxs_array[1]
            temp  = hdr.temperature
            ptable = 'ptable' if getattr(ace.unresolved_resonance, 'has_data', False) else ''

            name_noext, _ = os.path.splitext(new_filename)
            xsdir_path = os.path.join(sample_dir, f"{name_noext}.xsdir")

            line = (
                f"{zaid}{extension} "
                f"{awr:.6f} "
                f"{rel_path} "
                "0 1 1 "
                f"{nxs} 0 0 "
                f"{temp:.3E} "
                f"{ptable}"
            )
            with open(xsdir_path, 'w') as fx:
                fx.write(line + "\n")

            # If given an xsdir file, update it
            if xsdir_file:
                base = os.path.basename(xsdir_file)
                name, ext = os.path.splitext(base)
                new_master = f"{name}_{sample_str}{ext}"
                out_master = os.path.join(output_dir, new_master)

                source_xs = out_master if os.path.exists(out_master) else xsdir_file
                with open(source_xs, 'r') as fx:
                    xs_lines = fx.readlines()

                zaid_str = f"{zaid}{extension}"
                updated = []
                for l in xs_lines:
                    if l.startswith(zaid_str):
                        updated.append(line + "\n")
                    else:
                        updated.append(l)

                with open(out_master, 'w') as fx:
                    fx.writelines(updated)

    # After all perturbations, move master xsdir files to output_dir/xsdir
    if xsdir_file:
        base = os.path.basename(xsdir_file)
        name, ext = os.path.splitext(base)
        xsdir_dest = os.path.join(output_dir, 'xsdir')
        os.makedirs(xsdir_dest, exist_ok=True)
        pattern = os.path.join(output_dir, f"{name}_*{ext}")
        for xsfile in glob.glob(pattern):
            dst = os.path.join(xsdir_dest, os.path.basename(xsfile))
            shutil.move(xsfile, dst)

        
def apply_perturbation_factor_to_ace(ace, sample, energy_grid, mt_numbers):
    
    n_groups = len(energy_grid) - 1
    if sample.shape[0] != len(mt_numbers) * n_groups:
        raise ValueError(f"sample length {sample.shape[0]} ≠ {len(mt_numbers)}x{n_groups}")

    # For faster binning
    boundaries = np.asarray(energy_grid)

    for mt_idx, mt in enumerate(mt_numbers):
        # slice out the factors for this MT
        start = mt_idx * n_groups
        end   = start + n_groups
        factors = sample[start:end]  # shape == (n_groups,)

        # grab the reaction object
        reac = ace.cross_section.reaction[mt]
        energies = np.asarray(reac.energies)    # array of floats
        xs_entries = reac._xs_entries           # list of entry objects

        # assign each energy to a group index in [0..n_groups-1], or -1 if outside
        # digitize returns bins in 1..len(boundaries), so subtract 1 for 0‐based
        bin_indices = np.digitize(energies, boundaries) - 1

        # now multiply each entry’s value by its group factor
        for i, entry in enumerate(xs_entries):
            grp = bin_indices[i]
            # only apply if energy is within [E0,E_last)
            if 0 <= grp < n_groups:
                entry.value *= factors[grp]



