# run_my_perturb.py

import mcnpy
from mcnpy.sampling.perturb_parallel import parallel_perturb

# 1) Prepare inputs just like you did before
scale_file = '/share_snc/.../scale.rev05.44groupcov.txt'
ace_Fe56 = '/soft_snc/.../260560_80.06c'
ace_Fe54 = '/soft_snc/.../260540_80.06c'
ace_H1  = '/soft_snc/.../10010_80.06c'
ace_O16 = '/soft_snc/.../160160_80.06c'
acelist = [ace_Fe56, ace_Fe54, ace_H1, ace_O16]

covmat = mcnpy.read_scale_covmat(scale_file)
covmats = [covmat]*len(acelist)

mt_numbers   = [2,4,102,103,107]
energy_grid  = mcnpy.energy_grids.SCALE44
num_samples  = 24
output_dir   = "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB"
xsdir_file   = "/soft_snc/lib/xsdir/xsdir80"
seed         = 42
num_cpus     = 4

# 2) Call the new parallel_perturb
print(f"Generating {num_samples} perturbed ACE files...")
parallel_perturb(
    ace_file_path      = acelist,
    mt_numbers         = mt_numbers,
    energy_grid        = energy_grid,
    covmat             = covmats,
    num_samples        = num_samples,
    decomposition_method="svd",
    sampling_method    ="sobol",
    output_dir         = output_dir,
    xsdir              = xsdir_file,
    seed               = seed,
    verbose            = False,
    nprocs             = num_cpus,     
)
