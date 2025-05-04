import mcnpy
from mcnpy.sampling.perturb_ace_parallel import perturb_ace_files

# 1) Prepare inputs just like you did before
scale_file = '/share_snc/projets/INSIDER/SENSIBILITE/Covariances_scale_txt/scale.rev05.44groupcov.txt'
ace_Fe56 = '/soft_snc/lib/ace/80/600/260560_80.06c'
ace_Fe54 = '/soft_snc/lib/ace/80/600/260540_80.06c'
ace_H1 = '/soft_snc/lib/ace/80/600/10010_80.06c'
ace_O16 = '/soft_snc/lib/ace/80/600/80160_80.06c'
acelist = [ace_Fe56, ace_Fe54, ace_H1, ace_O16]

covmat = mcnpy.read_scale_covmat(scale_file)
covmats = [covmat]*len(acelist)

mt_numbers   = [2,4,102,103,107]
energy_grid  = mcnpy.energy_grids.SCALE44
num_samples  = 16
output_dir   = "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/parallel"
xsdir_file   = "/soft_snc/lib/xsdir/xsdir80"
seed         = 42
nprocs       = 4

# 2) Call the new perturb_ace_files
print(f"Generating {num_samples} perturbed ACE files...")
perturb_ace_files(
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
    verbose            = True,
    nprocs             = nprocs,     
)
