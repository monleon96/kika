import mcnpy
from mcnpy.sampling.ace_perturbation import perturb_ACE_files

# 1) Prepare inputs just like you did before
scale_file = '/share_snc/projets/INSIDER/SENSIBILITE/Covariances_scale_txt/scale.rev05.44groupcov.txt'
ace_Fe56 = '/soft_snc/lib/ace/80/600/260560_80.06c'
ace_Fe54 = '/soft_snc/lib/ace/80/600/260540_80.06c'
ace_H1 = '/soft_snc/lib/ace/80/600/10010_80.06c'
ace_O16 = '/soft_snc/lib/ace/80/600/80160_80.06c'
acelist = [ace_Fe56, ace_Fe54, ace_H1, ace_O16]

covmat = mcnpy.read_scale_covmat(scale_file)
covmatlist = [covmat]*len(acelist)

mt_numbers   = [2,4,102,103,107]
num_samples  = 16
output_dir   = "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/new_pert"
xsdir_file   = "/soft_snc/lib/xsdir/xsdir80"
seed         = 42
nprocs       = 4

# 2) Call the new perturb_ace_files
print(f"Generating {num_samples} perturbed ACE files...")
perturb_ACE_files(
    ace_files            = acelist,
    covmat               = covmatlist,
    mt_numbers           = mt_numbers,
    num_samples          = num_samples,
    output_dir           = output_dir,
    xsdir_file           = xsdir_file,
    sampling_method      = 'sobol',
    decomposition_method = 'svd',
    seed                 = seed,
    nprocs               = nprocs,
)
