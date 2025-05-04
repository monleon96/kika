import numpy as np
import scipy.stats
from scipy.stats import qmc
from typing import Optional, Tuple
from mcnpy.sampling.decomposition import svd_decomposition, cholesky_decomposition, eigen_decomposition

def generate_samples(
    cov: np.ndarray,
    n_samples: int,
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated samples for cross section perturbation.
    
    The samples are generated with mean 1.0 and have correlation structure
    defined by the provided covariance matrix.
    
    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix
    n_samples : int
        Number of samples to generate
    decomposition_method : str, default="svd"
        Method to decompose the covariance matrix ("svd", "cholesky", or "eigen")
    sampling_method : str, default="sobol"
        Method to generate the samples ("random", "lhs", or "sobol")
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Array of perturbation factors with shape (n_samples, n_dimensions)
    
    Raises
    ------
    ValueError
        If decomposition or sampling method is not supported
    """
    # Determine dimensions
    n_dimensions = cov.shape[0]
    
    cov, perm = reorder_to_descending(cov, n_groups=44, n_reactions=2)

    # Generate uncorrelated standard normal samples using the specified method
    if sampling_method.lower() == "random":
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        uncorrelated_samples = np.random.normal(0, 1, size=(n_samples, n_dimensions))
    
    elif sampling_method.lower() == "lhs":
        # Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
        uniform_samples = sampler.random(n=n_samples)
        # Transform uniform samples to standard normal using inverse CDF
        uncorrelated_samples = scipy.stats.norm.ppf(uniform_samples)
    
    elif sampling_method.lower() == "sobol":
        # Sobol sequence sampling
        sampler = qmc.Sobol(d=n_dimensions, scramble=True, seed=seed)
        uniform_samples = sampler.random(n=n_samples)
        # Transform uniform samples to standard normal using inverse CDF
        uncorrelated_samples = scipy.stats.norm.ppf(uniform_samples)

    else:
        raise ValueError(f"Unsupported sampling method: {sampling_method}. "
                        f"Choose from 'random', 'lhs', or 'sobol'.")

    # Apply correlation structure based on the decomposition method
    if decomposition_method.lower() == "svd":
        L = svd_decomposition(cov)
    elif decomposition_method.lower() == "cholesky":
        L = cholesky_decomposition(cov)
    elif decomposition_method.lower() == "eigen":
        L = eigen_decomposition(cov)
    else:
        raise ValueError(f"Unsupported decomposition method: {decomposition_method}. "
                        f"Choose from 'svd', 'cholesky', or 'eigen'.")
    
    # Transform uncorrelated samples to correlated samples with mean 1.0
    correlated_samples = uncorrelated_samples @ L.T
    
    # center the samples around 1.0
    correlated_samples_1 = correlated_samples + 1.0


    debug = True
    if debug:
        print(f"\n\nDEBUG: Uncorrelated samples:\n{uncorrelated_samples[-1]}\n")
        print(f"DEBUG: Correlated samples:\n{correlated_samples[-1]}\n")


        with open("debug_mcnpy.txt", "w") as f:
            f.write("L:\n")
            for item in L:
                f.write(f"{item}\n")


    return correlated_samples_1




def reorder_to_descending(cov: np.ndarray,
                          n_groups: int = 44,
                          n_reactions: int = 2
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return cov matrix with each reaction's energy‐groups reversed,
    plus the integer permutation array used.
    
    cov        : (n_groups*n_reactions, n_groups*n_reactions) matrix
    n_groups   : number of energy groups per reaction
    n_reactions: number of reactions (here 2)
    
    Returns
    -------
    cov_desc : cov with rows & cols permuted so that within each
               reaction block, energies go from high→low
    perm     : 1D array of length n_groups*n_reactions giving
               the new order of indices
    """
    # sanity check
    assert cov.shape == (n_groups*n_reactions, n_groups*n_reactions)
    
    perm = []
    for r in range(n_reactions):
        start = r * n_groups
        stop  = (r + 1) * n_groups
        block = np.arange(start, stop)
        perm.extend(block[::-1])   # reverse each block
    perm = np.array(perm, dtype=int)
    
    # apply to both axes
    cov_desc = cov[np.ix_(perm, perm)]
    return cov_desc, perm