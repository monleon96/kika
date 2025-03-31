import numpy as np
import scipy.stats
from scipy.stats import qmc
from typing import Optional, Literal
from mcnpy.sampling.decomposition import svd_decomposition, cholesky_decomposition

def generate_samples(
    cov: np.ndarray, 
    n_samples: int, 
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None
) -> np.ndarray:
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
        Method to decompose the covariance matrix ("svd" or "cholesky")
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
    else:
        raise ValueError(f"Unsupported decomposition method: {decomposition_method}. "
                        f"Choose from 'svd' or 'cholesky'.")
    
    # Transform uncorrelated samples to correlated samples with mean 1.0
    correlated_samples = uncorrelated_samples @ L.T + 1.0
    
    return correlated_samples
