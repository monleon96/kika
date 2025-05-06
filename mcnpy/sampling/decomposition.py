import numpy as np

def svd_decomposition(cov: np.ndarray) -> np.ndarray:
    """
    Decompose covariance matrix using Singular Value Decomposition.
    
    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix to decompose
        
    Returns
    -------
    np.ndarray
        Transformation matrix L where L @ L.T = cov
    """
    
    # Perform SVD
    U, s, Vt = np.linalg.svd(cov)
    # Return the transformation matrix
    return U @ np.diag(np.sqrt(s))

def eigen_decomposition(cov):
    eigvals, eigvecs = np.linalg.eigh(cov)
    if np.any(eigvals < 0):
        eigvals[eigvals < 0] = 0 
    return eigvecs @ np.diag(np.sqrt(eigvals))


def cholesky_decomposition(cov: np.ndarray) -> np.ndarray:
    """
    Decompose covariance matrix using Cholesky decomposition.
    
    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix to decompose (must be positive definite)
        
    Returns
    -------
    np.ndarray
        Lower triangular matrix L where L @ L.T = cov
    """
    try:
        # Attempt Cholesky decomposition
        L = np.linalg.cholesky(cov)
        return L
    except np.linalg.LinAlgError:
        # If fails, matrix is not positive definite
        raise ValueError("Covariance matrix is not positive definite. Try using SVD decomposition instead.")
