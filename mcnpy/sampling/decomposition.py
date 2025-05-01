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
    # Check if matrix is positive semi-definite
    #eigenvalues = np.linalg.eigvalsh(cov)
    #if np.any(eigenvalues < -1e-7):  # Allow for small numerical errors
    #    raise ValueError("Covariance matrix is not positive semi-definite")
    
    # Perform SVD
    U, s, Vt = np.linalg.svd(cov, full_matrices=False)
    
    # Remove very small negative eigenvalues (numerical artifacts)
    s = np.maximum(s, 0)
    
    # Return the transformation matrix
    return U @ np.diag(np.sqrt(s))

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
