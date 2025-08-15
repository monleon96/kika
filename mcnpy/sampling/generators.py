import numpy as np
from scipy import stats
from scipy.stats import qmc
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
from typing import List, Sequence, Optional, Tuple, Dict, Any

from mcnpy.cov.covmat import CovMat
from mcnpy.cov.decomposition import (
    verify_cholesky_decomposition,
    verify_eigen_decomposition, 
    verify_svd_decomposition,
    verify_pca_decomposition
)



# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _empirical_cov(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)

def _uncorrelated(
    dim: int,
    n: int,
    method: str,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Draw uncorrelated N(0, I) samples of shape (n, dim) via three methods:
      - 'random': plain RNG.normal
      - 'lhs': Latin Hypercube → inverse CDF
      - 'sobol': Sobol scramble → inverse CDF (with optional fast_forward)
    """
    m = method.lower()
    if m == "random":
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, dim))

    if m in ("lhs", "sobol"):
        if m == "lhs":
            sampler = qmc.LatinHypercube(d=dim, seed=seed)
        else:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        U = sampler.random(n)
        return stats.norm.ppf(U)

    raise ValueError("method must be 'random', 'lhs' or 'sobol'")




# ----------------------------------------------------------------------
# PCA decomposition (shared by linear & log spaces)
# ----------------------------------------------------------------------
def _pca_decomposition_sampling(
    cov_mat: np.ndarray,
    n_samples: int,
    sampling_method: str,
    seed: Optional[int],
    trunc_threshold: float,
    verbose: bool,
    space: str = "log",
    logger = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Draw N(0, Σ) samples via PCA truncation while capturing
    `trunc_threshold` of the variance.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, int]
        (samples, eigenvalues, eigenvectors, k) where k is number of components used
    """
    # 1) Force symmetry
    T = (cov_mat + cov_mat.T) / 2.0

    # 2) Eigen-decompose
    eigvals, eigvecs = np.linalg.eigh(T)

    # 3) Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 4) Clamp negatives (numerical noise)
    eigvals = np.clip(eigvals, 0.0, None)

    # 5) Choose rank k
    total_var = eigvals.sum()
    cumvar    = np.cumsum(eigvals)
    k = int(np.searchsorted(cumvar / total_var, trunc_threshold) + 1)
    if verbose:
        print(f"PCA: using k={k} components "
              f"({cumvar[k-1] / total_var:.4f} variance)")

    # 6) Verify PCA decomposition quality
    if verbose:
        verify_pca_decomposition(
            original_matrix=T,
            eigvals=eigvals,
            eigvecs=eigvecs,
            k=k,
            space=space,
            verbose=verbose,
            logger=logger
        )

    # 7) Build transform L and draw uncorrelated Z
    Vred    = eigvecs[:, :k]
    sqrt_D  = np.sqrt(eigvals[:k])
    L       = Vred @ np.diag(sqrt_D)
    Z       = _uncorrelated(k, n_samples, sampling_method, seed)

    samples = Z @ L.T                      # shape (n_samples, p)
    return samples, eigvals, eigvecs, k



# ----------------------------------------------------------------------
#  Linear-space sample diagnostics
# ----------------------------------------------------------------------
def _diagnostics_samples_linear(
    samples: np.ndarray,
    cov_lin: np.ndarray,
    param_pairs: List[Tuple[int, int]],
    num_groups: int,
    bins: Optional[np.ndarray],
    z_limit: float = 3.0,
    verbose: bool = True,
) -> None:

    if not verbose:
        return

    # Try to get logger from ace_perturbation module
    try:
        from mcnpy.sampling.ace_perturbation import _get_logger
        logger = _get_logger()
    except:
        logger = None

    separator = "-" * 60
    log_msg = f"\n[SAMPLING] [LINEAR DIAGNOSTICS]\n{separator}"
    
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    warnings = []
    means_f = samples.mean(axis=0)
    p, n = cov_lin.shape[0], samples.shape[0]

    # ---1) negatives ------------------------------------------------------
    neg_mask = samples < 0
    if np.any(neg_mask):
        frac_neg = np.mean(neg_mask, axis=0)
        for dim, frac in enumerate(frac_neg):
            if frac == 0.0:
                continue
            pair_idx, grp_idx = divmod(dim, num_groups)
            zaid, mt = param_pairs[pair_idx]

            if bins is not None:
                lo, hi = bins[grp_idx], bins[grp_idx + 1]
                primary = (f"(ZAID={zaid}, MT={mt}), "
                           f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
            else:
                primary = f"(ZAID={zaid}, MT={mt}), G={grp_idx}"

            warnings.append(
                f"  [WARNING] Negative values in {primary}: {frac:.2%} of samples"
            )

    # ---2) mean significantly off 1 --------------------------------------
    for dim in range(p):
        var_lin = cov_lin[dim, dim]
        if var_lin == 0.0:
            continue                      # nothing to test
        
        se_f = np.sqrt(var_lin / n)
        z = (means_f[dim] - 1.0) / se_f
        if abs(z) <= z_limit:
            continue

        pair_idx, grp_idx = divmod(dim, num_groups)
        zaid, mt = param_pairs[pair_idx]

        if bins is not None:
            lo, hi = bins[grp_idx], bins[grp_idx + 1]
            primary = (f"(ZAID={zaid}, MT={mt}), "
                       f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
        else:
            primary = f"(ZAID={zaid}, MT={mt}), G={grp_idx}"

        warnings.append(
            f"  [WARNING] Mean deviation in {primary}: |z|={abs(z):.2f}>{z_limit} "
            f"(mean={means_f[dim]:+.3e})"
        )

    # ---3) full-matrix reproduction --------------------------------------
    emp_cov = _empirical_cov(samples)
    frob_rel = (np.linalg.norm(emp_cov - cov_lin, ord='fro')
                / np.linalg.norm(cov_lin, ord='fro')) * 100.0
    
    result_msg = f"  Relative linear-cov error (Frobenius): {frob_rel:.2f}%"
    
    if logger:
        logger.info(result_msg)
    else:
        print(result_msg)

    # ---final report ------------------------------------------------------
    if warnings:
        warning_msg = "\n  Issues detected:"
        if logger:
            logger.info(warning_msg)
        else:
            print(warning_msg)
        for w in warnings:
            if logger:
                logger.info(w)
            else:
                print(w)
    else:
        ok_msg = "  All sample dimensions within thresholds."
        if logger:
            logger.info(ok_msg)
        else:
            print(ok_msg)
    
    end_msg = f"{separator}"
    if logger:
        logger.info(end_msg)
    else:
        print(end_msg)


def _diagnostics_samples_log(
    samples: np.ndarray,
    cov_log: np.ndarray,
    param_pairs: List[Tuple[int, int]],
    num_groups: int,
    bins: Optional[np.ndarray],
    z_limit: float = 3.0,
    verbose: bool = True,
) -> None:

    if not verbose:
        return

    # Try to get logger from ace_perturbation module
    try:
        from mcnpy.sampling.ace_perturbation import _get_logger
        logger = _get_logger()
    except:
        logger = None

    separator = "-" * 60
    log_msg = f"\n[SAMPLING] [LOG DIAGNOSTICS]\n{separator}"
    
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    warnings = []
    means_f = samples.mean(axis=0)
    p, n = cov_log.shape[0], samples.shape[0]

    for dim in range(p):
        var_log = cov_log[dim, dim]
        if var_log == 0.0:
            continue

        mean_th = 1.0
        se_f = np.sqrt((np.exp(var_log) - 1.0) / n)
        z = (means_f[dim] - mean_th) / se_f
        if abs(z) <= z_limit:
            continue

        pair_idx, grp_idx = divmod(dim, num_groups)
        zaid, mt = param_pairs[pair_idx]

        if bins is not None:
            lo, hi = bins[grp_idx], bins[grp_idx + 1]
            primary = (f"(ZAID={zaid}, MT={mt}), "
                       f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
        else:
            primary = f"(ZAID={zaid}, MT={mt}), G={grp_idx}"

        warnings.append(
            f"  [WARNING] Mean deviation in {primary}: |z|={abs(z):.2f}>{z_limit} "
            f"(mean={means_f[dim]:+.3e})"
        )

    emp_cov = _empirical_cov(np.log(samples))
    frob_rel = (np.linalg.norm(emp_cov - cov_log, ord='fro')
                / np.linalg.norm(cov_log, ord='fro')) * 100.0
    
    result_msg = f"  Relative log-cov error (Frobenius): {frob_rel:.2f}%"
    
    if logger:
        logger.info(result_msg)
    else:
        print(result_msg)

    if warnings:
        warning_msg = "\n  Issues detected:"
        if logger:
            logger.info(warning_msg)
        else:
            print(warning_msg)
        for w in warnings:
            if logger:
                logger.info(w)
            else:
                print(w)
    else:
        ok_msg = "  All sample dimensions within thresholds."
        if logger:
            logger.info(ok_msg)
        else:
            print(ok_msg)
    
    end_msg = f"{separator}"
    if logger:
        logger.info(end_msg)
    else:
        print(end_msg)


def _diagnostics_covariance(
    cov_lin: np.ndarray,
    param_pairs: List[Tuple[int, int]],
    num_groups: int,
    bins: Optional[np.ndarray],
    high_var_thr: float = 2.0,
    check_spd: bool = False,
    verbose: bool = True,
) -> None:
    if not verbose:
        return

    # Try to get logger from ace_perturbation module
    try:
        from mcnpy.sampling.ace_perturbation import _get_logger
        logger = _get_logger()
    except:
        logger = None

    separator = "-" * 60
    log_msg = f"\n[COVARIANCE] [DIAGNOSTICS]\n{separator}"
    
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    warnings = []
    diag = np.diag(cov_lin)
    p = cov_lin.shape[0]

    # ---1) large variances ------------------------------------------------
    for dim, var in enumerate(diag):
        if var <= high_var_thr:
            continue
        pair_idx, grp_idx = divmod(dim, num_groups)
        zaid, mt = param_pairs[pair_idx]

        if bins is not None:
            lo, hi = bins[grp_idx], bins[grp_idx + 1]
            primary = (f"(ZAID={zaid}, MT={mt}), "
                       f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
        else:
            primary = f"(ZAID={zaid}, MT={mt}), G={grp_idx}"

        warnings.append(
            f"  [WARNING] High variance in {primary}: σ²={var:.2f}>{high_var_thr}"
        )

    # ---2) SPD test (optional) --------------------------------------------
    if check_spd:
        lam_min = np.linalg.eigvalsh(cov_lin).min()
        if lam_min <= 0.0:
            warnings.append(
                f"  [WARNING] Covariance matrix is not SPD (λ_min={lam_min:.3e})"
            )

    # ---print outcome -----------------------------------------------------
    if warnings:
        warning_msg = "  Issues detected:"
        if logger:
            logger.info(warning_msg)
        else:
            print(warning_msg)
        for w in warnings:
            if logger:
                logger.info(w)
            else:
                print(w)
    else:
        ok_msg = "  No covariance issues detected."
        if logger:
            logger.info(ok_msg)
        else:
            print(ok_msg)
    
    end_msg = f"{separator}"
    if logger:
        logger.info(end_msg)
    else:
        print(end_msg)

class CovarianceFixError(Exception):
    """Exception raised when covariance matrix cannot be fixed to meet eigenvalue threshold."""
    pass

class SoftAutofixWarning(Exception):
    """Warning raised when soft autofix doesn't meet threshold but decomposition should still be attempted."""
    pass


# ----------------------------------------------------------------------
#  Main function
# ----------------------------------------------------------------------

def generate_samples(
    cov,
    n_samples: int,
    *,
    space: str = "log",          # "log" (default) or "linear"
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None,
    mt_numbers: Optional[Sequence[int]] = None,
    energy_grid: Optional[Sequence[float]] = None,
    autofix: Optional[str] = None,    # can be None/"soft"/"medium"/"hard"
    high_val_thresh: float = 5.0,
    accept_tol: float = -1.0e-4,  
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[List[int]], Optional[Dict[str, Any]]]: 
    """
    Draw multiplicative perturbation factors.

    Parameters
    ----------
    space : {"linear", "log"}
        * "linear": factors = 1 + X,   X ~ N(0, Σ_linear)
        * "log"   : factors = exp(Y),  Y ~ N(m, Σ_log) matched so that
                     Cov(factors) = Σ_linear and E[factors] = 1.
    autofix_level : {"soft", "medium", "hard"} or None/False
        If None or False, do not fix covariance. Otherwise, fix with the specified level.
    accept_tol : float
        Minimum eigenvalue threshold for accepting the covariance matrix
        
    Returns
    -------
    factors : np.ndarray
        Generated perturbation factors
    mt_numbers : Optional[List[int]]
        Final list of MT numbers (may be modified by autofix)
    fix_info : Optional[Dict[str, Any]]
        Information about covariance fixing, including removed correlations
    """
    # Try to get logger from ace_perturbation module
    try:
        from mcnpy.sampling.ace_perturbation import _get_logger
        logger = _get_logger()
    except:
        logger = None
    
    space  = space.lower()
    method = decomposition_method.lower()

    HIGH_VAR_LIN = 2.0
    HIGH_VAR_LOG = 2.0
    Z_LIMIT      = 3.0
    TRUNC_THRESHOLD = 0.999
    fix_info = None  # Initialize fix_info
    soft_autofix_failed = False  # Track if soft autofix failed to meet threshold

    # ------------------------------------------------------------------
    # 1. Fix the *linear* covariance if requested
    if autofix is not None:
        cov_fixed, fix_log = cov.fix_covariance(
            level=autofix, 
            high_val_thresh=high_val_thresh, 
            accept_tol=accept_tol,  
            verbose=verbose, 
            logger=logger
        )
        
        fix_info = fix_log  # Store the fix information
        
        # Check if covariance fixing was successful
        if not fix_log.get("converged", False):
            # For soft level, check if threshold was met
            if autofix.lower() == "soft" and not fix_log.get("soft_threshold_met", True):
                # Soft autofix didn't meet threshold, but we'll try decomposition anyway
                soft_autofix_failed = True
                min_eigenvalue = fix_log.get("min_eigenvalue", float('nan'))
                if logger:
                    logger.info(f"[COVARIANCE] [SOFT AUTOFIX] Threshold not met (λ_min={min_eigenvalue:.4e} < {accept_tol:.4e}), attempting decomposition anyway")
            else:
                # For medium/hard levels, this is a real failure
                min_eigenvalue = fix_log.get("min_eigenvalue", float('nan'))
                error_msg = (
                    f"Covariance matrix could not be fixed to meet eigenvalue threshold.\n"
                    f"  Final minimum eigenvalue: {min_eigenvalue:.4e}\n"
                    f"  Required threshold: {accept_tol:.4e}\n"
                    f"  Autofix level used: {autofix}\n"
                    f"  Suggestion: Try processing separately with a harder autofix level ('medium' or 'hard')"
                )
                
                # Only log to file if logger is available, let ace_perturbation handle console output
                if logger:
                    logger.info(f"[COVARIANCE] [ERROR] {error_msg}")
                else:
                    print(f"[ERROR] {error_msg}")
                    
                raise CovarianceFixError(f"min_eigenvalue={min_eigenvalue:.4e} below threshold={accept_tol:.4e}")
        
        # If we reach here and it's not a soft autofix failure, log success message
        if not soft_autofix_failed:
            final_eigenvalue = fix_log.get("min_eigenvalue", float('nan'))
            if verbose and logger:
                logger.info(f"[COVARIANCE] [SUCCESS] Matrix successfully fixed (final λ_min={final_eigenvalue:.4e})")
        
        cov_lin   = cov_fixed.covariance_matrix          # (p,p)             
        p         = cov_lin.shape[0]
        param_pairs = cov_fixed._get_param_pairs()
        num_groups  = cov_fixed.num_groups        

        if mt_numbers is not None and fix_log.get("removed_pairs"):
            # Extract removed MTs from the fix_log
            removed_pairs = fix_log.get("removed_pairs", [])
            removed_mts_from_autofix = set()
            
            # For "medium" level: look for removed block pairs and extract diagonal removals
            if autofix.lower() == "medium":
                for ra, rb in removed_pairs:
                    if ra == rb:  # Diagonal block removal means entire reaction removed
                        removed_mts_from_autofix.add(ra)
            
            # For "hard" level: look at removal_log for removed MTs
            elif autofix.lower() == "hard":
                removal_log = fix_log.get("removal_log", {})
                removed_mts_hard = removal_log.get("removed_mts", [])
                removed_mts_from_autofix.update(removed_mts_hard)
                
            if removed_mts_from_autofix:
                info_msg = f"  [INFO] MTs removed by fix_covariance: {sorted(removed_mts_from_autofix)}"
                if verbose:
                    if logger:
                        logger.info(info_msg)
                    else:
                        print(info_msg)
                        
                mt_numbers = [mt for mt in mt_numbers if mt not in removed_mts_from_autofix]
    else:
        cov_lin = cov.covariance_matrix
        p        = cov_lin.shape[0]
        param_pairs = cov._get_param_pairs()   
        num_groups  = cov.num_groups
        cov_fixed = cov.copy()   

    bins = np.asarray(energy_grid)

    # separate diagnostic for the input covariance
    _diagnostics_covariance(
        cov_lin, param_pairs, num_groups, bins,
        HIGH_VAR_LIN if space == "linear" else HIGH_VAR_LOG,
        check_spd=False,
        verbose=verbose
    )        

    # ------------------------------------------------------------------
    # 2. Decide which covariance to impose on the Gaussian draw
    if space == "linear":
        cov_mat = cov_lin
    elif space == "log":
        cov_mat = cov_fixed.log_covariance_matrix
    else:
        raise ValueError("space must be 'linear' or 'log'")

    # ------------------------------------------------------------------
    # 3. Draw uncorrelated N(0,1)
    Z = _uncorrelated(dim=p, n=n_samples,
                      method=sampling_method, seed=seed)

    # ------------------------------------------------------------------
    # 4. Impose correlation (any of the four decompositions)
    try:
        if method == "pca":
            Y = _pca_decomposition_sampling(
                cov_mat, n_samples, sampling_method, seed,
                TRUNC_THRESHOLD, verbose, space, logger
            )
        else:
            if method == "cholesky":
                try:
                    L = cov_fixed.cholesky_decomposition(space=space, verbose=verbose, logger=logger)
                    # Verify Cholesky decomposition quality
                    if verbose:
                        verify_cholesky_decomposition(
                            original_matrix=cov_mat,
                            L=L,
                            space=space,
                            verbose=verbose,
                            logger=logger
                        )
                except Exception as chol_err:
                    if verbose and logger:
                        logger.info(f"[DECOMPOSITION] [QUALITY] Cholesky verification skipped: decomposition failed ({str(chol_err)})")
                    elif verbose:
                        print(f"[DECOMPOSITION] [QUALITY] Cholesky verification skipped: decomposition failed ({str(chol_err)})")
                    raise  # Re-raise the original exception
            elif method == "eigen":
                eigvals, eigvecs = cov_fixed.eigen_decomposition(space=space, clip_negatives=True, verbose=verbose, logger=logger)
                # Verify eigendecomposition quality
                if verbose:
                    verify_eigen_decomposition(
                        original_matrix=cov_mat,
                        eigvals=eigvals,
                        eigvecs=eigvecs,
                        space=space,
                        verbose=verbose,
                        logger=logger
                    )
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            elif method == "svd":
                U, S, Vt = cov_fixed.svd_decomposition(space=space, clip_negatives=True, verbose=verbose, logger=logger)
                # Verify SVD quality
                if verbose:
                    verify_svd_decomposition(
                        original_matrix=cov_mat,
                        U=U,
                        S=S,
                        Vt=Vt,
                        space=space,
                        verbose=verbose,
                        logger=logger
                    )
                L = U @ np.diag(np.sqrt(S))
            else:
                raise ValueError(
                    "decomposition_method must be 'pca', 'cholesky', 'eigen' or 'svd'"
                )
            Y = Z @ L.T
    except Exception as e:
        # If decomposition fails and we had a soft autofix failure, raise special exception
        if soft_autofix_failed:
            min_eigenvalue = fix_info.get("min_eigenvalue", float('nan'))
            error_msg = f"Soft autofix failed to meet threshold (λ_min={min_eigenvalue:.4e} < {accept_tol:.4e}) and decomposition failed: {str(e)}"
            raise SoftAutofixWarning(error_msg)
        else:
            # Re-raise original exception for other cases
            raise e

    # ------------------------------------------------------------------
    # 5. Convert to multiplicative factors
    if space == "linear":
        factors = Y + 1.0

    else:  # log (moment-matched)
        m = -0.5 * np.diag(cov_mat)        # shift so mean → 1
        factors = np.exp(Y + m)            # strictly positive
    
    # Convert perturbation factors to float32 for memory efficiency
    factors = factors.astype(np.float32)

    # ------------------------------------------------------------------
    # 6. Diagnostics of the *samples* (same code you had, now renamed)

    if space == "linear":
        _diagnostics_samples_linear(
            factors, cov_mat,
            param_pairs, num_groups,
            bins,
            Z_LIMIT, verbose
        )
    else:
        _diagnostics_samples_log(
            factors, cov_mat,
            param_pairs, num_groups,
            bins,
            Z_LIMIT, verbose
        )
    
    # Update fix_info to include soft autofix status
    if soft_autofix_failed and fix_info:
        fix_info["soft_autofix_failed"] = True
        fix_info["decomposition_succeeded"] = True
        
    return factors, mt_numbers, fix_info  # Return fix_info as third value


# ----------------------------------------------------------------------
#  ENDF-specific sampling function for MF34 angular covariance data
# ----------------------------------------------------------------------

def generate_endf_samples(
    mf34_cov,
    n_samples: int,
    *,
    space: str = "log",          # "log" (default) or "linear"
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None,
    mt_numbers: Optional[Sequence[int]] = None,
    energy_grid: Optional[Sequence[float]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[List[int]]]: 
    """
    Draw multiplicative perturbation factors for ENDF angular distribution data (MF34).
    
    This function is specifically designed for MF34CovMat objects which contain
    covariance data for angular distributions with (isotope, reaction, legendre) triplets.
    
    Note: Unlike the standard generate_samples function, this does not include
    autofix capabilities since angular distribution covariance matrices have 
    different physical constraints than cross-section covariance matrices.

    Parameters
    ----------
    mf34_cov : MF34CovMat
        MF34 angular distribution covariance matrix object
    n_samples : int
        Number of perturbation factor samples to generate
    space : {"linear", "log"}
        * "linear": factors = 1 + X,   X ~ N(0, Σ_linear)
        * "log"   : factors = exp(Y),  Y ~ N(m, Σ_log) matched so that
                     Cov(factors) = Σ_linear and E[factors] = 1.
    decomposition_method : str
        Method for matrix decomposition: "svd", "cholesky", "eigen", or "pca"
    sampling_method : str
        Method for generating uncorrelated samples: "sobol", "random", "halton"
    seed : Optional[int]
        Random seed for reproducibility
    mt_numbers : Optional[Sequence[int]]
        List of MT numbers to include (for reference only, not used for filtering)
    energy_grid : Optional[Sequence[float]]
        Energy grid for diagnostics
    verbose : bool
        Whether to print diagnostic information
        
    Returns
    -------
    factors : np.ndarray
        Generated perturbation factors of shape (n_samples, n_parameters)
        where n_parameters corresponds to the flattened covariance matrix
    mt_numbers : Optional[List[int]]
        Unchanged list of MT numbers (returned for consistency with generate_samples)
    """
    # Try to get logger from endf_perturbation module
    try:
        from mcnpy.sampling.endf_perturbation import _get_logger
        logger = _get_logger()
    except:
        logger = None
    
    space = space.lower()
    method = decomposition_method.lower()

    HIGH_VAR_LIN = 2.0
    HIGH_VAR_LOG = 2.0
    Z_LIMIT = 3.0
    TRUNC_THRESHOLD = 0.999

    # ------------------------------------------------------------------
    # 1. Get covariance matrix and parameter information
    cov_lin = mf34_cov.covariance_matrix  # (p,p)
    p = cov_lin.shape[0]
    
    # For MF34 data, create parameter triplets and simplified param_pairs for diagnostics
    param_triplets = mf34_cov._get_param_triplets()  # List of (isotope, mt, legendre) triplets
    
    # Create param_pairs for diagnostics by converting triplets to (mt, legendre) pairs
    param_pairs = [(mt, l) for (iso, mt, l) in param_triplets]
    param_pairs = sorted(list(set(param_pairs)))
    
    # Calculate num_groups as maximum matrix size
    num_groups = max(matrix.shape[0] for matrix in mf34_cov.matrices) if mf34_cov.matrices else 0

    bins = np.asarray(energy_grid) if energy_grid is not None else None

    # Separate diagnostic for the input covariance
    if verbose:
        _diagnostics_endf_covariance(
            cov_lin, param_triplets, num_groups, bins,
            HIGH_VAR_LIN if space == "linear" else HIGH_VAR_LOG,
            verbose=verbose, logger=logger
        )

    # ------------------------------------------------------------------
    # 2. Decide which covariance to impose on the Gaussian draw
    if space == "linear":
        cov_mat = cov_lin
    elif space == "log":
        cov_mat = mf34_cov.log_covariance_matrix
    else:
        raise ValueError("space must be 'linear' or 'log'")

    # ------------------------------------------------------------------
    # 3. Draw uncorrelated N(0,1)
    Z = _uncorrelated(dim=p, n=n_samples,
                      method=sampling_method, seed=seed)

    # ------------------------------------------------------------------
    # 4. Impose correlation using matrix decomposition
    if method == "pca":
        Y = _pca_decomposition_sampling(
            cov_mat, n_samples, sampling_method, seed,
            TRUNC_THRESHOLD, verbose, space, logger
        )
    else:
        if method == "cholesky":
            try:
                L = mf34_cov.cholesky_decomposition(space=space, verbose=verbose, logger=logger)
                # Verify Cholesky decomposition quality
                if verbose:
                    verify_cholesky_decomposition(
                        original_matrix=cov_mat,
                        L=L,
                        space=space,
                        verbose=verbose,
                        logger=logger
                    )
            except Exception as chol_err:
                if verbose and logger:
                    logger.info(f"[DECOMPOSITION] [QUALITY] Cholesky verification skipped: decomposition failed ({str(chol_err)})")
                elif verbose:
                    print(f"[DECOMPOSITION] [QUALITY] Cholesky verification skipped: decomposition failed ({str(chol_err)})")
                raise  # Re-raise the original exception
        elif method == "eigen":
            eigvals, eigvecs = mf34_cov.eigen_decomposition(space=space, clip_negatives=True, verbose=verbose, logger=logger)
            # Verify eigendecomposition quality
            if verbose:
                verify_eigen_decomposition(
                    original_matrix=cov_mat,
                    eigvals=eigvals,
                    eigvecs=eigvecs,
                    space=space,
                    verbose=verbose,
                    logger=logger
                )
            L = eigvecs @ np.diag(np.sqrt(eigvals))
        elif method == "svd":
            U, S, Vt = mf34_cov.svd_decomposition(space=space, clip_negatives=True, verbose=verbose, logger=logger)
            # Verify SVD quality
            if verbose:
                verify_svd_decomposition(
                    original_matrix=cov_mat,
                    U=U,
                    S=S,
                    Vt=Vt,
                    space=space,
                    verbose=verbose,
                    logger=logger
                )
            L = U @ np.diag(np.sqrt(S))
        else:
            raise ValueError(
                "decomposition_method must be 'pca', 'cholesky', 'eigen' or 'svd'"
            )
        Y = Z @ L.T

    # ------------------------------------------------------------------
    # 5. Convert to multiplicative factors
    if space == "linear":
        factors = Y + 1.0
    else:  # log (moment-matched)
        m = -0.5 * np.diag(cov_mat)        # shift so mean → 1
        factors = np.exp(Y + m)            # strictly positive
    
    # Convert perturbation factors to float32 for memory efficiency
    factors = factors.astype(np.float32)

    # ------------------------------------------------------------------
    # 6. Diagnostics of the samples
    if verbose:
        if space == "linear":
            _diagnostics_endf_samples_linear(
                factors, cov_mat, param_triplets, num_groups,
                bins, Z_LIMIT, verbose=verbose, logger=logger
            )
        else:
            _diagnostics_endf_samples_log(
                factors, cov_mat, param_triplets, num_groups,
                bins, Z_LIMIT, verbose=verbose, logger=logger
            )
        
    return factors, mt_numbers


def _diagnostics_endf_covariance(
    cov_lin: np.ndarray,
    param_triplets: List[Tuple[int, int, int]],
    num_groups: int,
    bins: Optional[np.ndarray],
    high_var_thr: float = 2.0,
    verbose: bool = True,
    logger = None,
) -> None:
    """Diagnostic function for ENDF covariance matrices."""
    if not verbose:
        return

    separator = "-" * 60
    log_msg = f"\n[ENDF COVARIANCE] [DIAGNOSTICS]\n{separator}"
    
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    warnings = []
    diag = np.diag(cov_lin)
    p = cov_lin.shape[0]

    # Check for large variances
    for dim, var in enumerate(diag):
        if var <= high_var_thr:
            continue
        
        # For MF34, calculate triplet and group indices differently
        triplet_idx = dim // num_groups
        grp_idx = dim % num_groups
        
        if triplet_idx < len(param_triplets):
            isotope, mt, legendre = param_triplets[triplet_idx]
            
            if bins is not None and grp_idx < len(bins) - 1:
                lo, hi = bins[grp_idx], bins[grp_idx + 1]
                primary = (f"(ISO={isotope}, MT={mt}, L={legendre}), "
                          f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
            else:
                primary = f"(ISO={isotope}, MT={mt}, L={legendre}), G={grp_idx}"
        else:
            primary = f"Parameter {dim}"

        warnings.append(
            f"  [WARNING] High variance in {primary}: σ²={var:.2f}>{high_var_thr}"
        )

    # Check if matrix is positive semi-definite
    try:
        lam_min = np.linalg.eigvalsh(cov_lin).min()
        if lam_min <= 0.0:
            warnings.append(
                f"  [WARNING] Covariance matrix is not SPD (λ_min={lam_min:.3e})"
            )
    except Exception as e:
        warnings.append(f"  [WARNING] Could not check eigenvalues: {e}")

    # Print outcome
    if warnings:
        warning_msg = "  Issues detected:"
        if logger:
            logger.info(warning_msg)
        else:
            print(warning_msg)
        for w in warnings:
            if logger:
                logger.info(w)
            else:
                print(w)
    else:
        ok_msg = "  No covariance issues detected."
        if logger:
            logger.info(ok_msg)
        else:
            print(ok_msg)
    
    end_msg = f"{separator}"
    if logger:
        logger.info(end_msg)
    else:
        print(end_msg)


def _diagnostics_endf_samples_linear(
    samples: np.ndarray,
    cov_lin: np.ndarray,
    param_triplets: List[Tuple[int, int, int]],
    num_groups: int,
    bins: Optional[np.ndarray],
    z_limit: float = 3.0,
    verbose: bool = True,
    logger = None,
) -> None:
    """Linear space sample diagnostics for ENDF data."""
    if not verbose:
        return

    separator = "-" * 60
    log_msg = f"\n[ENDF SAMPLING] [LINEAR DIAGNOSTICS]\n{separator}"
    
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    warnings = []
    means_f = samples.mean(axis=0)
    p, n = cov_lin.shape[0], samples.shape[0]

    # NOTE: For ENDF Legendre coefficients, negative factors are physically valid
    # so we skip the negative factor check that is used for cross-sections

    # Check means significantly off 1
    for dim in range(p):
        var_lin = cov_lin[dim, dim]
        if var_lin == 0.0:
            continue
        
        se_f = np.sqrt(var_lin / n)
        z = (means_f[dim] - 1.0) / se_f
        if abs(z) <= z_limit:
            continue

        triplet_idx = dim // num_groups
        grp_idx = dim % num_groups
        
        if triplet_idx < len(param_triplets):
            isotope, mt, legendre = param_triplets[triplet_idx]
            
            if bins is not None and grp_idx < len(bins) - 1:
                lo, hi = bins[grp_idx], bins[grp_idx + 1]
                primary = (f"(ISO={isotope}, MT={mt}, L={legendre}), "
                          f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
            else:
                primary = f"(ISO={isotope}, MT={mt}, L={legendre}), G={grp_idx}"
        else:
            primary = f"Parameter {dim}"

        warnings.append(
            f"  [WARNING] Mean deviation in {primary}: |z|={abs(z):.2f}>{z_limit} "
            f"(mean={means_f[dim]:+.3e})"
        )

    # Full-matrix reproduction check
    emp_cov = _empirical_cov(samples)
    frob_rel = (np.linalg.norm(emp_cov - cov_lin, ord='fro')
                / np.linalg.norm(cov_lin, ord='fro')) * 100.0
    
    result_msg = f"  Relative linear-cov error (Frobenius): {frob_rel:.2f}%"
    
    if logger:
        logger.info(result_msg)
    else:
        print(result_msg)

    # Final report
    if warnings:
        warning_msg = "\n  Issues detected:"
        if logger:
            logger.info(warning_msg)
        else:
            print(warning_msg)
        for w in warnings:
            if logger:
                logger.info(w)
            else:
                print(w)
    else:
        ok_msg = "  All sample dimensions within thresholds."
        if logger:
            logger.info(ok_msg)
        else:
            print(ok_msg)
    
    end_msg = f"{separator}"
    if logger:
        logger.info(end_msg)
    else:
        print(end_msg)


def _diagnostics_endf_samples_log(
    samples: np.ndarray,
    cov_log: np.ndarray,
    param_triplets: List[Tuple[int, int, int]],
    num_groups: int,
    bins: Optional[np.ndarray],
    z_limit: float = 3.0,
    verbose: bool = True,
    logger = None,
) -> None:
    """Log space sample diagnostics for ENDF data."""
    if not verbose:
        return

    separator = "-" * 60
    log_msg = f"\n[ENDF SAMPLING] [LOG DIAGNOSTICS]\n{separator}"
    
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    warnings = []
    means_f = samples.mean(axis=0)
    p, n = cov_log.shape[0], samples.shape[0]

    for dim in range(p):
        var_log = cov_log[dim, dim]
        if var_log == 0.0:
            continue

        mean_th = 1.0
        se_f = np.sqrt((np.exp(var_log) - 1.0) / n)
        z = (means_f[dim] - mean_th) / se_f
        if abs(z) <= z_limit:
            continue

        triplet_idx = dim // num_groups
        grp_idx = dim % num_groups
        
        if triplet_idx < len(param_triplets):
            isotope, mt, legendre = param_triplets[triplet_idx]
            
            if bins is not None and grp_idx < len(bins) - 1:
                lo, hi = bins[grp_idx], bins[grp_idx + 1]
                primary = (f"(ISO={isotope}, MT={mt}, L={legendre}), "
                          f"G={grp_idx} [{lo:.2e},{hi:.2e}]")
            else:
                primary = f"(ISO={isotope}, MT={mt}, L={legendre}), G={grp_idx}"
        else:
            primary = f"Parameter {dim}"

        warnings.append(
            f"  [WARNING] Mean deviation in {primary}: |z|={abs(z):.2f}>{z_limit} "
            f"(mean={means_f[dim]:+.3e})"
        )

    emp_cov = _empirical_cov(np.log(samples))
    frob_rel = (np.linalg.norm(emp_cov - cov_log, ord='fro')
                / np.linalg.norm(cov_log, ord='fro')) * 100.0
    
    result_msg = f"  Relative log-cov error (Frobenius): {frob_rel:.2f}%"
    
    if logger:
        logger.info(result_msg)
    else:
        print(result_msg)

    if warnings:
        warning_msg = "\n  Issues detected:"
        if logger:
            logger.info(warning_msg)
        else:
            print(warning_msg)
        for w in warnings:
            if logger:
                logger.info(w)
            else:
                print(w)
    else:
        ok_msg = "  All sample dimensions within thresholds."
        if logger:
            logger.info(ok_msg)
        else:
            print(ok_msg)
    
    end_msg = f"{separator}"
    if logger:
        logger.info(end_msg)
    else:
        print(end_msg)