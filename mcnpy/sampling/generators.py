import numpy as np
from scipy import stats
from scipy.stats import qmc
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
from typing import List, Sequence, Optional, Tuple, Dict, Any

from mcnpy.cov.covmat import CovMat

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
) -> np.ndarray:
    """
    Draw N(0, Σ) samples via PCA truncation while capturing
    `trunc_threshold` of the variance.
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

    # 6) Build transform L and draw uncorrelated Z
    Vred    = eigvecs[:, :k]
    sqrt_D  = np.sqrt(eigvals[:k])
    L       = Vred @ np.diag(sqrt_D)
    Z       = _uncorrelated(k, n_samples, sampling_method, seed)

    return Z @ L.T                      # shape (n_samples, p)

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

    separator = "-" * 60
    print(f"\n[SAMPLING] [LINEAR DIAGNOSTICS]")
    print(f"{separator}")

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
    print(f"  Relative linear-cov error (Frobenius): {frob_rel:.2f}%")

    # ---final report ------------------------------------------------------
    if warnings:
        print("\n  Issues detected:")
        for w in warnings:
            print(w)
    else:
        print("  All sample dimensions within thresholds.")
    print(f"{separator}")


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

    separator = "-" * 60
    print(f"\n[SAMPLING] [LOG DIAGNOSTICS]")
    print(f"{separator}")

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
    print(f"  Relative log-cov error (Frobenius): {frob_rel:.2f}%")

    if warnings:
        print("\n  Issues detected:")
        for w in warnings:
            print(w)
    else:
        print("  All sample dimensions within thresholds.")
    print(f"{separator}")


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

    separator = "-" * 60
    print(f"\n[COVARIANCE] [DIAGNOSTICS]")
    print(f"{separator}")

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
        print("  Issues detected:")
        for w in warnings:
            print(w)
    else:
        print("  No covariance issues detected.")
    print(f"{separator}")


# ----------------------------------------------------------------------
# Unified sampler
# ----------------------------------------------------------------------
def generate_samples(
    cov,
    n_samples: int,
    *,
    space: str = "linear",          # "linear" or "log"
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None,
    mt_numbers: Optional[Sequence[int]] = None,
    energy_grid: Optional[Sequence[float]] = None,
    autofix: Optional[str] = None,    # can be None/"soft"/"medium"/"hard"
    high_val_thresh: float = 5.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[List[int]]]:
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
    """
    space  = space.lower()
    method = decomposition_method.lower()

    HIGH_VAR_LIN = 2.0
    HIGH_VAR_LOG = 2.0
    Z_LIMIT      = 3.0
    TRUNC_THRESHOLD = 0.999
    
    if verbose:
        print(f"[SAMPLING] [GENERATING] {n_samples} samples in {space} space using {method}/{sampling_method}")

    # ------------------------------------------------------------------
    # 1. Fix the *linear* covariance if requested
    if autofix is not None:
        cov_fixed, fix_log = cov.fix_covariance(
            level=autofix, high_val_thresh=high_val_thresh, verbose=verbose)
        cov_lin   = cov_fixed.covariance_matrix          # (p,p)             
        p         = cov_lin.shape[0]
        param_pairs = cov_fixed._get_param_pairs()
        num_groups  = cov_fixed.num_groups        

        if mt_numbers is not None and fix_log.get("removed_mts"):
            removed = fix_log["removed_mts"]
            if verbose:
                print(f"  [INFO] MTs removed by fix_covariance: {removed}")
                
                # Print more specific information about removals if available
                if "high_variance_mts" in fix_log:
                    print(f"  [INFO] MTs removed due to high variance: {fix_log['high_variance_mts']}")
                if "negative_eigenvalue_mts" in fix_log:
                    print(f"  [INFO] MTs removed due to negative eigenvalues: {fix_log['negative_eigenvalue_mts']}")
                    
            mt_numbers = [mt for mt in mt_numbers if mt not in removed]
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
    if method == "pca":
        Y = _pca_decomposition_sampling(
            cov_mat, n_samples, sampling_method, seed,
            TRUNC_THRESHOLD, verbose
        )
    else:
        if method == "cholesky":
            L = cov_fixed.cholesky_decomposition(space=space, verbose=verbose)
        elif method == "eigen":
            eigvals, eigvecs = cov_fixed.eigen_decomposition(space=space, clip_negatives=True, verbose=verbose)
            L = eigvecs @ np.diag(np.sqrt(eigvals))
        elif method == "svd":
            U, S, _ = cov_fixed.svd_decomposition(space=space, clip_negatives=True, verbose=verbose)
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
        
    return factors, mt_numbers