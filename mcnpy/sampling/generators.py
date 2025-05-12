import numpy as np
from scipy import stats
from scipy.stats import qmc
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
from typing import List, Sequence, Optional, Set

from mcnpy.sampling.decomposition import (
    svd_decomposition,
    cholesky_decomposition,
    eigen_decomposition,
)


def _empirical_cov(X: np.ndarray) -> np.ndarray:

    X_centered = X - X.mean(axis=0, keepdims=True)
    return (X_centered.T @ X_centered) / (X.shape[0] - 1)


def _decompose(cov: np.ndarray, method: str):

    key = method.lower()
    if key == "svd":
        return svd_decomposition(cov)
    if key == "cholesky":
        return cholesky_decomposition(cov)
    if key == "eigen":
        return eigen_decomposition(cov)
    raise ValueError("decomposition_method must be 'svd', 'cholesky' or 'eigen'")


def _uncorrelated(
    dim: int,
    n: int,
    method: str,
    seed: Optional[int],
    fast_forward: int = 0,
) -> np.ndarray:

    m = method.lower()
    if m == "random":
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, dim))

    if m == "lhs":
        U = qmc.LatinHypercube(d=dim, seed=seed).random(n)
        return stats.norm.ppf(U)

    if m == "sobol":
        sob = qmc.Sobol(d=dim, scramble=True, seed=seed)
        if fast_forward:
            sob.fast_forward(fast_forward)
        U = sob.random(n)
        return stats.norm.ppf(U)

    raise ValueError("sampling_method must be 'random', 'lhs' or 'sobol'")


def mt_blocks(
    cov_nd: np.ndarray,
    mt_numbers: Sequence[int],
    n_groups: int,
    tol: float = 1e-12,
) -> List[np.ndarray]:

    n_mt = len(mt_numbers)
    first_col = {k: k * n_groups for k in range(n_mt)}  # MT → first column index

    # Build adjacency matrix indicating interaction between maturities --------
    adj = np.zeros((n_mt, n_mt), dtype=bool)
    for i in range(n_mt):
        for j in range(i + 1, n_mt):
            sli = slice(first_col[i], first_col[i] + n_groups)
            slj = slice(first_col[j], first_col[j] + n_groups)
            if np.any(np.abs(cov_nd[sli, slj]) > tol):
                adj[i, j] = adj[j, i] = True

    n_comp, lbl = cs.connected_components(sp.csr_matrix(adj))
    blocks = [
        np.concatenate(
            [
                np.arange(first_col[m], first_col[m] + n_groups)
                for m in np.where(lbl == k)[0]
            ]
        )
        for k in range(n_comp)
    ]
    return blocks


def _related_mts(gdim: int, cov: np.ndarray, mt_numbers: Sequence[int],
                 n_groups: int, tol: float = 0.0) -> List[int]:
    """Return sorted list of MT codes whose covariance with *gdim* is non‑zero."""
    mask = np.abs(cov[gdim]) > tol
    idxs = np.where(mask)[0]
    mt_idx = (idxs // n_groups).tolist()
    rel = {mt_numbers[k] for k in mt_idx if k < len(mt_numbers)}
    return sorted(rel)


def generate_samples_blocked(
    cov: np.ndarray,
    blocks: List[np.ndarray],
    n_samples: int,
    *,
    decomposition_method: str = "svd",
    sampling_method: str = "sobol",
    seed: Optional[int] = None,
    mt_numbers: Optional[Sequence[int]] = None,
    energy_grid: Optional[Sequence[float]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Generate correlated samples and print concise diagnostics.

    Definitions
    ------------
    * **Mean**  : average of each *pre‑shift* correlated dimension (expected ≈0)
    * **SE**    : standard error = sqrt(variance / n_samples)
    * **z‑score** : mean / SE (units of SE)

    Flag rules
    ----------
    * High variance   : σ² > 1e4
    * High bias       : |z| > 3
    """

    HIGH_VAR = 1e4
    Z_LIMIT = 3.0

    p = cov.shape[0]
    samples = np.empty((n_samples, p))
    warnings: list[str] = []

    # Validate block coverage
    all_idx = np.concatenate(blocks)
    assert len(np.unique(all_idx)) == p, "Blocks wrongly defined (overlap or gaps)"

    # Uncorrelated base points
    if sampling_method.lower() == "sobol":
        sob = qmc.Sobol(d=p, scramble=True, seed=seed)
        Z_global = stats.norm.ppf(sob.random(n_samples))
    else:
        Z_global = None

    # Energy grid helpers
    if energy_grid is not None:
        bins = np.asarray(energy_grid)
        n_groups = len(bins) - 1
    else:
        bins = None
        n_groups = None

    # ------------------------------------------------------------------
    for idx in blocks:
        # Raw samples for this block
        if sampling_method.lower() == "sobol":
            z_b = Z_global[:, idx]
        else:
            seed_block = (seed or 0) + hash(tuple(idx)) % (2**31 - 1)
            z_b = _uncorrelated(len(idx), n_samples, sampling_method, seed_block)

        # Apply covariance
        L_b = _decompose(cov[np.ix_(idx, idx)], decomposition_method)
        x_b = z_b @ L_b.T

        # Stats per dimension
        means = x_b.mean(axis=0)
        for local_j, gdim in enumerate(idx):
            var = cov[gdim, gdim]
            if var == 0:
                continue  # deterministic → skip
            mu = means[local_j]
            se = np.sqrt(var / n_samples)
            z = mu / se
            high_var = var > HIGH_VAR
            high_z = abs(z) > Z_LIMIT
            if not (high_var or high_z):
                continue

            # Build primary label (MT, group, energy interval)
            if mt_numbers is not None and bins is not None:
                mt_idx = gdim // n_groups
                grp_idx = gdim % n_groups
                e_lo, e_hi = bins[grp_idx], bins[grp_idx + 1]
                primary_label = (f"MT={mt_numbers[mt_idx]}, G={grp_idx} "
                                  f"[{e_lo:.2e},{e_hi:.2e}]")
                related_mts = _related_mts(gdim, cov, mt_numbers, n_groups)
                # remove itself
                if mt_numbers[mt_idx] in related_mts:
                    related_mts.remove(mt_numbers[mt_idx])
                related_str = ", ".join(
                    [f"MT{i+1}={m}" for i, m in enumerate(related_mts)])
                if related_str:
                    primary_label = f"{primary_label}, {related_str}"
            else:
                primary_label = f"Dim={gdim}"

            # Compose warning message
            if high_var and high_z:
                msg = (f"σ²={var:.3e}>1e4 AND |z|={abs(z):.2f}>3 "
                       f"(mean={mu:+.3e}, SE={se:.3e})")
            elif high_var:
                msg = (f"σ²={var:.3e}>1e4 (mean={mu:+.3e}, z={z:.2f})")
            else:
                msg = (f"|z|={abs(z):.2f}>3 (mean={mu:+.3e}, SE={se:.3e})")

            warnings.append(f"WARNING: {primary_label}: {msg}")

        samples[:, idx] = x_b

    # Shift to mean 1
    samples_mean_1 = samples + 1.0

    # Empirical covariance check -------------------------------------------
    emp_cov = _empirical_cov(samples_mean_1)
    frob_rel = (np.linalg.norm(emp_cov - cov, ord="fro") /
                np.linalg.norm(cov, ord="fro") * 100.0)

    # Output summary
    if verbose:
        if warnings:
            print("Diagnostics – issues detected (see definitions below).")
            print("Mean ≈ 0, SE = sqrt(σ²/n), z = mean/SE.")
            print("\n".join(warnings))
        else:
            print("\nSampling completed successfully: all dimensions within thresholds.")
        print(f"\nRelative covariance error (Frobenius): {frob_rel:.2f}%")

    return samples_mean_1