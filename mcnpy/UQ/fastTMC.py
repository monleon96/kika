"""
Fast Total Monte Carlo (FTMC) uncertainty propagation analysis.

This module provides functions for decomposing uncertainty in Monte Carlo calculations
into statistical and nuclear data components using the Fast TMC methodology.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Optional, List


def fastTMC(
    y_val: Optional[Union[np.ndarray, list]] = None,
    y_sigma: Optional[Union[np.ndarray, list]] = None,
    results_df: Optional[pd.DataFrame] = None,
    uncertainties_df: Optional[pd.DataFrame] = None,
    columns: Optional[Union[List[str], List[int], str, int]] = None,
    n_samples: Optional[int] = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> Union[Dict, pd.DataFrame]:
    """
    Perform Fast Total Monte Carlo decomposition with bootstrap confidence intervals.
    
    This unified function can handle both single-tally analysis (using arrays/lists) 
    and multi-tally analysis (using DataFrames). Only one input mode should be used.
    
    Parameters
    ----------
    y_val : array-like, optional
        N sample results for single tally analysis. Cannot be used with DataFrames.
    y_sigma : array-like, optional
        N relative 1-σ uncertainties for single tally analysis (as fractions).
    results_df : pd.DataFrame, optional
        DataFrame with results for each tally (columns) and sample (rows).
    uncertainties_df : pd.DataFrame, optional
        DataFrame with uncertainties for each tally (columns) and sample (rows).
    columns : list or str or int, optional
        Specific columns to analyze. Can be column names, indices, or mix.
        If None, analyzes all columns in DataFrame mode.
    n_samples : int, optional
        Number of samples to use from DataFrames. If None, uses all available.
    n_bootstrap : int, optional
        Number of bootstrap replicates for confidence intervals. Default is 1000.
    ci_level : float, optional
        Desired confidence level (e.g., 0.95 for 95% CI). Default is 0.95.
    random_seed : int, optional
        Seed for reproducible bootstrap resampling. Default is None.
    verbose : bool, optional
        Whether to print detailed results. Default is True.
    
    Returns
    -------
    dict or pd.DataFrame
        For single tally: Dict with uncertainty decomposition results.
        For multiple tallies: DataFrame with summary results.
    
    Raises
    ------
    ValueError
        If both single-tally and DataFrame inputs are provided, or if neither is provided.
    
    Examples
    --------
    # Single tally analysis
    >>> result = fastTMC(y_val=[1.05, 0.98, 1.02], y_sigma=[0.02, 0.025, 0.018])
    
    # Multi-tally analysis
    >>> summary = fastTMC(results_df=df_results, uncertainties_df=df_uncertainties)
    
    # Specific columns only
    >>> summary = fastTMC(results_df=df_results, uncertainties_df=df_uncertainties, 
    ...                   columns=['F4_flux', 'F6_heating'])
    """
    # Input validation
    single_mode = y_val is not None or y_sigma is not None
    dataframe_mode = results_df is not None or uncertainties_df is not None
    
    if single_mode and dataframe_mode:
        raise ValueError(
            "Cannot use both single-tally mode (y_val, y_sigma) and DataFrame mode "
            "(results_df, uncertainties_df) simultaneously."
        )
    
    if not single_mode and not dataframe_mode:
        raise ValueError(
            "Must provide either single-tally inputs (y_val, y_sigma) or "
            "DataFrame inputs (results_df, uncertainties_df)."
        )
    
    if single_mode:
        # Single tally analysis
        if y_val is None or y_sigma is None:
            raise ValueError("Both y_val and y_sigma must be provided for single-tally analysis.")
        
        return _single_tally_analysis(
            y_val, y_sigma, n_bootstrap, ci_level, random_seed, verbose
        )
    
    else:
        # DataFrame analysis
        if results_df is None or uncertainties_df is None:
            raise ValueError("Both results_df and uncertainties_df must be provided for DataFrame analysis.")
        
        return _dataframe_analysis(
            results_df, uncertainties_df, columns, n_samples, 
            n_bootstrap, ci_level, random_seed, verbose
        )


def create_summary_table(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a summary table from Fast TMC analysis results with separate CI columns.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of FTMC results from fastTMC function.
    
    Returns
    -------
    pd.DataFrame
        Summary table with key uncertainty metrics and separate CI columns.
    """
    summary_data = []
    
    for tally_name, result in results_dict.items():
        if result['mean_tally'] == 0:
            continue
        
        summary_data.append({
            'Mean_Tally': result['mean_tally'],
            'Total_Unc_%': result['percent_unc_observed'],
            'Statistical_%': result['percent_unc_statistical'],
            'Nuclear_Data_%': result['percent_unc_nuclear_data'],
            'Nuclear_Data_CI_Lower_%': result['ci']['percent_unc_nuclear_data'][0],
            'Nuclear_Data_CI_Upper_%': result['ci']['percent_unc_nuclear_data'][1],
            'Mean_Tally_CI_Lower': result['ci']['mean_tally'][0],
            'Mean_Tally_CI_Upper': result['ci']['mean_tally'][1]
        })
    
    summary_df = pd.DataFrame(summary_data, index=list(results_dict.keys()))
    summary_df.index.name = 'Tally'
    
    return summary_df


def _single_tally_analysis(
    y_val: Union[np.ndarray, list],
    y_sigma: Union[np.ndarray, list],
    n_bootstrap: int,
    ci_level: float,
    random_seed: Optional[int],
    verbose: bool
) -> Dict:
    """Core Fast TMC analysis for a single tally."""
    # Convert to numpy arrays
    y = np.asarray(y_val, dtype=float)
    sig = np.asarray(y_sigma, dtype=float).copy()
    N = y.size
    
    # Validate input
    if N != sig.size:
        raise ValueError("y_val and y_sigma must have the same length")
    
    if sig.mean() > 1.0:
        raise ValueError(
            "Relative uncertainties should be in fraction form (e.g., 0.05 for 5%). "
            "If your uncertainties are in percent, divide by 100."
        )
    
    # Core FTMC calculation
    mean_tally = y.mean()
    abs_err = sig * y
    mean_var_within = np.mean(abs_err**2)
    var_observed = np.var(y, ddof=1)
    var_nuclear = max(var_observed - mean_var_within, 0.0)
    
    # Standard deviations
    std_obs = np.sqrt(var_observed)
    std_stat = np.sqrt(mean_var_within)
    std_nuc = np.sqrt(var_nuclear)
    
    # Convert to percentages
    pct_obs = std_obs / mean_tally * 100.0
    pct_stat = std_stat / mean_tally * 100.0
    pct_nuc = std_nuc / mean_tally * 100.0
    
    if verbose:
        _print_summary(mean_tally, pct_obs, pct_stat, pct_nuc)
    
    # Bootstrap analysis
    bootstrap_results = _bootstrap_analysis(
        y, sig, n_bootstrap, random_seed, verbose
    )
    
    # Compute confidence intervals
    ci = _compute_confidence_intervals(bootstrap_results, ci_level, verbose)
    
    return {
        'mean_tally': mean_tally,
        'percent_unc_observed': pct_obs,
        'percent_unc_statistical': pct_stat,
        'percent_unc_nuclear_data': pct_nuc,
        'bootstrap': bootstrap_results,
        'ci': ci
    }


def _dataframe_analysis(
    results_df: pd.DataFrame,
    uncertainties_df: pd.DataFrame,
    columns: Optional[Union[List[str], List[int], str, int]],
    n_samples: Optional[int],
    n_bootstrap: int,
    ci_level: float,
    random_seed: Optional[int],
    verbose: bool
) -> pd.DataFrame:
    """Core Fast TMC analysis for multiple tallies from DataFrames."""
    # Validate inputs
    if not all(col in uncertainties_df.columns for col in results_df.columns):
        raise ValueError("Column mismatch between results and uncertainties DataFrames")
    
    # Determine columns to analyze
    if columns is None:
        columns_to_analyze = list(results_df.columns)
    else:
        if isinstance(columns, (str, int)):
            columns = [columns]
        
        columns_to_analyze = []
        for col in columns:
            if isinstance(col, int):
                # Convert index to column name
                if col < len(results_df.columns):
                    columns_to_analyze.append(results_df.columns[col])
                else:
                    raise ValueError(f"Column index {col} out of range")
            else:
                # Assume it's a column name
                if col in results_df.columns:
                    columns_to_analyze.append(col)
                else:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Determine sample size
    if n_samples is None:
        n_samples = len(results_df)
    else:
        n_samples = min(n_samples, len(results_df))
    
    all_results = {}
    
    if verbose:
        print(f"\nPerforming Fast TMC analysis for {len(columns_to_analyze)} tallies "
              f"using {n_samples} samples...")
        print("=" * 70)
    
    for tally_name in columns_to_analyze:
        if verbose:
            print(f"\n--- Analyzing {tally_name} ---")
        
        # Extract data
        y_res = results_df[tally_name].iloc[:n_samples].values
        y_sig = uncertainties_df[tally_name].iloc[:n_samples].values
        
        # Skip if mean is zero
        if np.mean(y_res) == 0:
            if verbose:
                print(f"Skipping {tally_name}: mean tally is zero")
            continue
        
        # Perform analysis
        try:
            result = _single_tally_analysis(
                y_res, y_sig, n_bootstrap, ci_level, random_seed, verbose
            )
            all_results[tally_name] = result
        except Exception as e:
            if verbose:
                print(f"Error analyzing {tally_name}: {e}")
            continue
    
    # Return as summary DataFrame
    return create_summary_table(all_results)


def _print_summary(mean_tally: float, pct_obs: float, pct_stat: float, pct_nuc: float):
    """Print formatted summary of FTMC results."""
    print(f"Mean tally                   : {mean_tally:.5e}")
    print(f"Total 1-σ uncertainty        : {pct_obs:.2f}%")
    print(f"  • Statistical component    : {pct_stat:.2f}%")
    print(f"  • Nuclear-data component   : {pct_nuc:.2f}%")


def _bootstrap_analysis(
    y: np.ndarray,
    sig: np.ndarray,
    n_bootstrap: int,
    random_seed: Optional[int],
    verbose: bool
) -> Dict[str, np.ndarray]:
    """Perform bootstrap analysis for confidence intervals."""
    rng = np.random.default_rng(random_seed)
    N = len(y)
    
    bs_mean = np.empty(n_bootstrap)
    bs_nuc = np.empty(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Bootstrap resample
        idx = rng.integers(0, N, N)
        y_bs = y[idx]
        sig_bs = sig[idx]
        
        # Mean for this bootstrap sample
        bs_mean[i] = y_bs.mean()
        
        # Nuclear data uncertainty for this bootstrap sample
        abs_err_bs = sig_bs * y_bs
        var_within_bs = np.mean(abs_err_bs**2)
        var_obs_bs = np.var(y_bs, ddof=1)
        var_nuc_bs = max(var_obs_bs - var_within_bs, 0.0)
        std_nuc_bs = np.sqrt(var_nuc_bs)
        bs_nuc[i] = std_nuc_bs / y_bs.mean() * 100.0
    
    return {
        'mean_tally': bs_mean,
        'percent_unc_nuclear_data': bs_nuc
    }


def _compute_confidence_intervals(
    bootstrap_results: Dict[str, np.ndarray],
    ci_level: float,
    verbose: bool
) -> Dict[str, Tuple[float, float]]:
    """Compute percentile confidence intervals from bootstrap results."""
    alpha = 1 - ci_level
    lo, hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    
    ci = {
        'mean_tally': tuple(np.percentile(bootstrap_results['mean_tally'], [lo, hi])),
        'percent_unc_nuclear_data': tuple(
            np.percentile(bootstrap_results['percent_unc_nuclear_data'], [lo, hi])
        ),
    }
    
    if verbose:
        print(f"\nBootstrap {int(ci_level*100)}% CIs:")
        print(f"  • Mean tally                : {ci['mean_tally'][0]:.5e} – {ci['mean_tally'][1]:.5e}")
        print(f"  • Nuclear-data unc.         : {ci['percent_unc_nuclear_data'][0]:.2f}% – "
              f"{ci['percent_unc_nuclear_data'][1]:.2f}%")
    
    return ci
