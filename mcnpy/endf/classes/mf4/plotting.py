import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple, Dict, Any
from ...._plot_settings import setup_plot_style, format_axes


def plot_angular_distribution(
    mf4_mixed,
    energies: Optional[Union[float, Tuple[float, float], List[float]]] = None,
    energy_indices: Optional[Union[int, Tuple[int, int], List[int]]] = None,
    data_type: str = 'legendre',
    cosine_range: Tuple[float, float] = (-1.0, 1.0),
    n_points: int = 181,
    style: str = 'default',
    figsize: Tuple[float, float] = (8,5),
    legend_loc: str = 'best',
    **kwargs
) -> plt.Figure:
    """
    Plot angular distributions from MF4MTMixed data.
    
    Parameters
    ----------
    mf4_mixed : MF4MTMixed
        Mixed angular distribution data object
    energies : float, tuple of float, or list of float, optional
        - float: Find energy bin containing this value
        - tuple: Plot all energy bins in range [tuple[0], tuple[1]]
        - list: Specific energy values to plot (legacy support)
        If None, uses energy_indices
    energy_indices : int, tuple of int, or list of int, optional
        - int: Plot specific energy bin index
        - tuple: Plot energy bins from index tuple[0] to tuple[1] (inclusive)
        - list: Specific energy indices (legacy support)
        If None, plots first few energies
    data_type : str
        Type of data to plot: 'legendre', 'tabulated', or 'both'
    cosine_range : tuple
        Range of cosine values to evaluate (mu_min, mu_max)
    n_points : int
        Number of points for Legendre evaluation (tabulated uses all data points)
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    # Setup plot style
    plot_kwargs = setup_plot_style(style=style, figsize=figsize, **kwargs)
    fig = plot_kwargs['_fig']
    ax = plot_kwargs['ax']
    colors = plot_kwargs['_colors']
    
    # Determine which energies to plot based on data_type
    plot_energies = _determine_plot_energies_by_type(mf4_mixed, energies, energy_indices, data_type)
    
    # Plot each selected energy
    for i, (energy, energy_type, energy_range) in enumerate(plot_energies):
        color = colors[i % len(colors)]
        
        if energy_type == 'legendre' and data_type in ('legendre', 'both'):
            # Create label with energy range for Legendre (no suffix)
            if energy_range:
                label = f"E=[{energy:.3e}, {energy_range:.3e}) eV"
            else:
                label = f"E={energy:.3e} eV"
            
            _plot_legendre_distribution(
                ax, mf4_mixed, energy, cosine_range, n_points, 
                color, label
            )
        
        if energy_type == 'tabulated' and data_type in ('tabulated', 'both'):
            _plot_tabulated_distribution(
                ax, mf4_mixed, energy, color, f"E={energy:.3e} eV (Tab)"
            )
    
    # Format axes
    ax = format_axes(
        ax, style=style,
        x_label="Cosine of scattering angle (μ)",
        y_label="Probability density f(μ,E)",
        title=f"Angular Distribution - MT{mf4_mixed.number}",
        legend_loc=legend_loc
    )
    
    ax.set_xlim(cosine_range)
    
    return fig


def plot_legendre_coefficients(
    mf4_mixed,
    energies: Optional[Union[float, Tuple[float, float], List[float]]] = None,
    energy_indices: Optional[Union[int, Tuple[int, int], List[int]]] = None,
    max_order: Optional[int] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 5),  # Reduced from (12, 8)
    legend_loc: str = 'best',
    **kwargs
) -> plt.Figure:
    """
    Plot Legendre coefficients as a function of polynomial order.
    
    Parameters
    ----------
    mf4_mixed : MF4MTMixed
        Mixed angular distribution data object
    energies : float, tuple of float, or list of float, optional
        - float: Find energy bin containing this value
        - tuple: Plot all energy bins in range [tuple[0], tuple[1]]
        - list: Specific energy values (legacy support)
    energy_indices : int, tuple of int, or list of int, optional
        - int: Plot specific energy bin index
        - tuple: Plot energy bins from index tuple[0] to tuple[1] (inclusive)
        - list: Specific energy indices (legacy support)
    max_order : int, optional
        Maximum polynomial order to display
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    # Setup plot style
    plot_kwargs = setup_plot_style(style=style, figsize=figsize, **kwargs)
    fig = plot_kwargs['_fig']
    ax = plot_kwargs['ax']
    colors = plot_kwargs['_colors']
    
    # Get Legendre energies and coefficients
    legendre_energies = mf4_mixed.legendre_energies
    legendre_coeffs = mf4_mixed.legendre_coefficients
    
    if not legendre_energies:
        ax.text(0.5, 0.5, 'No Legendre coefficient data available', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Determine which energies to plot - only consider Legendre energies
    plot_indices = _determine_legendre_plot_indices(legendre_energies, energies, energy_indices)
    
    # Plot coefficients for each energy
    for i, energy_idx in enumerate(plot_indices):
        if energy_idx >= len(legendre_coeffs):
            continue
            
        coeffs = legendre_coeffs[energy_idx]
        energy = legendre_energies[energy_idx]
        
        if max_order is not None:
            coeffs = coeffs[:max_order+1]
        
        orders = range(len(coeffs))
        color = colors[i % len(colors)]
        
        ax.plot(orders, coeffs, 'o-', color=color, 
                label=f"E={energy:.3e} eV", linewidth=2, markersize=6)
    
    # Format axes
    ax = format_axes(
        ax, style=style,
        x_label="Legendre polynomial order (l)",
        y_label="Coefficient a_l",
        title=f"Legendre Coefficients - MT{mf4_mixed.number}",
        legend_loc=legend_loc
    )
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    return fig


def plot_energy_dependent_coefficients(
    mf4_mixed,
    orders: Optional[Union[int, List[int]]] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 5),  # Reduced from (10, 6)
    legend_loc: str = 'best',
    **kwargs
) -> plt.Figure:
    """
    Plot specific Legendre coefficients as a function of energy.
    
    Parameters
    ----------
    mf4_mixed : MF4MTMixed
        Mixed angular distribution data object
    orders : int or list of int, optional
        Polynomial orders to plot. If None, plots first few orders
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    # Setup plot style
    plot_kwargs = setup_plot_style(style=style, figsize=figsize, **kwargs)
    fig = plot_kwargs['_fig']
    ax = plot_kwargs['ax']
    colors = plot_kwargs['_colors']
    
    # Get data
    energies = mf4_mixed.legendre_energies
    coeffs_list = mf4_mixed.legendre_coefficients
    
    if not energies or not coeffs_list:
        ax.text(0.5, 0.5, 'No Legendre coefficient data available', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Determine which orders to plot
    if orders is None:
        max_available = max(len(coeffs) for coeffs in coeffs_list)
        orders = list(range(min(4, max_available)))  # Plot first 4 orders by default
    elif not isinstance(orders, list):
        orders = [orders]
    
    # Plot each order
    for i, order in enumerate(orders):
        color = colors[i % len(colors)]
        
        # Extract coefficient values for this order across all energies
        coeff_values = []
        energy_values = []
        
        for j, coeffs in enumerate(coeffs_list):
            if order < len(coeffs):
                coeff_values.append(coeffs[order])
                energy_values.append(energies[j])
        
        if energy_values:
            ax.plot(energy_values, coeff_values, 'o-', color=color,
                    label=f"a_{order}", linewidth=2, markersize=4)
    
    # Format axes
    ax = format_axes(
        ax, style=style, use_log_scale=True, is_energy_axis=True,
        x_label="Energy (eV)",
        y_label="Coefficient value",
        title=f"Energy-dependent Legendre Coefficients - MT{mf4_mixed.number}",
        legend_loc=legend_loc
    )
    
    return fig


def compare_angular_representations(
    mf4_mixed,
    energy: float,
    cosine_range: Tuple[float, float] = (-1.0, 1.0),
    n_points: int = 181,
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 5),  # Reduced from (10, 6)
    legend_loc: str = 'best',
    **kwargs
) -> plt.Figure:
    """
    Compare Legendre and tabulated representations at a specific energy.
    
    Parameters
    ----------
    mf4_mixed : MF4MTMixed
        Mixed angular distribution data object
    energy : float
        Energy value to find appropriate bin for comparison
    cosine_range : tuple
        Range of cosine values
    n_points : int
        Number of points for evaluation
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    # Setup plot style
    plot_kwargs = setup_plot_style(style=style, figsize=figsize, **kwargs)
    fig = plot_kwargs['_fig']
    ax = plot_kwargs['ax']
    colors = plot_kwargs['_colors']
    
    # Find appropriate energy bins for the given energy
    legendre_energy = _find_energy_bin(mf4_mixed.legendre_energies, energy)
    tabulated_energy = _find_energy_bin(mf4_mixed.tabulated_energies, energy)
    
    has_legendre = legendre_energy is not None
    has_tabulated = tabulated_energy is not None
    
    if not (has_legendre or has_tabulated):
        ax.text(0.5, 0.5, f'Energy {energy:.3e} eV not found in any data', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Plot Legendre representation
    if has_legendre:
        _plot_legendre_distribution(
            ax, mf4_mixed, legendre_energy, cosine_range, n_points, 
            colors[0], f"E={legendre_energy:.3e} eV"
        )
    
    # Plot tabulated representation
    if has_tabulated:
        _plot_tabulated_distribution(
            ax, mf4_mixed, tabulated_energy, colors[1], f"E={tabulated_energy:.3e} eV (Tab)"
        )
    
    # Format axes
    ax = format_axes(
        ax, style=style,
        x_label="Cosine of scattering angle (μ)",
        y_label="Probability density f(μ,E)",
        title=f"Angular Distribution Comparison for E={energy:.3e} eV - MT{mf4_mixed.number}",
        legend_loc=legend_loc
    )
    
    ax.set_xlim(cosine_range)
    
    return fig


def _find_energy_bin(energy_grid: List[float], target_energy: float) -> Optional[float]:
    """
    Find the energy bin that contains the target energy.
    
    For energy bins [E_i, E_{i+1}), find E_i such that E_i <= target_energy < E_{i+1}.
    For the last bin, include the upper boundary.
    
    Parameters
    ----------
    energy_grid : List[float]
        Sorted list of energy bin boundaries
    target_energy : float
        Target energy to find bin for
        
    Returns
    -------
    float or None
        Energy bin boundary (E_i) that contains target_energy, or None if not found
    """
    if not energy_grid or target_energy < energy_grid[0]:
        return None
    
    # Check if target is beyond the last energy
    if target_energy > energy_grid[-1]:
        return None
    
    # Find the appropriate bin
    for i in range(len(energy_grid) - 1):
        if energy_grid[i] <= target_energy < energy_grid[i + 1]:
            return energy_grid[i]
    
    # Special case: if target equals the last energy, return the second-to-last bin
    if len(energy_grid) >= 2 and target_energy == energy_grid[-1]:
        return energy_grid[-2]
    
    return None


def _determine_legendre_plot_indices(
    legendre_energies: List[float], 
    energies: Optional[Union[float, Tuple[float, float], List[float]]], 
    energy_indices: Optional[Union[int, Tuple[int, int], List[int]]]
) -> List[int]:
    """
    Determine which Legendre energy indices to plot.
    
    Returns list of indices into the legendre_energies list.
    """
    if energies is not None:
        if isinstance(energies, tuple) and len(energies) == 2:
            # Plot all energies in range [energies[0], energies[1]]
            e_min, e_max = energies
            indices = []
            for i, energy in enumerate(legendre_energies):
                if e_min <= energy <= e_max:
                    indices.append(i)
            return indices
        elif isinstance(energies, (int, float)):
            # Find energy bin containing this value
            target_energy = _find_energy_bin(legendre_energies, float(energies))
            if target_energy is not None:
                try:
                    return [legendre_energies.index(target_energy)]
                except ValueError:
                    return []
            return []
        elif isinstance(energies, list):
            # Legacy support - exact energy matches
            indices = []
            for energy in energies:
                try:
                    idx = legendre_energies.index(energy)
                    indices.append(idx)
                except ValueError:
                    continue
            return indices
    
    elif energy_indices is not None:
        if isinstance(energy_indices, tuple) and len(energy_indices) == 2:
            # Plot energy bins from index tuple[0] to tuple[1] (inclusive)
            start, end = energy_indices
            return list(range(start, min(end + 1, len(legendre_energies))))
        elif isinstance(energy_indices, int):
            # Plot specific energy bin index
            if 0 <= energy_indices < len(legendre_energies):
                return [energy_indices]
            return []
        elif isinstance(energy_indices, list):
            # Legacy support - specific indices
            return [idx for idx in energy_indices if 0 <= idx < len(legendre_energies)]
    
    # Default: plot first 5 energies
    return list(range(min(5, len(legendre_energies))))


def _determine_plot_energies_by_type(mf4_mixed, energies, energy_indices, data_type) -> List[Tuple[float, str, Optional[float]]]:
    """
    Determine which energies to plot based on data type.
    
    Returns list of (energy, type, next_energy) tuples where type is 'legendre' or 'tabulated'
    and next_energy is the upper bound for Legendre energy bins (None for tabulated).
    """
    plot_energies = []
    
    if energies is not None:
        if isinstance(energies, tuple) and len(energies) == 2:
            # Plot all energies in range [energies[0], energies[1]]
            e_min, e_max = energies
            
            if data_type in ('legendre', 'both'):
                # Add Legendre energies in range with upper bounds
                for i, energy in enumerate(mf4_mixed.legendre_energies):
                    if e_min <= energy <= e_max:
                        next_energy = (mf4_mixed.legendre_energies[i + 1] 
                                     if i + 1 < len(mf4_mixed.legendre_energies) 
                                     else None)
                        plot_energies.append((energy, 'legendre', next_energy))
            
            if data_type in ('tabulated', 'both'):
                # Add tabulated energies in range
                for energy in mf4_mixed.tabulated_energies:
                    if e_min <= energy <= e_max:
                        plot_energies.append((energy, 'tabulated', None))
                    
        elif isinstance(energies, (int, float)):
            # Find energy bins containing this value
            target_energy = float(energies)
            
            if data_type in ('legendre', 'both'):
                legendre_energy = _find_energy_bin(mf4_mixed.legendre_energies, target_energy)
                if legendre_energy is not None:
                    # Find the upper bound for this energy bin
                    idx = mf4_mixed.legendre_energies.index(legendre_energy)
                    next_energy = (mf4_mixed.legendre_energies[idx + 1] 
                                 if idx + 1 < len(mf4_mixed.legendre_energies) 
                                 else None)
                    plot_energies.append((legendre_energy, 'legendre', next_energy))
                
            if data_type in ('tabulated', 'both'):
                tabulated_energy = _find_energy_bin(mf4_mixed.tabulated_energies, target_energy)
                if tabulated_energy is not None:
                    plot_energies.append((tabulated_energy, 'tabulated', None))
                
        elif isinstance(energies, list):
            # Legacy support - exact energy matches
            if data_type in ('legendre', 'both'):
                for energy in energies:
                    if energy in mf4_mixed.legendre_energies:
                        idx = mf4_mixed.legendre_energies.index(energy)
                        next_energy = (mf4_mixed.legendre_energies[idx + 1] 
                                     if idx + 1 < len(mf4_mixed.legendre_energies) 
                                     else None)
                        plot_energies.append((energy, 'legendre', next_energy))
            
            if data_type in ('tabulated', 'both'):
                for energy in energies:
                    if energy in mf4_mixed.tabulated_energies:
                        plot_energies.append((energy, 'tabulated', None))
    
    elif energy_indices is not None:
        if isinstance(energy_indices, tuple) and len(energy_indices) == 2:
            # Plot energy bins from index tuple[0] to tuple[1] (inclusive)
            start, end = energy_indices
            
            if data_type in ('legendre', 'both'):
                # Add Legendre energies in index range
                for idx in range(start, min(end + 1, len(mf4_mixed.legendre_energies))):
                    energy = mf4_mixed.legendre_energies[idx]
                    next_energy = (mf4_mixed.legendre_energies[idx + 1] 
                                 if idx + 1 < len(mf4_mixed.legendre_energies) 
                                 else None)
                    plot_energies.append((energy, 'legendre', next_energy))
                
            if data_type in ('tabulated', 'both'):
                # Add tabulated energies in index range
                for idx in range(start, min(end + 1, len(mf4_mixed.tabulated_energies))):
                    energy = mf4_mixed.tabulated_energies[idx]
                    plot_energies.append((energy, 'tabulated', None))
                
        elif isinstance(energy_indices, int):
            # Plot specific energy bin indices
            if data_type in ('legendre', 'both') and energy_indices < len(mf4_mixed.legendre_energies):
                energy = mf4_mixed.legendre_energies[energy_indices]
                next_energy = (mf4_mixed.legendre_energies[energy_indices + 1] 
                             if energy_indices + 1 < len(mf4_mixed.legendre_energies) 
                             else None)
                plot_energies.append((energy, 'legendre', next_energy))
            
            if data_type in ('tabulated', 'both') and energy_indices < len(mf4_mixed.tabulated_energies):
                energy = mf4_mixed.tabulated_energies[energy_indices]
                plot_energies.append((energy, 'tabulated', None))
                
        elif isinstance(energy_indices, list):
            # Legacy support - specific indices
            if data_type in ('legendre', 'both'):
                for idx in energy_indices:
                    if idx < len(mf4_mixed.legendre_energies):
                        energy = mf4_mixed.legendre_energies[idx]
                        next_energy = (mf4_mixed.legendre_energies[idx + 1] 
                                     if idx + 1 < len(mf4_mixed.legendre_energies) 
                                     else None)
                        plot_energies.append((energy, 'legendre', next_energy))
            
            if data_type in ('tabulated', 'both'):
                for idx in energy_indices:
                    if idx < len(mf4_mixed.tabulated_energies):
                        energy = mf4_mixed.tabulated_energies[idx]
                        plot_energies.append((energy, 'tabulated', None))
    
    else:
        # Default behavior based on data_type
        if data_type == 'legendre':
            # Plot first 3 Legendre energies with upper bounds
            for i, energy in enumerate(mf4_mixed.legendre_energies[:3]):
                next_energy = (mf4_mixed.legendre_energies[i + 1] 
                             if i + 1 < len(mf4_mixed.legendre_energies) 
                             else None)
                plot_energies.append((energy, 'legendre', next_energy))
        elif data_type == 'tabulated':
            # Plot first 3 tabulated energies
            for energy in mf4_mixed.tabulated_energies[:3]:
                plot_energies.append((energy, 'tabulated', None))
        elif data_type == 'both':
            # Plot first few energies from both types
            for i, energy in enumerate(mf4_mixed.legendre_energies[:3]):
                next_energy = (mf4_mixed.legendre_energies[i + 1] 
                             if i + 1 < len(mf4_mixed.legendre_energies) 
                             else None)
                plot_energies.append((energy, 'legendre', next_energy))
            for energy in mf4_mixed.tabulated_energies[:3]:
                plot_energies.append((energy, 'tabulated', None))
    
    return plot_energies


def _plot_legendre_distribution(ax, mf4_mixed, energy, cosine_range, n_points, color, label):
    """Plot angular distribution from Legendre coefficients."""
    coeffs = mf4_mixed.get_legendre_coefficients(energy)
    if not coeffs:
        return
    
    # Generate cosine values
    mu = np.linspace(cosine_range[0], cosine_range[1], n_points)
    
    # Evaluate Legendre expansion: f(μ) = Σ aₗ × (2l+1)/2 × Pₗ(μ)
    # Note: a_0 = 1 is implicit and not included in coeffs list
    # coeffs[0] = a_1, coeffs[1] = a_2, etc.
    
    # Pre-compute all Legendre polynomials up to required order
    # We need polynomials from P_0 to P_(len(coeffs))
    max_order = len(coeffs) + 1  # +1 to include P_0
    legendre_polys = np.zeros((max_order, n_points))
    
    # P₀(μ) = 1
    legendre_polys[0, :] = 1.0
    
    # P₁(μ) = μ
    if max_order > 1:
        legendre_polys[1, :] = mu
    
    # Use recurrence relation: (l+1)Pₗ₊₁(μ) = (2l+1)μPₗ(μ) - lPₗ₋₁(μ)
    for l in range(2, max_order):
        legendre_polys[l, :] = ((2*l - 1) * mu * legendre_polys[l-1, :] - 
                                (l - 1) * legendre_polys[l-2, :]) / l
    
    # Start with a_0 = 1, P_0(μ) term
    prob_density = (2*0 + 1) / 2 * legendre_polys[0, :]  # a_0 * (1/2) * P_0(μ) = 1/2
    
    # Add contributions from a_1, a_2, ... (coeffs[0], coeffs[1], ...)
    for i, coeff in enumerate(coeffs):
        l = i + 1  # coeffs[0] corresponds to a_1 (l=1), coeffs[1] to a_2 (l=2), etc.
        prob_density += coeff * (2*l + 1) / 2 * legendre_polys[l, :]
    
    ax.plot(mu, prob_density, '-', color=color, label=label, linewidth=2)


def _plot_tabulated_distribution(ax, mf4_mixed, energy, color, label):
    """Plot tabulated angular distribution using all data points with interpolation."""
    cosines, probabilities = mf4_mixed.get_tabulated_distribution(energy)
    if not cosines or not probabilities:
        return
    
    # Get the interpolation scheme for this energy
    try:
        energy_idx = mf4_mixed.tabulated_energies.index(energy)
        ang_interp = (mf4_mixed.angular_interpolation[energy_idx] 
                     if energy_idx < len(mf4_mixed.angular_interpolation) 
                     else [])
    except (ValueError, IndexError):
        ang_interp = []
    
    # Plot the original data points
    ax.plot(cosines, probabilities, 'o', color=color, markersize=6, 
            markerfacecolor='white', markeredgecolor=color, markeredgewidth=2,
            label=label)
    
    # If we have interpolation information, create interpolated curve
    if ang_interp and len(cosines) > 1:
        # Create a dense grid for smooth interpolation
        mu_interp = np.linspace(min(cosines), max(cosines), 200)
        prob_interp = _interpolate_angular_data(cosines, probabilities, mu_interp, ang_interp)
        
        # Plot the interpolated curve
        ax.plot(mu_interp, prob_interp, '-', color=color, linewidth=2, alpha=0.7)
    else:
        # Simple linear interpolation between points if no scheme specified
        ax.plot(cosines, probabilities, '-', color=color, linewidth=2, alpha=0.7)


def _interpolate_angular_data(cosines, probabilities, mu_interp, ang_interp):
    """
    Interpolate angular distribution data using ENDF interpolation schemes.
    
    Parameters
    ----------
    cosines : list
        Original cosine values
    probabilities : list
        Original probability values
    mu_interp : array
        Cosine values to interpolate to
    ang_interp : list
        List of (NBT, INT) interpolation scheme pairs
        
    Returns
    -------
    array
        Interpolated probability values
    """
    if not ang_interp:
        # Default to linear interpolation
        return np.interp(mu_interp, cosines, probabilities)
    
    prob_interp = np.zeros_like(mu_interp)
    
    # Process interpolation regions
    start_idx = 0
    for nbt, int_code in ang_interp:
        end_idx = min(nbt, len(cosines))
        
        # Extract data for this region
        region_cosines = cosines[start_idx:end_idx]
        region_probs = probabilities[start_idx:end_idx]
        
        if len(region_cosines) < 2:
            start_idx = end_idx
            continue
        
        # Find interpolation points in this region
        mask = (mu_interp >= region_cosines[0]) & (mu_interp <= region_cosines[-1])
        if not np.any(mask):
            start_idx = end_idx
            continue
            
        mu_region = mu_interp[mask]
        
        # Apply interpolation scheme based on INT code
        if int_code == 1:  # Histogram (constant in each bin)
            # For histogram, use previous value
            prob_region = np.zeros_like(mu_region)
            for i, mu in enumerate(mu_region):
                # Find the bin this mu falls into
                bin_idx = np.searchsorted(region_cosines, mu, side='right') - 1
                bin_idx = max(0, min(bin_idx, len(region_probs) - 1))
                prob_region[i] = region_probs[bin_idx]
                
        elif int_code == 2:  # Linear-linear interpolation
            prob_region = np.interp(mu_region, region_cosines, region_probs)
            
        elif int_code == 3:  # Linear-log interpolation
            # y is linear, x is log - but cosines can be negative, so use linear
            prob_region = np.interp(mu_region, region_cosines, region_probs)
            
        elif int_code == 4:  # Log-linear interpolation
            # x is linear, y is log
            region_probs_pos = np.maximum(region_probs, 1e-10)  # Avoid log(0)
            log_probs = np.log(region_probs_pos)
            log_prob_region = np.interp(mu_region, region_cosines, log_probs)
            prob_region = np.exp(log_prob_region)
            
        elif int_code == 5:  # Log-log interpolation
            # Both x and y are log - but cosines can be negative, so use linear-log
            region_probs_pos = np.maximum(region_probs, 1e-10)  # Avoid log(0)
            log_probs = np.log(region_probs_pos)
            log_prob_region = np.interp(mu_region, region_cosines, log_probs)
            prob_region = np.exp(log_prob_region)
            
        else:
            # Default to linear interpolation for unknown schemes
            prob_region = np.interp(mu_region, region_cosines, region_probs)
        
        prob_interp[mask] = prob_region
        start_idx = end_idx
    
    return prob_interp
