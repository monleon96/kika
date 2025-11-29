# Script in development, not yet ready. Some fixes need to be done to visualizations.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
from typing import List, Optional, Union, Tuple, Dict, Any
from ...._plot_settings import setup_plot_style, format_axes

def plot_angular_distribution_3d(
    mf4_mixed,
    energies: Optional[Union[Tuple[float, float], List[float]]] = None,
    energy_indices: Optional[Union[Tuple[int, int], List[int]]] = None,
    data_type: str = 'legendre',
    cosine_range: Tuple[float, float] = (-1.0, 1.0),
    n_cosine_points: int = 101,
    n_energy_points: Optional[int] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (12, 8),
    colormap: str = 'viridis',
    view_angle: Tuple[float, float] = (30, 45),
    surface_alpha: float = 0.8,
    show_wireframe: bool = False,
    show_contours: bool = True,
    log_energy_scale: bool = True,
    interactive: bool = None,
    **kwargs
) -> plt.Figure:
    """
    Create a 3D surface plot of angular distribution probability density.
    
    Shows probability density f(μ,E) as a function of cosine angle (μ) and energy (E).
    
    Parameters
    ----------
    mf4_mixed : MF4MTMixed
        Mixed angular distribution data object
    energies : tuple of float or list of float, optional
        - tuple: Plot all energy bins in range [tuple[0], tuple[1]]
        - list: Specific energy values to plot
        If None, uses energy_indices or defaults to all available energies
    energy_indices : tuple of int or list of int, optional
        - tuple: Plot energy bins from index tuple[0] to tuple[1] (inclusive)
        - list: Specific energy indices
        If None, uses all available energies based on data_type
    data_type : str
        Type of data to plot: 'legendre', 'tabulated', or 'both'
    cosine_range : tuple
        Range of cosine values (mu_min, mu_max)
    n_cosine_points : int
        Number of cosine points for surface mesh
    n_energy_points : int, optional
        Number of energy points to interpolate between (if None, uses all available)
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    colormap : str
        Colormap for the surface ('viridis', 'plasma', 'coolwarm', etc.)
    view_angle : tuple
        3D view angles (elevation, azimuth)
    surface_alpha : float
        Transparency of the surface (0-1)
    show_wireframe : bool
        Whether to overlay wireframe on surface
    show_contours : bool
        Whether to show contour lines on bottom plane
    log_energy_scale : bool
        Whether to use logarithmic scale for energy axis
    interactive : bool, optional
        Whether to enable interactive matplotlib features (auto-detected if None)
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    plt.Figure
        The created figure with 3D plot
    """
    # Detect interactive mode properly
    from ...._plot_settings import _is_notebook, _detect_interactive_backend
    
    is_interactive = interactive
    if is_interactive is None:
        is_interactive = _is_notebook() and _detect_interactive_backend()
    
    # Use setup_plot_style but disable constrained_layout for 3D
    plot_kwargs = setup_plot_style(
        style=style, 
        figsize=figsize, 
        interactive=is_interactive,
        projection='3d',  # This tells setup_plot_style it's a 3D plot
        **kwargs
    )
    
    # Extract the created figure and axis
    fig = plot_kwargs['_fig']
    ax = plot_kwargs['ax']
    
    # Determine which energies to use
    plot_energies = _determine_3d_plot_energies(mf4_mixed, energies, energy_indices, data_type)
    
    if not plot_energies:
        ax.text(0.5, 0.5, 0.5, 'No energy data available for plotting', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Create cosine mesh
    mu_points = np.linspace(cosine_range[0], cosine_range[1], n_cosine_points)
    
    # Interpolate energies if needed
    if n_energy_points is not None and len(plot_energies) > n_energy_points:
        # Logarithmically space energy points for better visualization
        e_min, e_max = min(plot_energies), max(plot_energies)
        if log_energy_scale and e_min > 0:
            energy_points = np.logspace(np.log10(e_min), np.log10(e_max), n_energy_points)
        else:
            energy_points = np.linspace(e_min, e_max, n_energy_points)
    else:
        energy_points = np.array(plot_energies)
    
    # Create energy-cosine meshgrid
    E_mesh, Mu_mesh = np.meshgrid(energy_points, mu_points)
    
    # Calculate probability density for each point
    Z_mesh = np.zeros_like(E_mesh)
    
    for i, energy in enumerate(energy_points):
        # Find the closest available energy or interpolate
        prob_density = _evaluate_angular_distribution_at_energy(
            mf4_mixed, energy, mu_points, data_type
        )
        Z_mesh[:, i] = prob_density
    
    # Handle logarithmic energy scale for plotting
    if log_energy_scale and np.all(E_mesh > 0):
        E_plot = np.log10(E_mesh)
        energy_label = "log₁₀(Energy) [log₁₀(eV)]"
    else:
        E_plot = E_mesh
        energy_label = "Energy (eV)"
        log_energy_scale = False
    
    # Create the 3D surface plot
    surface = ax.plot_surface(
        Mu_mesh, E_plot, Z_mesh,
        cmap=colormap, alpha=surface_alpha,
        linewidth=0, antialiased=True,
        edgecolor='none',
        rasterized=not is_interactive  # Only rasterize for non-interactive backends
    )
    
    # Add wireframe overlay if requested
    if show_wireframe:
        ax.plot_wireframe(
            Mu_mesh, E_plot, Z_mesh,
            colors='black', alpha=0.3, linewidth=0.5
        )
    
    # Add contour lines on the bottom plane if requested
    if show_contours:
        # Project contours onto the bottom plane
        z_bottom = ax.get_zlim()[0]
        contour = ax.contour(
            Mu_mesh, E_plot, Z_mesh,
            levels=10, colors='gray', alpha=0.5,
            linestyles='solid', linewidths=1.0,
            zdir='z', offset=z_bottom
        )
    
    # Customize the plot appearance
    ax.set_xlabel("Cosine of scattering angle (μ)", labelpad=10)
    ax.set_ylabel(energy_label, labelpad=10)
    ax.set_zlabel("Probability density f(μ,E)", labelpad=8)
    
    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Set axis limits
    ax.set_xlim(cosine_range)
    if log_energy_scale:
        ax.set_ylim(np.log10(min(energy_points)), np.log10(max(energy_points)))
    else:
        ax.set_ylim(min(energy_points), max(energy_points))
    
    # Set title
    title_parts = [f"3D Angular Distribution - MT{mf4_mixed.number}"]
    if data_type != 'both':
        title_parts.append(f"({data_type.capitalize()} data)")
    ax.set_title('\n'.join(title_parts), pad=15)
    
    # Improve grid appearance
    ax.grid(True, alpha=0.3)
    
    # Apply style-specific customizations
    if style in ('paper', 'publication'):
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
    
    # Add colorbar with better positioning for interactive plots
    if is_interactive:
        # For interactive plots, use the default colorbar positioning
        cbar = fig.colorbar(surface, ax=ax, shrink=0.6, aspect=20)
    else:
        # For static plots, use manual positioning
        fig.subplots_adjust(left=0.05, right=0.80, top=0.90, bottom=0.15)
        cbar_ax = fig.add_axes([0.83, 0.25, 0.02, 0.5])
        cbar = fig.colorbar(surface, cax=cbar_ax)
    
    return fig


def plot_angular_distribution_heatmap(
    mf4_mixed,
    energies: Optional[Union[Tuple[float, float], List[float]]] = None,
    energy_indices: Optional[Union[Tuple[int, int], List[int]]] = None,
    data_type: str = 'legendre',
    cosine_range: Tuple[float, float] = (-1.0, 1.0),
    n_cosine_points: int = 101,
    n_energy_points: Optional[int] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (7, 5),
    colormap: str = 'viridis',
    log_energy_scale: bool = True,
    show_contours: bool = True,
    contour_levels: int = 15,
    interactive: bool = None,
    **kwargs
) -> plt.Figure:
    """
    Create a 2D heatmap of angular distribution probability density.
    
    Shows probability density f(μ,E) as a heatmap with cosine on x-axis and energy on y-axis.
    
    Parameters
    ----------
    mf4_mixed : MF4MTMixed
        Mixed angular distribution data object
    energies : tuple of float or list of float, optional
        Energy range or specific energies to plot
    energy_indices : tuple of int or list of int, optional
        Energy index range or specific indices to plot
    data_type : str
        Type of data to plot: 'legendre', 'tabulated', or 'both'
    cosine_range : tuple
        Range of cosine values (mu_min, mu_max)
    n_cosine_points : int
        Number of cosine points for heatmap
    n_energy_points : int, optional
        Number of energy points to interpolate between
    style : str
        Plot style from _plot_settings
    figsize : tuple
        Figure size
    colormap : str
        Colormap for the heatmap
    log_energy_scale : bool
        Whether to use logarithmic scale for energy axis
    show_contours : bool
        Whether to overlay contour lines
    contour_levels : int
        Number of contour levels to show
    interactive : bool, optional
        Whether to enable interactive features (auto-detected if None)
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    plt.Figure
        The created figure with heatmap
    """
    # Setup plot style - this handles all backend detection automatically
    plot_kwargs = setup_plot_style(
        style=style, 
        figsize=figsize, 
        interactive=interactive,
        **kwargs
    )
    fig = plot_kwargs['_fig']
    ax = plot_kwargs['ax']
    
    # Determine which energies to use
    plot_energies = _determine_3d_plot_energies(mf4_mixed, energies, energy_indices, data_type)
    
    if not plot_energies:
        ax.text(0.5, 0.5, 'No energy data available for plotting', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Create coordinate arrays
    mu_points = np.linspace(cosine_range[0], cosine_range[1], n_cosine_points)
    
    # Interpolate energies if needed
    if n_energy_points is not None and len(plot_energies) > n_energy_points:
        e_min, e_max = min(plot_energies), max(plot_energies)
        if log_energy_scale and e_min > 0:
            energy_points = np.logspace(np.log10(e_min), np.log10(e_max), n_energy_points)
        else:
            energy_points = np.linspace(e_min, e_max, n_energy_points)
    else:
        energy_points = np.array(plot_energies)
    
    # Create meshgrid
    Mu_mesh, E_mesh = np.meshgrid(mu_points, energy_points)
    
    # Calculate probability density
    Z_mesh = np.zeros_like(E_mesh)
    
    for i, energy in enumerate(energy_points):
        prob_density = _evaluate_angular_distribution_at_energy(
            mf4_mixed, energy, mu_points, data_type
        )
        Z_mesh[i, :] = prob_density
    
    # Create the heatmap
    if log_energy_scale and np.all(energy_points > 0):
        im = ax.imshow(
            Z_mesh, aspect='auto', origin='lower',
            extent=[cosine_range[0], cosine_range[1], 
                   np.log10(min(energy_points)), np.log10(max(energy_points))],
            cmap=colormap, interpolation='bilinear'
        )
        ax.set_ylabel("log₁₀(Energy) [log₁₀(eV)]")
    else:
        im = ax.imshow(
            Z_mesh, aspect='auto', origin='lower',
            extent=[cosine_range[0], cosine_range[1], 
                   min(energy_points), max(energy_points)],
            cmap=colormap, interpolation='bilinear'
        )
        ax.set_ylabel("Energy (eV)")
    
    # Add contour lines if requested
    if show_contours:
        if log_energy_scale and np.all(energy_points > 0):
            E_contour = np.log10(E_mesh)
        else:
            E_contour = E_mesh
            
        contours = ax.contour(
            Mu_mesh, E_contour, Z_mesh,
            levels=contour_levels, colors='white', alpha=0.7,
            linewidths=0.8
        )
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.3f')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Probability density f(μ,E)')
    
    # Format axes
    ax = format_axes(
        ax, style=style,
        x_label="Cosine of scattering angle (μ)",
        title=f"Angular Distribution Heatmap - MT{mf4_mixed.number}" + 
              (f" ({data_type.capitalize()})" if data_type != 'both' else "")
    )
    
    ax.set_xlim(cosine_range)
    
    return fig


def _determine_3d_plot_energies(mf4_mixed, energies, energy_indices, data_type) -> List[float]:
    """
    Determine which energies to use for 3D plotting.
    
    Returns a sorted list of energy values.
    Default behavior: use ALL available energies for the specified data_type.
    """
    plot_energies = []
    
    if energies is not None:
        if isinstance(energies, tuple) and len(energies) == 2:
            # Plot all energies in range
            e_min, e_max = energies
            
            if data_type in ('legendre', 'both'):
                for energy in mf4_mixed.legendre_energies:
                    if e_min <= energy <= e_max:
                        plot_energies.append(energy)
            
            if data_type in ('tabulated', 'both'):
                for energy in mf4_mixed.tabulated_energies:
                    if e_min <= energy <= e_max:
                        plot_energies.append(energy)
        
        elif isinstance(energies, list):
            # Specific energies
            if data_type in ('legendre', 'both'):
                for energy in energies:
                    if energy in mf4_mixed.legendre_energies:
                        plot_energies.append(energy)
            
            if data_type in ('tabulated', 'both'):
                for energy in energies:
                    if energy in mf4_mixed.tabulated_energies:
                        plot_energies.append(energy)
    
    elif energy_indices is not None:
        if isinstance(energy_indices, tuple) and len(energy_indices) == 2:
            # Index range
            start, end = energy_indices
            
            if data_type in ('legendre', 'both'):
                for idx in range(start, min(end + 1, len(mf4_mixed.legendre_energies))):
                    plot_energies.append(mf4_mixed.legendre_energies[idx])
            
            if data_type in ('tabulated', 'both'):
                for idx in range(start, min(end + 1, len(mf4_mixed.tabulated_energies))):
                    plot_energies.append(mf4_mixed.tabulated_energies[idx])
        
        elif isinstance(energy_indices, list):
            # Specific indices
            if data_type in ('legendre', 'both'):
                for idx in energy_indices:
                    if idx < len(mf4_mixed.legendre_energies):
                        plot_energies.append(mf4_mixed.legendre_energies[idx])
            
            if data_type in ('tabulated', 'both'):
                for idx in energy_indices:
                    if idx < len(mf4_mixed.tabulated_energies):
                        plot_energies.append(mf4_mixed.tabulated_energies[idx])
    
    else:
        # DEFAULT: use ALL available energies for the specified data_type
        if data_type == 'legendre':
            plot_energies.extend(mf4_mixed.legendre_energies)
        elif data_type == 'tabulated':
            plot_energies.extend(mf4_mixed.tabulated_energies)
        elif data_type == 'both':
            plot_energies.extend(mf4_mixed.legendre_energies)
            plot_energies.extend(mf4_mixed.tabulated_energies)
    
    # Remove duplicates and sort
    return sorted(list(set(plot_energies)))


def _evaluate_angular_distribution_at_energy(mf4_mixed, target_energy, mu_points, data_type):
    """
    Evaluate angular distribution at a specific energy across multiple cosine values.
    
    This function handles interpolation between available energy points and
    can work with both Legendre and tabulated data.
    """
    # Find the closest energy or interpolate
    legendre_energies = mf4_mixed.legendre_energies
    tabulated_energies = mf4_mixed.tabulated_energies
    
    prob_density = np.zeros(len(mu_points))
    
    # Try to use exact energy match first
    if data_type in ('legendre', 'both') and target_energy in legendre_energies:
        coeffs = mf4_mixed.get_legendre_coefficients(target_energy)
        if coeffs:
            prob_density += _evaluate_legendre_expansion(coeffs, mu_points)
    
    elif data_type in ('tabulated', 'both') and target_energy in tabulated_energies:
        cosines, probabilities = mf4_mixed.get_tabulated_distribution(target_energy)
        if cosines and probabilities:
            prob_density += np.interp(mu_points, cosines, probabilities)
    
    else:
        # Need to interpolate between available energies
        if data_type in ('legendre', 'both') and legendre_energies:
            # Find bounding energies for Legendre data
            lower_idx, upper_idx, weight = _find_interpolation_indices(legendre_energies, target_energy)
            
            if lower_idx is not None:
                lower_coeffs = mf4_mixed.legendre_coefficients[lower_idx]
                lower_density = _evaluate_legendre_expansion(lower_coeffs, mu_points)
                
                if upper_idx is not None and weight < 1.0:
                    upper_coeffs = mf4_mixed.legendre_coefficients[upper_idx]
                    upper_density = _evaluate_legendre_expansion(upper_coeffs, mu_points)
                    prob_density += (1 - weight) * lower_density + weight * upper_density
                else:
                    prob_density += lower_density
        
        if data_type in ('tabulated', 'both') and tabulated_energies:
            # Find bounding energies for tabulated data
            lower_idx, upper_idx, weight = _find_interpolation_indices(tabulated_energies, target_energy)
            
            if lower_idx is not None:
                lower_cosines, lower_probs = mf4_mixed.get_tabulated_distribution(tabulated_energies[lower_idx])
                lower_density = np.interp(mu_points, lower_cosines, lower_probs) if lower_cosines else np.zeros(len(mu_points))
                
                if upper_idx is not None and weight < 1.0:
                    upper_cosines, upper_probs = mf4_mixed.get_tabulated_distribution(tabulated_energies[upper_idx])
                    upper_density = np.interp(mu_points, upper_cosines, upper_probs) if upper_cosines else np.zeros(len(mu_points))
                    prob_density += (1 - weight) * lower_density + weight * upper_density
                else:
                    prob_density += lower_density
    
    return prob_density


def _evaluate_legendre_expansion(coeffs, mu_points):
    """
    Evaluate Legendre expansion at given cosine points.
    
    Assumes a_0 = 1 is implicit and coeffs contains a_1, a_2, ...
    """
    if not coeffs:
        return np.ones(len(mu_points)) / 2  # Just the a_0 term
    
    # Pre-compute Legendre polynomials
    max_order = len(coeffs) + 1  # +1 for P_0
    legendre_polys = np.zeros((max_order, len(mu_points)))
    
    # P_0(μ) = 1
    legendre_polys[0, :] = 1.0
    
    # P_1(μ) = μ
    if max_order > 1:
        legendre_polys[1, :] = mu_points
    
    # Recurrence relation for higher orders
    for l in range(2, max_order):
        legendre_polys[l, :] = ((2*l - 1) * mu_points * legendre_polys[l-1, :] - 
                                (l - 1) * legendre_polys[l-2, :]) / l
    
    # Start with a_0 = 1 term
    prob_density = 0.5 * legendre_polys[0, :]  # (2*0+1)/2 * P_0 = 1/2
    
    # Add higher order terms
    for i, coeff in enumerate(coeffs):
        l = i + 1  # coeffs[0] = a_1, coeffs[1] = a_2, etc.
        prob_density += coeff * (2*l + 1) / 2 * legendre_polys[l, :]
    
    return prob_density


def _find_interpolation_indices(energy_list, target_energy):
    """
    Find indices for linear interpolation between energy points.
    
    Returns (lower_idx, upper_idx, weight) where weight is for upper_idx.
    """
    if not energy_list:
        return None, None, 0.0
    
    # Convert to numpy array for easier handling
    energies = np.array(energy_list)
    
    if target_energy <= energies[0]:
        return 0, None, 0.0
    elif target_energy >= energies[-1]:
        return len(energies) - 1, None, 0.0
    else:
        # Find bounding indices
        upper_idx = np.searchsorted(energies, target_energy)
        lower_idx = upper_idx - 1
        
        # Calculate interpolation weight
        weight = (target_energy - energies[lower_idx]) / (energies[upper_idx] - energies[lower_idx])
        
        return lower_idx, upper_idx, weight
