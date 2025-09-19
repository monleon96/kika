"""
Comparison plotting utilities for angular distributions from multiple ACE objects.

These functions are designed to work with ACE objects that have already been loaded,
not with file paths. Use mcnpy.read_ace() to load your ACE files first, then pass
the resulting ACE objects to these comparison functions.
"""

from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np


def plot_ace_angular_comparison(ace_objects: List, 
                               mt: int, 
                               energy: float,
                               particle_type: str = 'neutron',
                               particle_idx: int = 0,
                               labels: Optional[List[str]] = None,
                               colors: Optional[List[str]] = None,
                               title: Optional[str] = None,
                               ax=None,
                               **kwargs) -> Optional[Tuple]:
    """
    Plot angular distributions from multiple ACE objects for comparison.
    
    Parameters
    ----------
    ace_objects : List[Ace]
        List of ACE objects to compare
    mt : int
        MT number for the reaction
    energy : float
        Incident energy to evaluate the distribution at
    particle_type : str, optional
        Type of particle: 'neutron', 'photon', or 'particle'
    particle_idx : int, optional
        Index of the particle type (used only for particle_type='particle')
    labels : List[str], optional
        List of labels for each ACE file. If None, uses file names or indices
    colors : List[str], optional
        List of colors for each ACE file. If None, uses default matplotlib colors
    title : str, optional
        Title for the plot. If None, generates a default title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure
    **kwargs : dict
        Additional keyword arguments passed to the plot function
        
    Returns
    -------
    tuple or None
        Tuple of (fig, ax) or None if matplotlib is not available
        
    Examples
    --------
    >>> # Compare two ACE objects
    >>> ace1 = mcnpy.read_ace('file1.ace')
    >>> ace2 = mcnpy.read_ace('file2.ace')
    >>> plot_ace_angular_comparison([ace1, ace2], mt=2, energy=1.0)
    
    >>> # Compare with custom labels
    >>> plot_ace_angular_comparison([ace1, ace2], 
    ...                            mt=2, energy=1.0,
    ...                            labels=['ENDF/B-VIII.0', 'JEFF-3.3'])
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Required package not available: {e}")
        return None
    
    # Use ACE objects directly (no file reading needed)
    processed_ace_objects = ace_objects
    
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Set up default labels and colors
    if labels is None:
        labels = [f"ACE object {i+1}" for i in range(len(ace_objects))]
    
    if colors is None:
        # Use matplotlib's default color cycle
        colors = plt.cm.tab10(np.linspace(0, 1, len(processed_ace_objects)))
    
    # Plot each ACE object's angular distribution
    success_count = 0
    for i, (ace_obj, label, color) in enumerate(zip(processed_ace_objects, labels, colors)):
        try:
            # Get the angular distribution data
            df = ace_obj.angular_distributions.to_dataframe(
                mt=mt, 
                energy=energy, 
                particle_type=particle_type, 
                particle_idx=particle_idx, 
                ace=ace_obj
            )
            
            if df is not None:
                # Set line properties
                line_kwargs = kwargs.copy()
                line_kwargs['color'] = color
                line_kwargs['label'] = label
                if 'linewidth' not in line_kwargs:
                    line_kwargs['linewidth'] = 2
                
                # Plot the data
                ax.plot(df['cosine'], df['pdf'], **line_kwargs)
                success_count += 1
            else:
                print(f"Warning: No data available for {label} at MT={mt}, energy={energy} MeV")
                
        except Exception as e:
            print(f"Error plotting {label}: {e}")
            continue
    
    if success_count == 0:
        print("No data could be plotted")
        return None
    
    # Set labels and formatting
    ax.set_xlabel('Cosine (μ)')
    ax.set_ylabel('Probability Density')
    
    if title is None:
        title = f'Angular Distribution Comparison - MT={mt} at {energy:.4g} MeV ({particle_type})'
    ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim(-1, 1)
    
    # Add legend and grid
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_ace_angular_energy_comparison(ace_objects: List, 
                                     mt: int, 
                                     energies: List[float],
                                     particle_type: str = 'neutron',
                                     particle_idx: int = 0,
                                     ace_labels: Optional[List[str]] = None,
                                     energy_labels: Optional[List[str]] = None,
                                     title: Optional[str] = None,
                                     **kwargs) -> Optional[Tuple]:
    """
    Plot angular distributions from multiple ACE objects at multiple energies.
    
    This creates a more complex comparison showing how angular distributions
    vary both between different ACE objects and across different energies.
    
    Parameters
    ----------
    ace_objects : List[Ace]
        List of ACE objects to compare
    mt : int
        MT number for the reaction
    energies : List[float]
        List of incident energies to evaluate the distribution at
    particle_type : str, optional
        Type of particle: 'neutron', 'photon', or 'particle'
    particle_idx : int, optional
        Index of the particle type (used only for particle_type='particle')
    ace_labels : List[str], optional
        List of labels for each ACE file
    energy_labels : List[str], optional
        List of labels for each energy
    title : str, optional
        Title for the plot
    **kwargs : dict
        Additional keyword arguments passed to the plot function
        
    Returns
    -------
    tuple or None
        Tuple of (fig, axes) where axes is a 2D array if multiple subplots
        
    Examples
    --------
    >>> # Compare two ACE files at multiple energies
    >>> plot_ace_angular_energy_comparison(['file1.ace', 'file2.ace'], 
    ...                                   mt=2, energies=[0.1, 1.0, 10.0])
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Required package not available: {e}")
        return None
    
    # Use ACE objects directly (no file reading needed)
    processed_ace_objects = ace_objects
    
    # Set up default labels
    if ace_labels is None:
        ace_labels = [f"ACE object {i+1}" for i in range(len(ace_objects))]
    
    if energy_labels is None:
        energy_labels = [f"{energy:.4g} MeV" for energy in energies]
    
    # Create subplots - one for each energy
    n_energies = len(energies)
    n_cols = min(3, n_energies)  # Max 3 columns
    n_rows = (n_energies + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_energies == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each energy in a separate subplot
    for i, (energy, energy_label) in enumerate(zip(energies, energy_labels)):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue
            
        # Plot all ACE objects for this energy
        plot_ace_angular_comparison(
            processed_ace_objects, 
            mt=mt, 
            energy=energy,
            particle_type=particle_type,
            particle_idx=particle_idx,
            labels=ace_labels,
            title=f"{energy_label}",
            ax=ax,
            **kwargs
        )
    
    # Hide unused subplots
    for i in range(n_energies, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    if title is None:
        title = f'Angular Distribution Energy Comparison - MT={mt} ({particle_type})'
    fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    return fig, axes


# Convenience function that automatically detects if it should do single or multi-energy comparison
def compare_ace_angular_distributions(ace_objects: List, 
                                    mt: int, 
                                    energy: Union[float, List[float]],
                                    **kwargs) -> Optional[Tuple]:
    """
    Convenience function to compare angular distributions from multiple ACE objects.
    
    Automatically chooses between single energy or multi-energy comparison based
    on the type of the energy parameter.
    
    Parameters
    ----------
    ace_objects : List[Ace]
        List of ACE objects to compare
    mt : int
        MT number for the reaction
    energy : float or List[float]
        Single energy or list of energies to compare
    **kwargs : dict
        Additional arguments passed to the appropriate plotting function
        
    Returns
    -------
    tuple or None
        Result from the appropriate plotting function
    """
    if isinstance(energy, (list, tuple, np.ndarray)):
        return plot_ace_angular_energy_comparison(ace_objects, mt, energy, **kwargs)
    else:
        return plot_ace_angular_comparison(ace_objects, mt, energy, **kwargs)