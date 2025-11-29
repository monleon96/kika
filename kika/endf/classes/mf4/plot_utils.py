"""
Utility functions to create PlotData objects from MF4 data.

These functions serve as bridges between the ENDF data structures
and the plotting infrastructure.
"""

from typing import Optional, List, Union
import numpy as np

from kika.plotting import LegendreCoeffPlotData, LegendreUncertaintyPlotData


def create_legendre_coeff_plot_data(
    mf4_data,
    order: int,
    label: Optional[str] = None,
    **styling_kwargs
) -> LegendreCoeffPlotData:
    """
    Create a LegendreCoeffPlotData object from MF4 data.
    
    Parameters
    ----------
    mf4_data : MF4MT object
        MF4 data object containing Legendre coefficients
    order : int
        Legendre polynomial order to extract
    label : str, optional
        Custom label. If None, auto-generates from isotope and order.
    **styling_kwargs
        Additional styling kwargs (color, linestyle, etc.)
        
    Returns
    -------
    LegendreCoeffPlotData
        Plottable data object
        
    Raises
    ------
    ValueError
        If the MF4 data doesn't contain Legendre coefficients
    """
    # Validate that this MF4 data has Legendre coefficients
    if not hasattr(mf4_data, 'legendre_energies') or not hasattr(mf4_data, 'legendre_coefficients'):
        obj_class_name = type(mf4_data).__name__
        raise ValueError(
            f"Cannot create Legendre coefficient plot data from {obj_class_name}. "
            f"This object does not contain Legendre coefficients."
        )
    
    energies = mf4_data.legendre_energies
    coeffs_list = mf4_data.legendre_coefficients
    
    if not energies or not coeffs_list:
        raise ValueError("No Legendre coefficient data available")
    
    # Extract coefficient values for the specified order
    coeff_values = []
    energy_values = []
    
    for j, coeffs in enumerate(coeffs_list):
        if order == 0:
            # a_0 is always 1 (implicit in ENDF format)
            coeff_values.append(1.0)
            energy_values.append(energies[j])
        elif order - 1 < len(coeffs):  # order 1 -> coeffs[0], order 2 -> coeffs[1], etc.
            coeff_values.append(coeffs[order - 1])
            energy_values.append(energies[j])
    
    if not energy_values:
        raise ValueError(f"No data available for Legendre order {order}")
    
    # Get isotope and MT information
    isotope = getattr(mf4_data, 'isotope', None)
    if isotope is None and hasattr(mf4_data, 'zaid'):
        # Try to construct isotope symbol from zaid if available
        isotope = str(mf4_data.zaid)
    
    mt = getattr(mf4_data, 'number', None)
    
    # Create and return the plot data object
    return LegendreCoeffPlotData(
        x=np.array(energy_values),
        y=np.array(coeff_values),
        order=order,
        isotope=isotope,
        mt=mt,
        energy_range=(min(energy_values), max(energy_values)),
        label=label,
        **styling_kwargs
    )


def create_legendre_uncertainty_plot_data(
    mf4_data,
    mf34_covmat,
    order: int,
    uncertainty_type: str = 'relative',
    label: Optional[str] = None,
    **styling_kwargs
) -> LegendreUncertaintyPlotData:
    """
    Create a LegendreUncertaintyPlotData object from MF4 and MF34 data.
    
    Parameters
    ----------
    mf4_data : MF4MT object
        MF4 data object containing Legendre coefficients
    mf34_covmat : MF34CovMat object
        Converted MF34 covariance matrix object
    order : int
        Legendre polynomial order
    uncertainty_type : str
        'relative' or 'absolute'
    label : str, optional
        Custom label. If None, auto-generates from isotope and order.
    **styling_kwargs
        Additional styling kwargs (color, linestyle, etc.)
        
    Returns
    -------
    LegendreUncertaintyPlotData
        Plottable data object for uncertainties
    """
    # Get isotope ID
    isotope_id = int(mf4_data.zaid) if hasattr(mf4_data, 'zaid') else 0
    mt = getattr(mf4_data, 'number', None)
    
    # Extract uncertainties from MF34
    uncertainties_data = mf34_covmat.get_uncertainties_for_legendre_coefficient(
        isotope=isotope_id,
        mt=mt,
        l_coefficient=order
    )
    
    if uncertainties_data is None:
        raise ValueError(f"No uncertainty data available for order {order}")
    
    unc_energies = uncertainties_data['energies']
    unc_values = uncertainties_data['uncertainties']
    
    # Find energy bin boundaries from the covariance matrix
    bin_boundaries = None
    for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c) in enumerate(zip(
        mf34_covmat.isotope_rows, mf34_covmat.reaction_rows, mf34_covmat.l_rows,
        mf34_covmat.isotope_cols, mf34_covmat.reaction_cols, mf34_covmat.l_cols
    )):
        if (iso_r == isotope_id and iso_c == isotope_id and 
            mt_r == mt and mt_c == mt and 
            l_r == order and l_c == order):
            bin_boundaries = np.array(mf34_covmat.energy_grids[i])
            break
    
    # Convert to appropriate units for plotting
    if uncertainty_type == 'relative':
        plot_values = np.array(unc_values) * 100  # Convert to percentage
    elif uncertainty_type == 'absolute':
        # Need to multiply by coefficient values
        energies = mf4_data.legendre_energies
        coeffs_list = mf4_data.legendre_coefficients
        
        # Extract coefficient values for this order
        coeff_values = []
        energy_values = []
        
        for j, coeffs in enumerate(coeffs_list):
            if order == 0:
                coeff_values.append(1.0)
                energy_values.append(energies[j])
            elif order - 1 < len(coeffs):
                coeff_values.append(coeffs[order - 1])
                energy_values.append(energies[j])
        
        # Interpolate to uncertainty energy grid
        interpolated_coeffs = []
        for unc_energy in unc_energies:
            closest_idx = min(range(len(energy_values)), 
                            key=lambda k: abs(energy_values[k] - unc_energy))
            interpolated_coeffs.append(abs(coeff_values[closest_idx]))
        
        plot_values = np.array(unc_values) * np.array(interpolated_coeffs)
    else:
        raise ValueError(f"uncertainty_type must be 'relative' or 'absolute', got '{uncertainty_type}'")
    
    # Get isotope symbol
    isotope = getattr(mf4_data, 'isotope', None)
    if isotope is None and hasattr(mf4_data, 'zaid'):
        isotope = str(mf4_data.zaid)
    
    # Create and return the plot data object
    return LegendreUncertaintyPlotData(
        x=np.array(unc_energies),
        y=plot_values,
        order=order,
        isotope=isotope,
        mt=mt,
        uncertainty_type=uncertainty_type,
        energy_bins=bin_boundaries,
        label=label,
        plot_type='step',
        step_where='post',
        **styling_kwargs
    )


def create_multiple_legendre_coeff_plot_data(
    mf4_data,
    orders: Optional[Union[int, List[int]]] = None,
    label_prefix: Optional[str] = None,
    **styling_kwargs
) -> List[LegendreCoeffPlotData]:
    """
    Create multiple LegendreCoeffPlotData objects for different orders.
    
    Parameters
    ----------
    mf4_data : MF4MT object
        MF4 data object containing Legendre coefficients
    orders : int, list of int, or None
        Orders to create plot data for. If None, creates for all available orders.
    label_prefix : str, optional
        Prefix for labels
    **styling_kwargs
        Common styling kwargs applied to all data objects
        
    Returns
    -------
    list of LegendreCoeffPlotData
        List of plottable data objects
    """
    # Determine orders
    if orders is None:
        coeffs_list = mf4_data.legendre_coefficients
        if coeffs_list:
            max_available = max(len(coeffs) for coeffs in coeffs_list)
            orders = list(range(max_available))
        else:
            return []
    elif isinstance(orders, int):
        orders = [orders]
    
    # Create plot data for each order
    plot_data_list = []
    for order in orders:
        try:
            label = None
            if label_prefix:
                label = f"{label_prefix} - L={order}"
            
            data = create_legendre_coeff_plot_data(
                mf4_data, order, label=label, **styling_kwargs
            )
            plot_data_list.append(data)
        except ValueError as e:
            print(f"Warning: Could not create plot data for order {order}: {e}")
            continue
    
    return plot_data_list


def create_multiple_legendre_uncertainty_plot_data(
    mf4_data,
    mf34_covmat,
    orders: Optional[Union[int, List[int]]] = None,
    uncertainty_type: str = 'relative',
    label_prefix: Optional[str] = None,
    **styling_kwargs
) -> List[LegendreUncertaintyPlotData]:
    """
    Create multiple LegendreUncertaintyPlotData objects for different orders.
    
    Parameters
    ----------
    mf4_data : MF4MT object
        MF4 data object containing Legendre coefficients
    mf34_covmat : MF34CovMat object
        Converted MF34 covariance matrix object
    orders : int, list of int, or None
        Orders to create plot data for. If None, creates for all available orders.
    uncertainty_type : str
        'relative' or 'absolute'
    label_prefix : str, optional
        Prefix for labels
    **styling_kwargs
        Common styling kwargs applied to all data objects
        
    Returns
    -------
    list of LegendreUncertaintyPlotData
        List of plottable data objects
    """
    # Determine orders
    if orders is None:
        coeffs_list = mf4_data.legendre_coefficients
        if coeffs_list:
            max_available = max(len(coeffs) for coeffs in coeffs_list)
            orders = list(range(max_available))
        else:
            return []
    elif isinstance(orders, int):
        orders = [orders]
    
    # Create plot data for each order
    plot_data_list = []
    for order in orders:
        try:
            label = None
            if label_prefix:
                label = f"{label_prefix} - L={order}"
            
            data = create_legendre_uncertainty_plot_data(
                mf4_data, mf34_covmat, order, 
                uncertainty_type=uncertainty_type,
                label=label, 
                **styling_kwargs
            )
            plot_data_list.append(data)
        except ValueError as e:
            print(f"Warning: Could not create uncertainty plot data for order {order}: {e}")
            continue
    
    return plot_data_list
