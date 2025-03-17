"""Utility functions for sensitivity analysis and processing.

This module contains functions for computing sensitivity coefficients from MCNP files,
creating SDF data objects from sensitivity data, and other related utility functions.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
from mcnpy.input.parse_input import read_mcnp
from mcnpy.mctal.parse_mctal import read_mctal
from mcnpy._constants import ATOMIC_NUMBER_TO_SYMBOL, MT_TO_REACTION
from mcnpy.sensitivities.sensitivity import SensitivityData, Coefficients, TaylorCoefficients
from mcnpy.sensitivities.sdf import SDFData, SDFReactionData
import math
import matplotlib.pyplot as plt


def compute_sensitivity(inputfile: str, mctalfile: str, tally: int, zaid: int, label: str) -> SensitivityData:
    """Compute sensitivity coefficients from MCNP input and output files.

    :param inputfile: Path to MCNP input file containing the PERT cards
    :type inputfile: str
    :param mctalfile: Path to MCNP MCTAL output file
    :type mctalfile: str
    :param tally: Tally number to analyze
    :type tally: int
    :param zaid: ZAID of the nuclide being perturbed
    :type zaid: int
    :param label: Label for the sensitivity data set
    :type label: str
    :returns: Object containing computed sensitivity coefficients
    :rtype: SensitivityData
    """
    input = read_mcnp(inputfile)
    mctal = read_mctal(mctalfile)
    
    pert_energies = input.perturbation.pert_energies
    reactions = input.perturbation.reactions
    group_dict_first = input.perturbation._group_perts_by_reaction(2)
    
    # Check if second-order perturbations are available
    # Use a more reliable method to check for method 3 perturbations
    has_second_order = False
    try:
        group_dict_second = input.perturbation._group_perts_by_reaction(3)
        has_second_order = bool(group_dict_second)  # True if dictionary is not empty
    except:
        group_dict_second = {}
    
    energy = mctal.tally[tally].energies 
    r0 = np.array(mctal.tally[tally].results)
    e0 = np.array(mctal.tally[tally].errors)
    
    # Prepare all the data first before creating the SensitivityData object
    full_data = {}
    coefficients = {}  # Store Taylor coefficients by energy and reaction

    for i in range(len(energy)):            # Loop over detector energies
        energy_data = {}
        coeff_data = {}
        
        # Calculate energy boundaries for the energy string
        if i == 0:
            lower_bound = 0.0
        else:
            lower_bound = energy[i-1]
        upper_bound = energy[i]
        # Format energy as string in the required format
        energy_str = f"{lower_bound:.2e}_{upper_bound:.2e}"
        
        for rxn in reactions:               # Loop over unique reaction
            # First-order processing
            sensCoef = np.zeros(len(group_dict_first[rxn]))
            sensErr = np.zeros(len(group_dict_first[rxn]))
            
            for j, pert in enumerate(group_dict_first[rxn]):
                c1 = mctal.tally[tally].perturbation[pert].results[i]
                e1 = mctal.tally[tally].perturbation[pert].errors[i]
                sensCoef[j] = c1/r0[i]
                sensErr[j] = np.sqrt(e0[i]**2 + e1**2)
            
            # Store first-order coefficients
            energy_data[rxn] = Coefficients(
                energy=energy_str,
                reaction=rxn,
                pert_energies=pert_energies,
                values=sensCoef,
                errors=sensErr,
                r0=float(r0[i]),
                e0=float(e0[i])
            )
            
            # Second-order processing (if available)
            if has_second_order and rxn in group_dict_second:
                c2_values = []
                c1_values = []  # Store the actual Taylor coefficients c1
                c1_errors = []  # Store errors of Taylor coefficients c1
                c2_errors = []  # Store errors of Taylor coefficients c2
                
                for j, pert in enumerate(group_dict_second[rxn]):
                    # Get first-order Taylor coefficient directly (not the sensitivity)
                    c1 = mctal.tally[tally].perturbation[group_dict_first[rxn][j]].results[i]
                    c1_err = mctal.tally[tally].perturbation[group_dict_first[rxn][j]].errors[i]
                    c1_values.append(c1)
                    c1_errors.append(c1_err)
                    
                    # Get second-order Taylor coefficient directly
                    c2 = mctal.tally[tally].perturbation[pert].results[i]
                    c2_err = mctal.tally[tally].perturbation[pert].errors[i]
                    c2_values.append(c2)
                    c2_errors.append(c2_err)
                
                # Calculate the ratio c2/c1 directly for each energy bin
                ratio_values = []
                for j in range(len(c1_values)):
                    if c1_values[j] != 0:  # Avoid division by zero
                        ratio_values.append(c2_values[j] / c1_values[j])
                    else:
                        ratio_values.append(float('nan'))
                
                coeff_data[rxn] = TaylorCoefficients(
                    energy=energy_str,
                    reaction=rxn,
                    pert_energies=pert_energies,
                    c1=c1_values,
                    c2=c2_values,
                    ratio=ratio_values,
                    c1_errors=c1_errors,
                    c2_errors=c2_errors
                )
        
        full_data[energy_str] = energy_data
        if coeff_data:  # Only add if there are any coefficients
            coefficients[energy_str] = coeff_data

    # Process integral results if available
    if mctal.tally[tally].integral_result is not None:
        integral_data = {}
        integral_coeff_data = {}
        integral_r0 = mctal.tally[tally].integral_result
        integral_e0 = mctal.tally[tally].integral_error
        
        for rxn in reactions:
            sensCoef_int = np.zeros(len(group_dict_first[rxn]))
            sensErr_int = np.zeros(len(group_dict_first[rxn]))
            
            # Process first-order coefficients for integral results
            for j, pert in enumerate(group_dict_first[rxn]):
                c1_int = mctal.tally[tally].perturbation[pert].integral_result
                e1_int = mctal.tally[tally].perturbation[pert].integral_error
                sensCoef_int[j] = c1_int / integral_r0
                sensErr_int[j] = np.sqrt(integral_e0**2 + e1_int**2)
            
            integral_data[rxn] = Coefficients(
                energy="integral",
                reaction=rxn,
                pert_energies=pert_energies,
                values=sensCoef_int,
                errors=sensErr_int,
                r0=integral_r0,
                e0=integral_e0
            )
            
            # Process second-order coefficients for integral results (if available)
            if has_second_order and rxn in group_dict_second:
                c2_values_int = []
                c1_values_int = []  # Store the actual Taylor coefficients
                c1_errors_int = []  # Store errors of Taylor coefficients c1
                c2_errors_int = []  # Store errors of Taylor coefficients c2
                
                for j, pert in enumerate(group_dict_second[rxn]):
                    # Get first-order Taylor coefficient directly
                    c1_int_val = mctal.tally[tally].perturbation[group_dict_first[rxn][j]].integral_result
                    c1_int_err = mctal.tally[tally].perturbation[group_dict_first[rxn][j]].integral_error
                    c1_values_int.append(c1_int_val)
                    c1_errors_int.append(c1_int_err)
                    
                    # Get second-order Taylor coefficient directly
                    c2_int_val = mctal.tally[tally].perturbation[pert].integral_result
                    c2_int_err = mctal.tally[tally].perturbation[pert].integral_error
                    c2_values_int.append(c2_int_val)
                    c2_errors_int.append(c2_int_err)
                
                # Calculate ratios for integral results
                ratio_values_int = []
                for j in range(len(c1_values_int)):
                    if c1_values_int[j] != 0:
                        ratio_values_int.append(c2_values_int[j] / c1_values_int[j])
                    else:
                        ratio_values_int.append(float('nan'))
                
                integral_coeff_data[rxn] = TaylorCoefficients(
                    energy="integral",
                    reaction=rxn,
                    pert_energies=pert_energies,
                    c1=c1_values_int,
                    c2=c2_values_int,
                    ratio=ratio_values_int,
                    c1_errors=c1_errors_int,
                    c2_errors=c2_errors_int
                )
        
        full_data["integral"] = integral_data
        if integral_coeff_data:
            coefficients["integral"] = integral_coeff_data
    
    # Create SensitivityData object after all data is prepared
    return SensitivityData(
        tally_id=tally,
        pert_energies=pert_energies,
        tally_name=mctal.tally[tally].name,
        zaid=zaid,
        label=label,
        data=full_data,
        coefficients=coefficients
    )


def plot_sens_comparison(sens_list: List[SensitivityData], 
                  energy: Union[str, List[str]] = None, 
                  reactions: Union[List[int], int] = None, 
                  xlim: tuple = None):
    """Plot comparison of multiple sensitivity datasets.

    :param sens_list: List of sensitivity datasets to compare
    :type sens_list: List[SensitivityData]
    :param energy: Energy string(s) to plot. If None, uses first dataset's energies
    :type energy: Union[str, List[str]], optional
    :param reactions: Reaction number(s) to plot. If None, uses reactions from first dataset
    :type reactions: Union[List[int], int], optional
    :param xlim: Optional x-axis limits as (min, max)
    :type xlim: tuple, optional
    """
    # If no energy specified, use all energies
    if energy is None:
        energy = list(sens_list[0].data.keys())
    elif not isinstance(energy, list):
        energy = [energy]
    
    # Ensure reactions is always a list
    if reactions is None:
        sample_energy = energy[0]
        reactions = list(sens_list[0].data[sample_energy].keys())
    elif not isinstance(reactions, list):
        reactions = [reactions]

    colors_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Create a separate figure for each energy
    for e in energy:
        n = len(reactions)
        
        # Use a single Axes if only one reaction
        if n == 1:
            fig, ax = plt.subplots(figsize=(5, 4))
            axes = [ax]
        else:
            cols = 3
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            # Ensure axes is a flat list of Axes objects
            if hasattr(axes, "flatten"):
                axes = list(axes.flatten())
            else:
                axes = [axes]
        
        # Modify title display based on energy string format
        if e == "integral":
            title_text = "Integral Result"
        else:
            # Parse the energy range from the string format
            try:
                lower, upper = e.split('_')
                title_text = f"Energy Range: {lower} - {upper} MeV"
            except ValueError:
                # Fallback if energy doesn't follow expected format
                title_text = f"Energy = {e}"
        
        # Raise the figure title position to avoid overlap with subplot titles
        fig.suptitle(title_text, y=1.01)
        
        for i, rxn in enumerate(reactions):
            ax = axes[i]
            has_data = False
            
            for idx, sens in enumerate(sens_list):
                if e in sens.data and rxn in sens.data[e]:
                    has_data = True
                    coef = sens.data[e][rxn]
                    color = colors_list[idx % len(colors_list)]
                    lp = np.array(coef.values_per_lethargy)
                    leth = np.array(coef.lethargy)
                    error_bars = np.array(coef.values) * np.array(coef.errors) / leth
                    x = np.array(coef.pert_energies)
                    y = np.append(lp, lp[-1])
                    ax.step(x, y, where='post', color=color, linewidth=2, label=sens.label)
                    x_mid = (x[:-1] + x[1:]) / 2.0
                    ax.errorbar(x_mid, lp, yerr=np.abs(error_bars), fmt=' ', 
                              elinewidth=1.5, ecolor=color, capsize=2.5)
            
            if not has_data:
                ax.text(0.5, 0.5, f"Reaction {rxn} not found", ha='center', va='center')
                ax.axis('off')
            else:
                ax.grid(True, alpha=0.3)
                ax.set_title(f"MT = {rxn}")
                ax.set_xlabel("Energy (MeV)")
                ax.set_ylabel("Sensitivity per lethargy")
                if xlim is not None:
                    ax.set_xlim(xlim)
                ax.legend()

        # Hide any extra subplots
        for j in range(n, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()







def create_sdf_data(
    sens_list: Union[List[SensitivityData], List[Tuple[SensitivityData, List[int]]]], 
    energy: str,
    title: str,
    response_values: Tuple[float, float] = None
    ) -> SDFData:
    """Create a SDFData object from a list of SensitivityData objects.
    
    :param sens_list: List of SensitivityData objects or tuples of (SensitivityData, reactions_list)
    :type sens_list: Union[List[SensitivityData], List[Tuple[SensitivityData, List[int]]]]
    :param energy: Energy value to use for sensitivity data
    :type energy: str
    :param title: Title for the SDF dataset
    :type title: str
    :param response_values: Optional tuple of (r0, e0) to override the reference values from sensitivity data.
                           This allows combining data from different sources that might have different base values.
                           r0 is the unperturbed tally result (reference response value),
                           e0 is the error of the unperturbed tally result.
                           Use this to ensure consistency when merging sensitivity data from different calculations.
    :type response_values: Tuple[float, float], optional
    :returns: SDFData object containing the combined sensitivity data
    :rtype: SDFData
    :raises ValueError: If pert_energies don't match across sensitivity data objects
    :raises ValueError: If r0 and e0 values don't match across sensitivity data objects and no response_values are provided
    """
    # Check if we have a list of SensitivityData objects or tuples
    has_tuples = any(isinstance(item, tuple) for item in sens_list)
    
    # Extract SensitivityData objects and reaction filters
    sens_data = []
    reaction_filters = []
    
    if has_tuples:
        for item in sens_list:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Expected tuple of (SensitivityData, List[int])")
            sens_obj, reactions = item
            sens_data.append(sens_obj)
            reaction_filters.append(reactions)
    else:
        sens_data = sens_list
        # No reaction filters means use all reactions for each SensitivityData
        reaction_filters = [None] * len(sens_data)
    
    # Verify that all sensitivity data objects have matching pert_energies
    reference_energies = sens_data[0].pert_energies
    for sd in sens_data[1:]:
        if sd.pert_energies != reference_energies:
            raise ValueError("All SensitivityData objects must have the same perturbation energies")
    
    # Determine r0 and e0 values (unperturbed tally result and its error)
    r0 = None
    e0 = None
    
    if response_values is not None:
        # Use provided response values
        r0, e0 = response_values
    else:
        # Verify that all sensitivity data objects have matching r0 and e0
        for sd in sens_data:
            # Find the first available reaction to get r0 and e0
            if energy in sd.data:
                for mt in sd.data[energy]:
                    if r0 is None and e0 is None:
                        # First sensitivity data object with reaction - set reference values
                        r0 = sd.data[energy][mt].r0
                        e0 = sd.data[energy][mt].e0
                    else:
                        # Compare with reference values
                        if sd.data[energy][mt].r0 != r0 or sd.data[energy][mt].e0 != e0:
                            raise ValueError(
                                "All SensitivityData objects must have the same r0 (unperturbed tally result) "
                                "and e0 (error) values. Use the response_values parameter to specify common values."
                            )
                    break  # Only need to check one reaction per sensitivity data object
    
    # Create a new SDFData object
    sdf_data = SDFData(
        title=title,
        energy=energy,
        pert_energies=reference_energies,
        r0=r0,
        e0=e0,
        data=[]
    )
    
    # Process each SensitivityData object
    for sd, reaction_filter in zip(sens_data, reaction_filters):
        # Check if energy exists in this sensitivity data
        if energy not in sd.data:
            continue
        
        # Get the reactions to process
        if reaction_filter is None:
            reactions_to_process = list(sd.data[energy].keys())
        else:
            reactions_to_process = [r for r in reaction_filter if r in sd.data[energy]]
        
        # Process each reaction
        for mt in reactions_to_process:
            coef_data = sd.data[energy][mt]
            
            # Check if all sensitivity coefficients are zero
            if all(value == 0.0 for value in coef_data.values):
                # Calculate the nuclide symbol for more informative message
                z = sd.zaid // 1000
                a = sd.zaid % 1000
                symbol = ATOMIC_NUMBER_TO_SYMBOL.get(z, f"unknown_{z}")
                nuclide = f"{symbol}-{a}"
                
                # Print message that reaction was skipped
                reaction_name = MT_TO_REACTION.get(mt, f"Unknown(MT={mt})")
                print(f"Skipping {nuclide} {reaction_name} (MT={mt}): All sensitivity coefficients are zero")
                continue
            
            # Create SDFReactionData object
            reaction_data = SDFReactionData(
                zaid=sd.zaid,
                mt=mt,
                sensitivity=coef_data.values,
                error=coef_data.errors
            )
            
            # Add to SDF data
            sdf_data.data.append(reaction_data)
    
    return sdf_data