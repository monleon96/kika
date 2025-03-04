from dataclasses import dataclass, field
from mcnpy.input.parse_input import read_mcnp  # Fix import path
from mcnpy.mctal.parse_mctal import read_mctal
from mcnpy._constants import ATOMIC_NUMBER_TO_SYMBOL
from typing import Dict, Union, List
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


@dataclass
class SensitivityData:
    """Container class for sensitivity analysis data.

    :ivar tally_id: ID of the tally used for sensitivity calculation
    :type tally_id: int
    :ivar pert_energies: List of perturbation energy boundaries
    :type pert_energies: List[float]
    :ivar zaid: ZAID of the nuclide for which sensitivities were calculated
    :type zaid: int
    :ivar label: Label for the sensitivity data set
    :type label: str
    :ivar tally_name: Name of the tally
    :type tally_name: str
    :ivar data: Nested dictionary containing sensitivity coefficients organized by energy and reaction number
    :type data: Dict[str, Dict[int, Coefficients]]
    :ivar lethargy: List of lethargy intervals between perturbation energies
    :type lethargy: List[float]
    :ivar energies: List of energy values used as keys in the data dictionary
    :type energies: List[str]
    :ivar reactions: Sorted list of unique reaction numbers found in the data
    :type reactions: List[int]
    :ivar nuclide: Nuclide symbol for the ZAID
    :type nuclide: str
    """
    tally_id: int
    pert_energies: list[float]
    zaid: int
    label: str
    tally_name: str = None
    data: Dict[str, Dict[int, 'Coefficients']] = None
    lethargy: List[float] = field(init=False, repr=False)
    energies: List[str] = field(init=False, repr=False)
    reactions: List[int] = field(init=False, repr=False)
    nuclide: str = field(init=False, repr=False)
    
    def __post_init__(self):
        """Compute attributes once after initialization"""
        # Calculate lethargy intervals
        self.lethargy = [np.log(self.pert_energies[i+1]/self.pert_energies[i]) 
                         for i in range(len(self.pert_energies)-1)]
        
        # Get energy keys
        self.energies = list(self.data.keys()) if self.data else []
        
        # Get unique reaction numbers
        if not self.data:
            self.reactions = []
        else:
            all_reactions = set()
            for energy_data in self.data.values():
                all_reactions.update(energy_data.keys())
            self.reactions = sorted(list(all_reactions))
        
        # Get nuclide symbol
        z = self.zaid // 1000
        a = self.zaid % 1000
        self.nuclide = f"{ATOMIC_NUMBER_TO_SYMBOL[z]}-{a}"

    def plot(self, energy: Union[str, List[str]] = None, 
             reactions: Union[List[int], int] = None, xlim: tuple = None):
        """Plot sensitivity coefficients for specified energies and reactions.

        :param energy: Energy string(s) to plot. If None, plots all energies
        :type energy: Union[str, List[str]], optional
        :param reactions: Reaction number(s) to plot. If None, plots all reactions
        :type reactions: Union[List[int], int], optional
        :param xlim: Optional x-axis limits as (min, max)
        :type xlim: tuple, optional
        :raises ValueError: If specified energies are not found in the data
        """
        # If no energy specified, use all energies
        if energy is None:
            energies = list(self.data.keys())
        else:
            # Ensure energy is always a list
            energies = [energy] if not isinstance(energy, list) else energy
            # Validate all energies exist in data
            invalid_energies = [e for e in energies if e not in self.data]
            if invalid_energies:
                raise ValueError(f"Energies {invalid_energies} not found in sensitivity data.")

        # Ensure reactions is always a list
        if reactions is None:
            # Get unique reactions from all energy data
            reactions = list(set().union(*[d.keys() for d in self.data.values()]))
            # Sort reactions in ascending numerical order
            reactions.sort()
        elif not isinstance(reactions, list):
            reactions = [reactions]

        # Create a separate figure for each energy
        for e in energies:
            coeffs_dict = self.data[e]
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
                if rxn not in coeffs_dict:
                    ax.text(0.5, 0.5, f"Reaction {rxn} not found", ha='center', va='center')
                    ax.axis('off')
                else:
                    coef = coeffs_dict[rxn]
                    coef.plot(ax=ax, xlim=xlim)

            # Hide any extra subplots
            for j in range(n, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            plt.show()

    @classmethod
    def plot_comparison(cls, sens_list: List['SensitivityData'], 
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

    def to_dataframe(self) -> pd.DataFrame:
        """Export sensitivity data as a pandas DataFrame for plotting.

        :returns: DataFrame with the following columns:
            - det_energy: Detector energy range string (e.g., "0.00e+00_1.00e-01") or 'integral'
            - energy_lower: Lower energy boundary parsed from det_energy string (None for 'integral')
            - energy_upper: Upper energy boundary parsed from det_energy string (None for 'integral')
            - reaction: Reaction number (MT)
            - e_lower: Lower boundary of perturbation energy bin
            - e_upper: Upper boundary of perturbation energy bin
            - sensitivity: Sensitivity per lethargy value
            - error: Relative error value for the sensitivity
            - label: Sensitivity data label
            - tally_name: Name of the tally
        :rtype: pd.DataFrame
        """
        data_records = []

        for det_energy, rxn_dict in self.data.items():
            # Parse energy bounds from energy string if not "integral"
            energy_lower = None
            energy_upper = None
            if det_energy != "integral":
                try:
                    energy_lower, energy_upper = map(float, det_energy.split('_'))
                except ValueError:
                    # Handle case where energy string doesn't match expected format
                    pass
            
            for rxn, coef in rxn_dict.items():
                energies = coef.pert_energies
                # Calculate values per lethargy
                lp = np.array(coef.values_per_lethargy)
                # Compute error bars from values, errors and lethargy
                leth = np.array(coef.lethargy)
                error_bars = (np.array(coef.values) * np.array(coef.errors) / leth).tolist()
                
                # Create records for each energy bin (using lower and upper boundaries)
                for i in range(len(energies) - 1):
                    data_records.append({
                        'det_energy': det_energy,
                        'energy_lower': energy_lower,
                        'energy_upper': energy_upper,
                        'reaction': rxn,
                        'e_lower': energies[i],
                        'e_upper': energies[i+1],
                        'sensitivity': lp[i],
                        'error': error_bars[i],
                        'label': self.label,
                        'tally_name': self.tally_name
                    })

        return pd.DataFrame(data_records)
        

@dataclass
class Coefficients:
    """Container for sensitivity coefficients for a specific energy and reaction.

    :ivar energy: Energy range string in format "lower_upper" (e.g., "0.00e+00_1.00e-01")
    :type energy: str
    :ivar reaction: Reaction number
    :type reaction: int
    :ivar pert_energies: Perturbation energy boundaries
    :type pert_energies: List[float]
    :ivar values: Raw Taylor coefficient values
    :type values: List[float]
    :ivar errors: Relative errors for the Taylor coefficients
    :type errors: List[float]
    :ivar r0: Unperturbed tally result
    :type r0: float
    :ivar e0: Unperturbed tally error
    :type e0: float
    """
    energy: str
    reaction: int
    pert_energies: list[float]
    values: list[float]
    errors: list[float]
    r0: float = None 
    e0: float = None 

    @property
    def lethargy(self):
        """Calculate lethargy intervals between perturbation energies.

        :returns: List of lethargy intervals
        :rtype: List[float]
        """
        return [np.log(self.pert_energies[i+1]/self.pert_energies[i]) for i in range(len(self.pert_energies)-1)]

    @property
    def values_per_lethargy(self):
        """Calculate sensitivity coefficients per unit lethargy.

        :returns: Sensitivity coefficients normalized by lethargy intervals
        :rtype: List[float]
        """
        lethargy_vals = self.lethargy
        return [self.values[i]/lethargy_vals[i] for i in range(len(lethargy_vals))]
    
    # New helper method to plot onto a provided axis
    def _plot_on_ax(self, ax, xlim=None):
        """Plot sensitivity coefficients on a given matplotlib axis.

        :param ax: The axis to plot on
        :type ax: matplotlib.axes.Axes
        :param xlim: Optional x-axis limits as (min, max)
        :type xlim: tuple, optional
        """
        # Compute values per lethargy and error ratios
        lp = np.array(self.values_per_lethargy)
        leth = np.array(self.lethargy)
        error_bars = np.array(self.values) * np.array(self.errors) / leth
        x = np.array(self.pert_energies)
        y = np.append(lp, lp[-1])
        color = 'blue'
        ax.step(x, y, where='post', color=color, linewidth=2)
        x_mid = (x[:-1] + x[1:]) / 2.0
        ax.errorbar(x_mid, lp, yerr=np.abs(error_bars), fmt=' ', elinewidth=1.5, ecolor=color, capsize=2.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"MT = {self.reaction}")
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Sensitivity per lethargy")
        if xlim is not None:
            ax.set_xlim(xlim)
        
    def plot(self, ax=None, xlim=None):
        """Create a new plot of sensitivity coefficients.

        :param ax: Optional existing axis to plot on
        :type ax: matplotlib.axes.Axes, optional
        :param xlim: Optional x-axis limits as (min, max)
        :type xlim: tuple, optional
        :returns: The axis containing the plot
        :rtype: matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        self._plot_on_ax(ax, xlim=xlim)
        return ax
    

def compute_sensitivity(input_path: str, mctal_path: str, tally: int, zaid: int, label: str) -> SensitivityData:
    """Compute sensitivity coefficients from MCNP input and output files.

    :param input_path: Path to MCNP input file containing the PERT cards
    :type input_path: str
    :param mctal_path: Path to MCNP MCTAL output file
    :type mctal_path: str
    :param tally: Tally number to analyze
    :type tally: int
    :param zaid: ZAID of the nuclide being perturbed
    :type zaid: int
    :param label: Label for the sensitivity data set
    :type label: str
    :returns: Object containing computed sensitivity coefficients
    :rtype: SensitivityData
    """
    input = read_mcnp(input_path)
    mctal = read_mctal(mctal_path)
    
    pert_energies = input.pert.pert_energies
    reactions = input.pert.reactions
    group_dict = input.pert.group_perts_by_reaction(2)
    
    energy = mctal.tally[tally].energies 
    r0 = np.array(mctal.tally[tally].results)
    e0 = np.array(mctal.tally[tally].errors)
    
    # Prepare all the data first before creating the SensitivityData object
    full_data = {}

    for i in range(len(energy)):            # Loop over detector energies
        energy_data = {}
        # Calculate energy boundaries for the energy string
        if i == 0:
            lower_bound = 0.0
        else:
            lower_bound = energy[i-1]
        upper_bound = energy[i]
        # Format energy as string in the required format
        energy_str = f"{lower_bound:.2e}_{upper_bound:.2e}"
        
        for rxn in reactions:               # Loop over unique reaction
            sensCoef = np.zeros(len(group_dict[rxn]))
            sensErr = np.zeros(len(group_dict[rxn]))
            for j, pert in enumerate(group_dict[rxn]):    # Loop over list of perturbations - one per pert energy bin
                c1 = mctal.tally[tally].pert_data[pert].results[i]
                e1 = mctal.tally[tally].pert_data[pert].errors[i]
                sensCoef[j] = c1/r0[i]
                sensErr[j] = np.sqrt(e0[i]**2 + e1**2)
            
            energy_data[rxn] = Coefficients(
                energy=energy_str,
                reaction=rxn,
                pert_energies=pert_energies,
                values=sensCoef,
                errors=sensErr,
                r0=float(r0[i]),  
                e0=float(e0[i])   
            )
        
        full_data[energy_str] = energy_data

    if mctal.tally[tally].integral_result is not None:
        integral_data = {}
        integral_r0 = mctal.tally[tally].integral_result
        integral_e0 = mctal.tally[tally].integral_error
        
        for rxn in reactions:
            sensCoef_int = np.zeros(len(group_dict[rxn]))
            sensErr_int = np.zeros(len(group_dict[rxn]))
            for j, pert in enumerate(group_dict[rxn]):
                c1_int = mctal.tally[tally].pert_data[pert].integral_result
                e1_int = mctal.tally[tally].pert_data[pert].integral_error
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
        full_data["integral"] = integral_data
    
    # Create SensitivityData object after all data is prepared
    return SensitivityData(
        tally_id=tally,
        pert_energies=pert_energies,
        tally_name=mctal.tally[tally].name,
        zaid=zaid,
        label=label,
        data=full_data  # Pass the fully populated data dictionary
    )