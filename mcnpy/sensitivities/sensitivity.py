from dataclasses import dataclass
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
    :type data: Dict[Union[float,str], Dict[int, Coefficients]]
    """
    tally_id: int
    pert_energies: list[float]
    zaid: int
    label: str
    tally_name: str = None
    data: Dict[Union[float,str], Dict[int, 'Coefficients']] = None

    @property
    def lethargy(self):
        """Calculate lethargy intervals between perturbation energies.

        :returns: List of lethargy intervals
        :rtype: List[float]
        """
        return [np.log(self.pert_energies[i+1]/self.pert_energies[i]) for i in range(len(self.pert_energies)-1)]
        
    @property
    def energies(self):
        """Get the list of energy values used as keys in the data dictionary.
        
        :returns: List of energy values or labels
        :rtype: List[Union[float, str]]
        """
        return list(self.data.keys()) if self.data else []
    
    @property
    def reactions(self):
        """Get a sorted list of unique reaction numbers found in the data.
        
        :returns: List of unique reaction numbers across all energies
        :rtype: List[int]
        """
        if not self.data:
            return []
        # Collect all reaction numbers from all energy entries
        all_reactions = set()
        for energy_data in self.data.values():
            all_reactions.update(energy_data.keys())
        return sorted(list(all_reactions))
    
    @property
    def nuclide(self):
        """Get the nuclide symbol for the ZAID used in the sensitivity calculation.
        
        :returns: Nuclide symbol
        :rtype: str
        """
        z = self.zaid // 1000
        a = self.zaid % 1000
        return f"{ATOMIC_NUMBER_TO_SYMBOL[z]}-{a}"

    def plot(self, energy: Union[float, str, List[float], List[str]] = None, 
             reactions: Union[List[int], int] = None, xlim: tuple = None):
        """Plot sensitivity coefficients for specified energies and reactions.

        :param energy: Energy value(s) to plot. If None, plots all energies
        :type energy: Union[float, str, List[float], List[str]], optional
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
            
            # Raise the figure title position to avoid overlap with subplot titles
            fig.suptitle(f"Energy = {e} MeV", y=1.01)
            
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
                      energy: Union[float, str, List[float], List[str]] = None, 
                      reactions: Union[List[int], int] = None, 
                      xlim: tuple = None):
        """Plot comparison of multiple sensitivity datasets.

        :param sens_list: List of sensitivity datasets to compare
        :type sens_list: List[SensitivityData]
        :param energy: Energy value(s) to plot. If None, uses first dataset's energies
        :type energy: Union[float, str, List[float], List[str]], optional
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
            
            # Raise the figure title position to avoid overlap with subplot titles
            fig.suptitle(f"Energy = {e} MeV", y=1.01)
            
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
            - det_energy: Detector energy bin value or 'integral' for integral results
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
                        'reaction': rxn,
                        'e_lower': energies[i],
                        'e_upper': energies[i+1],
                        'sensitivity': lp[i],
                        'error': error_bars[i],
                        'label': self.label,
                        'tally_name': self.tally_name
                    })

        return pd.DataFrame(data_records)

    @classmethod
    def to_sdf(cls, sens_list: List['SensitivityData'], output_path: str,
               group_inelastic: bool = True):
        """Export multiple sensitivity datasets to a single SDF file.

        :param sens_list: List of sensitivity datasets to export
        :type sens_list: List[SensitivityData]
        :param output_path: Path to write the SDF file
        :type output_path: str
        :param group_inelastic: Flag to group inelastic reactions (51-91) into MT 4
        :type group_inelastic: bool, optional
        """
        

@dataclass
class Coefficients:
    """Container for sensitivity coefficients for a specific energy and reaction.

    :ivar energy: Energy value or label
    :type energy: Union[float, str]
    :ivar reaction: Reaction number
    :type reaction: int
    :ivar pert_energies: Perturbation energy boundaries
    :type pert_energies: List[float]
    :ivar values: Raw Taylor coefficient values
    :type values: List[float]
    :ivar errors: Relative errors for the Taylor coefficients
    :type errors: List[float]
    """
    energy: Union[float, str]
    reaction: int
    pert_energies: list[float]
    values: list[float]
    errors: list[float]

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
    

def compute_senstivity(input_path: str, mctal_path: str, tally: int, zaid: int, label: str) -> SensitivityData:
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
       
    sens_result = SensitivityData(
        tally_id=tally,
        pert_energies=pert_energies,
        tally_name=mctal.tally[tally].name,
        zaid=zaid,
        label=label,
        data={}
    )

    for i in range(len(energy)):            # Loop over detector energies
        energy_data = {}
        for rxn in reactions:               # Loop over unique reaction
            sensCoef = np.zeros(len(group_dict[rxn]))
            sensErr = np.zeros(len(group_dict[rxn]))
            for j, pert in enumerate(group_dict[rxn]):    # Loop over list of perturbations - one per pert energy bin
                c1 = mctal.tally[tally].pert_data[pert].results[i]
                e1 = mctal.tally[tally].pert_data[pert].errors[i]
                sensCoef[j] = c1/r0[i]
                sensErr[j] = np.sqrt(e0[i]**2 + e1**2)
            
            energy_data[rxn] = Coefficients(
                energy=energy[i],
                reaction=rxn,
                pert_energies=pert_energies,
                values=sensCoef,
                errors=sensErr
            )
        
        sens_result.data[energy[i]] = energy_data

    if mctal.tally[tally].integral_result is not None:
        integral_data = {}
        for rxn in reactions:
            sensCoef_int = np.zeros(len(group_dict[rxn]))
            sensErr_int = np.zeros(len(group_dict[rxn]))
            for j, pert in enumerate(group_dict[rxn]):
                c1_int = mctal.tally[tally].pert_data[pert].integral_result
                e1_int = mctal.tally[tally].pert_data[pert].integral_error
                sensCoef_int[j] = c1_int / mctal.tally[tally].integral_result
                sensErr_int[j] = np.sqrt(mctal.tally[tally].integral_error**2 + e1_int**2)
            integral_data[rxn] = Coefficients(
                energy=mctal.tally[tally].integral_result,
                reaction=rxn,
                pert_energies=pert_energies,
                values=sensCoef_int,
                errors=sensErr_int
            )
        sens_result.data["integral"] = integral_data
    
    return sens_result
