from dataclasses import dataclass, field
from mcnpy._constants import ATOMIC_NUMBER_TO_SYMBOL
from typing import Dict, Union, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


@dataclass
class TaylorRatio:
    """Container for Taylor series expansion ratios between second and first-order coefficients.
    
    :ivar energy: Energy range string in format "lower_upper" (e.g., "0.00e+00_1.00e-01")
    :type energy: str
    :ivar reaction: Reaction number
    :type reaction: int
    :ivar pert_energies: Perturbation energy boundaries
    :type pert_energies: List[float]
    :ivar c1: First-order Taylor coefficients
    :type c1: List[float]
    :ivar c2: Second-order Taylor coefficients
    :type c2: List[float]
    :ivar ratio: Ratio of c2/c1 for each energy bin
    :type ratio: List[float]
    """
    energy: str
    reaction: int
    pert_energies: list[float]
    c1: list[float]
    c2: list[float]
    ratio: list[float]
    
    def calculate_nonlinearity(self, p: float) -> float:
        """Calculate the nonlinearity factor at a specific perturbation.
        
        The nonlinearity factor is (c2*p)/c1, which represents the ratio of 
        second-order to first-order term at perturbation magnitude p.
        
        :param p: Perturbation magnitude (0 to 100%)
        :type p: float
        :returns: Average nonlinearity across all energy bins
        :rtype: float
        """
        valid_ratios = [r for r in self.ratio if not np.isnan(r)]
        if not valid_ratios:
            return float('nan')
        # Convert p from percent to fraction (p/100)
        p_fraction = p / 100.0
        return np.mean(valid_ratios) * p_fraction
    
    def calculate_nonlinearity_by_bin(self, p: float) -> list:
        """Calculate the nonlinearity factor for each energy bin at specific perturbation.
        
        The nonlinearity factor is (c2*p)/c1, which represents the ratio of 
        second-order to first-order term at perturbation magnitude p.
        
        :param p: Perturbation magnitude (0 to 100%)
        :type p: float
        :returns: Nonlinearity factor for each energy bin
        :rtype: list
        """
        # Convert p from percent to fraction (p/100)
        p_fraction = p / 100.0
        return [r * p_fraction if not np.isnan(r) else np.nan for r in self.ratio]
    
    def plot(self, ax=None, title=None, top_n=5):
        """Plot the nonlinearity factor vs perturbation magnitude.
        
        The nonlinearity factor is plotted using absolute values for better comparison,
        as the sign of the ratio doesn't provide valuable information in this context.
        
        :param ax: Optional existing axis to plot on
        :type ax: matplotlib.axes.Axes, optional
        :param title: Optional custom title for the plot
        :type title: str, optional
        :param top_n: Number of top absolute ratios to plot (0 means plot all)
        :type top_n: int, optional
        :returns: The axis containing the plot
        :rtype: matplotlib.axes.Axes
        """
        if ax is None:
            # Use the same figure size as in SensitivityData.plot_ratio() for consistency
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate x values (perturbation magnitudes in percent)
        p_values = np.linspace(0, 20, 100)
        
        # Calculate nonlinearity for each energy bin and identify top N ratios by absolute value
        # Include zeros as valid candidates for top_n
        abs_ratios = [abs(r) if not np.isnan(r) else 0 for r in self.ratio]
        sorted_indices = np.argsort(abs_ratios)[::-1]  # Sort in descending order
        
        # If top_n is 0, plot all with labels
        if top_n == 0:
            top_indices = list(range(len(self.ratio)))
        else:
            top_indices = sorted_indices[:min(top_n, len(abs_ratios))]  # Get top N (or fewer if there aren't N)
        
        # Only plot lines for top_n energy bins
        for i in top_indices:
            if np.isnan(self.ratio[i]):
                continue
                
            # Calculate the nonlinearity values for this bin across p_values
            # Convert p_values from percent to fraction for calculation
            # Use absolute values for plotting
            abs_ratio = abs(self.ratio[i])
            y_values = [p/100.0 * abs_ratio * 100 for p in p_values]  # Convert to percentage
            # Include the actual ratio value in the label
            label = f"{self.pert_energies[i]:.2e}-{self.pert_energies[i+1]:.2e} MeV  ({abs_ratio:.2e})"
            
            ax.plot(p_values, y_values, label=label)
        
        # Add a horizontal line at 0
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        # Set plot style
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f"MT = {self.reaction} ({self.energy})", fontsize=14)
        
        ax.set_xlabel("Perturbation (%)", fontsize=12)
        ax.set_ylabel("Absolute Nonlinearity Factor (%)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add legend only if we have any top ratios
        if len(top_indices) > 0:
            legend_title = "All Energy Bins" if top_n == 0 else f"Top {len(top_indices)} Energy Bins"
            ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return ax


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
    :ivar ratios: Dictionary containing Taylor series ratios organized by energy and reaction number
    :type ratios: Dict[str, Dict[int, TaylorRatio]]
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
    ratios: Dict[str, Dict[int, TaylorRatio]] = field(default_factory=dict)
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

    def plot_sensitivity(self, energy: Union[str, List[str]] = None, 
             reaction: Union[List[int], int] = None, xlim: tuple = None):
        """Plot sensitivity coefficients for specified energies and reactions.

        :param energy: Energy string(s) to plot. If None, plots all energies
        :type energy: Union[str, List[str]], optional
        :param reaction: Reaction number(s) to plot. If None, plots all reactions
        :type reaction: Union[List[int], int], optional
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
        if reaction is None:
            # Get unique reactions from all energy data
            reaction = list(set().union(*[d.keys() for d in self.data.values()]))
            # Sort reactions in ascending numerical order
            reaction.sort()
        elif not isinstance(reaction, list):
            reaction = [reaction]

        # Create a separate figure for each energy
        for e in energies:
            coeffs_dict = self.data[e]
            n = len(reaction)
            
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
            
            for i, rxn in enumerate(reaction):
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

    def plot_ratio(self, energy: Union[str, List[str]] = None, reaction: Union[List[int], int] = None, top_n: int = 5):
        """Plot ratio of second-order to first-order sensitivity coefficients.
        
        The ratio is calculated as:
            R = (c2 * p) / c1
        
        Where:
            - c2 is the second-order coefficient
            - c1 is the first-order coefficient
            - p is the perturbation fraction

        :param energy: Energy string(s) to plot. If None, plots all energies
        :type energy: Union[str, List[str]], optional
        :param reaction: Reaction number(s) to plot. If None, plots all reactions
        :type reaction: Union[List[int], int], optional
        :param top_n: Number of top absolute ratios to plot with labels (0 means plot all)
        :type top_n: int, optional
        :raises ValueError: If sensitivity data does not contain Taylor ratios
        :raises ValueError: If specified energies are not found in the data
        """
        if not self.ratios:
            raise ValueError("Taylor ratios are required for ratio plots. Please recompute sensitivity with include_second_order=True.")
            
        # If no energy specified, use all energies
        if energy is None:
            energies = list(self.ratios.keys())
        else:
            # Ensure energy is always a list
            energies = [energy] if not isinstance(energy, list) else energy
            # Validate all energies exist in data
            invalid_energies = [e for e in energies if e not in self.ratios]
            if invalid_energies:
                raise ValueError(f"Energies {invalid_energies} not found in ratio data.")

        # Ensure reactions is always a list
        if reaction is None:
            # Get unique reactions from all energy data
            reaction = list(set().union(*[d.keys() for d in self.ratios.values()]))
            # Sort reactions in ascending numerical order
            reaction.sort()
        elif not isinstance(reaction, list):
            reaction = [reaction]

        # Create a separate figure for each energy
        for e in energies:
            rxn_dict = self.ratios[e]
            n = len(reaction)
            
            # Use a single Axes if only one reaction
            if n == 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                axes = [ax]
            else:
                cols = 1  # Changed from 2 to 1 to have only one figure per row
                rows = n  # Now rows equals the number of reactions
                fig, axes = plt.subplots(rows, cols, figsize=(10, 6 * rows))  # Adjust figure size
                # Ensure axes is a flat list of Axes objects
                if hasattr(axes, "flatten"):
                    axes = list(axes.flatten())
                else:
                    axes = [axes]
            
            # Energy group title
            if e == "integral":
                title_text = f"Order Ratio - Integral Result"
            else:
                # Parse the energy range from the string format
                try:
                    lower, upper = e.split('_')
                    title_text = f"Order Ratio - Energy Range: {lower} - {upper} MeV"
                except ValueError:
                    title_text = f"Order Ratio - Energy = {e}"
            
            fig.suptitle(title_text, y=1.01, fontsize=16)
            
            for i, rxn in enumerate(reaction):
                ax = axes[i]
                if rxn not in rxn_dict:
                    ax.text(0.5, 0.5, f"Reaction {rxn} not found", ha='center', va='center')
                    ax.axis('off')
                else:
                    # Get the TaylorRatio object and plot it
                    ratio_obj = rxn_dict[rxn]
                    ratio_obj.plot(ax=ax, title=f"MT = {rxn}", top_n=top_n)

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
        
    def __repr__(self):
        """Returns a formatted string representation of the sensitivity data.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :return: Formatted string representation of the sensitivity data
        :rtype: str
        """
        # Create a visually appealing header with a border
        header_width = 60
        header = "=" * header_width + "\n"
        header += f"{'Sensitivity Data for ' + self.nuclide:^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create aligned key-value pairs with consistent width
        label_width = 32  # Width for labels
        
        # Basic information
        info_lines = []
        info_lines.append(f"{'Label:':{label_width}} {self.label}")
        info_lines.append(f"{'Tally ID:':{label_width}} {self.tally_id}")
        
        if self.tally_name:
            info_lines.append(f"{'Tally Name:':{label_width}} {self.tally_name}")
        
        info_lines.append(f"{'Nuclide (ZAID):':{label_width}} {self.nuclide} ({self.zaid})")
        
        # Data overview
        num_energy_groups = len(self.energies)
        info_lines.append(f"{'Number of detector energy bins:':{label_width}} {num_energy_groups-1 if 'integral' in self.energies else num_energy_groups}")
        num_pert_bins = len(self.pert_energies) - 1
        info_lines.append(f"{'Number of perturbation bins:':{label_width}} {num_pert_bins}")
        info_lines.append(f"{'Reactions available:':{label_width}} {', '.join(map(str, self.reactions))}")
        
        # Add information about Taylor ratios if available
        if self.ratios:
            info_lines.append(f"{'Linearity ratios available:':{label_width}} Yes")
        else:
            info_lines.append(f"{'Linearity ratios available:':{label_width}} No")
        
        stats = "\n".join(info_lines)
        
        # Energy groups summary showing ALL energy-dependent results
        energy_info = "\n\nEnergy group ranges:\n"
        energy_groups = [e for e in self.energies if e != "integral"]
        if energy_groups:
            for e in energy_groups:
                energy_info += f"  - {e}\n"
        else:
            energy_info += "  No energy-dependent results available.\n"
        
        # Add integral entry at the end if it exists
        if "integral" in self.energies:
            energy_info += "  - integral\n"
        
        # Add an empty line before the footer with available methods
        footer = "\n\nAvailable methods:\n"
        footer += "- .plot_sensitivity(energy=None, reaction=None, xlim=None) - Plot sensitivity profiles\n"
        if self.ratios:
            footer += "- .plot_ratios(energy=None, reaction=None, p_range=None) - Plot Taylor ratio nonlinearity factors\n"
        footer += "- .to_dataframe() - Get full data as pandas DataFrame\n"
        
        # Add examples of accessing data
        examples = "\nExamples of accessing data:\n"
        examples += "- .data['0.00e+00_1.00e-01'][1] - Get coefficients for energy bin 0-0.1 MeV, reaction 1\n"
        if "integral" in self.energies:
            examples += "- .data['integral'][2] - Get integral coefficients for reaction 2\n"
        if self.ratios:
            energy_key = next(iter(self.ratios.keys()))
            rxn_key = next(iter(self.ratios[energy_key].keys()))
            examples += f"- .ratios['{energy_key}'][{rxn_key}] - Get Taylor ratio data for energy bin, reaction {rxn_key}\n"
        
        # Combine all sections
        return header + stats + energy_info + footer + examples


@dataclass
class Coefficients:
    """Container for sensitivity coefficients for a specific energy and reaction.

    :ivar energy: Energy range string in format "lower_upper" (e.g., "0.00e+00_1.00e-01")
    :type energy: str
    :ivar reaction: Reaction number
    :type reaction: int
    :ivar pert_energies: Perturbation energy boundaries
    :type pert_energies: List[float]
    :ivar values: Sensitivity coefficient values (first-order)
    :type values: List[float]
    :ivar errors: Relative errors for the sensitivity coefficients (first-order)
    :type errors: List[float]
    :ivar r0: Unperturbed tally result
    :type r0: float
    :ivar e0: Unperturbed tally error
    :type e0: float
    :ivar values_second: Second-order sensitivity coefficient values
    :type values_second: List[float], optional
    :ivar errors_second: Relative errors for the second-order sensitivity coefficients
    :type errors_second: List[float], optional
    """
    energy: str
    reaction: int
    pert_energies: list[float]
    values: list[float]
    errors: list[float]
    r0: float = None 
    e0: float = None
    values_second: list[float] = None
    errors_second: list[float] = None

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
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert coefficients data to a pandas DataFrame.
        
        :returns: DataFrame with columns:
            - energy: Energy range string (detector energy)
            - reaction: Reaction number (MT)
            - e_lower: Lower boundary of perturbation energy bin
            - e_upper: Upper boundary of perturbation energy bin
            - sensitivity: Sensitivity coefficient value
            - error: Relative error of the coefficient
        :rtype: pd.DataFrame
        """
        # Create a list to hold the data for each row
        data = []
        
        # Add each perturbation energy bin as a separate row
        for i in range(len(self.values)):
            data.append({
                'energy': self.energy,
                'reaction': self.reaction,
                'e_lower': self.pert_energies[i],
                'e_upper': self.pert_energies[i+1],
                'sensitivity': self.values[i],
                'error': self.errors[i]
            })
        
        # Create and return the DataFrame with the specified column order
        return pd.DataFrame(data, columns=[
            'energy', 'reaction', 'e_lower', 'e_upper', 'sensitivity', 'error'
        ])
    
    def __repr__(self):
        """Returns a formatted string representation of the coefficients.
        
        This method provides an informative overview of the coefficient data and available methods.
        
        :return: Formatted string representation of the coefficients
        :rtype: str
        """
        # Create a visually appealing header
        header_width = 50
        header = "=" * header_width + "\n"
        header += f"{'Sensitivity Coefficients':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Basic information section
        info_lines = []
        info_lines.append(f"Energy: {self.energy}")
        info_lines.append(f"Reaction Number (MT): {self.reaction}")
        info_lines.append(f"Number of perturbation bins: {len(self.pert_energies) - 1}")
        
        if self.r0 is not None:
            info_lines.append(f"Unperturbed result (R₀): {self.r0:.6e} ± {self.e0:.6e}")
        
        info = "\n".join(info_lines)
        
        # Data preview section - show first few and last few values
        n_preview = 3  # Number of values to show at beginning and end
        n_values = len(self.values)
        
        data_preview = "\n\nData preview (values and relative errors):\n\n"
        
        # Format as a small table
        data_preview += f"{'  Energy Bin':^19} | {'Value':^15} | {'  Rel. Error':^12}\n"
        data_preview += "-" * 46 + "\n"
        
        for i in range(min(n_preview, n_values)):
            e_low = f"{self.pert_energies[i]:.3e}"
            e_high = f"{self.pert_energies[i+1]:.3e}"
            data_preview += f"{e_low}-{e_high:^6} | {self.values[i]:15.6e} | {self.errors[i]:12.6f}\n"
        
        # Add ellipsis if there are more values than shown
        if n_values > 2 * n_preview:
            data_preview += "..." + " " * 43 + "\n"
            
            # Show last few values
            for i in range(max(n_preview, n_values - n_preview), n_values):
                e_low = f"{self.pert_energies[i]:.3e}"
                e_high = f"{self.pert_energies[i+1]:.3e}"
                data_preview += f"{e_low}-{e_high:^6} | {self.values[i]:15.6e} | {self.errors[i]:12.6f}\n"
        
        # Available methods section
        methods = "\n\nAvailable methods:\n"
        methods += "- .lethargy - Get lethargy intervals as property\n"
        methods += "- .values_per_lethargy - Get sensitivity per lethargy as property\n"
        methods += "- .plot(ax=None, xlim=None) - Plot sensitivity coefficients\n"
        methods += "- .to_dataframe() - Export data as pandas DataFrame with columns:\n"
        methods += "    energy, reaction, e_lower, e_upper, sensitivity, error\n"
        
        # Combine all sections
        return header + info + data_preview + methods
        
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