from dataclasses import dataclass
from typing import Optional, List, Dict
import matplotlib.pyplot as plt
import pandas as pd  # Add pandas import

@dataclass
class Mctal:
    """Container class for MCNP MCTAL file data.

    :ivar code_name: Name of the MCNP code version
    :type code_name: str
    :ivar ver: Version number of MCNP
    :type ver: str
    :ivar probid: Problem ID string
    :type probid: str
    :ivar knod: Code specific parameter
    :type knod: int
    :ivar nps: Number of particle histories
    :type nps: int
    :ivar rnr: Random number
    :type rnr: int
    :ivar problem_id: Problem identification line
    :type problem_id: str
    :ivar ntal: Number of tallies
    :type ntal: int
    :ivar npert: Number of perturbations
    :type npert: int
    :ivar tally_numbers: List of tally numbers
    :type tally_numbers: List[int]
    :ivar tally: Dictionary mapping tally numbers to Tally objects
    :type tally: Dict[int, Tally]
    """
    # Header information
    code_name: Optional[str] = None
    ver: Optional[str] = None
    probid: Optional[str] = None
    knod: Optional[int] = None
    nps: Optional[int] = None
    rnr: Optional[int] = None

    # Problem identification line
    problem_id: Optional[str] = None

    # Tally information
    ntal: Optional[int] = None  # Number of tallies
    npert: Optional[int] = 0    # Number of perturbations (optional)
    
    # Remove n and m as they were incorrectly used
    # Tally numbers list
    tally_numbers: List[int] = None

    # Change tally to be a dictionary
    tally: Dict[int, 'Tally'] = None

    def __post_init__(self):
        if self.tally_numbers is None:
            self.tally_numbers = []
        if self.tally is None:
            self.tally = {}  

    def __repr__(self):
        """Returns a formatted string representation of the MCTAL file data.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :return: Formatted string representation of the MCTAL file
        :rtype: str
        """
        # Create a visually appealing header with a border
        header_width = 60
        header = "=" * header_width + "\n"
        header += f"{'MCTAL File Summary':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create aligned key-value pairs with consistent width
        label_width = 20
        
        # General information section
        info_lines = []
        if self.code_name:
            info_lines.append(f"{'Code:':>{label_width}} {self.code_name} {self.ver}")
        if self.probid:
            info_lines.append(f"{'Problem ID:':>{label_width}} {self.probid}")
        if self.nps:
            info_lines.append(f"{'NPS:':>{label_width}} {self.nps:.2e}")
        if self.problem_id:
            info_lines.append(f"{'Title:':>{label_width}} {self.problem_id}")
        
        info_section = "\n".join(info_lines)
        
        # Tally type distribution
        tally_section = "\n\nTally Information:\n"
        tally_section += f"{'Total Tallies:':>{label_width}} {self.ntal}\n"
        
        if self.tally:
            # Count tallies by type (last digit)
            tally_type_counts = {}
            for tally_id in self.tally:
                # MCNP tally type is determined by the last digit of the tally number
                tally_type = f"F{tally_id % 10}"
                tally_type_counts[tally_type] = tally_type_counts.get(tally_type, 0) + 1
            
            if tally_type_counts:
                tally_section += "\nTally Type Distribution:\n"
                tally_section += "-" * 30 + "\n"
                tally_section += f"{'Tally Type':^15}|{'Count':^12}\n"
                tally_section += "-" * 30 + "\n"
                
                # Sort by tally type number
                for tally_type in sorted(tally_type_counts.keys()):
                    tally_section += f"{tally_type:^15}|{tally_type_counts[tally_type]:^12}\n"
                tally_section += "-" * 30 + "\n"
        
        # Perturbation summary
        pert_section = ""
        tallies_with_pert = [(tid, len(t.perturbation)) for tid, t in self.tally.items() if t.perturbation]
        if tallies_with_pert:
            pert_section = "\n\nPerturbation Data Summary:\n"
            pert_section += "-" * 35 + "\n"
            pert_section += f"{'Tally ID':^15}|{'Perturbations':^18}\n"
            pert_section += "-" * 35 + "\n"
            
            for tally_id, pert_count in sorted(tallies_with_pert):
                pert_section += f"{tally_id:^15}|{pert_count:^18}\n"
            
            pert_section += "-" * 35 + "\n"
            pert_section += f"Total: {sum(count for _, count in tallies_with_pert)} perturbations across {len(tallies_with_pert)} tallies\n"
        
        # Add information about methods
        footer = "\n\nAvailable methods:\n"
        footer += "- .tally[tally_id] - Access individual tallies\n"
        
        # Combine all sections
        return header + info_section + tally_section + pert_section + footer

class PerturbationCollection(dict):
    """A collection class for perturbation data that provides a nice summary representation.
    
    This class extends the standard dictionary with a custom __repr__ method
    to provide a formatted summary of the perturbation data.
    """
    
    def __repr__(self):
        """Returns a condensed formatted summary of the perturbation data.
        
        :return: Formatted string showing perturbation summary
        :rtype: str
        """
        if not self:
            return "No perturbation data available"
            
        # Get sorted perturbation numbers
        pert_nums = sorted(self.keys())
        
        # Find ranges of consecutive perturbation numbers
        ranges = []
        start = pert_nums[0]
        prev = pert_nums[0]
        
        for i in range(1, len(pert_nums)):
            # If there's a gap in sequence
            if pert_nums[i] > prev + 1:
                # Add the completed range
                if start == prev:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{prev}")
                # Start a new range
                start = pert_nums[i]
            prev = pert_nums[i]
        
        # Add the last range
        if start == prev:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{prev}")
        
        # Build the compact summary
        width = 60
        header = f"{'=' * width}\n"
        header += f"{'Perturbation Collection':^{width}}\n"
        header += f"{'=' * width}\n\n"
        
        # Summary line with total count
        summary = f"Total: {len(self)} perturbations\n"
        
        # Ranges information
        range_info = "Perturbation numbers: " + ", ".join(ranges) + "\n"
        
        # Access instructions
        instructions = "\nAccess:\n- perturbation[num] for individual perturbations\n"
        instructions += "- .to_dataframe() to convert all to DataFrame\n"
        
        return header + summary + range_info  + instructions
    
    def to_dataframe(self):
        """Converts all perturbation data to a pandas DataFrame.
        
        :return: DataFrame containing data from all perturbations
        :rtype: pandas.DataFrame
        """
        if not self:
            return pd.DataFrame()
        
        data = []
        for pert_num, pert_data in self.items():
            if hasattr(pert_data, 'to_dataframe'):
                df = pert_data.to_dataframe()
                df['Perturbation'] = pert_num  # Add the perturbation number
                data.append(df)
        
        if data:
            return pd.concat(data, ignore_index=True)
        else:
            return pd.DataFrame()

@dataclass
class Tally:
    """Container for MCNP tally data.

    :ivar tally_id: Unique identifier for the tally
    :type tally_id: int
    :ivar name: Name/description of the tally
    :type name: str
    :ivar energies: Energy bin boundaries
    :type energies: List[float]
    :ivar results: Tally results for each bin
    :type results: List[float]
    :ivar errors: Relative errors for each bin
    :type errors: List[float]
    :ivar integral_result: Integral result over all bins
    :type integral_result: float
    :ivar integral_error: Relative error of the integral result
    :type integral_error: float
    :ivar tfc_nps: Number of particles for TFC analysis
    :type tfc_nps: List[int]
    :ivar tfc_results: Results at each TFC step
    :type tfc_results: List[float]
    :ivar tfc_errors: Errors at each TFC step
    :type tfc_errors: List[float]
    :ivar tfc_fom: Figure of Merit at each TFC step
    :type tfc_fom: List[float]
    :ivar perturbation: Perturbation data keyed by perturbation index
    :type perturbation: PerturbationCollection
    """
    tally_id: int
    name: str = ""
    energies: List[float] = None
    results: List[float] = None
    errors: List[float] = None
    integral_result: Optional[float] = None
    integral_error: Optional[float] = None
    # TFC data (unperturbed)
    tfc_nps: List[int] = None
    tfc_results: List[float] = None
    tfc_errors: List[float] = None
    tfc_fom: List[float] = None
    # Perturbation data
    perturbation: PerturbationCollection = None
    
    def __post_init__(self):
        if self.energies is None:
            self.energies = []
        if self.results is None:
            self.results = []
        if self.errors is None:
            self.errors = []
        if self.tfc_nps is None:
            self.tfc_nps = []
        if self.tfc_results is None:
            self.tfc_results = []
        if self.tfc_errors is None:
            self.tfc_errors = []
        if self.tfc_fom is None:
            self.tfc_fom = []
        # Convert existing dictionary to PerturbationCollection or create a new one
        if self.perturbation is None:
            self.perturbation = PerturbationCollection()
        elif not isinstance(self.perturbation, PerturbationCollection):
            # Convert existing dict to PerturbationCollection
            self.perturbation = PerturbationCollection(self.perturbation)

    def __repr__(self):
        """Returns a formatted string representation of the tally.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :return: Formatted string representation of the tally
        :rtype: str
        """
        # Create a more visually appealing header with a border
        header_width = 60
        header = "=" * header_width + "\n"
        header += f"{'MCNP Tally ' + str(self.tally_id):^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create aligned key-value pairs with consistent width
        label_width = 25  # Width for labels
        
        info_lines = []
        if self.name:
            info_lines.append(f"{'Tally Name:':{label_width}} {self.name}")
        
        info_lines.append(f"{'Number of energy bins:':{label_width}} {len(self.energies)}")
        info_lines.append(f"{'Number of results:':{label_width}} {len(self.results)}")
        
        stats = "\n".join(info_lines)
        
        # Add an empty line between statistics and data table
        data_table = "\n\n"
        
        # Check if we have energy-dependent results
        if self.integral_result is not None:
            data_table += "  Energy (MeV)      Result          Rel. Error\n"
            data_table += "  ------------    ------------    ------------\n"
            
            # If we have energy bins, show all results
            if self.results and len(self.energies) > 0:
                for i in range(len(self.results)):
                    data_table += f"  {self.energies[i]:<15.6e} {self.results[i]:<15.6e} {self.errors[i]:<15.6e}\n"
                
                data_table += "  ------------------------------------------------\n"
            
            # Show the total row (either alone or after energy bins)
            data_table += f"  {'Total':<15} {self.integral_result:<15.6e} {self.integral_error:<15.6e}\n\n"
        
        # Additional information with aligned labels
        additional_info = []
        if self.perturbation:  # Update to use perturbation instead of pert_data
            additional_info.append(f"{'Perturbation data:':{label_width}} {len(self.perturbation)} perturbations")

        if self.tfc_nps:
            additional_info.append(f"{'TFC data available:':{label_width}} {len(self.tfc_nps)} points")
        
        additional = "\n".join(additional_info)
        
        # Add an empty line before the footer
        footer = "\n\nAvailable methods:\n"
        footer += "- .to_dataframe() - Get full data as DataFrame\n"
        footer += "- .plot_tfc_data() - Visualize convergence\n"
        if self.perturbation:  # Add note about accessing perturbations
            footer += "- .perturbation - Access perturbations\n"
        
        return header + stats + data_table + additional + footer

    def to_dataframe(self):
        """Converts tally data to a pandas DataFrame.
        
        Creates a DataFrame with three columns: 'Energy', 'Result', and 'Error',
        containing the energy bin values and corresponding results and errors.
        
        :return: DataFrame containing the tally data
        :rtype: pandas.DataFrame
        :raises ValueError: If data lengths are inconsistent
        """
        # Check that we have data to convert
        if not self.results:
            return pd.DataFrame(columns=['Energy', 'Result', 'Error'])
            
        # Create DataFrame
        df = pd.DataFrame({
            'Energy': self.energies,
            'Result': self.results,
            'Error': self.errors
        })
        
        return df

    def plot_tfc_data(self, figsize=(15, 5)):
        """Creates and displays plots showing TFC convergence data.

        This method creates a figure with three subplots showing the TFC convergence data:
        results vs NPS, relative errors vs NPS, and figure of merit vs NPS.
        The figure is displayed immediately.

        :param figsize: Figure size in inches as (width, height)
        :type figsize: tuple
        :raises ValueError: If no TFC data is available for plotting
        """
        if not self.tfc_nps:
            raise ValueError("No TFC data available for plotting")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot results
        ax1.plot(self.tfc_nps, self.tfc_results, 'b.-')
        ax1.set_xlabel('NPS')
        ax1.set_ylabel('Results')
        ax1.set_title('Result vs NPS')
        ax1.grid(True)
        
        # Plot errors
        ax2.plot(self.tfc_nps, self.tfc_errors, 'r.-')
        ax2.set_xlabel('NPS')
        ax2.set_ylabel('Relative Error')
        ax2.set_title('Error vs NPS')
        ax2.grid(True)
        
        # Plot FOM
        ax3.plot(self.tfc_nps, self.tfc_fom, 'g.-')
        ax3.set_xlabel('NPS')
        ax3.set_ylabel('Figure of Merit')
        ax3.set_title('FOM vs NPS')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

@dataclass
class TallyPert(Tally):
    """Container for perturbed tally data, inheriting from Tally.
    
    :ivar: Inherits all attributes from Tally class
    :ivar perturbation_number: The perturbation index number
    :type perturbation_number: int
    """
    perturbation_number: int = None
    
    def __post_init__(self):
        # Call parent post init to initialize lists/dict as needed.
        super().__post_init__()
        
    def __repr__(self):
        """Returns a concise string representation of the perturbed tally.
        
        :return: Formatted string representation of the perturbed tally
        :rtype: str
        """
        # Create a more visually appealing header with a border
        header_width = 60
        header = "=" * header_width + "\n"
        header += f"{'MCNP Tally ' + str(self.tally_id) + ' - Perturbation ' + str(self.perturbation_number):^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create aligned key-value pairs with consistent width
        label_width = 25  # Width for labels
        
        # Include only the most important information
        info_lines = []
        if self.name:
            info_lines.append(f"{'Tally Name:':{label_width}} {self.name}")
        
        info_lines.append(f"{'Number of energy bins:':{label_width}} {len(self.energies)}")
        info_lines.append(f"{'Number of results:':{label_width}} {len(self.results)}")
        
        stats = "\n".join(info_lines)
        
        # Simplified data preview - show only integral result if available
        data_preview = "\n\n"
        if self.integral_result is not None:
            data_preview += f"  {'Integral Result:':{label_width}} {self.integral_result:.6e}\n"
            data_preview += f"  {'Integral Error:':{label_width}} {self.integral_error:.6e}\n"
        
        # Clear footer with available methods
        footer = "\n\nAvailable methods:\n"
        footer += "- .to_dataframe() - Get full data as DataFrame\n"
        if self.tfc_nps:
            footer += "- .plot_tfc_data() - Visualize convergence\n"
        
        return header + stats + data_preview + footer
