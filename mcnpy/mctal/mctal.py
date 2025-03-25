from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Add xarray import for multidimensional labeled data
try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

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
        header_width = 70
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
        
        # Tally information section
        tally_section = "\n\nTally Information:\n"
        tally_section += f"{'Total Tallies:':>{label_width}} {self.ntal}\n"
        
        if self.tally:
            # Group tallies by type (last digit)
            tally_types = {}
            for tally_id in self.tally:
                # MCNP tally type is determined by the last digit of the tally number
                tally_type = f"F{tally_id % 10}"
                if tally_type not in tally_types:
                    tally_types[tally_type] = []
                tally_types[tally_type].append(tally_id)
            
            if tally_types:
                tally_section += "\nTally Type Distribution:\n"
                tally_section += "-" * header_width + "\n"
                tally_section += f"{'Type':^10}|{'Description':^30}|{'Tally IDs':^{header_width-42}}\n"
                tally_section += "-" * header_width + "\n"
                
                # Descriptions of MCNP tally types
                descriptions = {
                    'F1': 'Surface current',
                    'F2': 'Surface flux',
                    'F4': 'Cell flux',
                    'F5': 'Point detector',
                    'F6': 'Energy deposition',
                    'F7': 'Fission energy deposition',
                    'F8': 'Pulse height'
                }
                
                # Sort by tally type number
                for tally_type in sorted(tally_types.keys()):
                    description = descriptions.get(tally_type, 'Other')
                    
                    # Format the tally IDs in a readable way
                    tally_ids = sorted(tally_types[tally_type])
                    
                    # Group consecutive tally IDs for more compact display
                    grouped_ids = []
                    start_id = tally_ids[0]
                    prev_id = tally_ids[0]
                    
                    for i in range(1, len(tally_ids)):
                        if tally_ids[i] > prev_id + 1:  # Not consecutive
                            if start_id == prev_id:
                                grouped_ids.append(str(start_id))
                            else:
                                grouped_ids.append(f"{start_id}-{prev_id}")
                            start_id = tally_ids[i]
                        prev_id = tally_ids[i]
                    
                    # Add the last group
                    if start_id == prev_id:
                        grouped_ids.append(str(start_id))
                    else:
                        grouped_ids.append(f"{start_id}-{prev_id}")
                    
                    # Format the grouped IDs as a comma-separated list
                    tally_id_text = ", ".join(grouped_ids)
                    
                    # Handle wrapping for long tally ID lists
                    id_width = header_width - 42
                    if len(tally_id_text) > id_width:
                        # Wrap the text
                        wrapped_ids = []
                        current_line = ""
                        for group in grouped_ids:
                            if len(current_line) + len(group) + 2 > id_width:  # +2 for ", "
                                wrapped_ids.append(current_line)
                                current_line = group
                            else:
                                if current_line:
                                    current_line += ", " + group
                                else:
                                    current_line = group
                        if current_line:
                            wrapped_ids.append(current_line)
                        
                        # First line goes in the table row
                        tally_section += f"{tally_type:^10}|{description:^30}|{wrapped_ids[0]:<{id_width}}\n"
                        
                        # Continuation lines
                        for line in wrapped_ids[1:]:
                            tally_section += f"{' ':^10}|{' ':^30}|{line:<{id_width}}\n"
                    else:
                        tally_section += f"{tally_type:^10}|{description:^30}|{tally_id_text:<{id_width}}\n"
                
                tally_section += "-" * header_width + "\n"
        
        # Perturbation summary
        pert_section = ""
        tallies_with_pert = [(tid, len(t.perturbation)) for tid, t in self.tally.items() if t.perturbation]
        if tallies_with_pert:
            pert_section = "\n\nPerturbation Data Summary:\n"
            pert_section += "-" * header_width + "\n"
            pert_section += f"{'Tally ID':^15}|{'Perturbations':^25}|{'Perturbation IDs':^{header_width-42}}\n"
            pert_section += "-" * header_width + "\n"
            
            for tally_id, pert_count in sorted(tallies_with_pert):
                # Get and format perturbation IDs
                pert_ids = sorted(self.tally[tally_id].perturbation.keys())
                
                # Group consecutive perturbation IDs
                grouped_ids = []
                if pert_ids:
                    start_id = pert_ids[0]
                    prev_id = pert_ids[0]
                    
                    for i in range(1, len(pert_ids)):
                        if pert_ids[i] > prev_id + 1:  # Not consecutive
                            if start_id == prev_id:
                                grouped_ids.append(str(start_id))
                            else:
                                grouped_ids.append(f"{start_id}-{prev_id}")
                            start_id = pert_ids[i]
                        prev_id = pert_ids[i]
                    
                    # Add the last group
                    if start_id == prev_id:
                        grouped_ids.append(str(start_id))
                    else:
                        grouped_ids.append(f"{start_id}-{prev_id}")
                
                pert_id_text = ", ".join(grouped_ids)
                
                # Handle wrapping for long perturbation ID lists
                id_width = header_width - 42
                if len(pert_id_text) > id_width:
                    # First line with tally ID and count
                    pert_section += f"{tally_id:^15}|{pert_count:^25}|{pert_id_text[:id_width]:<{id_width}}\n"
                    
                    # Continuation lines
                    remaining = pert_id_text[id_width:]
                    while remaining:
                        next_chunk = remaining[:id_width]
                        remaining = remaining[id_width:]
                        pert_section += f"{' ':^15}|{' ':^25}|{next_chunk:<{id_width}}\n"
                else:
                    pert_section += f"{tally_id:^15}|{pert_count:^25}|{pert_id_text:<{id_width}}\n"
            
            pert_section += "-" * header_width + "\n"
            pert_section += f"Total: {sum(count for _, count in tallies_with_pert)} perturbations across {len(tallies_with_pert)} tallies\n"
        
        # Add improved table-formatted information about methods
        methods_section = "\n\nAvailable Methods:\n"
        methods_section += "-" * header_width + "\n"
        
        # Set column widths for method and description
        method_col_width = 25
        desc_col_width = header_width - method_col_width - 3  # -3 for spacing and formatting
        
        methods_section += "{:<{width1}} {:<{width2}}\n".format(
            "Method", "Description", width1=method_col_width, width2=desc_col_width)
        methods_section += "-" * header_width + "\n"
        
        # Function to add a method and description with proper wrapping
        def add_method(method, description):
            nonlocal methods_section
            if len(description) <= desc_col_width:
                methods_section += "{:<{width1}} {:<{width2}}\n".format(
                    method, description, width1=method_col_width, width2=desc_col_width)
            else:
                # Handle wrapping for long descriptions
                methods_section += "{:<{width1}} {:<{width2}}\n".format(
                    method, description[:desc_col_width], width1=method_col_width, width2=desc_col_width)
                remaining = description[desc_col_width:]
                while remaining:
                    chunk = remaining[:desc_col_width].strip()
                    remaining = remaining[len(chunk):].strip()
                    methods_section += "{:<{width1}} {:<{width2}}\n".format(
                        "", chunk, width1=method_col_width, width2=desc_col_width)
        
        # Add each method with its description
        add_method(".tally[tally_id]", "Access individual tallies")
        
        methods_section += "-" * header_width + "\n"
        
        # Combine all sections
        return header + info_section + tally_section + pert_section + methods_section

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
    :ivar n_cells_surfaces: Number of cells or surfaces where the tally is scored
    :type n_cells_surfaces: int
    :ivar cell_surface_ids: List of cell or surface IDs
    :type cell_surface_ids: List[int]
    :ivar n_direct_bins: Number of direct vs. total or flagged vs. unflagged bins
    :type n_direct_bins: int
    :ivar n_user_bins: Number of user bins
    :type n_user_bins: int
    :ivar _has_total_user_bin: Whether user bins include a total bin (private)
    :type _has_total_user_bin: bool
    :ivar _has_cumulative_user_bin: Whether user bins are cumulative (private)
    :type _has_cumulative_user_bin: bool
    :ivar n_segment_bins: Number of segment bins
    :type n_segment_bins: int
    :ivar _has_total_segment_bin: Whether segment bins include a total bin (private)
    :type _has_total_segment_bin: bool
    :ivar _has_cumulative_segment_bin: Whether segment bins are cumulative (private)
    :type _has_cumulative_segment_bin: bool
    :ivar n_multiplier_bins: Number of multiplier bins
    :type n_multiplier_bins: int
    :ivar _has_total_multiplier_bin: Whether multiplier bins include a total bin (private)
    :type _has_total_multiplier_bin: bool
    :ivar _has_cumulative_multiplier_bin: Whether multiplier bins are cumulative (private)
    :type _has_cumulative_multiplier_bin: bool
    :ivar n_cosine_bins: Number of cosine bins
    :type n_cosine_bins: int
    :ivar _has_total_cosine_bin: Whether cosine bins include a total bin (private)
    :type _has_total_cosine_bin: bool
    :ivar _has_cumulative_cosine_bin: Whether cosine bins are cumulative (private)
    :type _has_cumulative_cosine_bin: bool
    :ivar n_energy_bins: Number of energy bins
    :type n_energy_bins: int
    :ivar _has_total_energy_bin: Whether energy bins include a total bin (private)
    :type _has_total_energy_bin: bool
    :ivar _has_cumulative_energy_bin: Whether energy bins are cumulative (private)
    :type _has_cumulative_energy_bin: bool
    :ivar energies: Energy bin boundaries
    :type energies: List[float]
    :ivar n_time_bins: Number of time bins
    :type n_time_bins: int
    :ivar _has_total_time_bin: Whether time bins include a total bin (private)
    :type _has_total_time_bin: bool
    :ivar _has_cumulative_time_bin: Whether time bins are cumulative (private)
    :type _has_cumulative_time_bin: bool
    :ivar times: Time bin boundaries
    :type times: List[float]
    :ivar total_energy_result: Result for the total energy bin (if present)
    :type total_energy_result: float
    :ivar total_energy_error: Error for the total energy bin (if present)
    :type total_energy_error: float
    :ivar results: Tally results for each bin (excluding totals)
    :type results: List[float]
    :ivar errors: Relative errors for each bin (excluding totals)
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
    # Cell/surface attributes
    n_cells_surfaces: int = 0
    cell_surface_ids: List[int] = None
    # Direct/total bins
    n_direct_bins: int = 1
    # User bin attributes
    n_user_bins: int = 0
    _has_total_user_bin: bool = False
    _has_cumulative_user_bin: bool = False
    # Segment bin attributes
    n_segment_bins: int = 0
    _has_total_segment_bin: bool = False
    _has_cumulative_segment_bin: bool = False
    # Multiplier bin attributes
    n_multiplier_bins: int = 0
    _has_total_multiplier_bin: bool = False
    _has_cumulative_multiplier_bin: bool = False
    # Cosine bin attributes
    n_cosine_bins: int = 0
    _has_total_cosine_bin: bool = False
    _has_cumulative_cosine_bin: bool = False
    # Energy bin attributes
    n_energy_bins: int = 0
    _has_total_energy_bin: bool = False
    _has_cumulative_energy_bin: bool = False
    energies: List[float] = None
    # Time bin attributes
    n_time_bins: int = 0
    _has_total_time_bin: bool = False
    _has_cumulative_time_bin: bool = False
    times: List[float] = None
    # Separate storage for total energy bin
    total_energy_result: Optional[float] = None
    total_energy_error: Optional[float] = None
    # Result data
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
        if self.cell_surface_ids is None:
            self.cell_surface_ids = []
        if self.times is None:
            self.times = []
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
        # Create a visually appealing header with a border
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'MCNP Tally ' + str(self.tally_id):^{header_width}}\n"
        if self.name:
            header += f"{self.name:^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create simplified section for dimensions with improved formatting
        dimensions = self.get_dimensions()
        dim_section = "Dimensions:\n"
        dim_section += "-" * header_width + "\n"
        
        # Simplified table with just Dimension and Size columns
        dim_table = "{:<20} {:<15}\n".format("Dimension", "Size")
        dim_section += dim_table
        dim_section += "-" * header_width + "\n"
        
        if dimensions:
            # Add row for each dimension
            if 'cell' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("Cell/Surface", dimensions['cell'])
            if 'user' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("User", dimensions['user'])
            if 'segment' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("Segment", dimensions['segment'])
            if 'multiplier' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("Multiplier", dimensions['multiplier'])
            if 'cosine' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("Cosine", dimensions['cosine'])
            if 'energy' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("Energy", dimensions['energy'])
            if 'time' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("Time", dimensions['time'])
        else:
            dim_section += "{:<20} {:<15}\n".format("None", "0")
        
        dim_section += "-" * header_width + "\n"
        dim_section += f"Total Results: {len(self.results)}\n\n"
        
        # Additional information with aligned labels and bullet points
        additional_info = []
        if self._has_total_energy_bin and self.total_energy_result is not None:
            additional_info.append("• Integral energy data available")
        if self.perturbation:
            additional_info.append(f"• {len(self.perturbation)} perturbations")
        if self.tfc_nps:
            additional_info.append(f"• {len(self.tfc_nps)} TFC data points")
        
        additional = ""
        if additional_info:
            additional = "\nAdditional Information:\n"
            additional += "\n".join(additional_info)
            additional += "\n\n"
        
        # Available methods in improved table format with better wrapping
        methods_section = "\nAvailable Methods:\n"
        methods_section += "-" * header_width + "\n"
        
        # Improved method table with better spacing
        method_col_width = 33
        desc_col_width = header_width - method_col_width - 3  # -3 for spacing and formatting
        
        methods_section += "{:<{width1}} {:<{width2}}\n".format(
            "Method", "Description", width1=method_col_width, width2=desc_col_width)
        methods_section += "-" * header_width + "\n"
        
        # Function to add a method and description with proper wrapping
        def add_method(method, description):
            nonlocal methods_section
            if len(description) <= desc_col_width:
                methods_section += "{:<{width1}} {:<{width2}}\n".format(
                    method, description, width1=method_col_width, width2=desc_col_width)
            else:
                # Handle wrapping for long descriptions
                methods_section += "{:<{width1}} {:<{width2}}\n".format(
                    method, description[:desc_col_width], width1=method_col_width, width2=desc_col_width)
                remaining = description[desc_col_width:]
                while remaining:
                    chunk = remaining[:desc_col_width].strip()
                    remaining = remaining[len(chunk):].strip()
                    methods_section += "{:<{width1}} {:<{width2}}\n".format(
                        "", chunk, width1=method_col_width, width2=desc_col_width)
        
        # Add each method with its description
        add_method(".to_dataframe()", "Get results as pandas DataFrame")
        if XARRAY_AVAILABLE:
            add_method(".to_xarray()", "Get results as labeled multidimensional dataset")
            add_method(".get_slice(...)", "Extract specific slices of the data")
        add_method(".get_integral_energy_data()", "Get energy-integrated data as dictionary")
        add_method(".get_integral_energy_dataframe()", "Get energy-integrated data as DataFrame")
        add_method(".get_dimensions()", "Get dictionary of tally dimensions and sizes")
        add_method(".plot_tfc_data()", "Visualize convergence with plots")
        
        if self.perturbation:
            add_method(".perturbation", "Access perturbation collection")
        
        methods_section += "-" * header_width + "\n"
        
        return header + dim_section + additional + methods_section

    def _is_multidimensional(self) -> bool:
        """Internal method to determine if the tally has multiple dimensions.
        
        :return: True if tally has multiple non-trivial dimensions, False otherwise
        :rtype: bool
        """
        dimension_count = 0
        if self.n_cells_surfaces > 0:
            dimension_count += 1
        if self.n_user_bins > 0:
            dimension_count += 1
        if self.n_segment_bins > 0:
            dimension_count += 1
        if self.n_multiplier_bins > 0:
            dimension_count += 1
        if self.n_cosine_bins > 0:
            dimension_count += 1
        if self.n_energy_bins > 0:
            dimension_count += 1
        if self.n_time_bins > 0:
            dimension_count += 1
        return dimension_count > 1

    def get_dimensions(self) -> dict:
        """Get all dimensions of the tally with their sizes.
        
        :return: Dictionary with dimension names as keys and their sizes as values,
                 in order: cell, user, segment, multiplier, cosine, energy, time
        :rtype: dict
        """
        dimensions = {}
        if self.n_cells_surfaces > 0:
            dimensions['cell'] = self.n_cells_surfaces
        if self.n_user_bins > 0:
            dimensions['user'] = self.n_user_bins
        if self.n_segment_bins > 0:
            dimensions['segment'] = self.n_segment_bins
        if self.n_multiplier_bins > 0:
            dimensions['multiplier'] = self.n_multiplier_bins
        if self.n_cosine_bins > 0:
            dimensions['cosine'] = self.n_cosine_bins
        if self.n_energy_bins > 0:
            energy_bin_count = self.n_energy_bins
            if self._has_total_energy_bin:
                energy_bin_count -= 1
            dimensions['energy'] = energy_bin_count
        if self.n_time_bins > 0:
            dimensions['time'] = self.n_time_bins
        return dimensions

    def get_shaped_results(self) -> np.ndarray:
        """Reshape the flat results array into a multidimensional array.
        
        :return: Multidimensional array of results
        :rtype: numpy.ndarray
        """
        dimensions = self.get_dimensions()
        dims = list(dimensions.values())
        
        if not dims:
            return np.array(self.results)
            
        # Ensure the product of dimensions matches the result count
        expected_size = np.prod(dims)
        if len(self.results) != expected_size:
            # If sizes don't match, provide detailed error message
            raise ValueError(
                f"Cannot reshape results array of size {len(self.results)} into shape {tuple(dims)}. "
                f"Expected size: {expected_size}. Dimensions: {dimensions}"
            )
            
        return np.array(self.results).reshape(dims)

    def get_shaped_errors(self) -> np.ndarray:
        """Reshape the flat errors array into a multidimensional array.
        
        :return: Multidimensional array of errors
        :rtype: numpy.ndarray
        """
        dimensions = self.get_dimensions()
        dims = list(dimensions.values())
        
        if not dims:
            return np.array(self.errors)
            
        # Ensure the product of dimensions matches the error count
        expected_size = np.prod(dims)
        if len(self.errors) != expected_size:
            # If sizes don't match, provide detailed error message
            raise ValueError(
                f"Cannot reshape errors array of size {len(self.errors)} into shape {tuple(dims)}. "
                f"Expected size: {expected_size}. Dimensions: {dimensions}"
            )
            
        return np.array(self.errors).reshape(dims)

    def _debug_dimensions(self):
        """Debug helper to print dimension information.
        
        :return: Detailed dimension information
        :rtype: str
        """
        dimensions = self.get_dimensions()
        dims = list(dimensions.values())
        dim_names = list(dimensions.keys())
        
        info = []
        info.append(f"Results array size: {len(self.results)}")
        info.append(f"Errors array size: {len(self.errors)}")
        info.append(f"Dimension names: {dim_names}")
        info.append(f"Dimension sizes: {dims}")
        
        if dims:
            expected_size = np.prod(dims)
            info.append(f"Expected size from dimensions: {expected_size}")
            
            if expected_size != len(self.results):
                info.append("WARNING: Size mismatch between results and calculated dimensions!")
                
                # Add more detailed debug info
                info.append("\nDetailed dimension information:")
                if self.n_cells_surfaces > 0:
                    info.append(f"  cells/surfaces: {self.n_cells_surfaces}")
                if self.n_user_bins > 0:
                    info.append(f"  user bins: {self.n_user_bins} (has_total: {self._has_total_user_bin})")
                if self.n_segment_bins > 0:
                    info.append(f"  segment bins: {self.n_segment_bins} (has_total: {self._has_total_segment_bin})")
                if self.n_multiplier_bins > 0:
                    info.append(f"  multiplier bins: {self.n_multiplier_bins} (has_total: {self._has_total_multiplier_bin})")
                if self.n_cosine_bins > 0:
                    info.append(f"  cosine bins: {self.n_cosine_bins} (has_total: {self._has_total_cosine_bin})")
                if self.n_energy_bins > 0:
                    info.append(f"  energy bins: {self.n_energy_bins} (has_total: {self._has_total_energy_bin})")
                    info.append(f"  energy bin boundaries: {len(self.energies)}")
                    if self._has_total_energy_bin:
                        info.append(f"  total energy result: {self.total_energy_result is not None}")
                if self.n_time_bins > 0:
                    info.append(f"  time bins: {self.n_time_bins} (has_total: {self._has_total_time_bin})")
                
                # Add raw data size check
                if self._has_total_energy_bin:
                    n_energy_bins = self.n_energy_bins
                    expected_total_vals = expected_size // (n_energy_bins - 1) if n_energy_bins > 1 else 0
                    info.append(f"  expected total energy values: ~{expected_total_vals}")
        
        return "\n".join(info)

    def to_xarray(self):
        """Convert tally data to an xarray Dataset with labeled dimensions.
        
        :return: Dataset containing tally results and errors with labeled dimensions
        :rtype: xarray.Dataset
            
        :note: This method does not include energy-integrated data. Use get_integral_energy_data()
               to access energy-integrated results.
        """
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray is required for this method. Install with 'pip install xarray'")
        
        try:
            # Define dimension names and coordinates
            dimensions = self.get_dimensions()
            dims = list(dimensions.keys())
            coords = {}
            
            if 'cell' in dimensions:
                coords['cell'] = self.cell_surface_ids
            
            if 'user' in dimensions:
                # For user bins, we don't have specific coordinates, so use indices
                coords['user'] = list(range(dimensions['user']))
            
            if 'segment' in dimensions:
                coords['segment'] = list(range(dimensions['segment']))
            
            if 'multiplier' in dimensions:
                coords['multiplier'] = list(range(dimensions['multiplier']))
            
            if 'cosine' in dimensions:
                coords['cosine'] = list(range(dimensions['cosine']))
            
            if 'energy' in dimensions:
                # For energy dimension, use the actual energy values if available
                energy_bin_count = dimensions['energy']
                
                # Use energy values or indices as coordinates
                if len(self.energies) > 0:
                    coords['energy'] = self.energies
                else:
                    coords['energy'] = list(range(energy_bin_count))
            
            if 'time' in dimensions:
                # For time bins, use the actual time values if available
                if self.times:
                    coords['time'] = self.times
                else:
                    # Otherwise use indices
                    coords['time'] = list(range(dimensions['time']))
            
            # Get the shaped results and errors - these should now exclude total bins
            shaped_results = self.get_shaped_results()
            shaped_errors = self.get_shaped_errors()
            
            # Create the initial dataset with regular results
            if not dims:  # No dimensions
                data_vars = {
                    'result': ([], self.results[0] if self.results else np.nan),
                    'error': ([], self.errors[0] if self.errors else np.nan)
                }
            else:
                data_vars = {
                    'result': (dims, shaped_results),
                    'error': (dims, shaped_errors)
                }
            
            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            return ds
            
        except Exception as e:
            # Include detailed debug information in the error message
            error_msg = f"Failed to create xarray dataset: {str(e)}\n\n"
            error_msg += self._debug_dimensions()
            raise ValueError(error_msg)

    def get_slice(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a slice of results and errors by specifying dimension values.
        
        :param kwargs: Dimension name and value pairs. For dimensions with explicit 
                       coordinates (like energy or time), use the coordinate value.
                       For dimensions without explicit values, use the index.
        :type kwargs: dict
        :return: Tuple of (results, errors) arrays for the specified slice
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        
        :example:
            # Select by actual energy value (MeV)
            tally.get_slice(energy=1.0)  
            
            # Select by segment index
            tally.get_slice(segment=1)
            
            # Combine multiple dimensions
            tally.get_slice(energy=1.0, segment=1)
        """
        if XARRAY_AVAILABLE:
            # Use xarray for easier slicing with named dimensions
            ds = self.to_xarray()
            try:
                sliced = ds.sel(**kwargs)
                return sliced.result.values, sliced.error.values
            except Exception as e:
                raise ValueError(f"Failed to get slice: {str(e)}")
        else:
            # Fallback to numpy slicing
            results = self.get_shaped_results()
            errors = self.get_shaped_errors()
            
            dimensions = self.get_dimensions()
            dim_names = list(dimensions.keys())
            if not dim_names:
                return results, errors
            
            # Create slice objects for each dimension
            slices = [slice(None)] * len(dim_names)
            
            for dim_name, idx in kwargs.items():
                if dim_name in dim_names:
                    dim_idx = dim_names.index(dim_name)
                    slices[dim_idx] = idx
                else:
                    raise ValueError(f"Unknown dimension: {dim_name}. Available dimensions: {dim_names}")
            
            return results[tuple(slices)], errors[tuple(slices)]

    def to_dataframe(self):
        """Converts tally data to a pandas DataFrame.
        
        For simple tallies with only energy dependence, creates a DataFrame with
        'Energy', 'Result', and 'Error' columns. For multidimensional tallies,
        creates a flattened DataFrame with columns for each dimension.
        
        :return: DataFrame containing the tally data
        :rtype: pandas.DataFrame
            
        :note: This method does not include energy-integrated data. Use get_integral_energy_dataframe()
               to access energy-integrated results.
        """
        try:
            # For simple energy-only tallies with a straightforward structure
            if (self.n_cells_surfaces <= 1 and
                self.n_user_bins <= 1 and
                self.n_segment_bins <= 1 and
                self.n_multiplier_bins <= 1 and
                self.n_cosine_bins <= 1 and
                self.n_time_bins <= 1 and
                len(self.energies) > 0):
                
                # Create DataFrame with energy bins and their results/errors
                df = pd.DataFrame({
                    'Energy': self.energies,
                    'Result': self.results[:len(self.energies)],
                    'Error': self.errors[:len(self.energies)]
                })
                return df
            
            # For multidimensional tallies, try using xarray
            if XARRAY_AVAILABLE:
                try:
                    # Convert to xarray first (which handles dimension reshaping)
                    ds = self.to_xarray()
                    df = ds.to_dataframe().reset_index()
                    
                    # Convert all column names to title case
                    df.columns = [col.title() if col != 'cell' else 'Cell' for col in df.columns]
                    return df
                except Exception as e:
                    if "Failed to create xarray dataset" not in str(e):
                        # Log the error and fall back to the direct method
                        print(f"Warning: xarray conversion failed - {str(e)}")
            
            # Direct DataFrame creation (fallback approach)
            # For flat results (no dimensions)
            dimensions = self.get_dimensions()
            dim_names = list(dimensions.keys())
            if not dim_names:
                df = pd.DataFrame({
                    'Result': self.results,
                    'Error': self.errors
                })
                return df
            
            # Create a DataFrame directly from the numpy arrays
            # First get the shaped results - this should already exclude total bins
            results = self.get_shaped_results()
            errors = self.get_shaped_errors()
            
            # Build a DataFrame with one row per cell in the array
            rows = []
            for idx in np.ndindex(results.shape):
                row = {}
                # Add dimension coordinates
                for i, name in enumerate(dim_names):
                    if name == 'energy' and i < len(idx) and idx[i] < len(self.energies):
                        # Use actual energy value
                        row[name.title()] = self.energies[idx[i]]
                    elif name == 'time' and i < len(idx) and idx[i] < len(self.times):
                        # Use actual time value
                        row[name.title()] = self.times[idx[i]]
                    else:
                        # Use index for other dimensions
                        row[name.title()] = idx[i]
                
                # Add the result and error values
                row['Result'] = results[idx]
                row['Error'] = errors[idx]
                rows.append(row)
            
            return pd.DataFrame(rows)
            
        except Exception as e:
            # Create a detailed error message with debug info
            error_msg = f"Failed to convert to DataFrame: {str(e)}\n\n"
            error_msg += self._debug_dimensions()
            raise ValueError(error_msg)
    
    def get_integral_energy_data(self):
        """Get the results integrated over all energy bins.
        
        For multidimensional tallies, returns shaped arrays for all dimensions except energy.
        
        :return: Dictionary with 'Result' and 'Error' keys containing the energy-integrated data.
                 For multidimensional tallies, these will be numpy arrays.
        :rtype: dict or None
        """
        if not self._has_total_energy_bin or self.total_energy_result is None:
            return None
        
        # Get all dimensions except energy
        dimensions = self.get_dimensions()
        if 'energy' in dimensions:
            dimensions.pop('energy')
        
        # If no other dimensions or just a scalar, return simple dict
        if not dimensions:
            return {
                'Result': self.total_energy_result,
                'Error': self.total_energy_error
            }
        
        # For multidimensional tallies, extract and shape all total energy values
        dims = list(dimensions.values())
        
        # If total_energy_result is already a numpy array of the right shape, use it directly
        if isinstance(self.total_energy_result, np.ndarray) and self.total_energy_result.shape == tuple(dims):
            return {
                'Result': self.total_energy_result,
                'Error': self.total_energy_error
            }
        
        # If we have total_values attribute, use that
        if hasattr(self, 'total_energy_values') and self.total_energy_values is not None:
            results = self.total_energy_values['results']
            errors = self.total_energy_values['errors']
            
            # Check if we need to reshape
            expected_size = np.prod(dims)
            if len(results) == expected_size:
                shaped_results = np.array(results).reshape(dims)
                shaped_errors = np.array(errors).reshape(dims)
                
                return {
                    'Result': shaped_results,
                    'Error': shaped_errors
                }
        
        # Fallback: if we just have a scalar, but need a multidimensional array,
        # we'll create a constant array with the scalar value
        # This is a fallback that won't give accurate results for truly multidimensional tallies
        expected_size = np.prod(dims)
        result_array = np.full(dims, self.total_energy_result)
        error_array = np.full(dims, self.total_energy_error)
        
        return {
            'Result': result_array,
            'Error': error_array
        }
    
    def get_integral_energy_dataframe(self):
        """Get the energy-integrated results as a DataFrame.
        
        For multidimensional tallies, includes all dimensions except energy.
        
        :return: DataFrame containing the energy-integrated data
        :rtype: pandas.DataFrame
        """
        data = self.get_integral_energy_data()
        if data is None:
            return pd.DataFrame()
        
        # Simple case - just one result/error pair
        if np.isscalar(data['Result']):
            return pd.DataFrame({
                'Energy': ['Integral'],
                'Result': [data['Result']],
                'Error': [data['Error']]
            })
        
        # For multidimensional data, create a flattened DataFrame
        dimensions = self.get_dimensions()
        if 'energy' in dimensions:
            dimensions.pop('energy')
        
        # If no other dimensions, just return the simple case
        if not dimensions:
            return pd.DataFrame({
                'Energy': ['Integral'],
                'Result': [data['Result'].item() if isinstance(data['Result'], np.ndarray) else data['Result']],
                'Error': [data['Error'].item() if isinstance(data['Error'], np.ndarray) else data['Error']]
            })
        
        # Get the multidimensional arrays
        results = data['Result']
        errors = data['Error']
        
        # Create a DataFrame with one row per cell in the arrays
        dim_names = list(dimensions.keys())
        rows = []
        
        # For each cell in the multidimensional array
        for idx in np.ndindex(results.shape):
            row = {}
            # Add dimension coordinates
            for i, name in enumerate(dim_names):
                if name == 'cell' and self.cell_surface_ids and i < len(idx) and idx[i] < len(self.cell_surface_ids):
                    # Use actual cell/surface ID
                    row[name.title()] = self.cell_surface_ids[idx[i]]
                elif name == 'time' and self.times and i < len(idx) and idx[i] < len(self.times):
                    # Use actual time value
                    row[name.title()] = self.times[idx[i]]
                else:
                    # Use index for other dimensions
                    row[name.title()] = idx[i]
            
            # Add the result and error values
            row['Result'] = results[idx]
            row['Error'] = errors[idx]
            # Add indicator that this is energy-integrated data
            row['Energy'] = 'Integral'
            rows.append(row)
        
        return pd.DataFrame(rows)

    def plot_tfc_data(self, figsize=(15, 5), show_error_bars=True):
        """Creates and displays plots showing TFC convergence data.

        This method creates a figure with three subplots showing the TFC convergence data:
        results vs NPS (with optional error bars), relative errors vs NPS, and figure of merit vs NPS.
        The figure is displayed immediately.

        :param figsize: Figure size in inches as (width, height)
        :type figsize: tuple
        :param show_error_bars: Whether to display error bars on the results plot
        :type show_error_bars: bool
        :raises ValueError: If no TFC data is available for plotting
        :return: None
        """
        if not self.tfc_nps:
            raise ValueError("No TFC data available for plotting")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot results with or without error bars
        y_values = np.array(self.tfc_results)
        
        if show_error_bars:
            y_errors = np.array(self.tfc_errors) * np.abs(y_values)  # Convert relative errors to absolute
            ax1.errorbar(self.tfc_nps, y_values, yerr=y_errors, fmt='b.-', capsize=3, ecolor='lightblue')
            title_suffix = " (with error bars)"
        else:
            ax1.plot(self.tfc_nps, y_values, 'b.-')
            title_suffix = ""
            
        ax1.set_xlabel('NPS')
        ax1.set_ylabel('Results')
        ax1.set_title(f'Result vs NPS{title_suffix}')
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
        header_width = 70
        header = "=" * header_width + "\n"
        header += f"{'MCNP Tally ' + str(self.tally_id) + ' - Perturbation ' + str(self.perturbation_number):^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create simplified section for dimensions
        dimensions = self.get_dimensions()
        dim_section = "Dimensions:\n"
        dim_section += "-" * header_width + "\n"
        
        # Simplified table with just Dimension and Size columns
        dim_table = "{:<20} {:<15}\n".format("Dimension", "Size")
        dim_section += dim_table
        dim_section += "-" * header_width + "\n"
        
        if dimensions:
            # Add row for each dimension
            if 'cell' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("Cell/Surface", dimensions['cell'])
            if 'user' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("User", dimensions['user'])
            if 'segment' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("Segment", dimensions['segment'])
            if 'multiplier' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("Multiplier", dimensions['multiplier'])
            if 'cosine' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("Cosine", dimensions['cosine'])
            if 'energy' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("Energy", dimensions['energy'])
            if 'time' in dimensions:
                dim_section += "{:<20} {:<15}\n".format("Time", dimensions['time'])
        else:
            dim_section += "{:<20} {:<15}\n".format("None", "0")
        
        dim_section += "-" * header_width + "\n"
        dim_section += f"Total Results: {len(self.results)}\n\n"
        
        # Additional information with bullet points
        additional_info = []
        if self.integral_result is not None:
            additional_info.append(f"• Integral Result: {self.integral_result:.6e}")
            additional_info.append(f"• Integral Error: {self.integral_error:.6e}")
        if self.tfc_nps:
            additional_info.append(f"• {len(self.tfc_nps)} TFC data points")
        
        additional = ""
        if additional_info:
            additional = "\nAdditional Information:\n"
            additional += "\n".join(additional_info)
            additional += "\n\n"
        
        # Available methods in improved table format
        methods_section = "\nAvailable Methods:\n"
        methods_section += "-" * header_width + "\n"
        
        # Improved method table with better spacing
        method_col_width = 25
        desc_col_width = header_width - method_col_width - 3  # -3 for spacing and formatting
        
        methods_section += "{:<{width1}} {:<{width2}}\n".format(
            "Method", "Description", width1=method_col_width, width2=desc_col_width)
        methods_section += "-" * header_width + "\n"
        
        # Function to add a method and description with proper wrapping
        def add_method(method, description):
            nonlocal methods_section
            if len(description) <= desc_col_width:
                methods_section += "{:<{width1}} {:<{width2}}\n".format(
                    method, description, width1=method_col_width, width2=desc_col_width)
            else:
                # Handle wrapping for long descriptions
                methods_section += "{:<{width1}} {:<{width2}}\n".format(
                    method, description[:desc_col_width], width1=method_col_width, width2=desc_col_width)
                remaining = description[desc_col_width:]
                while remaining:
                    chunk = remaining[:desc_col_width].strip()
                    remaining = remaining[len(chunk):].strip()
                    methods_section += "{:<{width1}} {:<{width2}}\n".format(
                        "", chunk, width1=method_col_width, width2=desc_col_width)
        
        # Add each method with its description
        add_method(".to_dataframe()", "Get results as pandas DataFrame")
        if self.tfc_nps:
            add_method(".plot_tfc_data()", "Visualize convergence with plots")
        
        methods_section += "-" * header_width + "\n"
        
        return header + dim_section + additional + methods_section
