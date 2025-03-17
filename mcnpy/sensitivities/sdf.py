from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import os
from mcnpy._constants import MT_TO_REACTION, ATOMIC_NUMBER_TO_SYMBOL
from mcnpy.sensitivities.sensitivity import SensitivityData
from mcnpy.utils.energy_grids import _identify_energy_grid


@dataclass
class SDFReactionData:
    """Container for sensitivity data for a specific nuclide and reaction.
    
    :ivar zaid: ZAID of the nuclide
    :type zaid: int
    :ivar mt: MT reaction number
    :type mt: int
    :ivar sensitivity: List of sensitivity coefficients
    :type sensitivity: List[float]
    :ivar error: List of relative errors
    :type error: List[float]
    :ivar nuclide: Nuclide symbol (calculated from ZAID)
    :type nuclide: str
    :ivar reaction_name: Reaction name (calculated from MT)
    :type reaction_name: str
    """
    zaid: int
    mt: int
    sensitivity: List[float]
    error: List[float]
    nuclide: str = field(init=False)
    reaction_name: str = field(init=False)
    
    def __post_init__(self):
        """Calculate and store nuclide symbol and reaction name after initialization."""
        # Calculate nuclide symbol
        z = self.zaid // 1000
        a = self.zaid % 1000
        
        if z not in ATOMIC_NUMBER_TO_SYMBOL:
            raise KeyError(f"Atomic number {z} not found in ATOMIC_NUMBER_TO_SYMBOL dictionary")
            
        self.nuclide = f"{ATOMIC_NUMBER_TO_SYMBOL[z]}-{a}"
        
        # Calculate reaction name
        if self.mt not in MT_TO_REACTION:
            raise KeyError(f"MT number {self.mt} not found in MT_TO_REACTION dictionary")
            
        self.reaction_name = MT_TO_REACTION[self.mt]
    
    def __repr__(self) -> str:
        """Returns a formatted string representation of the reaction data.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :return: Formatted string representation of the reaction data
        :rtype: str
        """
        # Create a visually appealing header with a border
        header_width = 60
        header = "=" * header_width + "\n"
        header += f"{'SDF Reaction Data':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create aligned key-value pairs with consistent width
        label_width = 25  # Width for labels
        
        info_lines = []
        info_lines.append(f"{'Nuclide:':{label_width}} {self.nuclide} (ZAID {self.zaid})")
        info_lines.append(f"{'Reaction:':{label_width}} {self.reaction_name} (MT {self.mt})")
        info_lines.append(f"{'Energy groups:':{label_width}} {len(self.sensitivity)}")
        
        stats = "\n".join(info_lines)
        
        # Data preview - show first few and last few sensitivity values
        data_preview = "\n\nSensitivity coefficients (preview):\n"
        data_preview += "  Group      Sensitivity       Rel. Error\n"
        data_preview += "  -----    --------------    ------------\n"
        
        # Show first 3 and last 3 groups, if available
        n_groups = len(self.sensitivity)
        preview_count = min(3, n_groups)
        
        for i in range(preview_count):
            data_preview += f"  {i+1:<5d}    {self.sensitivity[i]:14.6e}    {self.error[i]:12.6e}\n"
            
        # Add ellipsis if there are more than 6 groups
        if n_groups > 6:
            data_preview += "  ...\n"
            
        # Show last 3 groups if there are more than 3 groups
        if n_groups > 3:
            for i in range(max(preview_count, n_groups-3), n_groups):
                data_preview += f"  {i+1:<5d}    {self.sensitivity[i]:14.6e}    {self.error[i]:12.6e}\n"
        
        return header + stats + data_preview


@dataclass
class SDFData:
    """Container for SDF data.
    
    :ivar title: Title of the SDF dataset
    :type title: str
    :ivar energy: Energy value or label
    :type energy: str
    :ivar pert_energies: List of perturbation energy boundaries
    :type pert_energies: List[float]
    :ivar r0: Unperturbed tally result (reference response value)
    :type r0: float
    :ivar e0: Error of the unperturbed tally result
    :type e0: float
    :ivar data: List of reaction-specific sensitivity data
    :type data: List[SDFReactionData]
    """
    title: str
    energy: str
    pert_energies: List[float]
    r0: float = None
    e0: float = None
    data: List[SDFReactionData] = field(default_factory=list)

    def __repr__(self) -> str:
        """Returns a detailed formatted string representation of the SDF data.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :return: Formatted string representation of the SDF data
        :rtype: str
        """
        # Create a visually appealing header with a border
        header_width = 70
        header = "=" * header_width + "\n"
        header += f"{'SDF Data: ' + self.title:^{header_width}}\n"
        header += f"{'Energy range: ' + self.energy:^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Create aligned key-value pairs with consistent width
        label_width = 25  # Width for labels
        
        # Basic information section
        info_lines = []
        info_lines.append(f"{'Response value:':{label_width}} {self.r0:.6e} ± {self.e0*100:.2f}% (rel)")
        info_lines.append(f"{'Energy groups:':{label_width}} {len(self.pert_energies) - 1}")
        
        # Add energy grid structure identification
        grid_name = _identify_energy_grid(self.pert_energies)
        if grid_name:
            info_lines.append(f"{'Energy structure:':{label_width}} {grid_name}")
            
        info_lines.append(f"{'Sensitivity profiles:':{label_width}} {len(self.data)}")
        
        # Count unique nuclides
        nuclides = {react.nuclide for react in self.data}
        info_lines.append(f"{'Unique nuclides:':{label_width}} {len(nuclides)}")
        
        stats = "\n".join(info_lines)
        
        # Energy grid preview
        energy_preview = "\n\nEnergy grid (preview):"
        energy_grid = "  " + ", ".join(f"{e:.6e}" for e in self.pert_energies[:3])
        
        if len(self.pert_energies) > 6:
            energy_grid += ", ... , " 
            energy_grid += ", ".join(f"{e:.6e}" for e in self.pert_energies[-3:])
        elif len(self.pert_energies) > 3:
            energy_grid += ", " + ", ".join(f"{e:.6e}" for e in self.pert_energies[3:])
            
        energy_preview += "\n  " + energy_grid
        
        # Data summary - most important nuclides and reactions with indices
        data_summary = "\n\nNuclides and reactions (with access indices):\n"
        
        # Group by nuclide
        nuclide_reactions = {}
        nuclide_indices = {}
        
        # Store all reaction data with their indices
        for idx, react in enumerate(self.data):
            if react.nuclide not in nuclide_reactions:
                nuclide_reactions[react.nuclide] = []
                nuclide_indices[react.nuclide] = []
            nuclide_reactions[react.nuclide].append((react.reaction_name, react.mt))
            nuclide_indices[react.nuclide].append(idx)
        
        # Determine width for consistent alignment
        reaction_width = 30  # Base width for reaction name + MT
        
        # Show data for each nuclide (limit to first 5 nuclides)
        for i, nuclide in enumerate(sorted(nuclide_reactions.keys())):
            if i >= 5:
                data_summary += f"\n  ... ({len(nuclides) - 5} more nuclides) ...\n"
                break
                
            data_summary += f"\n  {nuclide}:\n"
            reactions = nuclide_reactions[nuclide]
            indices = nuclide_indices[nuclide]
            
            # Sort by MT number but keep track of original indices
            sorted_data = sorted(zip(reactions, indices), key=lambda x: x[0][1])
            
            for j, ((name, mt), idx) in enumerate(sorted_data):
                if j >= 10:  # Limit to 10 reactions per nuclide
                    data_summary += f"    ... ({len(reactions) - 10} more reactions) ...\n"
                    break
                # Format the reaction info with consistent alignment for the "access with" part
                reaction_info = f"{name} (MT={mt})"
                data_summary += f"    {reaction_info:{reaction_width}} access with .data[{idx}]\n"
        
        # Footer with available methods
        footer = "\n\nAvailable methods:\n"
        footer += "- .write_file() - Write SDF data to a file\n"
        footer += "- .group_inelastic_reactions() - Group MT 51-91 into MT 4\n"
        
        return header + stats + energy_preview + data_summary + footer

    def write_file(self, output_dir: Optional[str] = None):
        """
        Write the SDF data to a file using the legacy format.
        
        :param output_dir: Directory where the SDF file will be written. If None, uses current directory.
        :type output_dir: Optional[str]
        """
        # Use current directory if output_dir is not provided
        if output_dir is None:
            output_dir = os.getcwd()
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a clean filename from title and energy
        filename = f"{self.title}_{self.energy}.sdf"
        # Ensure filename is valid by removing problematic characters
        filename = filename.replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        # Create full path to file
        filepath = os.path.join(output_dir, filename)
        
        ngroups = len(self.pert_energies) - 1
        nprofiles = len(self.data)
        
        # Sort the data by ZAID and then by MT number
        sorted_data = sorted(self.data, key=lambda x: (x.zaid, x.mt))
        
        with open(filepath, 'w') as file:
            # Write header
            file.write(f"{self.title} MCNP to SCALE sdf {ngroups}gr\n")
            file.write(f"       {ngroups} number of neutron groups\n")
            file.write(f"       {nprofiles}  number of sensitivity profiles         {nprofiles} are region integrated\n")
            
            # Ensure r0 and e0 are properly formatted
            r0_value = 0.0 if self.r0 is None else self.r0
            e0_value = 0.0 if self.e0 is None else self.e0
            file.write(f"  {r0_value:.6E} +/-   {e0_value:.6E}\n")
            
            # Write energy grid data - reversed to be in descending order
            file.write("energy boundaries:\n")
            energy_lines = ""
            # Create reversed list of energies for writing in descending order
            descending_energies = list(reversed(self.pert_energies))
            for idx, energy in enumerate(descending_energies):
                if idx > 0 and idx % 5 == 0:
                    energy_lines += "\n"
                energy_lines += f"{energy: >14.6E}"
            energy_lines += "\n"
            file.write(energy_lines)
            
            # Write sensitivity coefficient and standard deviations data for each reaction
            # using the sorted data
            for reaction in sorted_data:
                file.write(self._format_reaction_data(reaction))
        
        # Add print message indicating where the file was saved
        print(f"SDF file saved successfully: {filepath}")

    def _format_reaction_data(self, reaction: SDFReactionData) -> str:
        """
        Format a single SDFReactionData block to match the legacy file structure.
        
        :param reaction: The reaction data to format
        :type reaction: SDFReactionData
        :returns: Formatted string for the reaction data block
        :rtype: str
        """
        # Use the properties to get the nuclide symbol and reaction name
        form = reaction.nuclide
        reac = reaction.reaction_name
        
        # Format the header line for this reaction
        block = f"{form:<13}{reac:<17}{reaction.zaid:>5}{reaction.mt:>7}\n"
        block += "      0      0\n"
        block += "  0.000000E+00  0.000000E+00      0      0\n"
        
        # Use 0.0 for total sensitivity instead of calculating it incorrectly
            
        block += f"  0.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00  0.000000E+00\n"
        
        # Reverse sensitivity and error arrays to match the descending energy order
        reversed_sensitivity = list(reversed(reaction.sensitivity))
        reversed_error = list(reversed(reaction.error))
        
        # Write sensitivity coefficients with 5 per line (in reversed order)
        for idx, sens in enumerate(reversed_sensitivity):
            if idx > 0 and idx % 5 == 0:
                block += "\n"
            block += f"{sens: >14.6E}"
        block += "\n"
        
        # Write standard deviations with 5 per line (in reversed order)
        for idx, err in enumerate(reversed_error):
            if idx > 0 and idx % 5 == 0:
                block += "\n"
            block += f"{err: >14.6E}"
        block += "\n"
        return block

    def group_inelastic_reactions(self, replace: bool = False, remove_originals: bool = True) -> None:
        """Group inelastic reactions (MT 51-91) into MT 4 for each nuclide.
        
        This method combines all inelastic scattering reactions (MT 51-91) into 
        the total inelastic scattering reaction (MT 4) for each nuclide.
        
        :param replace: If True, replace existing MT 4 data if present.
                        If False, raise an error when MT 4 is already present.
        :type replace: bool, optional
        :param remove_originals: If True, remove the original MT 51-91 reactions
                                after combining them.
        :type remove_originals: bool, optional
        :raises ValueError: If MT 4 already exists for a nuclide and replace=False
        """
        # Group data by ZAID
        nuclide_reactions = {}
        for react in self.data:
            if react.zaid not in nuclide_reactions:
                nuclide_reactions[react.zaid] = []
            nuclide_reactions[react.zaid].append(react)
        
        # Process each nuclide
        for zaid, reactions in nuclide_reactions.items():
            # Find MT 4 if it exists
            mt4_exists = False
            mt4_reaction = None
            for react in reactions:
                if react.mt == 4:
                    mt4_exists = True
                    mt4_reaction = react
                    break
            
            # Find inelastic reactions (MT 51-91)
            inelastic_reactions = [r for r in reactions if 51 <= r.mt <= 91]
            
            # Skip if no inelastic reactions found for this nuclide
            if not inelastic_reactions:
                continue
            
            # Handle existing MT 4 reaction
            if mt4_exists and not replace:
                # Calculate the nuclide symbol for more informative error message
                z = zaid // 1000
                a = zaid % 1000
                symbol = ATOMIC_NUMBER_TO_SYMBOL.get(z, f"unknown_{z}")
                nuclide = f"{symbol}-{a}"
                
                raise ValueError(
                    f"MT 4 already exists for nuclide {nuclide} (ZAID {zaid}). "
                    f"Set replace=True to overwrite."
                )
            
            # Sum sensitivity and error values from all inelastic reactions
            n_groups = len(inelastic_reactions[0].sensitivity)
            summed_sensitivity = [0.0] * n_groups
            summed_error_squared = [0.0] * n_groups  
            
            for react in inelastic_reactions:
                for i in range(n_groups):
                    summed_sensitivity[i] += react.sensitivity[i]
                    # Convert relative error to absolute error (multiply by sensitivity), then square
                    absolute_error = react.sensitivity[i] * react.error[i]
                    summed_error_squared[i] += absolute_error ** 2 
            
            # Take square root of summed squared errors and convert back to relative errors
            summed_error = []
            for i in range(n_groups):
                absolute_error = summed_error_squared[i] ** 0.5
                # Convert back to relative error (divide by sensitivity)
                # Handle potential division by zero
                if summed_sensitivity[i] != 0:
                    relative_error = absolute_error / abs(summed_sensitivity[i])
                else:
                    relative_error = 0.0
                summed_error.append(relative_error)
            
            # Create or update MT 4 reaction
            if mt4_exists:
                mt4_reaction.sensitivity = summed_sensitivity
                mt4_reaction.error = summed_error
                print(f"Updated MT 4 for {mt4_reaction.nuclide} (ZAID {zaid})")
            else:
                new_mt4 = SDFReactionData(
                    zaid=zaid,
                    mt=4,
                    sensitivity=summed_sensitivity,
                    error=summed_error
                )
                self.data.append(new_mt4)
                print(f"Created MT 4 for {new_mt4.nuclide} (ZAID {zaid})")
            
            # Remove original MT 51-91 reactions if requested
            if remove_originals:
                mt_values = [r.mt for r in inelastic_reactions]
                print(f"Removed MT {', '.join(map(str, mt_values))} for {inelastic_reactions[0].nuclide} (ZAID {zaid})")
                
                # Remove the reactions from self.data
                self.data = [r for r in self.data if not (r.zaid == zaid and 51 <= r.mt <= 91)]
