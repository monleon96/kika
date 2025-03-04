from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import os
from mcnpy._constants import MT_TO_REACTION, ATOMIC_NUMBER_TO_SYMBOL
from mcnpy.sensitivities.sensitivity import SensitivityData


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
    """
    zaid: int
    mt: int
    sensitivity: List[float]
    error: List[float]
    
    @property
    def nuclide(self) -> str:
        """Get the nuclide symbol.
        
        :returns: Nuclide symbol
        :rtype: str
        :raises KeyError: If the atomic number is not found in ATOMIC_NUMBER_TO_SYMBOL
        """
        z = self.zaid // 1000
        a = self.zaid % 1000
        
        if z not in ATOMIC_NUMBER_TO_SYMBOL:
            raise KeyError(f"Atomic number {z} not found in ATOMIC_NUMBER_TO_SYMBOL dictionary")
            
        return f"{ATOMIC_NUMBER_TO_SYMBOL[z]}-{a}"
    
    @property
    def reaction_name(self) -> str:
        """Get the reaction name.
        
        :returns: Reaction name
        :rtype: str
        :raises KeyError: If the MT number is not found in MT_TO_REACTION
        """
        if self.mt not in MT_TO_REACTION:
            raise KeyError(f"MT number {self.mt} not found in MT_TO_REACTION dictionary")
            
        return MT_TO_REACTION[self.mt]


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

    def __str__(self) -> str:
        """Provide a concise human-readable summary of the SDFData.
        
        :returns: Simple summary of SDFData contents
        :rtype: str
        """
        ngroups = len(self.pert_energies) - 1
        nprofiles = len(self.data)
        
        # Count unique nuclides and organize by nuclide
        nuclide_reactions = {}
        for react in self.data:
            if react.nuclide not in nuclide_reactions:
                nuclide_reactions[react.nuclide] = []
            nuclide_reactions[react.nuclide].append((react.reaction_name, react.mt))
        
        # Build basic summary
        summary = [
            f"SDF: {self.title} ({self.energy})",
            f"Response: {self.r0:.4E} ± {self.e0:.4E}",
            f"Energy groups: {ngroups}, Profiles: {nprofiles}",
            f"Nuclides: {len(nuclide_reactions)}",
        ]
        
        # List nuclides and their reactions
        for nuclide, reactions in nuclide_reactions.items():
            reaction_str = ", ".join([f"{name}(MT={mt})" for name, mt in reactions])
            summary.append(f"  {nuclide}: {reaction_str}")
        
        return "\n".join(summary)

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
            
            # Write energy grid data
            file.write("energy boundaries:\n")
            energy_lines = ""
            for idx, energy in enumerate(self.pert_energies):
                if idx > 0 and idx % 5 == 0:
                    energy_lines += "\n"
                energy_lines += f"{energy: >14.6E}"
            energy_lines += "\n"
            file.write(energy_lines)
            
            # Write sensitivity coefficient and standard deviations data for each reaction
            # using the sorted data
            for reaction in sorted_data:
                file.write(self._format_reaction_data(reaction))

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
        
        # Calculate total sensitivity and error - proper error propagation
        total_sens = sum(reaction.sensitivity)
        
        # Convert relative errors to absolute errors, square them, sum them, and take the square root
        absolute_errors = [sens * err for sens, err in zip(reaction.sensitivity, reaction.error)]
        total_err = (sum(err**2 for err in absolute_errors))**0.5
            
        block += f"{total_sens: >14.6E}{total_err: >14.6E}  0.000000E+00  0.000000E+00  0.000000E+00\n"
        
        # Write sensitivity coefficients with 5 per line
        for idx, sens in enumerate(reaction.sensitivity):
            if idx > 0 and idx % 5 == 0:
                block += "\n"
            block += f"{sens: >14.6E}"
        block += "\n"
        
        # Write standard deviations with 5 per line
        for idx, err in enumerate(reaction.error):
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

def create_sdf_data(
    sensitivity_data_list: Union[List[SensitivityData], List[Tuple[SensitivityData, List[int]]]], 
    energy: str,
    title: str,
    response_values: Tuple[float, float] = None
    ) -> SDFData:
    """Create a SDFData object from a list of SensitivityData objects.
    
    :param sensitivity_data_list: List of SensitivityData objects or tuples of (SensitivityData, reactions_list)
    :type sensitivity_data_list: Union[List[SensitivityData], List[Tuple[SensitivityData, List[int]]]]
    :param energy: Energy value to use for sensitivity data
    :type energy: str
    :param title: Title for the SDF dataset
    :type title: str
    :param response_values: Optional tuple of (r0, e0) to override values from sensitivity data.
                           r0 is the unperturbed tally result (reference response value),
                           e0 is the error of the unperturbed tally result.
    :type response_values: Tuple[float, float], optional
    :returns: SDFData object containing the combined sensitivity data
    :rtype: SDFData
    :raises ValueError: If pert_energies don't match across sensitivity data objects
    :raises ValueError: If r0 and e0 values don't match across sensitivity data objects
    """
    # Check if we have a list of SensitivityData objects or tuples
    has_tuples = any(isinstance(item, tuple) for item in sensitivity_data_list)
    
    # Extract SensitivityData objects and reaction filters
    sens_data = []
    reaction_filters = []
    
    if has_tuples:
        for item in sensitivity_data_list:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Expected tuple of (SensitivityData, List[int])")
            sens_obj, reactions = item
            sens_data.append(sens_obj)
            reaction_filters.append(reactions)
    else:
        sens_data = sensitivity_data_list
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