import os
from .parse_input import read_mcnp, _read_material
from mcnpy._constants import MCNPY_HEADER, MCNPY_FOOTER, ATOMIC_MASS, N_AVOGADRO


def perturb_material(inputfile, material_number, density, nuclide, pert_mat_id=None, in_place=True, format=None):
    """Creates a perturbed material with 100% increase in the specified nuclide's fraction.
    
    Reads an MCNP input file, finds the specified material, and creates a new perturbed
    material with a 100% increase in the fraction of the specified nuclide. The new material
    is added to the input file right after the original material definition and saved to
    a new file.
    
    The function can handle materials defined with either atomic or weight fractions.
    The perturbed material will always be written in normalized atomic fractions, 
    regardless of how the original material was defined.
    
    :param inputfile: Path to the MCNP input file
    :type inputfile: str
    :param material_number: Material ID number to be perturbed
    :type material_number: int
    :param density: Density of the original material. If positive, interpreted as atoms/barn-cm,
                   if negative, interpreted as g/cm³
    :type density: float
    :param nuclide: ZAID of the nuclide to be perturbed
    :type nuclide: int
    :param pert_mat_id: Optional ID for the perturbed material. If None, uses material_number*100 + 1
    :type pert_mat_id: Optional[int]
    :param in_place: Whether to write the perturbed material to the original input file or a new file
    :type in_place: bool, optional
    :param format: Format to use when printing the perturbed material composition.
                         Can be 'atomic', 'weight' or None. If None, use the same format as original material.
    :type format: str, optional
    
    :returns: None
    
    :raises ValueError: If the material or nuclide is not found in the input file
    """
    # Parse the input file
    input_data = read_mcnp(inputfile)
    
    # Check if the material exists
    if material_number not in input_data.materials.mat:
        raise ValueError(f"Material {material_number} not found in input file")
    
    original_material = input_data.materials.mat[material_number]
    
    # Check if the nuclide exists in the material
    if nuclide not in original_material.nuclide:
        raise ValueError(f"Nuclide {nuclide} not found in material {material_number}")
    
    # Create a new material ID (original ID + 01 or user specified)
    new_material_id = pert_mat_id if pert_mat_id is not None else material_number * 100 + 1
    perturbed_material = original_material.copy(new_material_id)
    
    # Determine if original material uses weight fractions
    has_weight_fractions = any(nuclide.fraction < 0 for nuclide in perturbed_material.nuclide.values())
    
    # Ensure the material is in atomic fractions format for perturbation
    if has_weight_fractions:
        perturbed_material.to_atomic_fraction()
    
    # Calculate the sum of all fractions in the perturbed material (should be 1.0 if already normalized)
    total_fraction = sum(nuclide.fraction for nuclide in perturbed_material.nuclide.values())
    
    # Normalize the perturbed material composition to start with a normalized composition
    if abs(total_fraction - 1.0) > 1e-6:  # Check if normalization is needed
        normalization_factor = 1.0 / total_fraction
        for zaid in perturbed_material.nuclide:
            perturbed_material.nuclide[zaid].fraction *= normalization_factor
    
    # Now apply 100% perturbation to the specified nuclide (after normalization)
    perturbed_material.nuclide[nuclide].fraction *= 2.0
    
    # Calculate the sum of all fractions after perturbation
    new_total = sum(nuclide.fraction for nuclide in perturbed_material.nuclide.values())
    
    # Determine if density is in atoms/barn-cm or g/cm³
    is_atomic_density = density >= 0
    abs_density = abs(density)
    
    # Calculate average atomic mass for the material
    avg_atomic_mass = 0.0
    for zaid, nuclide_obj in original_material.nuclide.items():
        # Ensure we're working with positive fractions for the calculation
        fraction = abs(nuclide_obj.fraction)
        if has_weight_fractions:
            # Convert weight fraction to atomic fraction for density calculations
            atomic_number = zaid // 1000
            mass_number = zaid % 1000
            isotope_key = atomic_number * 1000 + mass_number
            atomic_mass = ATOMIC_MASS.get(isotope_key, float(mass_number) if mass_number > 0 else 1.0)
            fraction = fraction / atomic_mass  # Convert weight to atomic
            
        # Normalize to get proper weighting
        fraction = fraction / total_fraction
        
        # Get atomic mass
        if zaid in ATOMIC_MASS:
            atomic_mass = ATOMIC_MASS[zaid]
        else:
            # Approximate mass if not found in the dictionary
            atomic_number = zaid // 1000
            mass_number = zaid % 1000
            atomic_mass = float(mass_number) if mass_number > 0 else 1.0
            print(f"WARNING: Atomic mass not found for nuclide {zaid}. Using mass number {mass_number} as an approximation.")
        
        avg_atomic_mass += fraction * atomic_mass
    
    # Convert between atomic density and mass density
    if is_atomic_density:
        # Input is atoms/barn-cm
        atomic_density = abs_density
        # Convert to g/cm³: (atoms/barn-cm) * avg_atomic_mass / N_AVOGADRO * 1e24
        # 1e24 factor: 1 barn = 1e-24 cm²
        mass_density = atomic_density * avg_atomic_mass / N_AVOGADRO * 1e24
    else:
        # Input is g/cm³
        mass_density = abs_density
        # Convert to atoms/barn-cm: (g/cm³) * N_AVOGADRO / avg_atomic_mass * 1e-24
        atomic_density = mass_density * N_AVOGADRO / avg_atomic_mass * 1e-24
    
    # Calculate new densities after perturbation
    new_atomic_density = atomic_density * new_total
    
    # Recalculate average atomic mass for perturbed material
    new_avg_atomic_mass = 0.0
    for zaid, nuclide_obj in perturbed_material.nuclide.items():
        fraction = nuclide_obj.fraction  # Already normalized
        if zaid in ATOMIC_MASS:
            atomic_mass = ATOMIC_MASS[zaid]
        else:
            atomic_number = zaid // 1000
            mass_number = zaid % 1000
            atomic_mass = float(mass_number)
        
        new_avg_atomic_mass += fraction * atomic_mass
    
    # Calculate new mass density
    new_mass_density = new_atomic_density * new_avg_atomic_mass / N_AVOGADRO * 1e24
    
    # Re-normalize the perturbed material to maintain sum = 1.0
    renormalization_factor = 1.0 / new_total
    for zaid in perturbed_material.nuclide:
        perturbed_material.nuclide[zaid].fraction *= renormalization_factor
    
    # Determine output format based on parameter or original material format
    if format is None:
        # Use the same format as the original material
        if has_weight_fractions and not any(nuclide.fraction < 0 for nuclide in perturbed_material.nuclide.values()):
            # Original was in weight fractions but perturbed is now in atomic, so convert back
            perturbed_material.to_weight_fraction()
    elif format.lower() == 'weight':
        # Convert to weight fractions if not already
        if not any(nuclide.fraction < 0 for nuclide in perturbed_material.nuclide.values()):
            perturbed_material.to_weight_fraction()
    elif format.lower() == 'atomic':
        # Ensure using atomic fractions (already done during perturbation)
        pass
    else:
        print(f"WARNING: Unrecognized format '{format}'. Using atomic fractions.")
    
    # Create the original material string using __str__
    original_material_str = original_material.__str__()
    
    # Generate comment and perturbed material string using __str__
    comment = f"c Perturbed material with 100% increase in nuclide {nuclide}\n"
    perturbed_material_str = perturbed_material.__str__()
    
    # Generate density information for comments
    if is_atomic_density:
        density_str = f"c Density: {atomic_density:.6e} atoms/barn-cm | {mass_density:.6e} g/cm³\n"
        new_density_str = f"c Density: {new_atomic_density:.6e} atoms/barn-cm | {new_mass_density:.6e} g/cm³\n"
    else:
        density_str = f"c Density: {mass_density:.6e} g/cm³ | {atomic_density:.6e} atoms/barn-cm\n"
        new_density_str = f"c Density: {new_mass_density:.6e} g/cm³ | {new_atomic_density:.6e} atoms/barn-cm\n"
    
    # Generate separators and headers
    header_orig = f"c Original material being perturbed - rewritten by MCNPy\n{density_str}"
    header_pert = f"c Perturbed material generated by MCNPy (normalized)\n{new_density_str}"
    
    # Add format information to the header
    if format is None:
        format_str = "same as original"
    else:
        format_str = format.lower()
    header_pert += f"c Output format: {format_str} fractions\n"
    
    # Determine output file path
    if in_place:
        # Read the original file content
        with open(inputfile, 'r') as f:
            lines = f.readlines()
        
        # Find the position of the original material definition
        original_position = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(f"m{material_number} ") or line.strip() == f"m{material_number}":
                original_position = i
                break

        if original_position == -1:
            raise ValueError(f"Could not locate material {material_number} in input file")

        # Use _read_material to determine the end of the material block
        _, next_position = _read_material(lines, original_position)
        
        # Remove the original material lines from the file
        del lines[original_position:next_position]
        
        # Insert the original material and perturbed material at the original position
        lines.insert(original_position, MCNPY_HEADER)
        lines.insert(original_position + 1, "c \n")  # Add blank comment line after header
        lines.insert(original_position + 2, header_orig)
        lines.insert(original_position + 3, original_material_str + "\n")
        lines.insert(original_position + 4, "c \n")
        lines.insert(original_position + 5, header_pert)
        lines.insert(original_position + 6, comment)
        lines.insert(original_position + 7, perturbed_material_str + "\n")
        lines.insert(original_position + 8, "c \n")
        lines.insert(original_position + 9, MCNPY_FOOTER)
        
        final_path = inputfile
        
        # Write the modified content to the file
        with open(final_path, 'w') as f:
            f.writelines(lines)
    else:
        # Create the perturbed material file in the same directory as the inputfile
        input_dir = os.path.dirname(inputfile) or "."
        base_name = os.path.basename(inputfile)
        filename, ext = os.path.splitext(base_name)
        new_filename = f"{filename}_perturbed_m{material_number}_n{nuclide}{ext}"
        final_path = os.path.join(input_dir, new_filename)
        
        # Generate content for the new file (only the MCNPy part)
        content_to_write = []
        
        # Write header
        content_to_write.append(MCNPY_HEADER)
        content_to_write.append("c \n")  # Add blank comment line after header
        
        # Write original material info
        content_to_write.append(header_orig)
        content_to_write.append(original_material_str + "\n")
        content_to_write.append("c \n")
        
        # Write perturbed material info
        content_to_write.append(header_pert)
        content_to_write.append(comment)
        content_to_write.append(perturbed_material_str + "\n")
        content_to_write.append("c \n")
        
        # Write footer
        content_to_write.append(MCNPY_FOOTER)
        
        # Write the content to the new file
        with open(final_path, 'w') as f:
            f.writelines(content_to_write)
    
    # Print perturbation information before writing
    print(f"Perturbation details:")
    print(f"- Original material: {material_number}")
    print(f"- Perturbed material ID: {new_material_id}")
    print(f"- Perturbed nuclide: {nuclide}")
    
    if is_atomic_density:
        print(f"- Original density: {atomic_density:.6e} atoms/barn-cm | {mass_density:.6e} g/cm³")
        print(f"- Perturbed density: {new_atomic_density:.6e} atoms/barn-cm | {new_mass_density:.6e} g/cm³")
    else:
        print(f"- Original density: {mass_density:.6e} g/cm³ | {atomic_density:.6e} atoms/barn-cm")
        print(f"- Perturbed density: {new_mass_density:.6e} g/cm³ | {new_atomic_density:.6e} atoms/barn-cm")
    
    print(f"\nSuccess! Material written to: {final_path}")
    
    return

def generate_PERTcards(inputfile, cell, density, reactions, energies, material, order=2, errors=False, in_place=True):
    """Generates PERT cards for MCNP input files.

    Generates PERT cards based on the provided parameters. Can generate both first and
    second order perturbations, as well as cards for exact uncertainty calculations.
    Note that exact uncertainties are usually negligible, so verify their necessity
    before running long calculations.
    
    :param inputfile: Path to the MCNP input file
    :type inputfile: str
    :param cell: Cell number(s) for PERT card application
    :type cell: int or str or list[int]
    :param density: Density value for the perturbation
    :type density: float
    :param reactions: List of reaction identifiers
    :type reactions: list[str]
    :param energies: Energy values in eV units. Must be in ascending order. Used in consecutive pairs for energy bins
    :type energies: list[float]
    :param material: Material identifier for perturbation
    :type material: str or int
    :param order: Order of PERT card method (1 or 2), defaults to 2
    :type order: int, optional
    :param errors: Whether to include error methods (-2, -3, 1), defaults to False
    :type errors: bool, optional
    :param in_place: Whether to append PERT cards to original input file, defaults to True
    :type in_place: bool, optional
    
    :returns: None
    :raises ValueError: If energies are not in ascending order
    
    :note: Writes PERT cards with MCNPY header and footer to either the original file or a new file
    """
    # Validate that energies are in ascending order
    for i in range(len(energies)-1):
        if energies[i] >= energies[i+1]:
            raise ValueError(f"Energy values must be in ascending order. Found {energies[i]} >= {energies[i+1]} at positions {i} and {i+1}")
    
    # Determine output file path
    if in_place:
        # Read the original file content to ensure we start on a new line
        with open(inputfile, 'r') as f:
            content = f.read()
        
        # Append to the original input file
        output_file = inputfile
        mode = "a"  # Append mode
        
        # Check if file ends with a newline, if not add one
        needs_newline = not content.endswith('\n')
    else:
        # Create a new file in the same directory
        input_dir = os.path.dirname(inputfile) or "."
        base_name = os.path.basename(inputfile)
        filename, ext = os.path.splitext(base_name)
        new_filename = f"{filename}_pert_cards{ext}"
        output_file = os.path.join(input_dir, new_filename)
        mode = "w"  # Write mode
        needs_newline = False
    
    # Generate all the PERT card content in memory
    content_to_write = []
    
    # First add a newline if needed
    if needs_newline:
        content_to_write.append("\n")
    
    # Format cell parameter to string
    if isinstance(cell, list):
        cell_str = ','.join(map(str, cell))
    else:
        cell_str = str(cell)

    # Initialize the perturbation counter
    pert_counter = 1
    
    # Write header - always include it
    content_to_write.append(MCNPY_HEADER)
    content_to_write.append("c \n")
    
    # Loop over each combination of cell, density, and reaction
    for reaction in reactions:
        # Go through the energy list and use consecutive pairs
        for i in range(len(energies) - 1):
            E1 = energies[i]
            E2 = energies[i + 1]

            # Create properly formatted METHOD=2 PERT card
            pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material} RHO={density:.6e} METHOD=2 RXN={reaction} ERG={E1:.6e} {E2:.6e}"
            content_to_write.append(_format_mcnp_line(pert_card) + "\n")
            pert_counter += 1

            if order == 2:
                # Create properly formatted METHOD=3 PERT card
                pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material} RHO={density:.6e} METHOD=3 RXN={reaction} ERG={E1:.6e} {E2:.6e}"
                content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                pert_counter += 1
        
            if errors:
                # Create properly formatted METHOD=-2 PERT card
                pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material} RHO={density:.6e} METHOD=-2 RXN={reaction} ERG={E1:.6e} {E2:.6e}"
                content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                pert_counter += 1

                if order == 2:
                    # Create properly formatted METHOD=-3 PERT card
                    pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material} RHO={density:.6e} METHOD=-3 RXN={reaction} ERG={E1:.6e} {E2:.6e}"
                    content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                    pert_counter += 1

                    # Create properly formatted METHOD=1 PERT card
                    pert_card = f"PERT{pert_counter}:n CELL={cell_str} MAT={material} RHO={density:.6e} METHOD=1 RXN={reaction} ERG={E1:.6e} {E2:.6e}"
                    content_to_write.append(_format_mcnp_line(pert_card) + "\n")
                    pert_counter += 1
    
    # Write footer - always include it
    content_to_write.append("c \n")
    content_to_write.append(MCNPY_FOOTER)
    
    # Write all content at once
    with open(output_file, mode) as stream:
        stream.writelines(content_to_write)
    
    print(f"\nSuccess! PERT cards written to: {output_file}")
    
    return


def _format_mcnp_line(line, max_length=80):
    """Helper function to format MCNP input lines to stay under the character limit.
    
    :param line: The full line to format
    :type line: str
    :param max_length: Maximum line length (default: 80)
    :type max_length: int
    
    :return: Formatted string with proper line breaks
    :rtype: str
    """
    if len(line) <= max_length:
        return line
    
    result = []
    remaining = line.strip()
    
    while remaining:
        # If this is not the first line, add 5 spaces for indentation
        indent = 5 if result else 0
        available = max_length - indent
        
        if len(remaining) <= available:
            # The remaining content fits in one line
            if indent:
                result.append(" " * indent + remaining)
            else:
                result.append(remaining)
            break
        
        # Find a good splitting point
        split_pos = available
        
        # Try to find a space to split on
        while split_pos > 0 and remaining[split_pos] != " ":
            split_pos -= 1
        
        if split_pos == 0:
            # No good space found, force split at available length
            split_pos = available
        
        # Add the current line with a continuation character
        if indent:
            result.append(" " * indent + remaining[:split_pos].rstrip() + " &")
        else:
            result.append(remaining[:split_pos].rstrip() + " &")
        
        # Process the remaining part
        remaining = remaining[split_pos:].strip()
    
    return "\n".join(result)