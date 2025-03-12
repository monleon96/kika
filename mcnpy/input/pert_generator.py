import sys
import os
from mcnpy._grids import ENERGY_GRIDS
from .parse_input import read_mcnp, _read_material
from mcnpy._constants import MCNPY_HEADER, MCNPY_FOOTER, ATOMIC_MASS, N_AVOGADRO


def perturb_material(inputfile, material_number, density, nuclide, output_path=None, pert_mat_id=None):
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
                   if negative, interpreted as g/cm³ (absolute value is used)
    :type density: float
    :param nuclide: ZAID of the nuclide to be perturbed
    :type nuclide: int
    :param output_path: Optional path where to save the modified file. If None, rewrites original file
    :type output_path: Optional[str]
    :param pert_mat_id: Optional ID for the perturbed material. If None, uses material_number*100 + 1
    :type pert_mat_id: Optional[int]
    
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
    if nuclide not in original_material.nuclides:
        raise ValueError(f"Nuclide {nuclide} not found in material {material_number}")
    
    # Create a new material ID (original ID + 01 or user specified)
    new_material_id = pert_mat_id if pert_mat_id is not None else material_number * 100 + 1
    
    # Create a copy of the original material with the new ID using the copy method
    perturbed_material = original_material.copy(new_material_id)
    
    # Calculate the sum of all fractions in the original material
    total_fraction = sum(nuclide.fraction for nuclide in original_material.nuclides.values())
    
    # Normalize the perturbed material composition to start with a normalized composition
    if abs(total_fraction - 1.0) > 1e-6:  # Check if normalization is needed
        normalization_factor = 1.0 / total_fraction
        for zaid in perturbed_material.nuclides:
            perturbed_material.nuclides[zaid].fraction *= normalization_factor
    
    # Now apply 100% perturbation to the specified nuclide (after normalization)
    perturbed_material.nuclides[nuclide].fraction *= 2.0
    
    # Calculate the sum of all fractions after perturbation
    new_total = sum(nuclide.fraction for nuclide in perturbed_material.nuclides.values())
    
    # Determine if density is in atoms/barn-cm or g/cm³
    is_atomic_density = density >= 0
    abs_density = abs(density)
    
    # Calculate average atomic mass for the material
    avg_atomic_mass = 0.0
    for zaid, nuclide_obj in original_material.nuclides.items():
        fraction = nuclide_obj.fraction / total_fraction  # Normalize to get proper weighting
        if zaid in ATOMIC_MASS:
            atomic_mass = ATOMIC_MASS[zaid]
        else:
            # Approximate mass if not found in the dictionary
            atomic_number = zaid // 1000
            mass_number = zaid % 1000
            atomic_mass = float(mass_number)
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
    for zaid, nuclide_obj in perturbed_material.nuclides.items():
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
    for zaid in perturbed_material.nuclides:
        perturbed_material.nuclides[zaid].fraction *= renormalization_factor
    
    # Read the original input file content
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
    
    # Determine output file path
    if output_path is None:
        final_path = inputfile
    else:
        base_name = os.path.basename(inputfile)
        filename, ext = os.path.splitext(base_name)
        new_filename = f"{filename}_pert_{material_number}_{nuclide}{ext}"
        final_path = os.path.join(output_path, new_filename)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
    
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
    
    # Write the modified content to the file
    with open(final_path, 'w') as f:
        f.writelines(lines)
    
    print(f"\nSuccess! Material written to: {final_path}")
    
    return


def generate_PERTcards(cell, density, reactions, energies, mat=None, order=2, errors=False, output_path=None):
    """Generates PERT cards for MCNP input files.

    Generates PERT cards based on the provided parameters. Can generate both first and
    second order perturbations, as well as cards for exact uncertainty calculations.
    Note that exact uncertainties are usually negligible, so verify their necessity
    before running long calculations.

    :param cell: Cell number(s) for PERT card application
    :type cell: int or str or list[int]
    :param density: Density value for the perturbation
    :type density: float
    :param reactions: List of reaction identifiers
    :type reactions: list[str]
    :param energies: Energy values. Used in consecutive pairs for energy bins
    :type energies: list[float]
    :param mat: Material identifier, defaults to None
    :type mat: str, optional
    :param order: Order of PERT card method (1 or 2), defaults to 2
    :type order: int, optional
    :param errors: Whether to include error methods (-2, -3, 1), defaults to False
    :type errors: bool, optional
    :param output_path: Path to output file. If None, prints to stdout
    :type output_path: str, optional
    
    :returns: None
    
    :note: Prints PERT cards to either stdout or specified file with sequential numbering
    """
    # Determine output stream
    if output_path:
        stream = open(output_path, "w")
    else:
        stream = sys.stdout

    # Initialize the perturbation counter
    pert_counter = 1
    if type(cell) == list: 
        cell_str = ','.join(map(str, cell)) if isinstance(cell, list) else str(cell)
    else: 
        cell_str = str(cell)
    # Loop over each combination of cell, density, and reaction
    for reaction in reactions:
        # Go through the energy list and use consecutive pairs
        for i in range(len(energies) - 1):
            E1 = energies[i]
            E2 = energies[i + 1]

            if mat is None:
                # Print the output for METHOD=2
                stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={density:.6e} METHOD=2 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                pert_counter += 1

                # Print the output for METHOD=3
                if order == 2:
                    stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={density:.6e} METHOD=3 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                    pert_counter += 1

                if errors:
                    # Print the output for METHOD=-2
                    stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={density:.6e} METHOD=-2 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                    pert_counter += 1

                    if order == 2:
                        # Print the output for METHOD=-3
                        stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={density:.6e} METHOD=-3 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                        pert_counter += 1

                        # Print the output for METHOD=1
                        stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={density:.6e} METHOD=1 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                        pert_counter += 1

            else:
                # Print the output for METHOD=2 with MAT
                stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={density:.6e} METHOD=2 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                pert_counter += 1

                if order == 2:
                    # Print the output for METHOD=3 with MAT
                    stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={density:.6e} METHOD=3 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                    pert_counter += 1
            
                if errors:
                    # Print the output for METHOD=-2 with MAT
                    stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={density:.6e} METHOD=-2 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                    pert_counter += 1

                    if order == 2:
                        # Print the output for METHOD=-3 with MAT
                        stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={density:.6e} METHOD=-3 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                        pert_counter += 1

                        # Print the output for METHOD=1 with MAT
                        stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={density:.6e} METHOD=1 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                        pert_counter += 1

    # Close the file if it was opened
    if output_path:
        stream.close()