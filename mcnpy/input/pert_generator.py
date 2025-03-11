import sys
import os
from mcnpy._grids import ENERGY_GRIDS
from .parse_input import read_mcnp


def perturb_material(inputfile, material_number, nuclide):
    """Creates a perturbed material with 100% increase in the specified nuclide's fraction.
    
    Reads an MCNP input file, finds the specified material, and creates a new perturbed
    material with a 100% increase in the fraction of the specified nuclide. The new material
    is added to the input file right after the original material definition and saved to
    a new file.
    
    :param inputfile: Path to the MCNP input file
    :type inputfile: str
    :param material_number: Material ID number to be perturbed
    :type material_number: int
    :param nuclide: ZAID of the nuclide to be perturbed
    :type nuclide: int
    
    :returns: Path to the new file with the perturbed material
    :rtype: str
    
    :raises ValueError: If the material or nuclide is not found in the input file
    """
    # Parse the input file
    input_data = read_mcnp(inputfile)
    
    # Check if the material exists
    if material_number not in input_data.materials.mat:
        raise ValueError(f"Material {material_number} not found in input file")
    
    original_material = input_data.materials.mat[material_number]
    
    # Check if the nuclide exists in the material
    if nuclide not in original_material.components:
        raise ValueError(f"Nuclide {nuclide} not found in material {material_number}")
    
    # Create a new material ID (original ID + 01)
    new_material_id = material_number * 100 + 1
    
    # Create a copy of the original material with the new ID
    perturbed_material = input_data.materials.mat[material_number].__class__(
        id=new_material_id,
        nlib=original_material.nlib,
        plib=original_material.plib
    )
    
    # Copy all components to the new material
    for zaid, component in original_material.components.items():
        perturbed_material.add_component(
            zaid=zaid, 
            fraction=component['fraction'],
            library=component.get('nlib')
        )
    
    # Calculate the sum of all fractions to check if normalization is needed
    total_fraction = sum(comp['fraction'] for comp in original_material.components.values())
    
    # Apply 100% perturbation to the specified nuclide
    perturbed_material.components[nuclide]['fraction'] *= 2.0
    
    # Renormalize the composition to maintain the same total fraction
    new_total = sum(comp['fraction'] for comp in perturbed_material.components.values())
    normalization_factor = total_fraction / new_total
    
    for zaid in perturbed_material.components:
        perturbed_material.components[zaid]['fraction'] *= normalization_factor
    
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
    
    # Skip to the end of the original material definition
    next_position = original_position + 1
    while next_position < len(lines):
        line = lines[next_position].strip()
        # Skip empty or comment lines
        if not line or line.startswith("c") or line.startswith("C"):
            next_position += 1
            continue
            
        # If the line doesn't start with a space and isn't part of this material definition, we've reached the end
        if not line[0].isspace() and not (line.startswith(str(material_number)) and len(line) > len(str(material_number)) and line[len(str(material_number))].isspace()):
            break
            
        # Otherwise, it's part of the current material definition
        next_position += 1
    
    # Generate comment and perturbed material string using the __str__ method
    comment = f"c Perturbed material with 100% increase in nuclide {nuclide}\n"
    
    # Instead of using __str__ directly which might have duplicate components issues,
    # manually create the material card string
    perturbed_material_str = f"m{new_material_id}"
    if perturbed_material.nlib:
        perturbed_material_str += f" nlib={perturbed_material.nlib}"
    if perturbed_material.plib:
        perturbed_material_str += f" plib={perturbed_material.plib}"
    perturbed_material_str += "\n"
    
    # Add components without duplication
    for zaid, comp in perturbed_material.components.items():
        fraction = comp['fraction']
        if comp.get('nlib'):
            perturbed_material_str += f"    {zaid}.{comp['nlib']} {fraction:.6e}\n"
        else:
            perturbed_material_str += f"    {zaid} {fraction:.6e}\n"
    
    # Insert comment and perturbed material after original material
    lines.insert(next_position, "\n")  # Add a blank line for separation
    lines.insert(next_position + 1, comment)
    lines.insert(next_position + 2, perturbed_material_str)
    lines.insert(next_position + 3, "\n")  # Add another blank line
    
    # Create output filename based on input filename
    base_name = os.path.basename(inputfile)
    filename, ext = os.path.splitext(base_name)
    new_filename = f"{filename}_pert_{material_number}_{nuclide}{ext}"
    output_path = os.path.join(os.getcwd(), new_filename)
    
    # Write the modified content to the new file
    with open(output_path, 'w') as f:
        f.writelines(lines)
    
    # Print information about the operation
    if abs(total_fraction - 1.0) > 1e-10:
        print(f"Note: Original material {material_number} had a total fraction of {total_fraction:.6f}")
    
    print(f"Created perturbed material {new_material_id} with 100% increase in nuclide {nuclide}")
    print(f"Original nuclide fraction: {original_material.components[nuclide]['fraction']:.6e}")
    print(f"Perturbed nuclide fraction: {perturbed_material.components[nuclide]['fraction']:.6e}")
    print(f"New file created: {output_path}")
    
    return


def generate_PERTcards(cell, rho, reactions, energies, mat=None, order=2, errors=False, output_path=None):
    """Generates PERT cards for MCNP input files.

    Generates PERT cards based on the provided parameters. Can generate both first and
    second order perturbations, as well as cards for exact uncertainty calculations.
    Note that exact uncertainties are usually negligible, so verify their necessity
    before running long calculations.

    :param cell: Cell number(s) for PERT card application
    :type cell: int or str or list[int]
    :param rho: Density value for the perturbation
    :type rho: float
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
    # Loop over each combination of cell, rho, and reaction
    for reaction in reactions:
        # Go through the energy list and use consecutive pairs
        for i in range(len(energies) - 1):
            E1 = energies[i]
            E2 = energies[i + 1]

            if mat is None:
                # Print the output for METHOD=2
                stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={rho:.6e} METHOD=2 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                pert_counter += 1

                # Print the output for METHOD=3
                if order == 2:
                    stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={rho:.6e} METHOD=3 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                    pert_counter += 1

                if errors:
                    # Print the output for METHOD=-2
                    stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={rho:.6e} METHOD=-2 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                    pert_counter += 1

                    if order == 2:
                        # Print the output for METHOD=-3
                        stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={rho:.6e} METHOD=-3 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                        pert_counter += 1

                        # Print the output for METHOD=1
                        stream.write(f"PERT{pert_counter}:n CELL={cell_str} &\nRHO={rho:.6e} METHOD=1 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                        pert_counter += 1

            else:
                # Print the output for METHOD=2 with MAT
                stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={rho:.6e} METHOD=2 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                pert_counter += 1

                if order == 2:
                    # Print the output for METHOD=3 with MAT
                    stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={rho:.6e} METHOD=3 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                    pert_counter += 1
            
                if errors:
                    # Print the output for METHOD=-2 with MAT
                    stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={rho:.6e} METHOD=-2 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                    pert_counter += 1

                    if order == 2:
                        # Print the output for METHOD=-3 with MAT
                        stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={rho:.6e} METHOD=-3 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                        pert_counter += 1

                        # Print the output for METHOD=1 with MAT
                        stream.write(f"PERT{pert_counter}:n CELL={cell_str} MAT={mat} &\nRHO={rho:.6e} METHOD=1 RXN={reaction} ERG={E1:.6e} {E2:.6e}\n")
                        pert_counter += 1

    # Close the file if it was opened
    if output_path:
        stream.close()