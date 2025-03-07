from .input import Input, Perturbation, Pert, Mat, Materials, Component
import re

def _read_PERT(lines, start_index):
    """Internal helper function to read and parse a PERT card from MCNP input lines.

    :param lines: List of input lines from the MCNP input file
    :type lines: List[str]
    :param start_index: Starting index of the PERT card in the lines list
    :type start_index: int

    :returns: A tuple containing:
        - Perturbation object with parsed data, or None if parsing fails
        - The new index after processing the PERT card
    :rtype: tuple[Perturbation, int]
    """
    i = start_index
    line = lines[i].strip()
    card_lines = []
    
    # Accumulate lines that end with '&'
    while line.endswith("&"):
        card_lines.append(line.rstrip("&").strip())
        i += 1
        if i < len(lines):
            line = lines[i].strip()
        else:
            break
    card_lines.append(line)
    full_line = " ".join(card_lines)
    
    # Tokenize the full card line
    tokens = full_line.split()
    # Parse header: e.g. "PERT1:n"
    header_match = re.match(r'PERT(\d+):(\w)', tokens[0])
    if not header_match:
        return None, i + 1
        
    pert_num = int(header_match.group(1))
    particle = header_match.group(2)
    
    allowed_keywords = {"CELL", "MAT", "RHO", "METHOD", "RXN", "ERG"}
    pert_attrs = {}
    j = 1
    while j < len(tokens):
        token = tokens[j]
        if "=" in token:
            key, val = token.split("=", 1)
            key = key.upper()
            if key not in allowed_keywords:
                break
            if key == "CELL":
                cell_vals = val.replace(',', ' ').split()
                pert_attrs['cell'] = [int(x) for x in cell_vals]
            elif key == "MAT":
                pert_attrs['material'] = int(val)
            elif key == "RHO":
                pert_attrs['rho'] = float(val)
            elif key == "METHOD":
                pert_attrs['method'] = int(val)
            elif key == "RXN":
                pert_attrs['reaction'] = int(val)
            elif key == "ERG":
                erg_numbers = [val]
                j += 1
                while j < len(tokens) and "=" not in tokens[j]:
                    erg_numbers.append(tokens[j])
                    j += 1
                if len(erg_numbers) >= 2:
                    pert_attrs['energy'] = (float(erg_numbers[0]), float(erg_numbers[1]))
                else:
                    pert_attrs['energy'] = None
                continue
            j += 1
        else:
            break
    
    return Perturbation(id=pert_num, particle=particle, **pert_attrs), i + 1

def _read_material(lines, start_index):
    """Internal helper function to read and parse a material card from MCNP input lines.

    :param lines: List of input lines from the MCNP input file
    :type lines: List[str]
    :param start_index: Starting index of the material card in the lines list
    :type start_index: int

    :returns: A tuple containing:
        - Mat object with parsed data, or None if parsing fails
        - The new index after processing the material card
    :rtype: tuple[Mat, int]
    """
    i = start_index
    line = lines[i].strip()
    
    # Remove comments
    if "$" in line:
        line = line.split("$")[0].strip()
    
    # Parse material ID from first line (e.g., "m200100")
    match = re.match(r'^m(\d+)', line)
    if not match:
        return None, i + 1
    
    material_id = int(match.group(1))
    material_obj = Mat(id=material_id)
    
    # Process default library if specified on the first line
    if "lib=" in line:
        lib_match = re.search(r'([a-z])lib=(\S+)', line)
        if lib_match:
            material_obj.library = lib_match.group(2)
    
    # Move to the next line to start parsing nuclides
    i += 1
    
    # Parse nuclides
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip comment lines
        if line.startswith("c ") or not line:
            i += 1
            continue
        
        # Remove comments if present
        if "$" in line:
            line = line.split("$")[0].strip()
            
        # Skip if line is now empty after removing comments
        if not line:
            i += 1
            continue
        
        # Stop if we hit a line that's not a nuclide entry
        if not re.match(r'^\s*\d+', line):
            break
        
        # Parse nuclide and fraction
        parts = line.split()
        if len(parts) >= 2:
            try:
                nuclide_spec = parts[0]
                specific_lib = None
                
                # Check if nuclide has explicit library notation (e.g., 1001.06c)
                if '.' in nuclide_spec:
                    nuclide_parts = nuclide_spec.split('.')
                    nuclide = int(nuclide_parts[0])
                    specific_lib = nuclide_parts[1]
                else:
                    nuclide = int(nuclide_spec)
                
                fraction = float(parts[1])
                material_obj.add_component(zaid=nuclide, fraction=fraction, library=specific_lib)
                
            except (ValueError, IndexError):
                pass  # Skip if can't parse
        
        i += 1
    
    return material_obj, i

def read_mcnp(file_path):
    """Reads and parses an MCNP input file.

    This function reads an MCNP input file and parses its contents, focusing on
    PERT cards and material definitions. It creates an Input object containing 
    all parsed data.

    :param file_path: Path to the MCNP input file
    :type file_path: str

    :returns: An Input object containing all parsed data
    :rtype: Input
    """
    inst = Input()  # instance of the input class
    inst.pert = Pert()
    inst.materials = Materials()
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("PERT"):
            pert_obj, i = _read_PERT(lines, i)  
            if pert_obj:
                inst.pert.perturbation[pert_obj.id] = pert_obj
        elif line.startswith("m"):
            material_obj, i = _read_material(lines, i)
            if material_obj:
                inst.materials.mat[material_obj.id] = material_obj
        else:
            i += 1
            
    return inst

