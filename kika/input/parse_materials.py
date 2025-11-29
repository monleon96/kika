from .material import Mat
import re

def read_material(lines, start_index):
    """Read and parse a material card from MCNP input lines.

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
    
    # Remove comments
    if "$" in full_line:
        full_line = full_line.split("$")[0].strip()
    
    # Parse material ID from first line (e.g., "m200100")
    tokens = full_line.split()
    match = re.match(r'^m(\d+)', tokens[0])
    if not match:
        return None, i + 1
    
    material_id = int(match.group(1))
    material_obj = Mat(id=material_id)
    
    # Process library specifications
    for token in tokens[1:]:
        if token.startswith("nlib="):
            material_obj.nlib = token.split("=")[1]
        elif token.startswith("plib="):
            material_obj.plib = token.split("=")[1]
        elif token.startswith("ylib="):
            material_obj.ylib = token.split("=")[1]
        elif token.startswith("lib="):
            lib_value = token.split("=")[1]
            # Determine library type by examining the last character
            if lib_value.endswith('c'):
                material_obj.nlib = lib_value
            elif lib_value.endswith('p'):
                material_obj.plib = lib_value
            elif lib_value.endswith('y'):
                material_obj.ylib = lib_value
        else:
            # Start processing nuclides
            break
    
    # Find index where nuclide data begins
    nuclide_start_idx = 1  # Default to after material ID
    for idx, token in enumerate(tokens[1:], 1):
        if not (token.startswith("nlib=") or token.startswith("plib=") or 
                token.startswith("ylib=") or token.startswith("lib=")):
            nuclide_start_idx = idx
            break
    
    # Process nuclides in the main definition
    idx = nuclide_start_idx
    while idx < len(tokens):
        try:
            nuclide_spec = tokens[idx]
            specific_lib = None
            
            # Check if nuclide has explicit library notation (e.g., 1001.06c)
            if '.' in nuclide_spec:
                nuclide_parts = nuclide_spec.split('.')
                nuclide = int(nuclide_parts[0])
                specific_lib = nuclide_parts[1]
            else:
                try:
                    nuclide = int(nuclide_spec)
                except ValueError:
                    idx += 1
                    continue
            
            # Check if we have a fraction
            if idx + 1 < len(tokens):
                try:
                    fraction = float(tokens[idx + 1])
                    
                    # Check for consistency in fraction types
                    if material_obj.nuclide:
                        existing_is_negative = next(iter(material_obj.nuclide.values())).fraction < 0
                        current_is_negative = fraction < 0
                        if existing_is_negative != current_is_negative:
                            raise ValueError(f"Material {material_id} has inconsistent fraction types (mixed weight and atomic fractions)")
                    
                    material_obj.add_nuclide(zaid=nuclide, fraction=fraction, library=specific_lib)
                    idx += 2  # Move past both nuclide and fraction
                except ValueError:
                    # If the next token is not a valid fraction, move on
                    idx += 1
            else:
                idx += 1
                
        except (ValueError, IndexError):
            idx += 1  # Skip if can't parse
    
    # Continue parsing additional lines for nuclides
    i += 1
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
            
        # Check if this is another material definition
        if re.match(r'^m\d+', line):
            break
        
        # Process additional nuclide lines
        parts = line.split()
        idx = 0
        while idx < len(parts):
            try:
                nuclide_spec = parts[idx]
                specific_lib = None
                
                # Check if nuclide has explicit library notation (e.g., 1001.06c)
                if '.' in nuclide_spec:
                    nuclide_parts = nuclide_spec.split('.')
                    nuclide = int(nuclide_parts[0])
                    specific_lib = nuclide_parts[1]
                else:
                    nuclide = int(nuclide_spec)
                
                # Check if we have a fraction
                if idx + 1 < len(parts):
                    try:
                        fraction = float(parts[idx + 1])
                        
                        # Check for consistency in fraction types
                        if material_obj.nuclide:
                            existing_is_negative = next(iter(material_obj.nuclide.values())).fraction < 0
                            current_is_negative = fraction < 0
                            if existing_is_negative != current_is_negative:
                                raise ValueError(f"Material {material_id} has inconsistent fraction types (mixed weight and atomic fractions)")
                        
                        material_obj.add_nuclide(zaid=nuclide, fraction=fraction, library=specific_lib)
                        idx += 2  # Move past both nuclide and fraction
                    except ValueError:
                        # If the next token is not a valid fraction, move on
                        idx += 1
                else:
                    idx += 1
                    
            except (ValueError, IndexError):
                idx += 1  # Skip if can't parse
        
        i += 1
    
    return material_obj, i
