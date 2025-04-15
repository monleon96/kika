"""
Parser for MF4 (Angular Distributions) sections in ENDF files.

MF4 contains angular distribution data for neutron-induced reactions.
Different formats are used based on the LTT flag:
- LTT=0: All angular distributions are isotropic
- LTT=1: Data given as Legendre expansion coefficients
- LTT=2: Data given as tabulated probability distributions
- LTT=3: Mixed representation (Legendre + tabulated)
"""
from typing import List

from ..classes.mf import MF
from ..classes.mf4.base import MF4MT 
from ..classes.mf4.isotropic import MF4MTIsotropic
from ..classes.mf4.polynomial import MF4MTLegendre
from ..classes.mf4.tabulated import MF4MTTabulated
from ..classes.mf4.mixed import MF4MTMixed
from ..utils import parse_line, parse_endf_id, group_lines_by_mt_with_positions


def parse_mf4(lines: List[str], file_offset: int = 0) -> MF:
    """
    Parse MF4 (Angular Distributions) data.
    
    Args:
        lines: List of string lines from the MF4 section
        file_offset: Line number offset from the start of the original file
        
    Returns:
        MF object with parsed MF4 data
    """
    mf = MF(number=4)
    
    # Record number of lines
    mf.num_lines = len(lines)
    
    # Group lines by MT sections with line counting
    mt_groups, line_counts = group_lines_by_mt_with_positions(lines)
    
    # Parse each MT section
    for mt, mt_lines in mt_groups.items():
        # Skip MT=0 since these are section end markers, not actual data
        if mt == 0:
            print(f"DEBUG - Ignoring MT=0 marker lines (end of section indicators)")
            continue
        
        try:
            mt_section = parse_mf4_mt(mt_lines, mt)
            mf.add_section(mt_section)
            
            # Add line count information if available
            if mt in line_counts:
                mt_section.num_lines = line_counts[mt]
        except Exception as e:
            print(f"WARNING - Error parsing MT{mt} in MF4: {e}")
    
    return mf

def parse_mf4_mt(lines: List[str], mt: int) -> MF4MT:
    """
    Parse a MT section from MF4, determining the appropriate format based on LTT.
    
    Args:
        lines: List of string lines from the MT section
        mt: The MT section number
        
    Returns:
        MF4MT object of the appropriate type based on LTT
    """
    # Need at least one line for the header
    if not lines:
        raise ValueError(f"No data provided for MT{mt} section in MF4")
    
    # Parse the header line
    header = parse_line(lines[0])
    za = header.get("C1")
    awr = header.get("C2")
    # C3 is zero (unused)
    ltt = header.get("C4")
    # C5 and C6 are zero (unused)
    
    # Get material number
    mat, mf, mt = parse_endf_id(lines[0])
    
    # Check which format we're dealing with and parse accordingly
    if ltt == 0:
        return _parse_mf4_mt_isotropic(lines, za, awr, mt, mat)
    elif ltt == 1:
        return _parse_mf4_mt_legendre(lines, za, awr, mt, mat)
    elif ltt == 2:
        return _parse_mf4_mt_tabulated(lines, za, awr, mt, mat)
    elif ltt == 3:
        return _parse_mf4_mt_mixed(lines, za, awr, mt, mat)
    else:
        raise ValueError(f"Unknown LTT value: {ltt} in MT{mt}")


def _parse_mf4_mt_isotropic(lines: List[str], za: float, awr: float, 
                          mt: int, mat: int) -> MF4MTIsotropic:
    """
    Parse MT section with isotropic angular distributions (LTT=0).
    
    Args:
        lines: List of string lines from the MT section
        za: ZA value from header
        awr: AWR value from header
        mt: MT number
        mat: MAT number
        
    Returns:
        MF4MTIsotropic object
    """
    # For isotropic, we just need the header and any energy points
    # Parse the second line for NE (number of energies)
    if len(lines) < 2:
        raise ValueError(f"Incomplete MT{mt} section in MF4 (LTT=0)")
    
    # Parse the second line according to specifications
    line2 = parse_line(lines[1])
    # C1 is 0.0 (unused)
    # C2 AWR already parsed
    li = line2.get("C3")     # Flag for isotropic (1 = all isotropic)
    lct = line2.get("C4")    # Frame of reference (1=LAB system, 2=CM system)
    # C5 is 0 (unused)
    # C6 is 0 (unused)
    
    # Create the MT section object
    mt_section = MF4MTIsotropic(
        number=mt, 
        _za=za, 
        _awr=awr, 
        _li=li,
        _lct=lct,
        _mat=mat
    )
    
    # The energy grid will be parsed once specifications are provided
    
    return mt_section


def _parse_mf4_mt_legendre(lines: List[str], za: float, awr: float,
                         mt: int, mat: int) -> MF4MTLegendre:
    """
    Parse MT section with Legendre expansions (LTT=1).
    
    Args:
        lines: List of string lines from the MT section
        za: ZA value from header
        awr: AWR value from header
        mt: MT number
        mat: MAT number
        
    Returns:
        MF4MTLegendre object with coefficients for each energy
    """
    if len(lines) < 3:
        raise ValueError(f"Incomplete MT{mt} section in MF4 (LTT=1)")
    
    # Parse the second line according to specifications
    line2 = parse_line(lines[1])
    # C1 is 0.0 (unused)
    # C2 AWR already parsed
    li = line2.get("C3")     # Flag for isotropic (1 = all isotropic)
    lct = line2.get("C4")    # Frame of reference (1=LAB system, 2=CM system)
    # C5 is 0 (unused)
    # C6 is 0 (unused)

    # Parse the third line with NR and NE
    line3 = parse_line(lines[2])
    nr = line3.get("C5")    # Number of interpolation regions
    ne = line3.get("C6")    # Number of energy points
    
    # Create the MT section object
    mt_section = MF4MTLegendre(
        number=mt, 
        _za=za, 
        _awr=awr, 
        _li=li,
        _lct=lct,
        _nr=nr,
        _ne=ne,
        _mat=mat
    )
    
    # Parse the interpolation scheme pairs
    current_line = 3  # Start at 4th line (index 3)
    interpolation_pairs = []
    
    # Skip if no interpolation regions
    if nr and nr > 0:
        pairs_to_read = nr
        
        # Read pairs across multiple lines if necessary
        while pairs_to_read > 0 and current_line < len(lines):
            line_data = parse_line(lines[current_line])
            
            # Each line can contain up to 3 pairs (6 values)
            pairs_in_line = min(3, pairs_to_read)
            
            for i in range(pairs_in_line):
                nbt = line_data.get(f"C{i*2+1}")
                interp = line_data.get(f"C{i*2+2}")
                if nbt is not None and interp is not None:
                    interpolation_pairs.append((nbt, interp))
            
            pairs_to_read -= pairs_in_line
            current_line += 1
        
        mt_section._interpolation = interpolation_pairs
    
    # Parse Legendre coefficients for each energy point
    energies = []
    legendre_coeffs = []
    
    # Process ne energy points
    for _ in range(ne):
        if current_line >= len(lines):
            break
            
        # Read the header line for this energy point
        header_line = parse_line(lines[current_line])
        current_line += 1
        
        t = header_line.get("C1")      # Temperature (normally 0)
        energy = header_line.get("C2")  # Energy value
        lt = header_line.get("C3")     # Temperature dependence test (normally 0)
        # C4 is 0 (unused)
        nl = header_line.get("C5")     # Highest order Legendre polynomial at this energy
        # C6 is 0 (unused)
        
        # Add this energy to our grid
        if energy is not None:
            energies.append(energy)
        else:
            continue  # Skip if energy is missing
        
        # Read the Legendre coefficients for this energy
        coeffs = []
        
        # Calculate number of lines needed for nl coefficients (6 per line)
        num_coef_lines = (nl + 5) // 6  # Integer division with ceiling
        
        for _ in range(num_coef_lines):
            if current_line >= len(lines):
                break
                
            coef_line = parse_line(lines[current_line])
            current_line += 1
            
            # Read up to 6 coefficients from this line
            for i in range(1, 7):
                if len(coeffs) < nl:
                    coef = coef_line.get(f"C{i}")
                    if coef is not None:
                        coeffs.append(coef)
        
        # Store the coefficients for this energy
        legendre_coeffs.append(coeffs)
    
    # Store the parsed energy grid and Legendre coefficients
    mt_section._energies = energies
    mt_section._legendre_coeffs = legendre_coeffs
    
    return mt_section


def _parse_mf4_mt_tabulated(lines: List[str], za: float, awr: float,
                          mt: int, mat: int) -> MF4MTTabulated:
    """
    Parse MT section with tabulated distributions (LTT=2).
    
    Args:
        lines: List of string lines from the MT section
        za: ZA value from header
        awr: AWR value from header
        mt: MT number
        mat: MAT number
        
    Returns:
        MF4MTTabulated object with probability tables for each energy
    """
    if len(lines) < 3:
        raise ValueError(f"Incomplete MT{mt} section in MF4 (LTT=2)")
    
    # Parse the second line according to specifications
    line2 = parse_line(lines[1])
    # C1 is 0.0 (unused)
    # C2 AWR already parsed
    li = line2.get("C3")     # Flag for isotropic (1 = all isotropic)
    lct = line2.get("C4")    # Frame of reference (1=LAB system, 2=CM system)
    # C5 is 0 (unused)
    # C6 is 0 (unused)
    
    # Parse the third line with NR and NE
    line3 = parse_line(lines[2])
    nr = line3.get("C5")    # Number of interpolation regions
    ne = line3.get("C6")    # Number of energy points
    
    # Create the MT section object
    mt_section = MF4MTTabulated(
        number=mt, 
        _za=za, 
        _awr=awr, 
        _li=li,
        _lct=lct,
        _nr=nr,
        _ne=ne,
        _mat=mat
    )
    
    # Parse the interpolation scheme pairs
    current_line = 3  # Start at 4th line (index 3)
    interpolation_pairs = []
    
    # Skip if no interpolation regions
    if nr and nr > 0:
        pairs_to_read = nr
        
        # Read pairs across multiple lines if necessary
        while pairs_to_read > 0 and current_line < len(lines):
            line_data = parse_line(lines[current_line])
            
            # Each line can contain up to 3 pairs (6 values)
            pairs_in_line = min(3, pairs_to_read)
            
            for i in range(pairs_in_line):
                nbt = line_data.get(f"C{i*2+1}")
                interp = line_data.get(f"C{i*2+2}")
                if nbt is not None and interp is not None:
                    interpolation_pairs.append((nbt, interp))
            
            pairs_to_read -= pairs_in_line
            current_line += 1
        
        mt_section._interpolation = interpolation_pairs
    
    # Parse tabulated distributions for each energy point
    energies = []
    cosines = []
    probabilities = []
    angular_interpolation = []
    
    # Process ne energy points
    for _ in range(ne):
        if current_line >= len(lines):
            break
        
        # Read the header line for this energy point
        point_header = parse_line(lines[current_line])
        current_line += 1
        
        # C1 is 0.0 (unused)
        energy = point_header.get("C2")  # Energy value
        # C3 is 0 (unused)
        # C4 is 0 (unused)
        nr_ang = point_header.get("C5")  # Number of interpolation regions for angular data
        np_val = point_header.get("C6")  # Number of angular points (cosines)
        
        # Add this energy to our grid
        if energy is not None:
            energies.append(energy)
        else:
            continue  # Skip if energy is missing
        
        # Process angular interpolation information
        ang_interp_pairs = []
        
        if nr_ang and nr_ang > 0:
            pairs_to_read = nr_ang
            
            # Read pairs across multiple lines if necessary
            while pairs_to_read > 0 and current_line < len(lines):
                line_data = parse_line(lines[current_line])
                current_line += 1
                
                # Each line can contain up to 3 pairs (6 values)
                pairs_in_line = min(3, pairs_to_read)
                
                for i in range(pairs_in_line):
                    nbt = line_data.get(f"C{i*2+1}")
                    interp = line_data.get(f"C{i*2+2}")
                    if nbt is not None and interp is not None:
                        ang_interp_pairs.append((nbt, interp))
                
                pairs_to_read -= pairs_in_line
        
        # Store the angular interpolation scheme for this energy
        angular_interpolation.append(ang_interp_pairs)
        
        # Now read NP pairs of cosine and probability values
        energy_cosines = []
        energy_probabilities = []
        
        # Calculate number of lines needed for np pairs (3 pairs per line)
        num_pair_lines = (np_val + 2) // 3  # Integer division with ceiling
        
        for _ in range(num_pair_lines):
            if current_line >= len(lines):
                break
                
            pair_line = parse_line(lines[current_line])
            current_line += 1
            
            # Read up to 3 pairs from this line
            for i in range(3):
                if len(energy_cosines) < np_val:
                    mu_idx = i * 2 + 1
                    prob_idx = i * 2 + 2
                    
                    mu = pair_line.get(f"C{mu_idx}")
                    prob = pair_line.get(f"C{prob_idx}")
                    
                    if mu is not None and prob is not None:
                        energy_cosines.append(mu)
                        energy_probabilities.append(prob)
        
        # Store the tabulated data for this energy
        cosines.append(energy_cosines)
        probabilities.append(energy_probabilities)
    
    # Store the parsed data in the object
    mt_section._energies = energies
    mt_section._cosines = cosines
    mt_section._probabilities = probabilities
    mt_section._angular_interpolation = angular_interpolation
    
    return mt_section


def _parse_mf4_mt_mixed(lines: List[str], za: float, awr: float,
                      mt: int, mat: int) -> MF4MTMixed:
    """
    Parse MT section with mixed representation (LTT=3).
    
    Args:
        lines: List of string lines from the MT section
        za: ZA value from header
        awr: AWR value from header
        mt: MT number
        mat: MAT number
        
    Returns:
        MF4MTMixed object with Legendre and tabulated components
    """
    if len(lines) < 3:
        raise ValueError(f"Incomplete MT{mt} section in MF4 (LTT=3)")
    
    # Parse the second line according to specifications
    line2 = parse_line(lines[1])
    # C1 is 0.0 (unused)
    # C2 AWR already parsed
    li = line2.get("C3")     # Flag for isotropic (1 = all isotropic)
    lct = line2.get("C4")    # Frame of reference (1=LAB system, 2=CM system)
    # C5 is 0 (unused)
    nm = line2.get("C6")    # Maximum order of Legendre polynomial
    
    # ======================================
    # Parsing of Legendre coefficients
    # ======================================

    # Parse the third line containing interpolation information
    if len(lines) < 3:
        raise ValueError(f"Missing third line in MT{mt} section in MF4 (LTT=3)")
    
    line3 = parse_line(lines[2])
    # C1 is 0.0 (unused)
    # C2 is 0.0 (unused)
    # C3 is 0 (unused)
    # C4 is 0 (unused)
    nr = line3.get("C5")    # Number of different interpolation intervals
    ne1 = line3.get("C6")   # Number of energy points for Legendre coefficients
    
    # Create the MT section object
    mt_section = MF4MTMixed(
        number=mt, 
        _za=za, 
        _awr=awr, 
        _li=li,
        _lct=lct,
        _nm=nm,
        _nr=nr,
        _ne1=ne1,
        _mat=mat
    )
    
    # Parse the interpolation scheme pairs
    current_line = 3  # Start at 4th line (index 3)
    interpolation_pairs = []
    
    # Skip if no interpolation regions
    if nr and nr > 0:
        pairs_to_read = nr
        
        # Read pairs across multiple lines if necessary
        while pairs_to_read > 0 and current_line < len(lines):
            line_data = parse_line(lines[current_line])
            
            # Each line can contain up to 3 pairs (6 values)
            pairs_in_line = min(3, pairs_to_read)
            
            for i in range(pairs_in_line):
                nbt = line_data.get(f"C{i*2+1}")
                interp = line_data.get(f"C{i*2+2}")
                if nbt is not None and interp is not None:
                    interpolation_pairs.append((nbt, interp))
            
            pairs_to_read -= pairs_in_line
            current_line += 1
        
        mt_section._interpolation = interpolation_pairs
    
    # Now parse the Legendre coefficients
    energies = []
    legendre_coeffs = []
    
    # Process ne1 energy points for Legendre coefficients
    for _ in range(ne1):
        if current_line >= len(lines):
            break
            
        # Read the header line for this energy point
        header_line = parse_line(lines[current_line])
        current_line += 1
        
        t = header_line.get("C1")      # Temperature (normally 0)
        energy = header_line.get("C2")  # Energy value
        lt = header_line.get("C3")     # Temperature dependence test (normally 0)
        # C4 is 0 (unused)
        nl = header_line.get("C5")     # Highest order Legendre polynomial at this energy
        # C6 is 0 (unused)
        
        # Add this energy to our grid
        if energy is not None:
            energies.append(energy)
        else:
            continue  # Skip if energy is missing
        
        # Read the Legendre coefficients for this energy
        coeffs = []
        
        # Calculate number of lines needed for nl coefficients (6 per line)
        num_coef_lines = (nl + 5) // 6  # Integer division with ceiling
        
        for _ in range(num_coef_lines):
            if current_line >= len(lines):
                break
                
            coef_line = parse_line(lines[current_line])
            current_line += 1
            
            # Read up to 6 coefficients from this line
            for i in range(1, 7):
                if len(coeffs) < nl:
                    coef = coef_line.get(f"C{i}")
                    if coef is not None:
                        coeffs.append(coef)
        
        # Store the coefficients for this energy
        legendre_coeffs.append(coeffs)
    
    # Store the parsed energy grid and Legendre coefficients
    mt_section._energies = energies
    mt_section._legendre_coeffs = legendre_coeffs
    
    # ======================================
    # Now parse the tabulated distribution part
    # ======================================
    
    # Parse the header line for tabulated data
    tab_header_line = parse_line(lines[current_line])
    current_line += 1
    
    # C1 is 0.0 (unused)
    # C2 is 0.0 (unused)
    # C3 is 0 (unused)
    # C4 is 0 (unused)
    nr_tab = tab_header_line.get("C5")    # Number of different interpolation intervals for tabulated data
    ne2 = tab_header_line.get("C6")       # Number of energy points for tabulated distributions
    
    mt_section._ne2 = ne2
    mt_section._nr_tab = nr_tab  # Store the nr_tab value
    
    # Skip if no tabulated energy points
    if not ne2 or ne2 <= 0:
        return mt_section
    
    # Process interpolation information for tabulated data
    tab_interpolation_pairs = []
    if nr_tab and nr_tab > 0:
        pairs_to_read = nr_tab
        
        # Read pairs across multiple lines if necessary
        while pairs_to_read > 0 and current_line < len(lines):
            line_data = parse_line(lines[current_line])
            current_line += 1
            
            # Each line can contain up to 3 pairs (6 values per line)
            pairs_in_line = min(3, pairs_to_read)
            
            for i in range(pairs_in_line):
                nbt = line_data.get(f"C{i*2+1}")
                interp = line_data.get(f"C{i*2+2}")
                if nbt is not None and interp is not None:
                    tab_interpolation_pairs.append((nbt, interp))
                
                pairs_to_read -= 1
        
        # Store the tabulated energy interpolation scheme
        mt_section._tab_interpolation = tab_interpolation_pairs
    
    # Now read NE2 tabulated distributions
    tabulated_energies = []
    tabulated_cosines = []
    tabulated_probabilities = []
    angular_interpolation = []
    
    for _ in range(ne2):
        if current_line >= len(lines):
            break
        
        # Read the header line for this energy point
        tab_point_header = parse_line(lines[current_line])
        current_line += 1
        
        t = tab_point_header.get("C1")      # Temperature (normally 0)
        energy = tab_point_header.get("C2")  # Energy value
        lt = tab_point_header.get("C3")      # Temperature dependence test (normally 0)
        # C4 is 0 (unused)
        nr_ang = tab_point_header.get("C5")  # Number of interpolation regions for angular data
        np_val = tab_point_header.get("C6")  # Number of angular points (cosines)
        
        # Add this energy to the tabulated energies
        if energy is not None:
            tabulated_energies.append(energy)
        else:
            continue  # Skip if energy is missing
        
        # Process angular interpolation information
        ang_interp_pairs = []
        
        if nr_ang and nr_ang > 0:
            pairs_to_read = nr_ang
            
            # Read pairs across multiple lines if necessary
            while pairs_to_read > 0 and current_line < len(lines):
                line_data = parse_line(lines[current_line])
                current_line += 1
                
                # Each line can contain up to 3 pairs (6 values)
                pairs_in_line = min(3, pairs_to_read)
                
                for i in range(pairs_in_line):
                    nbt = line_data.get(f"C{i*2+1}")
                    interp = line_data.get(f"C{i*2+2}")
                    if nbt is not None and interp is not None:
                        ang_interp_pairs.append((nbt, interp))
                
                pairs_to_read -= pairs_in_line
        
        # Store the angular interpolation scheme for this energy
        angular_interpolation.append(ang_interp_pairs)
        
        # Now read NP pairs of cosine and probability values
        cosines = []
        probabilities = []
        
        # Calculate number of lines needed for np pairs (3 pairs per line)
        num_pair_lines = (np_val + 2) // 3  # Integer division with ceiling
        
        for _ in range(num_pair_lines):
            if current_line >= len(lines):
                break
                
            pair_line = parse_line(lines[current_line])
            current_line += 1
            
            # Read up to 3 pairs from this line
            for i in range(3):
                if len(cosines) < np_val:
                    mu_idx = i * 2 + 1
                    prob_idx = i * 2 + 2
                    
                    mu = pair_line.get(f"C{mu_idx}")
                    prob = pair_line.get(f"C{prob_idx}")
                    
                    if mu is not None and prob is not None:
                        cosines.append(mu)
                        probabilities.append(prob)
        
        # Store the tabulated data for this energy
        tabulated_cosines.append(cosines)
        tabulated_probabilities.append(probabilities)
    
    # Store the tabulated data in the object
    mt_section._tabulated_energies = tabulated_energies
    mt_section._tabulated_cosines = tabulated_cosines
    mt_section._tabulated_probabilities = tabulated_probabilities
    mt_section._angular_interpolation = angular_interpolation
    
    return mt_section
