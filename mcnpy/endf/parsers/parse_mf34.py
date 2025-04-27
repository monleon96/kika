"""
Parser for MF34 (Angular Distribution Covariances) sections in ENDF files.

MF34 contains covariance data for angular distributions of secondary particles.
"""
from typing import List

from ..classes.mf import MF
from ..classes.mf34.mf34 import MF34MT, Subsection, SubSubsection, SubSubsectionRecord
from ..utils import parse_line, parse_endf_id, group_lines_by_mt_with_positions


def parse_mf34(lines: List[str]) -> MF:
    """
    Parse MF34 (Angular Distribution Covariances) data.
    
    Args:
        lines: List of string lines from the MF34 section
        
    Returns:
        MF object with parsed MF34 data
    """
    mf = MF(number=34)
    
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
            mt_section = parse_mf34_mt(mt_lines, mt)
            mf.add_section(mt_section)
            
            # Add line count information if available
            if mt in line_counts:
                mt_section.num_lines = line_counts[mt]
        except Exception as e:
            print(f"WARNING - Error parsing MT{mt} in MF34: {e}")
    
    return mf


def parse_mf34_mt(lines: List[str], mt: int) -> MF34MT:
    """
    Parse a MT section from MF34.
    
    Args:
        lines: List of string lines from the MT section
        mt: The MT section number
        
    Returns:
        MF34MT object with parsed data
    """
    # Need at least two lines for header and first subsection
    if len(lines) < 2:
        raise ValueError(f"Insufficient data provided for MT{mt} section in MF34")
    
    # Parse the header line
    header = parse_line(lines[0])
    za = header.get("C1")
    awr = header.get("C2")
    # C3 is zero (unused)
    ltt = header.get("C4")
    # C5 is zero (unused)
    nmt1 = header.get("C6")  # Number of subsections
    

    print(f"DEBUG - Parsing MF34 MT{mt} with ZA={za}, AWR={awr}, LTT={ltt}, NMT1={nmt1}")


    # Get material number
    mat, mf, mt = parse_endf_id(lines[0])
    
    # Create the MT section object
    mt_section = MF34MT(
        number=mt, 
        _za=za, 
        _awr=awr, 
        _ltt=ltt,
        _nmt1=nmt1,
        _mat=mat
    )
    
    # Parse each subsection
    current_line = 1  # Start with the first subsection header
    
    for _ in range(nmt1 if nmt1 else 0):
        if current_line >= len(lines):
            break
            
        # Parse subsection header (second line of first subsection or subsequent subsections)
        subsec_line = parse_line(lines[current_line])
        current_line += 1
        
        # C1 and C2 are 0.0 (unused)
        mat1 = subsec_line.get("C3")  # MAT1 = 0.0 typically
        mt1 = subsec_line.get("C4")
        nl = subsec_line.get("C5")
        nl1 = subsec_line.get("C6")
        
        print(f"DEBUG - Parsing subsection with MAT1={mat1}, MT1={mt1}, NL={nl}, NL1={nl1}")

        # Create subsection
        subsection = Subsection(
            mt1=mt1,
            nl=nl,
            nl1=nl1,
            mat1=mat1
        )
        
        # Calculate number of sub-subsections
        if mt1 == mt:
            # Only upper triangle is given when MT1=MT to avoid redundancy
            nss = (nl * (nl + 1)) // 2
        else:
            nss = nl * nl1
        
        # Parse sub-subsections
        for _ in range(nss):
            if current_line >= len(lines):
                break
                
            # Parse sub-subsection header
            sub_header = parse_line(lines[current_line])
            current_line += 1
            
            # C1 and C2 are 0.0 (unused)
            l = sub_header.get("C3")
            l1 = sub_header.get("C4")
            lct = sub_header.get("C5")
            ni = sub_header.get("C6")  # Number of LIST records
            
            print(f"DEBUG - Parsing sub-subsection with L={l}, L1={l1}, LCT={lct}, NI={ni}")

            # Create sub-subsection
            sub_subsection = SubSubsection(
                l=l,
                l1=l1,
                lct=lct,
                ni=ni
            )
            
            # Parse LIST records
            for _ in range(ni):
                if current_line >= len(lines):
                    break
                    
                # Parse LIST record header
                list_header = parse_line(lines[current_line])
                current_line += 1
                
                # C1 and C2 are 0.0 (unused)
                ls = list_header.get("C3")  # Flag for symmetric matrix
                lb = list_header.get("C4")  # Flag for covariance pattern
                nt = list_header.get("C5")  # Total number of items in list
                
                # Create LIST record
                list_record = SubSubsectionRecord(
                    ls=ls,
                    lb=lb,
                    nt=nt
                )
                
                if lb >= 0 and lb <= 4:
                    # For LB=0 to LB=4
                    np_value = list_header.get("C6")  # Number of pairs
                    list_record.np = np_value
                    list_record.lt = ls  # In LB=0-4, LT is stored in C3 position
                    
                    # Total values to read
                    values_to_read = nt
                    
                    # Calculate sizes of tables
                    if list_record.lt == 0:
                        # Only one table: {E_k, F_k}
                        num_k_pairs = np_value
                        num_l_pairs = 0
                    else:
                        # Two tables: {E_k, F_k}{E_l, F_l}
                        num_l_pairs = list_record.lt
                        num_k_pairs = np_value - num_l_pairs
                    
                    # Read values
                    all_values = []
                    values_read = 0
                    
                    while values_read < values_to_read and current_line < len(lines):
                        value_line = parse_line(lines[current_line])
                        current_line += 1
                        
                        # Read up to 6 values per line
                        for i in range(1, 7):
                            if values_read < values_to_read:
                                value = value_line.get(f"C{i}")
                                if value is not None:
                                    all_values.append(value)
                                    values_read += 1
                    
                    # Distribute values to appropriate tables
                    for i in range(num_k_pairs):
                        list_record.e_table_k.append(all_values[2*i])
                        list_record.f_table_k.append(all_values[2*i+1])
                    
                    # If there's a second table
                    if num_l_pairs > 0:
                        offset = 2 * num_k_pairs
                        for i in range(num_l_pairs):
                            list_record.e_table_l.append(all_values[offset + 2*i])
                            list_record.f_table_l.append(all_values[offset + 2*i+1])
                    
                    print(f"DEBUG - Parsed LB={lb} record with NP={np_value}, LT={list_record.lt}")
                    print(f"DEBUG - First table: {num_k_pairs} pairs, Second table: {num_l_pairs} pairs")
                
                elif lb == 5:
                    # For LB=5 (original implementation)
                    ne = list_header.get("C6")  # Number of energy entries
                    list_record.ne = ne
                    
                    # Calculate number of matrix elements based on LS and NE for LB=5
                    if ls == 0:  # Asymmetric matrix
                        num_matrix_elements = (ne - 1) ** 2
                    elif ls == 1:  # Symmetric matrix
                        num_matrix_elements = (ne * (ne - 1)) // 2
                    else:
                        raise ValueError(f"Unknown LS value: {ls}. Should be 0 or 1")
                    
                    # Number of values to read (energies + matrix elements)
                    values_to_read = ne + num_matrix_elements
                    
                    # Read energy grid and matrix values
                    energies = []
                    matrix_values = []
                    values_read = 0
                    
                    while values_read < values_to_read and current_line < len(lines):
                        value_line = parse_line(lines[current_line])
                        current_line += 1
                        
                        # Read up to 6 values per line
                        for i in range(1, 7):
                            if values_read < values_to_read:
                                value = value_line.get(f"C{i}")
                                if value is not None:
                                    if values_read < ne:
                                        energies.append(value)
                                    else:
                                        matrix_values.append(value)
                                    values_read += 1
                    
                    # Store the parsed energy grid and matrix values
                    list_record.energies = energies
                    list_record.matrix = matrix_values
                    
                    print(f"DEBUG - Parsed LB=5 record with LS={ls}, NE={ne}")
                else:
                    raise ValueError(f"Unsupported LB value: {lb}. Only LB=0-5 are currently supported")
                
                # Add the list record to the sub-subsection
                sub_subsection.records.append(list_record)
            
            # Add the sub-subsection to the subsection
            subsection.sub_subsections.append(sub_subsection)
        
        # Add the subsection to the MT section
        mt_section.add_subsection(subsection)
    
    return mt_section