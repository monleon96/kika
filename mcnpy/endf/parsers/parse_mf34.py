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
                ls = list_header.get("C3")
                lb = list_header.get("C4")
                nt = list_header.get("C5")
                ne_field = list_header.get("C6")

                # Create LIST record
                list_record = SubSubsectionRecord(ls=ls, lb=lb, nt=nt)

                if lb in (0, 1, 2):
                    # one (E,F) table: NT=2*NE
                    list_record.ne = ne_field
                    list_record.lt = 0
                    list_record.np = list_record.ne
                    values_to_read = nt
                    all_values = []
                    while len(all_values) < values_to_read and current_line < len(lines):
                        vl = parse_line(lines[current_line]); current_line += 1
                        for i in range(1,7):
                            if len(all_values) < values_to_read:
                                v = vl.get(f"C{i}")
                                if v is not None:
                                    all_values.append(v)
                    # distribute into E_k, F_k
                    for i in range(list_record.ne):
                        list_record.e_table_k.append(all_values[2*i])
                        list_record.f_table_k.append(all_values[2*i+1])
                    print(f"DEBUG - Parsed LB={lb} one-table record NE={list_record.ne}")

                elif lb == 5:
                    # For LB=5 (original implementation)
                    ne = list_header.get("C6")  # Number of energy entries
                    list_record.ne = ne
                    
                    print(f"DEBUG LB=5 - NE={ne}")
                    
                    # Calculate number of matrix elements based on LS and NE for LB=5
                    # Matrix is (NE-1) x (NE-1) for intervals
                    m = ne - 1  # Number of intervals = matrix dimension
                    if ls == 0:  # Asymmetric matrix
                        num_matrix_elements = m * m  # (NE-1)²
                        print(f"DEBUG LB=5 - Asymmetric matrix: {m}x{m} = {num_matrix_elements} elements")
                    elif ls == 1:  # Symmetric matrix - upper triangle including diagonal
                        num_matrix_elements = m * (m + 1) // 2  # (NE-1)(NE-1+1)/2
                        print(f"DEBUG LB=5 - Symmetric matrix: {m}x{m}, upper triangle = {num_matrix_elements} elements")
                    else:
                        raise ValueError(f"Unknown LS value: {ls}. Should be 0 or 1")
                    
                    # Total values to read: energy grid + matrix elements
                    values_to_read = ne + num_matrix_elements
                    print(f"DEBUG LB=5 - Total values to read: {ne} (energies) + {num_matrix_elements} (matrix) = {values_to_read}")
                    
                    # Check NT field
                    print(f"DEBUG LB=5 - NT field says: {nt} values")
                    if nt != values_to_read:
                        print(f"WARNING LB=5 - NT mismatch! Expected {values_to_read}, got {nt}")
                    
                    # Read all the data values following the LIST header
                    all_values = []
                    remaining_values = nt
                    print(f"DEBUG LB=5 - Starting to read {remaining_values} values from data lines...")
                    
                    while remaining_values > 0 and current_line < len(lines):
                        data_line = parse_line(lines[current_line])
                        current_line += 1
                        
                        # Extract all 6 values from this line
                        line_values = [
                            data_line.get("C1"), data_line.get("C2"), data_line.get("C3"),
                            data_line.get("C4"), data_line.get("C5"), data_line.get("C6")
                        ]
                        
                        # Only take the values we actually need
                        values_from_this_line = min(6, remaining_values)
                        all_values.extend(line_values[:values_from_this_line])
                        remaining_values -= values_from_this_line
                        
                        print(f"DEBUG LB=5 - Read line {current_line-1}: {line_values[:values_from_this_line]} (remaining: {remaining_values})")
                    
                    print(f"DEBUG LB=5 - Total values read: {len(all_values)}")
                    
                    # Split into energies and matrix values
                    energies = all_values[:ne]
                    matrix_values = all_values[ne:]
                    
                    print(f"DEBUG LB=5 - Energy grid: {len(energies)} values")
                    print(f"DEBUG LB=5 - Matrix values: {len(matrix_values)} values")
                    print(f"DEBUG LB=5 - First few energies: {energies[:5]}")
                    print(f"DEBUG LB=5 - First few matrix values: {matrix_values[:10]}")
                    
                    # Store in record
                    list_record.energies = energies
                    list_record.matrix = matrix_values
                    
                    print(f"DEBUG LB=5 - Stored {len(list_record.energies)} energies and {len(list_record.matrix)} matrix values")
                elif lb == 6:
                    # rectangular matrix: NT=1+NER*NEC
                    ner = ne_field
                    nec = (nt - 1) // ner
                    list_record.ne = ner
                    values_to_read = nt
                    all_values = []
                    while len(all_values) < values_to_read and current_line < len(lines):
                        vl = parse_line(lines[current_line]); current_line += 1
                        for i in range(1,7):
                            if len(all_values) < values_to_read:
                                v = vl.get(f"C{i}")
                                if v is not None:
                                    all_values.append(v)
                    # split row energies, col energies, matrix
                    list_record.row_energies = all_values[:ner]
                    list_record.col_energies = all_values[ner:ner+nec]
                    list_record.rect_matrix   = all_values[ner+nec:]
                    print(f"DEBUG - Parsed LB=6 record NER={ner}, NEC={nec}")

                else:
                    raise ValueError(f"Unsupported LB value: {lb}")
                
                # Add the list record to the sub-subsection
                sub_subsection.records.append(list_record)
            
            # Add the sub-subsection to the subsection
            subsection.sub_subsections.append(sub_subsection)
        
        # Add the subsection to the MT section
        mt_section.add_subsection(subsection)
    
    return mt_section