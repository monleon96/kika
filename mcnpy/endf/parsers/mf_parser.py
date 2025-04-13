"""
Parser functions for specific MF sections in ENDF files.
"""
from typing import Dict, List, Optional

from ..classes.mf import MF
from ..classes.mt import MT, MT451
from ..utils import parse_line, parse_endf_id


def parse_mf1(lines: List[str]) -> MF:
    """
    Parse MF1 (General Information) data.
    
    Args:
        lines: List of string lines from the MF1 section
        
    Returns:
        MF object with parsed MF1 data
    """
    mf = MF(number=1)
    
    # Group lines by MT sections
    mt_groups = _group_lines_by_mt(lines)
    
    # Parse MT451 (General Information)
    if 451 in mt_groups:
        # Directly add the MT451 object to the MF sections
        mt451 = parse_mt451(mt_groups[451])
        mf.add_section(mt451)
    
    # Parse other MT sections in MF1 if present
    # Example: if 452 in mt_groups:
    #     mt452 = parse_mt452(mt_groups[452])
    #     mf.add_section(mt452)
    
    return mf


def parse_mt451(lines: List[str]) -> MT451:
    """
    Parse MT451 (General Information) data.
    
    Args:
        lines: List of string lines from the MT451 section
        
    Returns:
        MT451 object with parsed data
    """
    mt451 = MT451()
    
    if len(lines) < 4:  # Need at least 4 lines for basic metadata
        return mt451
    
    # Parse first line - numeric data
    line1 = parse_line(lines[0])
    mt451._za = line1.get("C1")
    mt451._awr = line1.get("C2")
    mt451._lrp = line1.get("C3")
    mt451._lfi = line1.get("C4")
    mt451._nlib = line1.get("C5")
    mt451._nmod = line1.get("C6")
    
    # Parse second line - numeric data
    line2 = parse_line(lines[1])
    mt451._elis = line2.get("C1")
    mt451._sta = line2.get("C2")
    mt451._lis = line2.get("C3")
    mt451._liso = line2.get("C4")
    # Skip C5 as it's a placeholder (0)
    mt451._nfor = line2.get("C6")
    
    # Parse third line - numeric data
    line3 = parse_line(lines[2])
    mt451._awi = line3.get("C1")
    mt451._emax = line3.get("C2")
    mt451._lrel = line3.get("C3")
    # Skip C4 as it's a placeholder (0)
    mt451._nsub = line3.get("C5")
    mt451._nver = line3.get("C6")
    
    # Parse fourth line - numeric data
    line4 = parse_line(lines[3])
    mt451._temp = line4.get("C1")
    # Skip C2 as it's a placeholder (0.0)
    mt451._ldrv = line4.get("C3")
    # Skip C4 as it's a placeholder (0)
    mt451._nwd = line4.get("C5")
    mt451._nxc = line4.get("C6")
    
    # Get MAT number from first line
    mat, mf, mt = parse_endf_id(lines[0])
    mt451._mat = mat
    
    # Determine how many lines are in the MT451 section
    nwd = mt451._nwd if mt451._nwd is not None else 0
    nxc = mt451._nxc if mt451._nxc is not None else 0
    
    # Store the raw text section (NWD lines after the 4 header lines)
    if len(lines) >= 4 + nwd:
        mt451._text_lines = lines[:4+nwd]  # Include the 4 header lines + text lines
    
    # Extract basic text fields for convenience (if lines are available)
    if len(lines) >= 5:
        line = lines[4]
        if len(line) >= 66:
            mt451._zsymam = line[:11].strip()    # cols 1-11
            mt451._alab = line[11:22].strip()    # cols 12-22
            mt451._edate = line[22:32].strip()   # cols 23-32
            mt451._auth = line[33:66].strip()    # cols 34-66
    
    if len(lines) >= 6:
        line = lines[5]
        if len(line) >= 66:
            mt451._ref = line[1:22].strip()      # cols 2-22
            mt451._ddate = line[22:32].strip()   # cols 23-32
            mt451._rdate = line[33:43].strip()   # cols 34-43
            mt451._endate = line[55:63].strip()  # cols 56-63
    
    # Parse directory entries
    dir_start = 4 + nwd  # Directory starts after header + text lines
    for i in range(dir_start, min(dir_start + nxc, len(lines))):
        line_data = parse_line(lines[i])
        mf_val = line_data.get("C3")
        mt_val = line_data.get("C4")
        nc_val = line_data.get("C5")
        mod_val = line_data.get("C6")
        
        # If we hit end marker, break
        if mt_val == 0:
            break
            
        # Add valid entries to directory
        if mf_val is not None and mt_val is not None and nc_val is not None and mod_val is not None:
            mt451.add_directory_entry(mf_val, mt_val, nc_val, mod_val)
    
    return mt451


def parse_mf2(lines: List[str]) -> MF:
    """
    Parse MF2 (Resonance Parameters) data.
    
    Args:
        lines: List of string lines from the MF2 section
        
    Returns:
        MF object with parsed MF2 data
    """
    mf = MF(number=2)
    # Placeholder for MF2 parsing logic
    return mf


def _group_lines_by_mt(lines: List[str]) -> Dict[int, List[str]]:
    """
    Group lines by MT section.
    
    Args:
        lines: List of lines for a single MF section
        
    Returns:
        Dictionary mapping MT numbers to lists of lines
    """
    result: Dict[int, List[str]] = {}
    current_mt = None
    current_lines: List[str] = []
    
    for line in lines:
        # Get the MT number from the line
        if len(line) >= 75:
            try:
                mt = int(line[72:75])
                
                # Handle section changes and end markers
                if mt == 0:
                    # End of section marker
                    if current_mt is not None and current_lines:
                        if current_mt not in result:
                            result[current_mt] = []
                        result[current_mt].extend(current_lines)
                    break
                
                if current_mt is None:
                    # First section
                    current_mt = mt
                    current_lines = [line]
                elif current_mt != mt:
                    # Section change
                    if current_mt not in result:
                        result[current_mt] = []
                    result[current_mt].extend(current_lines)
                    
                    # Start new section
                    current_mt = mt
                    current_lines = [line]
                else:
                    # Continue current section
                    current_lines.append(line)
            except ValueError:
                # If MT can't be parsed, just add to current group
                if current_mt is not None:
                    current_lines.append(line)
        else:
            # Short line, add to current group
            if current_mt is not None:
                current_lines.append(line)
    
    # Add the last section if needed
    if current_mt is not None and current_lines:
        if current_mt not in result:
            result[current_mt] = []
        result[current_mt].extend(current_lines)
    
    return result
