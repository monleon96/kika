"""
ENDF Parser - Functions for parsing Evaluated Nuclear Data Files (ENDF)
"""
import os
from typing import Dict, List, Optional

from ..classes.endf import ENDF
from ..classes.mf import MF
from ..classes.mt import MT
from ..utils import parse_endf_id
from .mf_parser import parse_mf1, parse_mf2


# Dictionary mapping MF numbers to their parser functions
MF_PARSERS = {
    1: parse_mf1,
    2: parse_mf2,
    # Additional parsers will be registered here
}


def parse_endf_file(filepath: str) -> ENDF:
    """
    Parse a complete ENDF file.
    
    Args:
        filepath: Path to the ENDF file
        
    Returns:
        ENDF object with parsed data
    """
    endf = ENDF()
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # Group lines by MF and MT
        mf_mt_groups = _group_lines_by_mf_mt(lines)
        
        # Parse each MF section
        for mf_number, mt_groups in mf_mt_groups.items():
            if mf_number in MF_PARSERS:
                # Extract just the lines for this MF
                mf_lines = []
                for mt_lines in mt_groups.values():
                    mf_lines.extend(mt_lines)
                
                # Parse the MF section
                mf = MF_PARSERS[mf_number](mf_lines)
                endf.add_file(mf)
    
    return endf


def parse_mf_from_file(filepath: str, mf_number: int) -> Optional[MF]:
    """
    Parse only a specific MF section from an ENDF file.
    
    Args:
        filepath: Path to the ENDF file
        mf_number: MF number to parse
        
    Returns:
        MF object if found and parsed, None otherwise
    """
    if mf_number not in MF_PARSERS:
        raise ValueError(f"No parser available for MF{mf_number}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ENDF file not found at {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            # Find all MF sections matching the specified MF number
            mf_lines = []
            for line in lines:
                if len(line) >= 72:
                    try:
                        current_mf = int(line[70:72])
                        if current_mf == mf_number:
                            mf_lines.append(line)
                    except ValueError:
                        pass
            
            # Parse the MF section if found
            if mf_lines:
                return MF_PARSERS[mf_number](mf_lines)
            
            # Fallback to the group_by method
            mf_mt_groups = _group_lines_by_mf_mt(lines)
            
            # Check if the requested MF is in the file
            if mf_number in mf_mt_groups:
                # Extract just the lines for this MF
                mf_lines = []
                for mt_lines in mf_mt_groups[mf_number].values():
                    mf_lines.extend(mt_lines)
                
                # Parse the MF section
                return MF_PARSERS[mf_number](mf_lines)
    except Exception as e:
        raise RuntimeError(f"Error parsing file: {e}")
    
    return None


def _group_lines_by_mf_mt(lines: List[str]) -> Dict[int, Dict[int, List[str]]]:
    """
    Group ENDF file lines by MF and MT numbers.
    
    Args:
        lines: List of all lines from an ENDF file
        
    Returns:
        Nested dictionary: {mf_number: {mt_number: [lines]}}
    """
    result: Dict[int, Dict[int, List[str]]] = {}
    current_mf = None
    current_mt = None
    current_lines: List[str] = []
    
    for line in lines:
        # Parse identification fields
        mat, mf, mt = parse_endf_id(line)
        
        # Handle end markers or section changes
        if mf == 0 and mt == 0:
            # End of material or file
            if current_mf is not None and current_mt is not None:
                if current_mf not in result:
                    result[current_mf] = {}
                if current_mt not in result[current_mf]:
                    result[current_mf][current_mt] = []
                result[current_mf][current_mt].extend(current_lines)
            break
        
        if current_mf is None or current_mt is None:
            # First section
            current_mf = mf
            current_mt = mt
            current_lines = [line]
        elif current_mf != mf or current_mt != mt:
            # Section change
            if current_mf not in result:
                result[current_mf] = {}
            if current_mt not in result[current_mf]:
                result[current_mf][current_mt] = []
            result[current_mf][current_mt].extend(current_lines)
            
            # Start new section
            current_mf = mf
            current_mt = mt
            current_lines = [line]
        else:
            # Continue current section
            current_lines.append(line)
    
    # Add the last section if needed
    if current_mf is not None and current_mt is not None and current_lines:
        if current_mf not in result:
            result[current_mf] = {}
        if current_mt not in result[current_mf]:
            result[current_mf][current_mt] = []
        result[current_mf][current_mt].extend(current_lines)
    
    return result
