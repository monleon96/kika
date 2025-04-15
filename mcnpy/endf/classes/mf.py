"""
MF file for ENDF files.

MF files contain related nuclear data sections grouped by MT numbers.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, TypeVar, Tuple, List

from .mt import MT
from .mf1.mf1mt import MT451

# Type for any MT section class (MT, MT451, etc.)
MTSection = TypeVar('MTSection', bound=MT)

@dataclass
class MF:
    """
    Data class representing an MF file in ENDF format.
    """
    number: int
    sections: Dict[int, Union[MT, MT451]] = field(default_factory=dict)
    
    # Line count
    num_lines: int = 0  # Number of lines in this MF section
    
    def add_section(self, section: Union[MT, MT451]) -> None:
        """
        Add an MT section to this MF file
        
        Args:
            section: The MT section object to add
        """
        self.sections[section.number] = section
    
    def get_section(self, mt_number: int) -> Optional[Union[MT, MT451]]:
        """
        Get an MT section by number
        
        Args:
            mt_number: The MT section number to retrieve
            
        Returns:
            The MT section object or None if not found
        """
        return self.sections.get(mt_number)
    
    @property
    def mt(self) -> Dict[int, Union[MT, MT451]]:
        """Direct access to MT sections dictionary"""
        return self.sections
    
    @property
    
    def __repr__(self):
        return f"MF({self.number}, {len(self.sections)} sections)"
    
    def __getitem__(self, mt_number: int) -> Union[MT, MT451]:
        """Allow accessing MT sections like: mf[451]"""
        if mt_number not in self.sections:
            raise KeyError(f"MT section {mt_number} not found in MF{self.number}")
        return self.sections[mt_number]
        
    def __str__(self) -> str:
        """
        Convert the MF file to an ENDF format string.
        
        Returns:
            A string containing all MT sections in ENDF format, sorted by MT number
        """
        # Import inside the method to avoid circular imports
        from ..utils import format_endf_data_line, ENDF_FORMAT_INT
        
        # Get all MT sections and sort them by MT number
        sorted_mts = sorted(self.sections.items(), key=lambda x: x[0])
        
        # Convert each MT section to a string and join them
        mt_strings = [str(mt) for _, mt in sorted_mts]
        
        # Join all MT sections with newlines
        result = "\n".join(mt_strings)
        
        # Add the required end-of-file marker - a blank data line with MAT and zeros for MF, MT
        # Use the MAT number from the last MT section if available, otherwise default to 0
        mat = 0
        if sorted_mts:
            _, last_mt = sorted_mts[-1]
            mat = getattr(last_mt, "_mat", 0) or 0
        
        # Format end-of-file marker
        end_line = format_endf_data_line(
            [0, 0, 0, 0, 0, 0],
            mat, 0, 0, 0,  # MF=0, MT=0 for end of file marker
            formats=[ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        
        # Add end-of-file marker to the result
        result += "\n" + end_line
        
        return result
