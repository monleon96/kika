"""
MF file for ENDF files.

MF files contain related nuclear data sections grouped by MT numbers.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Union, TypeVar

from .mt import MT, MT451

# Type for any MT section class (MT, MT451, etc.)
MTSection = TypeVar('MTSection', bound=MT)

@dataclass
class MF:
    """
    Data class representing an MF file in ENDF format.
    """
    number: int
    sections: Dict[int, Union[MT, MT451]] = field(default_factory=dict)
    
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
    
    def __repr__(self):
        return f"MF({self.number}, {len(self.sections)} sections)"
    
    def __getitem__(self, mt_number: int) -> Union[MT, MT451]:
        """Allow accessing MT sections like: mf[451]"""
        if mt_number not in self.sections:
            raise KeyError(f"MT section {mt_number} not found in MF{self.number}")
        return self.sections[mt_number]
