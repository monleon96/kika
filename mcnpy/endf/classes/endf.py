"""
ENDF file representation.

Contains multiple MF files organized in a dictionary.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from .mf import MF


@dataclass
class ENDF:
    """
    Data class representing an ENDF file.
    """
    files: Dict[int, MF] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_file(self, mf: MF) -> None:
        """Add an MF file to this ENDF file"""
        self.files[mf.number] = mf
    
    def get_file(self, mf_number: int) -> Optional[MF]:
        """Get an MF file by number"""
        return self.files.get(mf_number)
    
    @property
    def mf(self) -> Dict[int, MF]:
        """
        Direct access to MF files dictionary.
        
        This allows accessing MF files like: endf.mf[1]
        """
        return self.files
    
    def __repr__(self):
        return f"ENDF({len(self.files)} files)"
    
    def __getitem__(self, mf_number: int) -> MF:
        """
        Allow accessing MF files using dictionary-like syntax: endf[1]
        
        Args:
            mf_number: The MF file number to retrieve
            
        Returns:
            The requested MF file
            
        Raises:
            KeyError: If the MF file doesn't exist
        """
        if mf_number not in self.files:
            raise KeyError(f"MF file {mf_number} not found in ENDF")
        return self.files[mf_number]
