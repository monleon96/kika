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
    mat: Optional[int] = None  # MAT number from ENDF file
    
    def add_file(self, mf: MF) -> None:
        """Add an MF file to this ENDF file"""
        self.files[mf.number] = mf
    
    def get_file(self, mf_number: int) -> Optional[MF]:
        """Get an MF file by number"""
        return self.files.get(mf_number)
    
    @property
    def zaid(self) -> Optional[int]:
        """
        Get the ZAID number derived from the MAT number.
        
        Returns
        -------
        int or None
            ZAID number if MAT is available and in the mapping, None otherwise
        """
        if self.mat is not None:
            from mcnpy._constants import ENDF_MAT_TO_ZAID
            return ENDF_MAT_TO_ZAID.get(self.mat, None)
        return None
    
    def get_isotope_symbol(self) -> Optional[str]:
        """
        Get the isotope symbol (e.g., 'Fe56') from the ZAID.
        
        Returns
        -------
        str or None
            Isotope symbol like 'Fe56' if ZAID is available, None otherwise
        """
        if self.zaid is not None:
            from mcnpy._utils import zaid_to_symbol
            return zaid_to_symbol(self.zaid)
        return None
    
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
