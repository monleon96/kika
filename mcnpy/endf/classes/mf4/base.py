"""
Classes for MT sections within MF4 (Angular Distributions) in ENDF files.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

from ..mt import MT


@dataclass
class MF4MT(MT):
    """
    Base class for MT sections within MF4 (Angular Distributions).
    
    This class provides a common interface for all angular distribution formats.
    """
    _za: float = None    # ZA identifier
    _awr: float = None   # Atomic weight ratio
    _ltt: int = None     # Angular distribution format flag
    _li: int = None      # Flag for identical particles
    _lct: int = None     # Frame of reference (1=LAB, 2=CM)
    _mat: int = None     # Material identifier
    
    # Line count
    num_lines: int = 0  # Number of lines in this MT section
    
    @property
    def zaid(self) -> float:
        """ZA identifier (1000*Z+A)"""
        return self._za
    
    @property
    def atomic_weight_ratio(self) -> float:
        """Atomic weight ratio"""
        return self._awr
    
    @property
    def distribution_type(self) -> str:
        """Angular distribution format flag"""
        dist_type = ''
        if self._ltt == 0: dist_type = 'Isotropic'
        elif self._ltt == 1: dist_type = 'Legendre'
        elif self._ltt == 2: dist_type = 'Tabulated'
        elif self._ltt == 3: dist_type = 'Legendre and Tabulated'
        else:
            raise ValueError(f"Invalid LTT value: {self._ltt}. Expected 0, 1, 2, or 3.")
        return dist_type
    
    @property
    def is_isotropic(self) -> bool:
        """Flag for identical particles (0=not all isotropic, 1=all isotropic)"""
        if self._li == 0:
            return False
        elif self._li == 1:
            return True
        else:
            raise ValueError(f"Invalid value for LI: {self._li}. Expected 0 or 1.")
    
    @property
    def reference_frame(self) -> str:
        """Frame of reference (1=LAB system, 2=CM system)"""
        if self._lct == 1: 
            return "LAB"
        elif self._lct == 2:
            return "CM"
        else:
            raise ValueError(f"Invalid value for LCT: {self._lct}. Expected 1 or 2.")
    
    @property
    def all_isotropic(self) -> bool:
        """Whether all angular distributions are isotropic"""
        return self._li == 1
    
    @property
    def reference_frame(self) -> str:
        """Reference frame used for angular distributions (LAB or CM)"""
        return "LAB" if self._lct == 1 else "CM" if self._lct == 2 else "UNKNOWN"
    
    def __str__(self) -> str:
        """
        Convert the MF4MT object back to ENDF format string.
        
        Returns:
            Multi-line string in ENDF format
        """
        # Import inside the method to avoid circular imports
        from ...utils import format_endf_data_line, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_BLANK
        
        mat = self._mat if self._mat is not None else 0
        mf = 4
        mt = self.number
        lines = []
        
        # Format first line - header
        line1 = format_endf_data_line(
            [self._za, self._awr, 0, self._ltt, 0, 0],
            mat, mf, mt, 1,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line1)
        
        return "\n".join(lines)










