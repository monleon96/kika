"""
MT section for ENDF files.

MT sections contain specific types of nuclear data within an MF file.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

@dataclass
class MT:
    """
    Base class for all MT sections within an MF file.
    """
    number: int
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Line count
    num_lines: int = 0  # Number of lines in this MT section
    
    def __repr__(self):
        return f"MT({self.number})"


