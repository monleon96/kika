"""
MF file for ENDF files.

MF files contain related nuclear data sections grouped by MT numbers.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, TypeVar, Tuple, List, Any

from .mt import MT
from .mf1.mf1mt import MT451
from ...cov.mf34_covmat import MF34CovMat 

# Type for any MT section class (MT, MT451, etc.)
MTSection = TypeVar('MTSection', bound=MT)

@dataclass
class MF:
    """
    Data class representing an MF file in ENDF format.
    """
    number: int
    sections: Dict[int, Union[MT, MT451]] = field(default_factory=dict)
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
        
    def to_ang_covmat(self) -> MF34CovMat:
        """
        Convert MF34 data to an MF34CovMat object that contains data from all MT sections.
        
        This method aggregates angular covariance data from all MT sections in this MF file
        (if it's MF34) and returns a combined MF34CovMat object.
        
        Returns:
            MF34CovMat object containing data from all MT sections, or None if not MF34
            
        Raises:
            ValueError: If this method is called on an MF that is not MF34
        """
        # Check if this is an MF34 file
        if self.number != 34:
            raise ValueError(f"The to_ang_covmat method is only available for MF34, not MF{self.number}")
        
        # Import here to avoid circular imports
        from ...cov.mf34_covmat import MF34CovMat
        
        # Create a new MF34CovMat object to store the combined data
        combined_ang_covmat = MF34CovMat()
        
        # Loop through all MT sections and combine their data
        for mt_number, mt_section in self.sections.items():
            # Get the individual MF34CovMat for this MT section
            try:
                # The MT section must be an MF34MT object with to_ang_covmat method
                mt_ang_covmat: MF34CovMat = mt_section.to_ang_covmat()
                
                # Add all matrices and their energy grids from this MT section
                for i in range(mt_ang_covmat.num_matrices):
                    combined_ang_covmat.add_matrix(
                        mt_ang_covmat.isotope_rows[i],
                        mt_ang_covmat.reaction_rows[i],
                        mt_ang_covmat.l_rows[i],
                        mt_ang_covmat.isotope_cols[i],
                        mt_ang_covmat.reaction_cols[i],
                        mt_ang_covmat.l_cols[i],
                        mt_ang_covmat.matrices[i],
                        mt_ang_covmat.energy_grids[i],
                        mt_ang_covmat.is_relative[i],
                        mt_ang_covmat.frame[i]
                    )
            except (AttributeError, ValueError) as e:
                # Catch potential errors if a section isn't a valid MF34MT or conversion fails
                print(f"Warning: Could not convert MT{mt_number} to MF34CovMat: {e}")
        
        return combined_ang_covmat
