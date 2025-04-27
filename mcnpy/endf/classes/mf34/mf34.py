"""
Classes for MT sections within MF34 (Angular Distribution Covariances) in ENDF files.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from ..mt import MT
from ....cov.covmat import Ang_CovMat

@dataclass
class SubSubsectionRecord:
    """Record representing a LIST within a sub-subsection"""
    ls: int = None       # Flag for symmetric matrix (1=yes, 0=no)
    lb: int = None       # Flag for covariance pattern
    nt: int = None       # Total number of items in the list
    ne: int = None       # Number of energy entries
    
    # For LB=5 (original format)
    energies: List[float] = field(default_factory=list)   # Energy grid
    matrix: List[float] = field(default_factory=list)     # Fk,k' covariance matrix values
    
    # For LB=0-4
    lt: int = None       # Number of pairs in the second array
    np: int = None       # Total number of pairs
    e_table_k: List[float] = field(default_factory=list)  # Energies in first table
    f_table_k: List[float] = field(default_factory=list)  # F-values in first table
    e_table_l: List[float] = field(default_factory=list)  # Energies in second table (when LT > 0)
    f_table_l: List[float] = field(default_factory=list)  # F-values in second table (when LT > 0)


@dataclass
class SubSubsection:
    """Sub-subsection for a particular (L,L1) pair"""
    l: int = None        # Index of Legendre coefficient for reaction MT
    l1: int = None       # Index of Legendre coefficient for reaction MT1
    lct: int = None      # Frame of reference flag
    ni: int = None       # Number of LIST records
    records: List[SubSubsectionRecord] = field(default_factory=list)  # LIST records in this sub-subsection


@dataclass
class Subsection:
    """Subsection for a particular MT1 reaction"""
    mt1: int = None      # 'Other' reaction type 
    nl: int = None       # Number of Legendre coefficients for MT
    nl1: int = None      # Number of Legendre coefficients for MT1
    mat1: float = None   # MAT1 value (normally 0.0)
    sub_subsections: List[SubSubsection] = field(default_factory=list)  # Sub-subsections for this subsection


@dataclass
class MF34MT(MT):
    """
    Class for MT sections within MF34 (Angular Distribution Covariances).
    
    This class stores covariance data for angular distributions.
    """
    _za: float = None      # ZA identifier
    _awr: float = None     # Atomic weight ratio
    _ltt: int = None       # Flag to specify representation used 
                           # 1 = data given for Legendre coefficients starting with a1 or higher order
                           # 2 = data given for Legendre coefficients starting with a0
                           # 3 = if either L or L1 = 0 anywhere in the section
    _nmt1: int = None      # Number of subsections present in this MT section
    _mat: int = None       # Material identifier
    
    # Subsections (one per MT1)
    _subsections: List[Subsection] = field(default_factory=list)
    
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
    def representation_flag(self) -> str:
        """Flag indicating which Legendre representation is used"""
        if self._ltt == 1:
            return "Legendre coefficients starting with a1"
        elif self._ltt == 2:
            return "Legendre coefficients starting with a0"
        elif self._ltt == 3:
            return "Either L or L1 = 0"
        else:
            raise ValueError(f"Invalid LTT value: {self._ltt}. Expected 1, 2, or 3.")
    
    @property
    def num_subsections(self) -> int:
        """Number of subsections"""
        return self._nmt1
    
    @property
    def subsections(self) -> List[Subsection]:
        """Subsections data"""
        return self._subsections
    
    def add_subsection(self, subsection: Subsection) -> None:
        """Add a subsection to this MT"""
        self._subsections.append(subsection)
        
    def get_subsection(self, mt1: int) -> Optional[Subsection]:
        """Get a subsection by MT1 value"""
        for subsection in self._subsections:
            if subsection.mt1 == mt1:
                return subsection
        return None
    
    def __str__(self) -> str:
        """
        Convert the MF34MT object back to ENDF format string.
        
        Returns:
            Multi-line string in ENDF format
        """
        # Import inside the method to avoid circular imports
        from ...utils import format_endf_data_line, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_BLANK
        
        mat = self._mat if self._mat is not None else 0
        mf = 34
        mt = self.number
        lines = []
        
        # Helper function to remove line number field and replace with spaces
        def blank_line_number(line: str) -> str:
            # Replace line number field (positions 76-80) with spaces
            return line[:75] + "     "
        
        # Format first line - header - ZA, AWR as float, rest as integers with zeros printed
        line1 = format_endf_data_line(
            [self._za, self._awr, 0, self._ltt, 0, self._nmt1],
            mat, mf, mt, 0,  # Use 0 for line number, will be replaced with blanks
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, 
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT]
        )
        lines.append(blank_line_number(line1))
        
        # Format subsections
        for subsection in self._subsections:
            # Format subsection header line
            subsec_header = format_endf_data_line(
                [0.0, 0.0, subsection.mat1, subsection.mt1, subsection.nl, subsection.nl1],
                mat, mf, mt, 0,  # Use 0 for line number, will be replaced with blanks
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                         ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
            )
            lines.append(blank_line_number(subsec_header))
            
            # Format sub-subsections
            for sub_subsec in subsection.sub_subsections:
                # Format sub-subsection header
                format_lct = ENDF_FORMAT_INT_ZERO if sub_subsec.lct == 0 else ENDF_FORMAT_INT
                sub_header = format_endf_data_line(
                    [0.0, 0.0, sub_subsec.l, sub_subsec.l1, sub_subsec.lct, sub_subsec.ni],
                    mat, mf, mt, 0,  # Use 0 for line number, will be replaced with blanks
                    formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                             ENDF_FORMAT_INT, format_lct, ENDF_FORMAT_INT]
                )
                lines.append(blank_line_number(sub_header))
                
                # Format LIST records
                for record in sub_subsec.records:
                    # Different formatting based on LB value
                    if record.lb >= 0 and record.lb <= 4:
                        # Format LIST record header for LB=0-4
                        # For LB=0-4, LT is in C3 position, and NP is in C6 position
                        record_header = format_endf_data_line(
                            [0.0, 0.0, record.lt, record.lb, record.nt, record.np],
                            mat, mf, mt, 0,  # Use 0 for line number, will be replaced with blanks
                            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                                     ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                        )
                        lines.append(blank_line_number(record_header))
                        
                        # Format E-table and F-table values in blocks of 6
                        # Combine all E-k, F-k and E-l, F-l values in the correct order
                        all_values = []
                        
                        # Add E-k, F-k pairs
                        for i in range(len(record.e_table_k)):
                            all_values.append(record.e_table_k[i])
                            all_values.append(record.f_table_k[i])
                        
                        # Add E-l, F-l pairs if they exist
                        if record.lt > 0:
                            for i in range(len(record.e_table_l)):
                                all_values.append(record.e_table_l[i])
                                all_values.append(record.f_table_l[i])
                        
                        # Format in blocks of 6
                        current_values = []
                        for val in all_values:
                            current_values.append(val)
                            if len(current_values) == 6:
                                value_line = format_endf_data_line(current_values, mat, mf, mt, 0)
                                lines.append(blank_line_number(value_line))
                                current_values = []
                        
                        # Add any remaining values
                        if current_values:
                            # Pad with None for blanks
                            while len(current_values) < 6:
                                current_values.append(None)
                            value_line = format_endf_data_line(
                                current_values, 
                                mat, mf, mt, 0,
                                formats=[ENDF_FORMAT_FLOAT] * len(current_values) + [ENDF_FORMAT_BLANK] * (6 - len(current_values))
                            )
                            lines.append(blank_line_number(value_line))
                            
                    elif record.lb == 5:
                        # Format LIST record header for LB=5 (original format)
                        record_header = format_endf_data_line(
                            [0.0, 0.0, record.ls, record.lb, record.nt, record.ne],
                            mat, mf, mt, 0,  # Use 0 for line number, will be replaced with blanks
                            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT,
                                     ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                        )
                        lines.append(blank_line_number(record_header))
                        
                        # Format energy grid and matrix values in blocks of 6
                        all_values = record.energies + record.matrix
                        current_values = []
                        
                        for val in all_values:
                            current_values.append(val)
                            if len(current_values) == 6:
                                value_line = format_endf_data_line(current_values, mat, mf, mt, 0)
                                lines.append(blank_line_number(value_line))
                                current_values = []
                        
                        # Add any remaining values
                        if current_values:
                            # Pad with None for blanks
                            while len(current_values) < 6:
                                current_values.append(None)
                            value_line = format_endf_data_line(
                                current_values, 
                                mat, mf, mt, 0,
                                formats=[ENDF_FORMAT_FLOAT] * len(current_values) + [ENDF_FORMAT_BLANK] * (6 - len(current_values))
                            )
                            lines.append(blank_line_number(value_line))
        
        # End of section marker - all integers - this one keeps the line number (99999)
        end_line = format_endf_data_line(
            [0, 0, 0, 0, 0, 0],
            mat, mf, 0, 99999,  # Note MT=0 for end of section, use 99999 for the standard end marker
            formats=[ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, 
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        lines.append(end_line)
        
        return "\n".join(lines)
    
    def to_ang_covmat(self) -> 'Ang_CovMat':
        """
        Convert the MF34MT data to an Ang_CovMat object.
        
        This method creates an angular covariance matrix representation that can be used
        for uncertainty analysis. It converts both symmetric and asymmetric matrices
        to their full form.
        
        Returns
        -------
        Ang_CovMat
            Angular distribution covariance matrix object
        """
        # Import at function level to avoid circular imports
        from mcnpy.cov.covmat import Ang_CovMat
        import numpy as np
        
        # Get the isotope ID from the ZA value
        isotope = int(self._za)
        reaction = self.number
        
        # Create a new Ang_CovMat object
        ang_covmat = Ang_CovMat()
        
        # Process each subsection (different MT1 values)
        for subsection in self._subsections:
            mt1 = subsection.mt1
            
            # Process each sub-subsection (different L, L1 combinations)
            for sub_subsec in subsection.sub_subsections:
                l = sub_subsec.l
                l1 = sub_subsec.l1
                
                # Process each LIST record in the sub-subsection
                for record in sub_subsec.records:
                    # Set the number of energy points if not already set
                    if ang_covmat.num_energies == 0:
                        ang_covmat.num_energies = record.ne
                    
                    ls = record.ls  # Flag for symmetric matrix
                    ne = record.ne  # Number of energy points
                    energies = record.energies  # Energy grid
                    
                    # Extract the raw matrix values
                    raw_matrix = np.array(record.matrix)
                    
                    # Calculate the full matrix shape and indices based on symmetry
                    if ls == 1:  # Symmetric matrix - upper triangular stored
                        # Create a full matrix from the upper triangular part
                        full_matrix = np.zeros((ne, ne))
                        idx = 0
                        
                        # Fill the upper triangular part and mirror to lower triangular
                        for i in range(ne):
                            for j in range(i, ne):
                                if idx < len(raw_matrix):
                                    full_matrix[i, j] = raw_matrix[idx]
                                    # Mirror to ensure symmetry
                                    full_matrix[j, i] = raw_matrix[idx]
                                    idx += 1
                    else:  # Asymmetric matrix - full matrix stored
                        # Reshape the raw matrix into the full matrix form
                        full_matrix = np.zeros((ne, ne))
                        idx = 0
                        
                        # Fill the matrix row by row
                        for i in range(ne):
                            for j in range(ne):
                                if idx < len(raw_matrix):
                                    full_matrix[i, j] = raw_matrix[idx]
                                    idx += 1
                    
                    # Add the matrix to the Ang_CovMat object
                    ang_covmat.add_matrix(
                        isotope, reaction, l,
                        isotope, mt1, l1,
                        full_matrix
                    )
        
        return ang_covmat
