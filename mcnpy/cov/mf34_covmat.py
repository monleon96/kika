from typing import List, Dict, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from mcnpy._utils import create_repr_section # Import the utility function

@dataclass
class MF34CovMat: # Renamed from Ang_CovMat
    """
    Class for storing angular distribution covariance matrix data (MF34).
    
    Attributes
    ----------
    isotope_rows : List[int]
        List of row isotope IDs
    reaction_rows : List[int]
        List of row reaction MT numbers
    l_rows : List[int]
        List of row Legendre coefficient indices
    isotope_cols : List[int]
        List of column isotope IDs
    reaction_cols : List[int]
        List of column reaction MT numbers
    l_cols : List[int]
        List of column Legendre coefficient indices
    energy_grids : List[List[float]]
        List of energy grids for each covariance matrix
    matrices : List[np.ndarray]
        List of covariance matrices
    """
    isotope_rows: List[int] = field(default_factory=list)
    reaction_rows: List[int] = field(default_factory=list)
    l_rows: List[int] = field(default_factory=list)
    isotope_cols: List[int] = field(default_factory=list)
    reaction_cols: List[int] = field(default_factory=list)
    l_cols: List[int] = field(default_factory=list)
    energy_grids: List[List[float]] = field(default_factory=list)
    matrices: List[np.ndarray] = field(default_factory=list)
    
    def add_matrix(self, 
                  isotope_row: int, 
                  reaction_row: int,
                  l_row: int,
                  isotope_col: int, 
                  reaction_col: int,
                  l_col: int,
                  matrix: np.ndarray,
                  energy_grid: List[float]) -> None:
        """
        Add an angular covariance matrix to the collection.
        
        Parameters
        ----------
        isotope_row : int
            Row isotope ID
        reaction_row : int
            Row reaction MT number
        l_row : int
            Row Legendre coefficient index
        isotope_col : int
            Column isotope ID
        reaction_col : int
            Column reaction MT number
        l_col : int
            Column Legendre coefficient index
        matrix : np.ndarray
            Covariance matrix 
        energy_grid : List[float]
            Energy grid for this covariance matrix
        """
        # No validation on matrix shape as each matrix can have a different size
        
        self.isotope_rows.append(isotope_row)
        self.reaction_rows.append(reaction_row)
        self.l_rows.append(l_row)
        self.isotope_cols.append(isotope_col)
        self.reaction_cols.append(reaction_col)
        self.l_cols.append(l_col)
        self.energy_grids.append(energy_grid)
        self.matrices.append(matrix)
    
    @property
    def num_matrices(self) -> int:
        """Number of matrices stored."""
        return len(self.matrices)
    
    @property
    def unique_isotopes(self) -> Set[int]:
        """Set of unique isotope IDs."""
        return set(self.isotope_rows + self.isotope_cols)
    
    @property
    def unique_reactions(self) -> Set[int]:
        """Set of unique reaction MT numbers."""
        return set(self.reaction_rows + self.reaction_cols)
    
    @property
    def unique_legendre_indices(self) -> Set[int]:
        """Set of unique Legendre coefficient indices."""
        return set(self.l_rows + self.l_cols)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the angular covariance matrix data to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the covariance matrix data with columns:
            ISO_H, REAC_H, L_H, ISO_V, REAC_V, L_V, ENE, STD
        """
        # Convert matrices to Python lists for storing in DataFrame
        matrix_lists = [matrix.tolist() for matrix in self.matrices]
        
        # Create DataFrame
        data = {
            "ISO_H": self.isotope_rows,
            "REAC_H": self.reaction_rows,
            "L_H": self.l_rows,
            "ISO_V": self.isotope_cols,
            "REAC_V": self.reaction_cols,
            "L_V": self.l_cols,
            "ENE": self.energy_grids,
            "STD": matrix_lists
        }
        
        return pd.DataFrame(data)
    
    def __str__(self) -> str:
        """String representation showing summary information."""
        unique_isos = len(self.unique_isotopes)
        unique_mts = len(self.unique_reactions)
        unique_ls = len(self.unique_legendre_indices)
        
        return (f"MF34 Angular Covariance Matrix Data:\n" # Updated name
                f"- {self.num_matrices} matrices\n"
                f"- {unique_isos} unique isotopes\n"
                f"- {unique_mts} unique reaction types\n"
                f"- {unique_ls} unique Legendre indices")
    
    def __repr__(self) -> str:
        """
        Get a detailed string representation of the MF34CovMat object.
        
        Returns
        -------
        str
            String representation with content summary
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'MF34 Angular Distribution Covariance Information':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of MF34 covariance matrix data
        description = (
            "This object contains covariance matrix data for angular distributions (MF34).\n"
            "Each matrix represents the covariance between Legendre coefficients for specific\n"
            "isotope-reaction pairs across energy groups.\n\n"
        )
        
        # Create a summary table of data information
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        info_table = "MF34 Covariance Data Summary:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Property", "Value", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        # Add summary information
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Covariance Matrices", self.num_matrices, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Unique Isotopes", len(self.unique_isotopes), 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Unique Reactions", len(self.unique_reactions), 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Unique Legendre Indices", len(self.unique_legendre_indices), 
            width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for data access using create_repr_section
        data_access = {
            ".num_matrices": "Get total number of covariance matrices",
            ".unique_isotopes": "Get set of unique isotope IDs",
            ".unique_reactions": "Get set of unique reaction MT numbers",
            ".unique_legendre_indices": "Get set of unique Legendre indices (L values)"
        }
        
        data_access_section = create_repr_section(
            "How to Access Covariance Data:", 
            data_access, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Add a blank line after the section
        data_access_section += "\n"
        
        # Create a section for available methods using create_repr_section
        methods = {
            ".to_dataframe()": "Convert all MF34 covariance data to DataFrame"
            # Add other methods here if they are implemented later
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        return header + description + info_table + data_access_section + methods_section
