from typing import List, Dict, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import os

from mcnpy._constants import MT_TO_REACTION
from mcnpy._utils import create_repr_section



@dataclass
class CovMat:
    """
    Class for storing covariance matrix data from nuclear data files (SCALE, NJOY, etc).
    
    Attributes
    ----------
    num_groups : int
        Number of energy groups in the covariance matrices
    energy_grid : Optional[List[float]]
        List of energy grid boundaries (if available)
    isotope_rows : List[int]
        List of row isotope IDs
    reaction_rows : List[int]
        List of row reaction MT numbers
    isotope_cols : List[int]
        List of column isotope IDs
    reaction_cols : List[int]
        List of column reaction MT numbers
    matrices : List[np.ndarray]
        List of covariance matrices (one for each row-column combination)
    cov_type : Optional[str]
        Type of covariance file ("SCALE", "NJOY", or None/other)
    """
    num_groups: int = 0
    energy_grid: Optional[List[float]] = None 
    isotope_rows: List[int] = field(default_factory=list)
    reaction_rows: List[int] = field(default_factory=list)
    isotope_cols: List[int] = field(default_factory=list)
    reaction_cols: List[int] = field(default_factory=list)
    matrices: List[np.ndarray] = field(default_factory=list)
    cov_type: Optional[str] = None  # "SCALE", "NJOY", or None

    @property
    def type(self) -> str:
        """
        Returns the type of covariance matrix ("SCALE", "NJOY", or "UNKNOWN").
        """
        if self.cov_type is not None:
            return self.cov_type.upper()
        return "UNKNOWN"

    def add_matrix(self, 
                  isotope_row: int, 
                  reaction_row: int, 
                  isotope_col: int, 
                  reaction_col: int, 
                  matrix: np.ndarray) -> None:
        """
        Add a covariance matrix to the collection.
        
        Parameters
        ----------
        isotope_row : int
            Row isotope ID
        reaction_row : int
            Row reaction MT number
        isotope_col : int
            Column isotope ID
        reaction_col : int
            Column reaction MT number
        matrix : np.ndarray
            Covariance matrix of shape (num_groups, num_groups)
        """
        # Validate matrix shape
        if matrix.shape != (self.num_groups, self.num_groups):
            raise ValueError(f"Matrix shape {matrix.shape} does not match expected shape ({self.num_groups}, {self.num_groups})")
        
        self.isotope_rows.append(isotope_row)
        self.reaction_rows.append(reaction_row)
        self.isotope_cols.append(isotope_col)
        self.reaction_cols.append(reaction_col)
        self.matrices.append(matrix)
    
    @property
    def num_matrices(self) -> int:
        """
        Get the number of covariance matrices stored.
        
        Returns
        -------
        int
            Number of matrices
        """
        return len(self.matrices)
    
    @property
    def unique_isotopes(self) -> Set[int]:
        """
        Get the set of unique isotope IDs in the covariance matrices.
        
        Returns
        -------
        Set[int]
            Set of unique isotope IDs
        """
        return set(self.isotope_rows + self.isotope_cols)
    
    @property
    def unique_reactions(self) -> Set[int]:
        """
        Get the set of unique reaction MT numbers in the covariance matrices.
        
        Returns
        -------
        Set[int]
            Set of unique reaction MT numbers
        """
        return set(self.reaction_rows + self.reaction_cols)
    
    def get_isotope_reactions(self) -> Dict[int, Set[int]]:
        """
        Get a mapping of isotopes to their available reactions.
        
        Returns
        -------
        Dict[int, Set[int]]
            Dictionary mapping isotope IDs to sets of MT numbers
        """
        result = {}
        
        # Process all row combinations
        for i, iso in enumerate(self.isotope_rows):
            if iso not in result:
                result[iso] = set() # Initialize set if isotope is new
            result[iso].add(self.reaction_rows[i])
        
        # Process all column combinations
        for i, iso in enumerate(self.isotope_cols):
            if iso not in result:
                result[iso] = set() # Initialize set if isotope is new
            result[iso].add(self.reaction_cols[i]) # Add reaction from column pair
        
        return result
    
    def get_reactions_summary(self) -> pd.DataFrame:
        """
        Get a summary of available reactions for each isotope.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'Isotope' and 'Reactions', where 'Reactions' 
            contains the list of available MT numbers for each isotope
        """
        isotope_reactions = self.get_isotope_reactions()
        
        isotopes = []
        reactions_list = []
        
        for iso, reactions in sorted(isotope_reactions.items()):
            isotopes.append(iso) # Add isotope
            reactions_list.append(sorted(list(reactions))) # Add sorted list of reactions
        
        # Create DataFrame with isotopes and their reactions
        df = pd.DataFrame({
            'Isotope': isotopes,
            'Reactions': reactions_list
        })
        
        return df
    
    def get_matrix(self, 
                  isotope_row: int, 
                  reaction_row: int, 
                  isotope_col: int, 
                  reaction_col: int) -> Optional[np.ndarray]:
        """
        Get a specific covariance matrix by isotope and reaction IDs.
        
        Parameters
        ----------
        isotope_row : int
            Row isotope ID
        reaction_row : int
            Row reaction MT number
        isotope_col : int
            Column isotope ID
        reaction_col : int
            Column reaction MT number
            
        Returns
        -------
        np.ndarray or None
            Covariance matrix if found, None otherwise
        """
        for i in range(self.num_matrices):
            if (self.isotope_rows[i] == isotope_row and
                self.reaction_rows[i] == reaction_row and
                self.isotope_cols[i] == isotope_col and
                self.reaction_cols[i] == reaction_col):
                return self.matrices[i]
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the covariance matrix data to a pandas DataFrame.

        Includes an extra row at the beginning to store the energy grid if available.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the covariance matrix data with columns:
            ISO_H, REAC_H, ISO_V, REAC_V, STD
        """
        # Convert matrices to Python lists for storing in DataFrame
        matrix_lists = [matrix.tolist() for matrix in self.matrices]

        # Create DataFrame for covariance matrices
        data = {
            "ISO_H": self.isotope_rows,
            "REAC_H": self.reaction_rows,
            "ISO_V": self.isotope_cols,
            "REAC_V": self.reaction_cols,
            "STD": matrix_lists
        }
        df = pd.DataFrame(data)

        # Add energy grid row if available
        if self.energy_grid is not None:
            energy_grid_row = pd.DataFrame({
                "ISO_H": [0],
                "REAC_H": [0],
                "ISO_V": [0],
                "REAC_V": [0],
                "STD": [self.energy_grid] # Store the list directly
            })
            # Concatenate the energy grid row at the beginning
            df = pd.concat([energy_grid_row, df], ignore_index=True)

        return df
    
    def save_excel(self, output_path: str) -> None:
        """
        Save the covariance matrix data to an Excel file.
        
        Parameters
        ----------
        output_path : str
            Path to save the Excel file
        """
        df = self.to_dataframe()
        
        if output_path.endswith(".xlsx"):
            df.to_excel(output_path, index=False)
        elif os.path.isdir(output_path):
            # Default filename if only directory is provided
            filename = os.path.join(output_path, "covariance_data.xlsx")
            df.to_excel(filename, index=False)
        else:
            # Assume it's a filename without extension, add .xlsx
            df.to_excel(output_path + ".xlsx", index=False)
    
    def get_isotope_covariance_matrix(self, isotope: int, mt_list: Optional[List[int]] = None) -> np.ndarray:
        """
        Build a combined covariance matrix for all specified reactions of a given isotope.
        
        This method constructs a block matrix where each block represents the covariance
        between two specific reactions. The resulting matrix can be used for uncertainty
        propagation calculations.
        
        Parameters
        ----------
        isotope : int
            Isotope ID to build the covariance matrix for
        mt_list : List[int], optional
            List of MT reaction numbers to include. If None, all available reactions for the isotope are used.
            
        Returns
        -------
        np.ndarray
            Combined covariance matrix with shape (N*G, N*G) where N is the number of reactions
            and G is the number of energy groups
            
        Raises
        ------
        ValueError
            If the isotope has no covariance data or if none of the specified reactions are available
        """
        # Get all reactions for this isotope
        isotope_reactions = self.get_isotope_reactions().get(isotope, set())
        
        if not isotope_reactions:
            raise ValueError(f"Isotope {isotope} has no covariance data.")
        
        # If mt_list is not provided, use all available reactions for this isotope
        if mt_list is None:
            mt_list = sorted(list(isotope_reactions))
        else:
            # Filter mt_list to only include available reactions
            mt_list = sorted([mt for mt in mt_list if mt in isotope_reactions])
            
        if not mt_list:
            raise ValueError(f"None of the specified reactions are available for isotope {isotope}.")
        
        # Initialize the combined covariance matrix
        n_reactions = len(mt_list)
        n_total = n_reactions * self.num_groups
        combined_cov = np.zeros((n_total, n_total))
        
        # Build the covariance matrix
        for i, mt_i in enumerate(mt_list):
            for j, mt_j in enumerate(mt_list):
                # Get the sub-matrix C(i, j)
                matrix_ij = self.get_matrix(isotope, mt_i, isotope, mt_j)
                if matrix_ij is not None:
                    # Place it in the correct block
                    row_start = i * self.num_groups
                    row_end = row_start + self.num_groups
                    col_start = j * self.num_groups
                    col_end = col_start + self.num_groups
                    combined_cov[row_start:row_end, col_start:col_end] = matrix_ij
                # else: block remains zero
        
        return combined_cov

    
    def __repr__(self) -> str:
        """
        Get a detailed string representation of the CovMat object.
        
        Returns
        -------
        str
            String representation with content summary
        """
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Covariance Matrix Information':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Description of covariance matrix data
        description = (
            "This object contains covariance matrix data from nuclear data files (SCALE, NJOY, etc).\n"
            "Each matrix represents the covariance between cross sections for specific\n"
            "isotope-reaction pairs across energy groups.\n\n"
            f"Covariance type: {self.type}\n\n"
        )
        
        # Create a summary table of data information
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        info_table = "Covariance Data Summary:\n"
        info_table += "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Property", "Value", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        # Add summary information
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Energy Groups", self.num_groups, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Covariance Matrices", self.num_matrices, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Unique Isotopes", len(self.unique_isotopes), 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Unique Reactions", len(self.unique_reactions), 
            width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for data access using create_repr_section
        data_access = {
            ".unique_isotopes": "Get set of unique isotope IDs",
            ".unique_reactions": "Get set of unique reaction MT numbers",
            ".num_matrices": "Get total number of covariance matrices",
            ".num_groups": "Get number of energy groups"
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
            ".get_matrix(...)": "Get specific covariance matrix",
            ".get_isotope_reactions()": "Get mapping of isotopes to their reactions",
            ".get_reactions_summary()": "Get DataFrame of isotopes with their reactions",
            ".get_isotope_covariance_matrix(...)": "Build combined covariance matrix for an isotope",
            ".to_dataframe()": "Convert all covariance data to DataFrame",
            ".save_excel()": "Save covariance data to Excel file"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        return header + description + info_table + data_access_section + methods_section
