import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from mcnpy._constants import MT_TO_REACTION
from mcnpy.cov.covmat_repr import covmat_repr

@dataclass
class CovMat:
    """
    Class for storing covariance matrix data from SCALE format.
    
    Attributes
    ----------
    num_groups : int
        Number of energy groups in the covariance matrices
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
    """
    num_groups: int = 0
    isotope_rows: List[int] = field(default_factory=list)
    reaction_rows: List[int] = field(default_factory=list)
    isotope_cols: List[int] = field(default_factory=list)
    reaction_cols: List[int] = field(default_factory=list)
    matrices: List[np.ndarray] = field(default_factory=list)
    
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
                result[iso] = set()
            result[iso].add(self.reaction_rows[i])
        
        # Process all column combinations
        for i, iso in enumerate(self.isotope_cols):
            if iso not in result:
                result[iso] = set()
            result[iso].add(self.reaction_cols[i])
        
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
            isotopes.append(iso)
            # Convert set to sorted list for better readability
            mt_list = sorted(reactions)
            reactions_list.append(mt_list)
        
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
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the covariance matrix data with columns:
            ISO_H, REAC_H, ISO_V, REAC_V, STD
        """
        # Convert matrices to Python lists for storing in DataFrame
        matrix_lists = [matrix.tolist() for matrix in self.matrices]
        
        # Create DataFrame
        data = {
            "ISO_H": self.isotope_rows,
            "REAC_H": self.reaction_rows,
            "ISO_V": self.isotope_cols,
            "REAC_V": self.reaction_cols,
            "STD": matrix_lists
        }
        
        return pd.DataFrame(data)
    
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
            df.to_excel(output_path)
        elif os.path.isdir(output_path):
            df.to_excel(os.path.join(output_path, "SCALE_COVMAT.xlsx"))
        else:
            df.to_excel(output_path + ".xlsx")
    
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
            raise ValueError(f"No covariance data available for isotope {isotope}")
        
        # If mt_list is not provided, use all available reactions for this isotope
        if mt_list is None:
            mt_list = sorted(isotope_reactions)
        else:
            # Filter to only include reactions that exist for this isotope
            mt_list = [mt for mt in mt_list if mt in isotope_reactions]
            
        if not mt_list:
            raise ValueError(f"None of the specified reactions are available for isotope {isotope}")
        
        # Initialize the combined covariance matrix
        n_reactions = len(mt_list)
        n_total = n_reactions * self.num_groups
        combined_cov = np.zeros((n_total, n_total))
        
        # Build the covariance matrix
        for i, mt_i in enumerate(mt_list):
            for j, mt_j in enumerate(mt_list):
                # Calculate starting indices for the blocks
                idx_i = i * self.num_groups
                idx_j = j * self.num_groups
                
                # To ensure we get the correct matrix, we need to handle the ordering
                # Order the MT numbers since the covariance matrix may be stored in a specific order
                mt_row, mt_col = (mt_i, mt_j) if mt_i <= mt_j else (mt_j, mt_i)
                
                # Get the covariance matrix for this reaction pair
                cov_matrix = self.get_matrix(isotope, mt_row, isotope, mt_col)
                
                if cov_matrix is not None:
                    # If we had to swap the order, we need to transpose the matrix
                    if mt_i > mt_j:
                        cov_matrix = cov_matrix.T
                        
                    # Insert the covariance block into the combined matrix
                    combined_cov[idx_i:idx_i+self.num_groups, idx_j:idx_j+self.num_groups] = cov_matrix
                    
                    # Ensure symmetry (may not be needed if the original data is symmetric)
                    if i != j:
                        combined_cov[idx_j:idx_j+self.num_groups, idx_i:idx_i+self.num_groups] = cov_matrix.T
        
        return combined_cov
    
    # Use the external repr function
    __repr__ = covmat_repr
    
    def __str__(self) -> str:
        return self.__repr__()
