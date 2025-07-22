from typing import List, Dict, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mcnpy._utils import create_repr_section # Import the utility function

@dataclass
class MF34CovMat: 
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
    def isotopes(self) -> Set[int]:
        """Set of unique isotope IDs."""
        return sorted(set(self.isotope_rows + self.isotope_cols))
    
    @property
    def reactions(self) -> Set[int]:
        """Set of unique reaction MT numbers."""
        return sorted(set(self.reaction_rows + self.reaction_cols))
    
    @property
    def legendre_indices(self) -> Set[int]:
        """Set of unique Legendre coefficient indices."""
        return sorted(set(self.l_rows + self.l_cols))
    
    def filter_by_isotope_reaction(self, isotope: int, mt: int) -> "MF34CovMat":
        """
        Return a new MF34CovMat containing only matrices for the given isotope and reaction.
        
        Parameters
        ----------
        isotope : int
            Isotope ID to filter by
        mt : int
            Reaction MT number to filter by
            
        Returns
        -------
        MF34CovMat
            New object with filtered matrices
        """
        new_mf34 = MF34CovMat()
        
        for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c, energy_grid, matrix) in enumerate(zip(
            self.isotope_rows, self.reaction_rows, self.l_rows,
            self.isotope_cols, self.reaction_cols, self.l_cols,
            self.energy_grids, self.matrices
        )):
            if iso_r == isotope and iso_c == isotope and mt_r == mt and mt_c == mt:
                new_mf34.add_matrix(iso_r, mt_r, l_r, iso_c, mt_c, l_c, matrix, energy_grid)
        
        return new_mf34

    def _get_legendre_pairs(self, isotope: int, mt: int) -> List[int]:
        """
        Get sorted list of unique Legendre coefficients for given isotope and MT.
        
        Parameters
        ----------
        isotope : int
            Isotope ID
        mt : int
            Reaction MT number
            
        Returns
        -------
        List[int]
            Sorted list of unique Legendre coefficients
        """
        legendre_set = set()
        
        for iso_r, mt_r, l_r, iso_c, mt_c, l_c in zip(
            self.isotope_rows, self.reaction_rows, self.l_rows,
            self.isotope_cols, self.reaction_cols, self.l_cols
        ):
            if iso_r == isotope and iso_c == isotope and mt_r == mt and mt_c == mt:
                legendre_set.add(l_r)
                legendre_set.add(l_c)
        
        return sorted(list(legendre_set))

    def get_matrix_for_legendre_pair(self, isotope: int, mt: int, l_row: int, l_col: int) -> Tuple[np.ndarray, List[float]]:
        """
        Get the covariance matrix and energy grid for a specific Legendre coefficient pair.
        
        Parameters
        ----------
        isotope : int
            Isotope ID
        mt : int
            Reaction MT number
        l_row : int
            Row Legendre coefficient
        l_col : int
            Column Legendre coefficient
            
        Returns
        -------
        Tuple[np.ndarray, List[float]]
            Covariance matrix and corresponding energy grid
        """
        for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c) in enumerate(zip(
            self.isotope_rows, self.reaction_rows, self.l_rows,
            self.isotope_cols, self.reaction_cols, self.l_cols
        )):
            if (iso_r == isotope and iso_c == isotope and 
                mt_r == mt and mt_c == mt and 
                l_r == l_row and l_c == l_col):
                return self.matrices[i], self.energy_grids[i]
        
        raise ValueError(f"No matrix found for isotope={isotope}, MT={mt}, L_row={l_row}, L_col={l_col}")

    def build_full_correlation_matrix(self, isotope: int, mt: int, legendre_coeffs: List[int]) -> Tuple[np.ndarray, List[float]]:
        """
        Build the full correlation matrix for specified Legendre coefficients.
        
        Parameters
        ----------
        isotope : int
            Isotope ID
        mt : int
            Reaction MT number
        legendre_coeffs : List[int]
            List of Legendre coefficients to include
            
        Returns
        -------
        Tuple[np.ndarray, List[float]]
            Full correlation matrix and energy grid
        """
        # Get all available diagonal matrices to find the maximum energy grid size
        diagonal_matrices = {}
        energy_grids = {}
        max_G = 0
        
        for l_val in legendre_coeffs:
            try:
                matrix, energy_grid = self.get_matrix_for_legendre_pair(isotope, mt, l_val, l_val)
                diagonal_matrices[l_val] = matrix
                energy_grids[l_val] = energy_grid
                max_G = max(max_G, matrix.shape[0])
            except ValueError:
                # If diagonal doesn't exist, we'll handle this below
                pass
        
        if not diagonal_matrices:
            raise ValueError(f"No diagonal matrices found for any Legendre coefficients")
            
        # Use the energy grid from the first available diagonal matrix
        first_l = min(diagonal_matrices.keys())
        reference_energy_grid = energy_grids[first_l]
        G = diagonal_matrices[first_l].shape[0]
        
        # Verify all diagonal matrices have the same size
        for l_val, matrix in diagonal_matrices.items():
            if matrix.shape[0] != G:
                raise ValueError(f"Inconsistent matrix sizes: L={l_val} has {matrix.shape[0]} groups, expected {G}")
        
        # Build the full covariance matrix
        N = len(legendre_coeffs)
        full_cov = np.zeros((N * G, N * G))
        
        for i, l_i in enumerate(legendre_coeffs):
            for j, l_j in enumerate(legendre_coeffs):
                try:
                    matrix, _ = self.get_matrix_for_legendre_pair(isotope, mt, l_i, l_j)
                    if matrix.shape[0] != G or matrix.shape[1] != G:
                        print(f"Warning: Matrix for L=({l_i},{l_j}) has shape {matrix.shape}, expected ({G},{G})")
                        continue
                    full_cov[i*G:(i+1)*G, j*G:(j+1)*G] = matrix
                except ValueError:
                    # If matrix doesn't exist, leave as zeros
                    pass
        
        # Convert to correlation matrix
        std = np.sqrt(np.diag(full_cov))
        denom = np.outer(std, std)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.divide(full_cov, denom, out=np.zeros_like(full_cov), where=denom>0)
        
        # Handle NaN values
        corr[~np.isfinite(corr)] = 0.0
        np.fill_diagonal(corr, 1.0)
        
        return corr, reference_energy_grid

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
        unique_isos = len(self.isotopes)
        unique_mts = len(self.reactions)
        unique_ls = len(self.legendre_indices)
        
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
            "Number of Unique Isotopes", len(self.isotopes), 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Unique Reactions", len(self.reactions), 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Unique Legendre Indices", len(self.legendre_indices), 
            width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for data access using create_repr_section
        data_access = {
            ".num_matrices": "Get total number of covariance matrices",
            ".isotopes": "Get set of unique isotope IDs",
            ".reactions": "Get set of unique reaction MT numbers",
            ".legendre_indices": "Get set of unique Legendre indices (L values)"
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

    def plot_covariance_heatmap(
        self,
        isotope: int,
        mt: int,
        legendre_coeffs: Union[int, List[int], Tuple[int, int]],
        ax: Optional['plt.Axes'] = None,
        *,
        style: str = "default",
        figsize: Tuple[float, float] = (6, 6),
        dpi: int = 300,
        font_family: str = "serif",
        vmax: Optional[float] = None,
        vmin: Optional[float] = None,
        show_energy_ticks: bool = False,
        show_uncertainties: bool = True,
        cmap: Optional[any] = None,
        **imshow_kwargs,
    ) -> Union['plt.Axes', Tuple['plt.Axes', List['plt.Axes']]]:
        """
        Draw a correlation-matrix heat-map for MF34 angular distribution data.

        Parameters
        ----------
        isotope : int
            Isotope ID
        mt : int
            Reaction MT number
        legendre_coeffs : int, list of int, or tuple of (row_l, col_l)
            Legendre coefficient(s). Can be:
            - Single int: diagonal block for that L
            - List of ints: diagonal blocks for those L values
            - Tuple of (row_l, col_l): off-diagonal block between row and column L
        ax : plt.Axes, optional
            Matplotlib axes to draw into (only used when show_uncertainties=False)
        style : str
            Plot style: 'default', 'dark', 'paper', 'publication', 'presentation'
        figsize : tuple
            Figure size in inches (width, height)
        dpi : int
            Dots per inch for figure resolution
        font_family : str
            Font family for text elements
        vmax, vmin : float, optional
            Color scale limits
        show_energy_ticks : bool
            Whether to show subtle energy group tick marks at the heatmap borders
        show_uncertainties : bool
            Whether to show uncertainty plots above the heatmap
        **imshow_kwargs
            Additional arguments passed to imshow

        Returns
        -------
        plt.Axes or tuple
            If show_uncertainties=False: returns the heatmap axes
            If show_uncertainties=True: returns (heatmap_axes, uncertainty_axes_list)
        """
        from mcnpy.cov.mf34cov_heatmap import plot_mf34_covariance_heatmap

        return plot_mf34_covariance_heatmap(
            mf34_covmat=self,
            isotope=isotope,
            mt=mt,
            legendre_coeffs=legendre_coeffs,
            ax=ax,
            style=style,
            figsize=figsize,
            dpi=dpi,
            font_family=font_family,
            vmax=vmax,
            vmin=vmin,
            show_energy_ticks=show_energy_ticks,
            show_uncertainties=show_uncertainties,
            cmap=cmap,
            **imshow_kwargs
        )


