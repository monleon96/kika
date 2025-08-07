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



    # ------------------------------------------------------------------
    # Basic methods
    # ------------------------------------------------------------------

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





    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
    
    @property
    def covariance_matrix(self) -> np.ndarray:
        """
        Return the full covariance matrix of shape (N·G) × (N·G),
        where N = number of unique (iso, rxn, l) triplets and G = max number of energy groups.
        """
        param_triplets = self._get_param_triplets()
        idx_map = {p: i for i, p in enumerate(param_triplets)}

        # Find maximum energy grid size across all matrices
        max_G = max(matrix.shape[0] for matrix in self.matrices) if self.matrices else 0
        N = len(param_triplets) * max_G
        full = np.zeros((N, N), dtype=float)

        for ir, rr, lr, ic, rc, lc, matrix in zip(
            self.isotope_rows, self.reaction_rows, self.l_rows,
            self.isotope_cols, self.reaction_cols, self.l_cols,
            self.matrices
        ):
            i = idx_map[(ir, rr, lr)]
            j = idx_map[(ic, rc, lc)]
            G = matrix.shape[0]  # Current matrix size
            r0, r1 = i*max_G, i*max_G + G
            c0, c1 = j*max_G, j*max_G + G

            full[r0:r1, c0:c1] = matrix
            if i != j:
                full[c0:c1, r0:r1] = matrix.T

        return full

    @property 
    def correlation_matrix(self) -> np.ndarray:
        """
        Return the correlation matrix computed from the covariance matrix.
        Diagonal elements are forced to 1.0, undefined entries become NaN.
        """
        from mcnpy.cov.decomposition import compute_correlation
        return compute_correlation(self, clip=False, force_diagonal=True)

    @property
    def clipped_correlation_matrix(self) -> np.ndarray:
        """
        Return the correlation matrix clipped to [-1, 1] range.
        Diagonal elements are forced to 1.0, undefined entries become NaN.
        """
        from mcnpy.cov.decomposition import compute_correlation
        return compute_correlation(self, clip=True, force_diagonal=True)

    @property
    def log_covariance_matrix(self) -> np.ndarray:
        """
        Return the log-space covariance matrix.
        Converts relative covariance to log-space using log1p transformation.
        """
        cov_rel = self.covariance_matrix
        Sigma_log = np.log1p(cov_rel)
        return Sigma_log





    # ------------------------------------------------------------------
    # General methods
    # ------------------------------------------------------------------

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
    
    def plot_covariance_heatmap(
        self,
        isotope: int,
        mt: int,
        legendre_coeffs: Union[int, List[int], Tuple[int, int]],
        ax: Optional['plt.Axes'] = None,
        *,
        matrix_type: str = "corr",
        style: str = "default",
        figsize: Tuple[float, float] = (6, 6),
        dpi: int = 300,
        font_family: str = "serif",
        vmax: Optional[float] = None,
        vmin: Optional[float] = None,
        show_uncertainties: bool = False,
        cmap: Optional[any] = None,
        **imshow_kwargs,
    ) -> 'plt.Figure':
        """
        Draw a covariance or correlation matrix heat-map for MF34 angular distribution data.

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
        matrix_type : str
            Type of matrix to plot: "cov" for covariance (uses linear scale) or "corr" for correlation
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
        show_uncertainties : bool
            Whether to show uncertainty plots above the heatmap
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap to use for the heatmap. Can be a string name of any matplotlib 
            colormap (e.g., 'viridis', 'plasma', 'RdYlBu', 'coolwarm') or a matplotlib 
            Colormap object. If None, defaults to 'RdYlGn' for correlation matrices 
            and 'viridis' for covariance matrices.
        **imshow_kwargs
            Additional arguments passed to imshow

        Returns
        -------
        plt.Figure
            The matplotlib figure containing the heatmap and optional uncertainty plots
        """
        from mcnpy.cov.mf34cov_heatmap import plot_mf34_covariance_heatmap

        return plot_mf34_covariance_heatmap(
            mf34_covmat=self,
            isotope=isotope,
            mt=mt,
            legendre_coeffs=legendre_coeffs,
            ax=ax,
            matrix_type=matrix_type,
            style=style,
            figsize=figsize,
            dpi=dpi,
            font_family=font_family,
            vmax=vmax,
            vmin=vmin,
            show_uncertainties=show_uncertainties,
            cmap=cmap,
            **imshow_kwargs
        )

    def get_uncertainties_for_legendre_coefficient(
        self, 
        isotope: int, 
        mt: int, 
        l_coefficient: Union[int, List[int]],
    ) -> Union[Optional[Dict[str, np.ndarray]], Dict[int, Optional[Dict[str, np.ndarray]]]]:
        """
        Extract standard uncertainties (square root of diagonal variance) for Legendre coefficient(s).
        
        Parameters
        ----------
        isotope : int
            Isotope ID
        mt : int
            Reaction MT number
        l_coefficient : int or list of int
            Legendre coefficient index (L value) or list of L values
        return_relative : bool, optional
            If True, return relative uncertainties (fractional). If False, return the raw
            square root of diagonal variance elements (default behavior).
            
        Returns
        -------
        dict or dict of dicts
            For single int: Dictionary containing 'energies' and 'uncertainties' arrays if found, None otherwise.
            For list of ints: Dictionary mapping L coefficient to uncertainty data (or None if not found).
            The uncertainties are the square root of the diagonal elements of the covariance matrix.
            Note: MF34 covariance data is typically stored as relative covariances (fractional uncertainties).
        """
        # Handle single coefficient case
        if isinstance(l_coefficient, int):
            # Find the matrix for this specific (isotope, mt, l_coefficient) combination
            for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c, energy_grid, matrix) in enumerate(zip(
                self.isotope_rows, self.reaction_rows, self.l_rows,
                self.isotope_cols, self.reaction_cols, self.l_cols,
                self.energy_grids, self.matrices
            )):
                # Look for diagonal variance matrix (L = L) for the specified parameters
                if (iso_r == isotope and iso_c == isotope and 
                    mt_r == mt and mt_c == mt and 
                    l_r == l_coefficient and l_c == l_coefficient):
                    
                    # Extract diagonal elements (variances) and take square root
                    diagonal_variances = np.diag(matrix)
                    
                    # Check for negative variances (which shouldn't happen for diagonal blocks)
                    if np.any(diagonal_variances < 0):
                        # Handle negative variances by setting them to zero
                        diagonal_variances = np.maximum(diagonal_variances, 0.0)
                    
                    uncertainties = np.sqrt(diagonal_variances)
                    
                    # Handle energy grid vs matrix size difference
                    # Energy grid defines bin boundaries (N+1 points), matrix defines bin uncertainties (N×N)
                    energy_array = np.array(energy_grid)
                    
                    if len(energy_array) == len(uncertainties) + 1:
                        # Standard case: energy grid has bin boundaries, calculate bin centers
                        energy_array = (energy_array[:-1] + energy_array[1:]) / 2.0
                    elif len(energy_array) == len(uncertainties):
                        # Energy points already represent bin centers or midpoints
                        pass
                    elif len(energy_array) > len(uncertainties):
                        # More energy points than uncertainty values - truncate energy grid
                        if len(energy_array) == len(uncertainties) + 1:
                            # Convert to bin centers
                            energy_array = (energy_array[:-1] + energy_array[1:]) / 2.0
                        else:
                            # Simple truncation
                            energy_array = energy_array[:len(uncertainties)]
                    else:
                        # More uncertainty values than energy points - truncate uncertainties
                        uncertainties = uncertainties[:len(energy_array)]
                    
                    return {
                        'energies': energy_array,
                        'uncertainties': uncertainties
                    }
            
            return None
        
        # Handle list of coefficients case
        elif isinstance(l_coefficient, (list, tuple)):
            result = {}
            for l_coeff in l_coefficient:
                result[l_coeff] = self.get_uncertainties_for_legendre_coefficient(isotope, mt, l_coeff)
            return result
        
        else:
            raise TypeError(f"l_coefficient must be int or list of int, got {type(l_coefficient)}")




    # ------------------------------------------------------------------
    # Decomposition methods
    # ------------------------------------------------------------------

    def cholesky_decomposition(
        self,
        *,
        space: str = "log",        # "log" (default) or "linear"
        jitter_scale: float = 1e-10,
        max_jitter_ratio: float = 1e-3,
        verbose: bool = True,
        logger = None,
    ) -> np.ndarray:
        """
        Robust Cholesky factor L such that M ≈ L L^T.
        
        Parameters
        ----------
        space : str
            "linear" or "log" space for decomposition
        jitter_scale : float
            Base jitter scale for PSD correction
        max_jitter_ratio : float
            Maximum jitter relative to matrix norm
        verbose : bool
            Whether to log progress
        logger : optional
            Logger instance for output
            
        Returns
        -------
        np.ndarray
            Lower triangular Cholesky factor L
        """
        from mcnpy.cov.decomposition import cholesky_decomposition
        return cholesky_decomposition(
            self, 
            space=space, 
            jitter_scale=jitter_scale,
            max_jitter_ratio=max_jitter_ratio,
            verbose=verbose, 
            logger=logger
        )

    def eigen_decomposition(
        self,
        *,
        space: str = "log",
        clip_negatives: bool = True,
        verbose: bool = True,
        logger = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigendecomposition with optional clipping instead of jitter.

        If ``clip_negatives`` is *True*, negative eigenvalues are set to
        zero and the user is informed of the number of clips and the minimum
        original value.
        
        Parameters
        ----------
        space : str
            "linear" or "log" space for decomposition
        clip_negatives : bool
            Whether to clip negative eigenvalues to zero
        verbose : bool
            Whether to log progress
        logger : optional
            Logger instance for output
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Eigenvalues and eigenvectors
        """
        from mcnpy.cov.decomposition import eigen_decomposition
        return eigen_decomposition(
            self,
            space=space,
            clip_negatives=clip_negatives,
            verbose=verbose,
            logger=logger
        )

    def svd_decomposition(
        self,
        *,
        space: str = "log",
        clip_negatives: bool = True,
        verbose: bool = True,
        full_matrices: bool = False,
        logger = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SVD with pre-clipping using eigendecomposition.

        For symmetric matrices, SVD and eigen are equivalent. If
        ``clip_negatives`` is activated, a preliminary eigendecomposition,
        clipping, and reconstruction step is performed before applying SVD,
        ensuring singular values consistent with a PSD matrix.
        
        Parameters
        ----------
        space : str
            "linear" or "log" space for decomposition
        clip_negatives : bool
            Whether to clip negative eigenvalues before SVD
        verbose : bool
            Whether to log progress
        full_matrices : bool
            Whether to return full-sized U and V matrices
        logger : optional
            Logger instance for output
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            U, singular values, V^T matrices
        """
        from mcnpy.cov.decomposition import svd_decomposition
        return svd_decomposition(
            self,
            space=space,
            clip_negatives=clip_negatives,
            verbose=verbose,
            full_matrices=full_matrices,
            logger=logger
        )






    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

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





    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _get_param_triplets(self) -> List[Tuple[int, int, int]]:
        """
        Return a list of all (isotope, reaction, legendre) triplets present,
        sorted first by isotope, then by reaction, then by legendre coefficient.
        """
        triplets = set(zip(self.isotope_rows, self.reaction_rows, self.l_rows)) \
                 | set(zip(self.isotope_cols, self.reaction_cols, self.l_cols))
        return sorted(triplets, key=lambda t: (t[0], t[1], t[2]))
