"""
Multigroup MF34 angular distribution covariance matrix data structure.

This module contains the MGMF34CovMat class for storing multigroup covariance
matrices derived from MF34 angular distribution data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional, Dict
from matplotlib import pyplot as plt
from .._utils import create_repr_section


@dataclass
class MGMF34CovMat:
    """
    Multigroup covariance matrix class for MF34 angular distribution data.
    
    This class stores multigroup covariance matrices derived from MF34 data,
    maintaining similar structure to MF34CovMat but for multigroup data.
    
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
    energy_grid : np.ndarray
        Energy group boundaries [G_0, G_1, ..., G_n] (n+1 edges for n groups)
    relative_matrices : List[np.ndarray]
        List of relative covariance matrices
    absolute_matrices : List[np.ndarray]
        List of absolute covariance matrices
    _mg_means_row : List[np.ndarray]
        Multigroup means for row Legendre coefficients (bar{a}_l,g) - private
    _mg_means_col : List[np.ndarray]
        Multigroup means for column Legendre coefficients (bar{a}_l',g') - private
    frame : List[str]
        List of reference frames for each matrix
    weighting_function : str
        Description of weighting function used (e.g., "constant", "flux-weighted")
    legendre_coefficients : Dict[Tuple[int, int, int], np.ndarray]
        Dictionary mapping (isotope, mt, l) to multigroup Legendre coefficients
    """
    isotope_rows: List[int] = field(default_factory=list)
    reaction_rows: List[int] = field(default_factory=list)
    l_rows: List[int] = field(default_factory=list)
    isotope_cols: List[int] = field(default_factory=list)
    reaction_cols: List[int] = field(default_factory=list)
    l_cols: List[int] = field(default_factory=list)
    energy_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    relative_matrices: List[np.ndarray] = field(default_factory=list)
    absolute_matrices: List[np.ndarray] = field(default_factory=list)
    _mg_means_row: List[np.ndarray] = field(default_factory=list)
    _mg_means_col: List[np.ndarray] = field(default_factory=list)
    frame: List[str] = field(default_factory=list)
    weighting_function: str = "constant"
    legendre_coefficients: Dict[Tuple[int, int, int], np.ndarray] = field(default_factory=dict)

    @property
    def num_matrices(self) -> int:
        """Number of matrices stored."""
        return len(self.relative_matrices)
    
    @property
    def num_groups(self) -> int:
        """Number of energy groups."""
        return len(self.energy_grid) - 1 if len(self.energy_grid) > 0 else 0
    
    @property
    def isotopes(self) -> List[int]:
        """Set of unique isotope IDs."""
        return sorted(set(self.isotope_rows + self.isotope_cols))
    
    @property
    def reactions(self) -> List[int]:
        """Set of unique reaction MT numbers."""
        return sorted(set(self.reaction_rows + self.reaction_cols))
    
    @property
    def legendre_indices(self) -> List[int]:
        """Set of unique Legendre coefficient indices."""
        return sorted(set(self.l_rows + self.l_cols))

    def add_matrix(self, 
                  isotope_row: int, 
                  reaction_row: int,
                  l_row: int,
                  isotope_col: int, 
                  reaction_col: int,
                  l_col: int,
                  relative_matrix: np.ndarray,
                  absolute_matrix: np.ndarray,
                  mg_means_row: np.ndarray,
                  mg_means_col: np.ndarray,
                  frame: str):
        """
        Add a multigroup covariance matrix to the collection.
        
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
        relative_matrix : np.ndarray
            Multigroup relative covariance matrix
        absolute_matrix : np.ndarray
            Multigroup absolute covariance matrix
        mg_means_row : np.ndarray
            Multigroup means for row Legendre coefficients
        mg_means_col : np.ndarray
            Multigroup means for column Legendre coefficients
        frame : str
            Reference frame
        """
        self.isotope_rows.append(isotope_row)
        self.reaction_rows.append(reaction_row)
        self.l_rows.append(l_row)
        self.isotope_cols.append(isotope_col)
        self.reaction_cols.append(reaction_col)
        self.l_cols.append(l_col)
        self.relative_matrices.append(relative_matrix)
        self.absolute_matrices.append(absolute_matrix)
        self._mg_means_row.append(mg_means_row)
        self._mg_means_col.append(mg_means_col)
        self.frame.append(frame)
        
        # Update legendre_coefficients dictionary
        key_row = (isotope_row, reaction_row, l_row)
        key_col = (isotope_col, reaction_col, l_col)
        
        # Store the multigroup Legendre coefficients
        if key_row not in self.legendre_coefficients:
            self.legendre_coefficients[key_row] = mg_means_row.copy()
        if key_col not in self.legendre_coefficients and key_col != key_row:
            self.legendre_coefficients[key_col] = mg_means_col.copy()

    def summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame with one row per matrix.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with matrix summary information
        """
        data = {
            "isotope_row": self.isotope_rows,
            "MT_row": self.reaction_rows,
            "L_row": self.l_rows,
            "isotope_col": self.isotope_cols,
            "MT_col": self.reaction_cols,
            "L_col": self.l_cols,
            "num_groups": [matrix.shape[0] for matrix in self.relative_matrices],
            "frame": self.frame
        }
        return pd.DataFrame(data)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the multigroup angular covariance matrix data to a pandas DataFrame.
        
        This method creates a DataFrame with the same format as MF34CovMat.to_dataframe()
        but adapted for multigroup data structure.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the multigroup covariance matrix data with columns:
            ISO_H, REAC_H, L_H, ISO_V, REAC_V, L_V, ENE, STD
            where ENE contains the energy group boundaries and STD contains the 
            relative covariance matrices.
        """
        # Convert relative matrices to Python lists for storing in DataFrame
        matrix_lists = [matrix.tolist() for matrix in self.relative_matrices]
        
        # For multigroup data, ENE contains the energy group boundaries (same for all matrices)
        energy_group_lists = [self.energy_grid.tolist() for _ in range(self.num_matrices)]
        
        # Create DataFrame with same column names as MF34CovMat
        data = {
            "ISO_H": self.isotope_rows,
            "REAC_H": self.reaction_rows,
            "L_H": self.l_rows,
            "ISO_V": self.isotope_cols,
            "REAC_V": self.reaction_cols,
            "L_V": self.l_cols,
            "ENE": energy_group_lists,
            "STD": matrix_lists
        }
        
        return pd.DataFrame(data)

    def filter_by_isotope_reaction(self, isotope: int, mt: int) -> "MGMF34CovMat":
        """
        Return a new MGMF34CovMat containing only matrices for the specified isotope and MT reaction.
        
        This method filters the covariance matrices to include only those where both
        row and column parameters match the specified isotope and MT reaction.
        
        Parameters
        ----------
        isotope : int
            Isotope ID to filter by
        mt : int
            Reaction MT number to filter by
            
        Returns
        -------
        MGMF34CovMat
            New MGMF34CovMat object containing only the filtered matrices
        """
        # Find indices where both row and column match the specified isotope and MT
        matching_indices = []
        for i, (iso_r, mt_r, iso_c, mt_c) in enumerate(zip(
            self.isotope_rows, self.reaction_rows, 
            self.isotope_cols, self.reaction_cols
        )):
            if iso_r == isotope and mt_r == mt and iso_c == isotope and mt_c == mt:
                matching_indices.append(i)
        
        # Create new MGMF34CovMat with filtered data
        filtered_mg = MGMF34CovMat()
        
        # Copy the energy group edges (same for all matrices)
        filtered_mg.energy_grid = self.energy_grid.copy()
        filtered_mg.weighting_function = self.weighting_function
        
        # Copy relevant legendre_coefficients
        for key, coeffs in self.legendre_coefficients.items():
            iso_key, mt_key, l_key = key
            if iso_key == isotope and mt_key == mt:
                filtered_mg.legendre_coefficients[key] = coeffs.copy()
        
        for i in matching_indices:
            filtered_mg.isotope_rows.append(self.isotope_rows[i])
            filtered_mg.reaction_rows.append(self.reaction_rows[i])
            filtered_mg.l_rows.append(self.l_rows[i])
            filtered_mg.isotope_cols.append(self.isotope_cols[i])
            filtered_mg.reaction_cols.append(self.reaction_cols[i])
            filtered_mg.l_cols.append(self.l_cols[i])
            filtered_mg.relative_matrices.append(self.relative_matrices[i])
            filtered_mg.absolute_matrices.append(self.absolute_matrices[i])
            filtered_mg._mg_means_row.append(self._mg_means_row[i])
            filtered_mg._mg_means_col.append(self._mg_means_col[i])
            filtered_mg.frame.append(self.frame[i])
        
        return filtered_mg

    @property
    def covariance_matrix(self) -> np.ndarray:
        """
        Return the full multigroup covariance matrix of shape (N·G) × (N·G),
        where N = number of unique (iso, rxn, l) triplets and G = number of energy groups.
        
        For multigroup data, this constructs a block matrix where each block corresponds
        to covariance between different Legendre coefficients.
        """
        if not self.relative_matrices:
            return np.array([])
            
        param_triplets = self._get_param_triplets()
        idx_map = {p: i for i, p in enumerate(param_triplets)}

        G = self.num_groups  # Number of energy groups (same for all matrices)
        N = len(param_triplets)
        full = np.zeros((N * G, N * G), dtype=float)

        for ir, rr, lr, ic, rc, lc, matrix in zip(
            self.isotope_rows, self.reaction_rows, self.l_rows,
            self.isotope_cols, self.reaction_cols, self.l_cols,
            self.relative_matrices
        ):
            i = idx_map[(ir, rr, lr)]
            j = idx_map[(ic, rc, lc)]
            r0, r1 = i * G, (i + 1) * G
            c0, c1 = j * G, (j + 1) * G

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
        cov_matrix = self.covariance_matrix
        if cov_matrix.size == 0:
            return cov_matrix
            
        # Compute correlation matrix
        diag_sqrt = np.sqrt(np.diag(cov_matrix))
        
        # Handle zero or negative diagonal elements
        valid_diag = diag_sqrt > 0
        corr_matrix = np.full_like(cov_matrix, np.nan)
        
        if np.any(valid_diag):
            # Create outer product for normalization
            diag_outer = np.outer(diag_sqrt, diag_sqrt)
            
            # Only compute correlation for valid diagonal elements
            valid_mask = np.outer(valid_diag, valid_diag)
            corr_matrix[valid_mask] = (cov_matrix[valid_mask] / diag_outer[valid_mask])
            
            # Set diagonal to 1.0 for valid elements
            np.fill_diagonal(corr_matrix, 1.0)
        
        return corr_matrix

    @property
    def energy_grids(self) -> List[List[float]]:
        """
        Return energy grids in the format expected by plotting functions.
        For multigroup data, this returns the energy group boundaries for each matrix.
        """
        return [self.energy_grid.tolist() for _ in range(self.num_matrices)]

    @property
    def matrices(self) -> List[np.ndarray]:
        """
        Return matrices in the format expected by plotting functions.
        For compatibility, this returns the relative matrices.
        """
        return self.relative_matrices

    def _get_param_triplets(self) -> List[Tuple[int, int, int]]:
        """
        Return a list of all (isotope, reaction, legendre) triplets present,
        sorted first by isotope, then by reaction, then by legendre coefficient.
        """
        triplets = set(zip(self.isotope_rows, self.reaction_rows, self.l_rows)) \
                 | set(zip(self.isotope_cols, self.reaction_cols, self.l_cols))
        return sorted(triplets, key=lambda t: (t[0], t[1], t[2]))

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
        Draw a covariance or correlation matrix heat-map for multigroup MF34 angular distribution data.

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

    def plot_uncertainties(
        self,
        isotope: int,
        mt: int,
        legendre_coeffs: Union[int, List[int]],
        ax: Optional['plt.Axes'] = None,
        *,
        uncertainty_type: str = "relative",
        style: str = "default",
        figsize: Tuple[float, float] = (8, 5),
        dpi: int = 100,
        font_family: str = "serif",
        legend_loc: str = "best",
        energy_range: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> 'plt.Figure':
        """
        Plot uncertainties for multigroup MF34 angular distribution data for specific Legendre coefficients.
        
        This method extracts and plots the diagonal uncertainties from the multigroup covariance matrix
        for the specified isotope, MT reaction, and Legendre coefficients.
        
        Parameters
        ----------
        isotope : int
            Isotope ID
        mt : int
            Reaction MT number
        legendre_coeffs : int or list of int
            Legendre coefficient(s) to plot uncertainties for.
            Can be a single int or a list of ints.
        ax : plt.Axes, optional
            Matplotlib axes to draw into. If None, creates new figure.
        uncertainty_type : str, default "relative"
            Type of uncertainty to plot: "relative" (%) or "absolute"
        style : str, default "default"
            Plot style: 'default', 'dark', 'paper', 'publication', 'presentation'
        figsize : tuple, default (8, 5)
            Figure size in inches (width, height)
        dpi : int, default 100
            Dots per inch for figure resolution
        font_family : str, default "serif"
            Font family for text elements
        legend_loc : str, default "best"
            Legend location
        energy_range : tuple of float, optional
            Energy range (min, max) for x-axis. If None, uses the full data range.
            Values are used directly without clamping to data range.
        **kwargs
            Additional arguments passed to matplotlib plot functions
        
        Returns
        -------
        plt.Figure
            The matplotlib figure containing the uncertainty plots
        
        Examples
        --------
        Plot relative uncertainties for Legendre coefficients L=1,2,3:
        
        >>> fig = mg_mf34_covmat.plot_uncertainties(isotope=92235, mt=2, 
        ...                                        legendre_coeffs=[1, 2, 3])
        >>> fig.show()
        
        Plot absolute uncertainties for a single Legendre coefficient:
        
        >>> fig = mg_mf34_covmat.plot_uncertainties(isotope=92235, mt=2,
        ...                                        legendre_coeffs=1, 
        ...                                        uncertainty_type="absolute")
        >>> fig.show()
        """
        from mcnpy.cov.mf34cov_heatmap import plot_mf34_uncertainties

        return plot_mf34_uncertainties(
            mf34_covmat=self,
            isotope=isotope,
            mt=mt,
            legendre_coeffs=legendre_coeffs,
            ax=ax,
            uncertainty_type=uncertainty_type,
            style=style,
            figsize=figsize,
            dpi=dpi,
            font_family=font_family,
            legend_loc=legend_loc,
            energy_range=energy_range,
            **kwargs
        )

    def __str__(self) -> str:
        """String representation showing summary information."""
        return (f"Multigroup MF34 Angular Covariance Matrix Data:\n"
                f"- {self.num_matrices} matrices\n"
                f"- {self.num_groups} energy groups\n"
                f"- {len(self.isotopes)} unique isotopes\n"
                f"- {len(self.reactions)} unique reaction types\n"
                f"- {len(self.legendre_indices)} unique Legendre indices\n"
                f"- Weighting: {self.weighting_function}")

    def __repr__(self) -> str:
        """Detailed string representation."""
        header_width = 85
        header = "=" * header_width + "\n"
        header += f"{'Multigroup MF34 Angular Distribution Covariance Information':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        description = (
            "This object contains multigroup covariance matrix data for angular distributions\n"
            "derived from MF34 data. Each matrix represents covariance between Legendre\n"
            "coefficients for specific isotope-reaction pairs across energy groups.\n\n"
        )
        
        # Create summary table
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3
        
        info_table = "-" * header_width + "\n"
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Property", "Value", width1=property_col_width, width2=value_col_width)
        info_table += "-" * header_width + "\n"
        
        # Add summary information
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Covariance Matrices", self.num_matrices, 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Energy Groups", self.num_groups, 
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
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Weighting Function", self.weighting_function, 
            width1=property_col_width, width2=value_col_width)
        
        info_table += "-" * header_width + "\n\n"
        
        # Create a section for data access using create_repr_section
        data_access = {
            ".num_matrices": "Get total number of covariance matrices",
            ".num_groups": "Get number of energy groups",
            ".isotopes": "Get set of unique isotope IDs",
            ".reactions": "Get set of unique reaction MT numbers",
            ".legendre_indices": "Get set of unique Legendre indices (L values)"
        }
        
        data_access_section = create_repr_section(
            "How to Access Multigroup Covariance Data:", 
            data_access, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Add a blank line after the section
        data_access_section += "\n"
        
        # Create a section for available methods using create_repr_section
        methods = {
            ".to_dataframe()": "Convert all multigroup MF34 covariance data to DataFrame",
            ".summary()": "Get summary DataFrame with matrix metadata",
            ".plot_covariance_heatmap()": "Plot covariance/correlation matrix heatmap",
            ".plot_uncertainties()": "Plot uncertainty curves for Legendre coefficients"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        return header + description + info_table + data_access_section + methods_section
