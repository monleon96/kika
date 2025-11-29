"""
Multigroup MF34 angular distribution covariance matrix data structure.

This module contains the MGMF34CovMat class for storing multigroup covariance
matrices derived from MF34 angular distribution data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional, Dict, TYPE_CHECKING
from matplotlib import pyplot as plt
from ..._utils import create_repr_section

if TYPE_CHECKING:
    from ..mf34_covmat import MF34CovMat

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
    relative_normalization: str = "mf34_cell"
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
                Return the full multigroup RELATIVE covariance matrix (block assembled).

                Shape: (N·G) × (N·G), where
                    N = number of unique (isotope, reaction, Legendre) triplets
                    G = number of energy groups

                Blocks: each (i,j) block is the stored relative covariance between triplet i and j
                across energy groups. Off-diagonal blocks are symmetrized.
        
                Note: These are RELATIVE covariances (dimensionless) consistent with the
                MF34 convention. Use `absolute_covariance_matrix` for absolute units.
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
    def absolute_covariance_matrix(self) -> np.ndarray:
        """Return the full multigroup ABSOLUTE covariance matrix assembled as blocks.

        Mirrors the logic of `covariance_matrix` but using `absolute_matrices`.
        Returns an empty array if no absolute matrices are stored.
        """
        if not self.absolute_matrices:
            return np.array([])

        param_triplets = self._get_param_triplets()
        idx_map = {p: i for i, p in enumerate(param_triplets)}

        G = self.num_groups
        N = len(param_triplets)
        full = np.zeros((N * G, N * G), dtype=float)

        for ir, rr, lr, ic, rc, lc, matrix in zip(
            self.isotope_rows, self.reaction_rows, self.l_rows,
            self.isotope_cols, self.reaction_cols, self.l_cols,
            self.absolute_matrices
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

    def _get_param_triplets(self) -> List[Tuple[int, int, int]]:
        """
        Return a list of all (isotope, reaction, legendre) triplets present,
        sorted first by isotope, then by reaction, then by legendre coefficient.
        """
        triplets = set(zip(self.isotope_rows, self.reaction_rows, self.l_rows)) \
                 | set(zip(self.isotope_cols, self.reaction_cols, self.l_cols))
        return sorted(triplets, key=lambda t: (t[0], t[1], t[2]))

    def reactions_by_isotope(self, isotope: Optional[int] = None) -> Union[Dict[int, List[int]], List[int]]:
        """
        Get a mapping of isotopes to their available reactions, or list of reactions for a specific isotope.
        
        This method provides compatibility with the CovMat interface for the sandwich uncertainty 
        propagation function.

        Parameters
        ----------
        isotope : Optional[int]
            If provided, return reactions only for this isotope.

        Returns
        -------
        Dict[int, List[int]] or List[int]
            Mapping from isotope IDs to sorted lists of MT numbers, or list of MT numbers for the specified isotope.
        """
        result: Dict[int, set] = {}

        # Process all row combinations
        for i, iso in enumerate(self.isotope_rows):
            result.setdefault(iso, set()).add(self.reaction_rows[i])

        # Process all column combinations
        for i, iso in enumerate(self.isotope_cols):
            result.setdefault(iso, set()).add(self.reaction_cols[i])

        # Convert sets to sorted lists
        sorted_dict: Dict[int, List[int]] = {iso: sorted(reactions) for iso, reactions in result.items()}

        if isotope is not None:
            # Return the list for the specified isotope, or empty list if not found
            return sorted_dict.get(isotope, [])

        return True

    def to_plot_data(
        self,
        isotope: int,
        mt: int,
        order: int,
        sigma: float = 1.0,
        label: str = None,
        **styling_kwargs
    ):
        """
        Create PlotData objects for multigroup Legendre coefficients with uncertainties.
        
        Returns both the Legendre coefficient data and uncertainty band, following the
        unified API pattern. Always returns a tuple (legendre_data, unc_band) where
        either can be None if data is not available.
        
        Parameters
        ----------
        isotope : int
            Isotope ID
        mt : int
            Reaction MT number
        order : int
            Legendre polynomial order
        sigma : float, default 1.0
            Sigma level for uncertainty bands (e.g., 1.0 for 1σ, 2.0 for 2σ)
        label : str, optional
            Custom label for the plot. If None, auto-generates from isotope and order.
        **styling_kwargs
            Additional styling kwargs (color, linestyle, linewidth, etc.)
            
        Returns
        -------
        tuple of (LegendreCoeffPlotData, UncertaintyBand)
            Tuple containing:
            - legendre_data: PlotData for Legendre coefficients (None if not available)
            - unc_band: UncertaintyBand with relative uncertainties (None if not available)
            
        Raises
        ------
        ValueError
            If neither Legendre coefficient nor covariance data is available
            
        Examples
        --------
        >>> # Extract data for L=1 coefficient
        >>> legendre_data, unc_band = mg_covmat.to_plot_data(
        ...     isotope=26056, mt=2, order=1)
        >>> 
        >>> # Build a plot with PlotBuilder
        >>> from kika.plotting import PlotBuilder
        >>> fig = (PlotBuilder()
        ...        .add_data(legendre_data, uncertainty=unc_band)
        ...        .build())
        """
        from kika.plotting import LegendreCoeffPlotData, UncertaintyBand
        from kika._utils import zaid_to_symbol
        
        # Check if Legendre coefficient data exists
        key = (isotope, mt, order)
        if key not in self.legendre_coefficients:
            raise ValueError(
                f"No Legendre coefficient data available for "
                f"isotope={isotope}, MT={mt}, order={order}"
            )
        
        # Extract Legendre coefficients
        coeffs = np.asarray(self.legendre_coefficients[key], dtype=float)
        energy_grid = np.asarray(self.energy_grid, dtype=float)
        
        # Validate data consistency
        if coeffs.size != energy_grid.size - 1:
            raise ValueError(
                f"Legendre coefficient size ({coeffs.size}) does not match "
                f"energy grid groups ({energy_grid.size - 1})"
            )
        
        # For step plots with where='post', extend y to match x length
        # This creates the proper step representation for histogram-like data
        coeffs_extended = np.append(coeffs, coeffs[-1])
        
        # Generate label if not provided
        if label is None:
            try:
                isotope_symbol = zaid_to_symbol(isotope)
            except Exception:
                isotope_symbol = f"Isotope {isotope}"
            
            label = f"{isotope_symbol} - $a_{{{order}}}$"
            if sigma == 1.0:
                label += " (±1σ)"
            else:
                label += f" (±{sigma}σ)"
        
        # Create LegendreCoeffPlotData
        legendre_data = LegendreCoeffPlotData(
            x=energy_grid,  # Bin boundaries for step plots
            y=coeffs_extended,  # Extended to match x length for step='post'
            label=label,
            order=order,
            isotope=zaid_to_symbol(isotope) if isotope else None,
            mt=mt,
            plot_type='step',
            **styling_kwargs
        )
        
        # Create uncertainty PlotData instead of UncertaintyBand
        unc_data = self._create_uncertainty_plotdata(
            isotope=isotope,
            mt=mt,
            order=order,
            energy_grid=energy_grid,
            sigma=sigma,
            label=label,
            **styling_kwargs
        )
        
        return legendre_data, unc_data
    
    def _create_uncertainty_plotdata(
        self,
        isotope: int,
        mt: int,
        order: int,
        energy_grid: np.ndarray,
        sigma: float,
        label: str = None,
        **styling_kwargs
    ):
        """
        Helper method to create LegendreUncertaintyPlotData from covariance matrix.
        
        Returns None if covariance data is not available.
        """
        from kika.plotting import LegendreUncertaintyPlotData
        from kika._utils import zaid_to_symbol
        
        # Find the covariance matrix for this order
        for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c) in enumerate(zip(
            self.isotope_rows, self.reaction_rows, self.l_rows,
            self.isotope_cols, self.reaction_cols, self.l_cols
        )):
            if (iso_r == isotope and mt_r == mt and l_r == order and
                iso_c == isotope and mt_c == mt and l_c == order):
                # Found diagonal block
                cov_matrix = self.relative_matrices[i]
                diag = np.diag(cov_matrix)
                
                # Extract relative uncertainties (square root of diagonal)
                # This has n values (one per energy group)
                rel_unc = np.sqrt(diag)
                
                # Convert to percentage and apply sigma multiplier
                rel_unc_pct = rel_unc * 100.0 * sigma
                
                # Convert to energy group centers for plotting
                energy_centers = np.sqrt(energy_grid[:-1] * energy_grid[1:])
                
                # Generate label if not provided
                if label is None:
                    try:
                        isotope_symbol = zaid_to_symbol(isotope)
                    except Exception:
                        isotope_symbol = f"Isotope {isotope}"
                    
                    sigma_str = f"{sigma}σ" if sigma != 1.0 else "1σ"
                    label = f"{isotope_symbol} - $a_{{{order}}}$ Uncertainty ({sigma_str})"
                
                # Create LegendreUncertaintyPlotData
                return LegendreUncertaintyPlotData(
                    x=energy_centers,
                    y=rel_unc_pct,
                    label=label,
                    order=order,
                    isotope=zaid_to_symbol(isotope) if isotope else None,
                    mt=mt,
                    uncertainty_type='relative',
                    energy_bins=energy_grid,
                    step_where='post',
                    **styling_kwargs
                )
        
        # No covariance data found
        return None

    def plot_legendre_coefficients(
        self,
        isotope: int,
        mt: int,
        orders: Optional[Union[int, List[int]]] = None,
        style: str = 'default',
        figsize: Tuple[float, float] = (10, 6),
        legend_loc: str = 'best',
        marker: bool = True,
        include_uncertainties: bool = False,
        uncertainty_sigma: float = 1.0,
        **kwargs
    ) -> plt.Figure:
        """
        Plot multigroup Legendre coefficients for this covariance matrix object.
        
        This is a convenience method that calls the standalone plotting function
        with this object as input.
        
        Parameters
        ----------
        isotope : int
            Isotope ID to plot
        mt : int
            Reaction MT number to plot
        orders : int or list of int, optional
            Legendre orders to plot. If None, plots all available orders
        style : str
            Plot style from _plot_settings
        figsize : tuple
            Figure size
        legend_loc : str
            Legend location
        marker : bool
            Whether to include markers on the plot lines
        include_uncertainties : bool
            Whether to include uncertainty bands if available
        uncertainty_sigma : float
            Number of sigma levels for uncertainty bands
        **kwargs
            Additional plotting arguments
        
        Returns
        -------
        plt.Figure
            The matplotlib figure containing the plot
        """
        from .plotting_mg import plot_mg_legendre_coefficients
        
        return plot_mg_legendre_coefficients(
            mg_covmat=self,
            isotope=isotope,
            mt=mt,
            orders=orders,
            style=style,
            figsize=figsize,
            legend_loc=legend_loc,
            marker=marker,
            include_uncertainties=include_uncertainties,
            uncertainty_sigma=uncertainty_sigma,
            **kwargs
        )

    def plot_vs_endf(
        self,
        endf: object,
        isotope: int,
        mt: int,
        orders: Optional[Union[int, List[int]]] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        style: str = 'default',
        figsize: Tuple[float, float] = (12, 8),
        legend_loc: str = 'best',
        mg_marker: bool = True,
        include_uncertainties: bool = False,
        uncertainty_sigma: float = 1.0,
        **kwargs
    ) -> plt.Figure:
        """
        Compare multigroup Legendre coefficients with ENDF data.
        
        This is a convenience method that calls the comparison plotting function
        with this object and the provided ENDF object.
        
        Parameters
        ----------
        endf : ENDF object
            Original ENDF data object containing MF4 data
        isotope : int
            Isotope ID to plot
        mt : int
            Reaction MT number to plot
        orders : int or list of int, optional
            Legendre orders to plot. If None, plots all available orders
        energy_range : tuple of float, optional
            Energy range for plotting ENDF data
        style : str
            Plot style from _plot_settings
        figsize : tuple
            Figure size
        legend_loc : str
            Legend location
        mg_marker : bool
            Whether to include markers for multigroup data
        include_uncertainties : bool
            If True, display ±σ uncertainty bands for both MG and ENDF (if MF34 present)
        uncertainty_sigma : float
            Sigma multiplier for uncertainty bands (default 1.0)
        **kwargs
            Additional plotting arguments
        
        Returns
        -------
        plt.Figure
            The matplotlib figure containing the plot
        """
        from .plotting_mg import plot_mg_vs_endf_comparison
        
        return plot_mg_vs_endf_comparison(
            mg_covmat=self,
            endf=endf,
            isotope=isotope,
            mt=mt,
            orders=orders,
            energy_range=energy_range,
            style=style,
            figsize=figsize,
            legend_loc=legend_loc,
            mg_marker=mg_marker,
            include_uncertainties=include_uncertainties,
            uncertainty_sigma=uncertainty_sigma,
            **kwargs
        )

    def plot_covariance_heatmap(
        self,
        isotope: int,
        mt: int,
        orders: Optional[Union[int, List[int]]] = None,
        matrix_type: str = 'cov',
        covariance_type: str = 'rel',
        style: str = 'default',
        figsize: Tuple[float, float] = (10, 8),
        colormap: Optional[str] = None,
        show_colorbar: bool = True,
        annotate: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        symmetric_scale: bool = True,
        use_log_scale: bool = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot a heatmap of the covariance or correlation matrix for specific isotope, MT, and Legendre orders.
        
        This is a convenience method that calls the standalone heatmap plotting function
        with this object as input.
        
        Parameters
        ----------
        isotope : int
            Isotope ID to plot
        mt : int
            Reaction MT number to plot
        orders : int or list of int, optional
            Legendre orders to include. If None, includes all available orders
        matrix_type : str, default 'cov'
            Type of matrix to plot: 'cov' for covariance, 'corr' for correlation
        covariance_type : str, default 'rel'
            Type of covariance matrix: 'rel' for relative, 'abs' for absolute
        style : str
            Plot style from _plot_settings
        figsize : tuple
            Figure size
        colormap : str, optional
            Matplotlib colormap name. If None, uses 'RdYlGn' for correlation and 'viridis' for covariance
        show_colorbar : bool
            Whether to show the colorbar
        annotate : bool
            Whether to annotate matrix values on the heatmap
        vmin : float, optional
            Minimum value for color scaling
        vmax : float, optional
            Maximum value for color scaling  
        symmetric_scale : bool
            Whether to use symmetric color scale around zero (for correlation matrices)
        use_log_scale : bool, optional
            Whether to use logarithmic color scale. If None, uses linear for correlation and log for covariance
        **kwargs
            Additional plotting arguments
        
        Returns
        -------
        plt.Figure
            The matplotlib figure containing the heatmap
        """
        from .plotting_mg import plot_mg_covariance_heatmap
        
        return plot_mg_covariance_heatmap(
            mg_covmat=self,
            isotope=isotope,
            mt=mt,
            orders=orders,
            matrix_type=matrix_type,
            covariance_type=covariance_type,
            style=style,
            figsize=figsize,
            colormap=colormap,
            show_colorbar=show_colorbar,
            annotate=annotate,
            vmin=vmin,
            vmax=vmax,
            symmetric_scale=symmetric_scale,
            use_log_scale=use_log_scale,
            **kwargs
        )

    def plot_uncertainties_comparison(
        self,
        endf_data: Union["MF34CovMat", object],
        isotope: int,
        mt: int,
        orders: Optional[Union[int, List[int]]] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        style: str = 'default',
        figsize: Tuple[float, float] = (10, 6),
        legend_loc: str = 'best',
        mg_marker: bool = True,
        uncertainty_type: str = "relative",
        **kwargs
    ) -> plt.Figure:
        """
        Compare multigroup uncertainties with ENDF uncertainties for Legendre coefficients.
        
        This is a convenience method that calls the standalone uncertainty comparison
        plotting function with this object as input.
        
        Parameters
        ----------
        endf_data : MF34CovMat or ENDF object
            Either:
            - MF34CovMat: Original ENDF MF34 covariance data object
            - ENDF object: ENDF object containing MF34 data (will extract MF34CovMat automatically)
        isotope : int
            Isotope ID to plot
        mt : int
            Reaction MT number to plot
        orders : int or list of int, optional
            Legendre orders to plot. If None, plots all available orders
        energy_range : tuple of float, optional
            Energy range for plotting. If None, uses full multigroup range
        style : str
            Plot style from _plot_settings
        figsize : tuple
            Figure size
        legend_loc : str
            Legend location
        mg_marker : bool
            Whether to include markers for multigroup data
        uncertainty_type : str
            Type of uncertainty to plot: "relative" (%) or "absolute"
        **kwargs
            Additional plotting arguments
        
        Returns
        -------
        plt.Figure
            The matplotlib figure containing the comparison plot
        """
        from .plotting_mg import plot_mg_vs_endf_uncertainties_comparison
        
        return plot_mg_vs_endf_uncertainties_comparison(
            mg_covmat=self,
            endf_data=endf_data,
            isotope=isotope,
            mt=mt,
            orders=orders,
            energy_range=energy_range,
            style=style,
            figsize=figsize,
            legend_loc=legend_loc,
            mg_marker=mg_marker,
            uncertainty_type=uncertainty_type,
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
                f"- Weighting: {self.weighting_function}\n"
                f"- Normalization: {self.relative_normalization}")

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
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Relative Normalization", self.relative_normalization, 
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
            ".plot_legendre_coefficients()": "Plot multigroup Legendre coefficients vs energy",
            ".plot_vs_endf()": "Compare multigroup with continuous ENDF coefficients",
            ".filter_by_isotope_reaction()": "Filter matrices by isotope and reaction",
            ".covariance_matrix": "Get full covariance matrix property",
            ".correlation_matrix": "Get full correlation matrix property"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        return header + description + info_table + data_access_section + methods_section
