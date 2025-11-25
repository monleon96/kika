from typing import List, Dict, Optional, Set, Tuple, Union, Any
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
    is_relative : List[bool]
        List of flags indicating if matrix values are relative (True) or absolute (False)
        False only when LB=0 is present
    frame : List[str]
        List of reference frames for each matrix:
        - "same-as-MF4" when LCT=0
        - "LAB" when LCT=1  
        - "CM" when LCT=2
    """
    isotope_rows: List[int] = field(default_factory=list)
    reaction_rows: List[int] = field(default_factory=list)
    l_rows: List[int] = field(default_factory=list)
    isotope_cols: List[int] = field(default_factory=list)
    reaction_cols: List[int] = field(default_factory=list)
    l_cols: List[int] = field(default_factory=list)
    energy_grids: List[List[float]] = field(default_factory=list)
    matrices: List[np.ndarray] = field(default_factory=list)

    # Metadata fields
    is_relative: List[bool] = field(default_factory=list)
    frame: List[str] = field(default_factory=list)

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
                  energy_grid: List[float],
                  is_relative: bool,
                  frame: str):
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
        is_relative : bool
            True if matrix values are relative, False if absolute (LB=0 present)
        frame : str
            Reference frame: "same-as-MF4", "LAB", "CM", or "unknown LCT=X"
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
        
        # Store metadata
        self.is_relative.append(is_relative)
        self.frame.append(frame)
        

    # ------------------------------------------------------------------
    # User-friendly methods
    # ------------------------------------------------------------------

    def summary(self) -> 'pd.DataFrame':
        """
        Create a summary DataFrame with one row per matrix.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: isotope_row, reaction_row, L_row, isotope_col, 
            reaction_col, L_col, NE (len(energy_grid)), M (NE-1), is_relative, frame
        """
        
        data = {
            "isotope_row": self.isotope_rows,
            "MT_row": self.reaction_rows, 
            "L_row": self.l_rows,
            "isotope_col": self.isotope_cols,
            "MT_col": self.reaction_cols,
            "L_col": self.l_cols,
            "NE": [len(grid) for grid in self.energy_grids],
            "is_relative": self.is_relative,
            "frame": self.frame
        }
        
        return pd.DataFrame(data)

    def describe(self, i: int) -> str:
        """
        Pretty single-matrix summary in plain text.
        
        Parameters
        ----------
        i : int
            Index of the matrix to describe
            
        Returns
        -------
        str
            Human-readable description of the matrix
        """
        if i < 0 or i >= len(self.matrices):
            return f"Matrix index {i} out of range [0, {len(self.matrices)-1}]"
        
        matrix = self.matrices[i]
        energy_grid = self.energy_grids[i]
        
        desc = [
            f"Matrix {i}:",
            f"  Reaction: {self.isotope_rows[i]} MT{self.reaction_rows[i]} (L={self.l_rows[i]}) ↔ {self.isotope_cols[i]} MT{self.reaction_cols[i]} (L={self.l_cols[i]})",
            f"  Shape: {matrix.shape}, Energy grid: {len(energy_grid)} points ({len(energy_grid)-1} intervals)",
            f"  Type: {'Relative' if self.is_relative[i] else 'Absolute'}",
            f"  Reference frame: {self.frame[i]}",
        ]
        
        return '\n'.join(desc)



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
        param_triplets = self._get_param_triplets()
        idx_map = {p: i for i, p in enumerate(param_triplets)}
        unions = getattr(self, "_union_grids", None) or self.compute_union_energy_grids()
        # number of bins (not points) per triplet on the union
        Gmap = {t: len(unions[t]) - 1 for t in param_triplets}
        max_G = max(Gmap.values()) if Gmap else 0
        N = len(param_triplets) * max_G
        full = np.zeros((N, N), dtype=float)

        for ir, rr, lr, ic, rc, lc, matrix, grid in zip(
            self.isotope_rows, self.reaction_rows, self.l_rows,
            self.isotope_cols, self.reaction_cols, self.l_cols,
            self.matrices, self.energy_grids
        ):
            tr = (ir, rr, lr); tc = (ic, rc, lc)
            i, j = idx_map[tr], idx_map[tc]
            # lift Σ to (union_r × union_c)
            Ar = self._lift_matrix(np.asarray(grid), unions[tr])
            Ac = self._lift_matrix(np.asarray(grid), unions[tc])
            Sigma = Ar @ matrix @ Ac.T

            Gr, Gc = Gmap[tr], Gmap[tc]
            r0, r1 = i*max_G, i*max_G + Gr
            c0, c1 = j*max_G, j*max_G + Gc
            full[r0:r1, c0:c1] = Sigma
            if i != j:
                full[c0:c1, r0:r1] = Sigma.T
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

    def has_uniform_energy_grid(self) -> bool:
        """
        Check if all matrices have the same energy grid.
        
        Returns
        -------
        bool
            True if all energy grids are identical, False otherwise.
            Returns True for empty collections (vacuous truth).
        """
        if not self.energy_grids:
            return True
        
        # Compare all grids to the first one
        first_grid = self.energy_grids[0]
        
        for grid in self.energy_grids[1:]:
            # Check if lengths are different
            if len(grid) != len(first_grid):
                return False
            
            # Check if values are different (using numpy for numerical comparison)
            if not np.allclose(grid, first_grid, rtol=1e-15, atol=1e-15):
                return False
        
        return True

    def get_union_energy_grids(self):
        return getattr(self, "_union_grids", None) or self.compute_union_energy_grids()


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
        Plot uncertainties for MF34 angular distribution data for specific Legendre coefficients.
        
        This method extracts and plots the diagonal uncertainties from the covariance matrix
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
        
        >>> fig = mf34_covmat.plot_uncertainties(isotope=92235, mt=2, 
        ...                                     legendre_coeffs=[1, 2, 3])
        >>> fig.show()
        
        Plot absolute uncertainties for a single Legendre coefficient:
        
        >>> fig = mf34_covmat.plot_uncertainties(isotope=92235, mt=2,
        ...                                     legendre_coeffs=1, 
        ...                                     uncertainty_type="absolute")
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

    def to_plot_data(
        self,
        isotope: int,
        mt: int,
        order: int,
        uncertainty_type: str = 'relative',
        label: str = None,
        **styling_kwargs
    ):
        """
        Create a PlotData object for Legendre coefficient uncertainties.
        
        This is a convenience method to easily convert MF34 covariance data into
        a plottable format using the new plotting infrastructure.
        
        Parameters
        ----------
        isotope : int
            Isotope ID
        mt : int
            Reaction MT number
        order : int
            Legendre polynomial order
        uncertainty_type : str, default 'relative'
            Type of uncertainty: 'relative' (%) or 'absolute'
        label : str, optional
            Custom label for the plot. If None, auto-generates from isotope and order.
        **styling_kwargs
            Additional styling kwargs (color, linestyle, linewidth, etc.)
            
        Returns
        -------
        LegendreUncertaintyPlotData
            Plot data object ready to be added to a PlotBuilder
            
        Raises
        ------
        ValueError
            If uncertainty data is not available for the specified parameters
            
        Examples
        --------
        >>> # Extract uncertainty data from MF34CovMat
        >>> mf34_covmat = endf.mf[34].mt[2].to_ang_covmat()
        >>> unc_data = mf34_covmat.to_plot_data(isotope=26056, mt=2, order=1)
        >>> 
        >>> # Build a plot
        >>> from mcnpy.plotting import PlotBuilder
        >>> fig = PlotBuilder().add_data(unc_data).build()
        """
        from mcnpy.plotting import LegendreUncertaintyPlotData
        from mcnpy._utils import zaid_to_symbol
        
        # Get uncertainty data
        unc_data = self.get_uncertainties_for_legendre_coefficient(isotope, mt, order)
        
        if unc_data is None:
            raise ValueError(
                f"No uncertainty data available for isotope={isotope}, MT={mt}, L={order}"
            )
        
        # Extract energies and uncertainties
        energies = unc_data['energies']
        uncertainties = unc_data['uncertainties']
        
        # Get energy bin boundaries
        energy_bins = None
        for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c) in enumerate(zip(
            self.isotope_rows, self.reaction_rows, self.l_rows,
            self.isotope_cols, self.reaction_cols, self.l_cols
        )):
            # Look for diagonal variance matrix (L = L) for the specified parameters
            if (iso_r == isotope and iso_c == isotope and 
                mt_r == mt and mt_c == mt and 
                l_r == order and l_c == order):
                energy_bins = np.array(self.energy_grids[i])
                break
        
        # Convert to percentage if relative
        if uncertainty_type.lower() == 'relative':
            uncertainties = uncertainties * 100.0  # Convert to percentage
        
        # Generate label if not provided
        if label is None:
            isotope_symbol = zaid_to_symbol(isotope)
            if uncertainty_type.lower() == 'relative':
                label = f"{isotope_symbol} MT={mt} L={order} (σ %)"
            else:
                label = f"{isotope_symbol} MT={mt} L={order} (σ abs)"
        
        # For step plots with histogram data:
        # - energies has N+1 bin boundaries
        # - uncertainties has N values (one per bin)
        # For proper step plotting with where='post', we need to duplicate the last
        # uncertainty value so that the last bin is drawn extending to the last boundary
        if len(energies) == len(uncertainties) + 1:
            # Append the last uncertainty value to match the energy boundaries length
            uncertainties = np.append(uncertainties, uncertainties[-1])
        
        # Create and return PlotData object
        return LegendreUncertaintyPlotData(
            x=energies,
            y=uncertainties,
            label=label,
            order=order,
            isotope=zaid_to_symbol(isotope),
            mt=mt,
            uncertainty_type=uncertainty_type,
            energy_bins=energy_bins,
            **styling_kwargs
        )

    def filter_by_isotope_reaction(self, isotope: int, mt: int) -> "MF34CovMat":
        """
        Return a new MF34CovMat containing only matrices for the specified isotope and MT reaction.
        
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
        MF34CovMat
            New MF34CovMat object containing only the filtered matrices
        """
        # Find indices where both row and column match the specified isotope and MT
        matching_indices = []
        for i, (iso_r, mt_r, iso_c, mt_c) in enumerate(zip(
            self.isotope_rows, self.reaction_rows, 
            self.isotope_cols, self.reaction_cols
        )):
            if iso_r == isotope and mt_r == mt and iso_c == isotope and mt_c == mt:
                matching_indices.append(i)
        
        # Create new MF34CovMat with filtered data
        filtered_mf34 = MF34CovMat()
        
        for i in matching_indices:
            filtered_mf34.isotope_rows.append(self.isotope_rows[i])
            filtered_mf34.reaction_rows.append(self.reaction_rows[i])
            filtered_mf34.l_rows.append(self.l_rows[i])
            filtered_mf34.isotope_cols.append(self.isotope_cols[i])
            filtered_mf34.reaction_cols.append(self.reaction_cols[i])
            filtered_mf34.l_cols.append(self.l_cols[i])
            filtered_mf34.energy_grids.append(self.energy_grids[i])
            filtered_mf34.matrices.append(self.matrices[i])
        
        return filtered_mf34

    def get_uncertainties_for_legendre_coefficient(
        self, 
        isotope: int, 
        mt: int, 
        l_coefficient: Union[int, List[int]],
    ) -> Union[Optional[Dict[str, np.ndarray]], Dict[int, Optional[Dict[str, np.ndarray]]]]:
        """
        Extract standard uncertainties (square root of diagonal variance) for Legendre coefficient(s).
        
        **IMPORTANT**: MF34 data is typically stored as RELATIVE covariances (fractional uncertainties δA_ℓ/A_ℓ).
        This method returns the uncertainties as stored in MF34, along with an 'is_relative' flag.
        
        To convert relative uncertainties to absolute: σ_abs = σ_rel × |A_ℓ|
        where A_ℓ are the Legendre coefficients from ENDF MF=4 data.
        
        Parameters
        ----------
        isotope : int
            Isotope ID
        mt : int
            Reaction MT number
        l_coefficient : int or list of int
            Legendre coefficient index (L value) or list of L values
            
        Returns
        -------
        dict or dict of dicts
            For single int: Dictionary containing:
                - 'energies': np.ndarray - Energy bin boundaries (N+1 points for N bins) in eV or MeV
                - 'uncertainties': np.ndarray - Uncertainties (√diagonal of covariance) for each bin
                - 'is_relative': bool - True if relative (δA_ℓ/A_ℓ), False if absolute (δA_ℓ)
            For list of ints: Dictionary mapping L coefficient to uncertainty data (or None if not found).
            
        Notes
        -----
        - If is_relative=True, you must convert to absolute uncertainties by multiplying
          by the Legendre coefficients A_ℓ from ENDF MF=4 before using in propagation formulas.
        - The LB flag in ENDF-6 format determines if data is relative (LB=1,2,5) or absolute (LB=0).
        - Energies are returned as BIN BOUNDARIES, not bin centers. Each uncertainty value applies
          to the energy bin defined by consecutive boundary pairs [E[i], E[i+1]).
        """
        # Handle single coefficient case
        if isinstance(l_coefficient, int):
            # Find the matrix for this specific (isotope, mt, l_coefficient) combination
            matrix_is_relative = None
            for i, (iso_r, mt_r, l_r, iso_c, mt_c, l_c, energy_grid, matrix) in enumerate(zip(
                self.isotope_rows, self.reaction_rows, self.l_rows,
                self.isotope_cols, self.reaction_cols, self.l_cols,
                self.energy_grids, self.matrices
            )):
                # Look for diagonal variance matrix (L = L) for the specified parameters
                if (iso_r == isotope and iso_c == isotope and 
                    mt_r == mt and mt_c == mt and 
                    l_r == l_coefficient and l_c == l_coefficient):
                    
                    # Store whether this matrix is relative
                    matrix_is_relative = self.is_relative[i]
                    
                    # Extract diagonal elements (variances) and take square root
                    diagonal_variances = np.diag(matrix)
                    
                    # Check for negative variances (which shouldn't happen for diagonal blocks)
                    if np.any(diagonal_variances < 0):
                        # Handle negative variances by setting them to zero
                        diagonal_variances = np.maximum(diagonal_variances, 0.0)
                    
                    uncertainties = np.sqrt(diagonal_variances)
                    
                    # Energy grid contains bin boundaries directly
                    energy_array = np.array(energy_grid)
                    
                    # Ensure we have the correct number of boundaries for uncertainties
                    if len(energy_array) == len(uncertainties) + 1:
                        # Perfect: N+1 boundaries for N uncertainties
                        pass
                    elif len(energy_array) > len(uncertainties) + 1:
                        # Too many energy points - truncate to N+1 boundaries
                        import warnings
                        warnings.warn(
                            f"MF34 data for isotope={isotope}, MT={mt}, L={l_coefficient}: "
                            f"Energy grid has {len(energy_array)} points but only {len(uncertainties)} uncertainties. "
                            f"Expected {len(uncertainties) + 1} energy points. Truncating energy grid.",
                            UserWarning
                        )
                        energy_array = energy_array[:len(uncertainties) + 1]
                    elif len(energy_array) < len(uncertainties) + 1:
                        # Too few energy points - truncate uncertainties to match
                        import warnings
                        warnings.warn(
                            f"MF34 data for isotope={isotope}, MT={mt}, L={l_coefficient}: "
                            f"Energy grid has {len(energy_array)} points but {len(uncertainties)} uncertainties. "
                            f"Expected {len(uncertainties) + 1} energy points. Truncating uncertainties.",
                            UserWarning
                        )
                        uncertainties = uncertainties[:len(energy_array) - 1]
                    
                    return {
                        'energies': energy_array,  # Bin boundaries (N+1 points for N bins)
                        'uncertainties': uncertainties,
                        'is_relative': matrix_is_relative
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

    def compute_union_energy_grids(self, atol: float = 1e-12):
        """
        Compute union energy grids for all parameter triplets.
        
        This method creates a unified energy grid for each (isotope, reaction, legendre) 
        triplet by merging all energy grids that involve that triplet, removing duplicates
        within tolerance.
        
        Parameters
        ----------
        atol : float, default 1e-12
            Absolute tolerance for merging energy points
            
        Returns
        -------
        Dict[Tuple[int, int, int], np.ndarray]
            Dictionary mapping (isotope, reaction, legendre) triplets to union energy grids
        """
        triplets = self._get_param_triplets()
        unions = {t: [] for t in triplets}
        for i, grid in enumerate(self.energy_grids):
            row = (self.isotope_rows[i], self.reaction_rows[i], self.l_rows[i])
            col = (self.isotope_cols[i], self.reaction_cols[i], self.l_cols[i])
            unions[row].extend(grid); unions[col].extend(grid)
        # deduplicate with tolerance
        for t, g in unions.items():
            g = np.unique(np.asarray(g, dtype=float))
            merged = [g[0]]
            for x in g[1:]:
                if not np.isclose(x, merged[-1], rtol=0.0, atol=atol):
                    merged.append(x)
            unions[t] = np.array(merged, dtype=float)
        self._union_grids = unions
        return unions

    def validate_union_grids(self, verbose: bool = True) -> bool:
        """
        Validate that union grids are properly constructed and aligned.
        
        Parameters
        ----------
        verbose : bool, default True
            Whether to print validation details
            
        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        try:
            param_triplets = self._get_param_triplets()
            union_grids = self.get_union_energy_grids()
            
            if verbose:
                print(f"Validating union grids for {len(param_triplets)} parameter triplets")
            
            # Check that all triplets have union grids
            missing_grids = [t for t in param_triplets if t not in union_grids]
            if missing_grids:
                if verbose:
                    print(f"ERROR: Missing union grids for {len(missing_grids)} triplets")
                return False
            
            # Check grid properties
            max_G = 0
            for triplet, grid in union_grids.items():
                if len(grid) < 2:
                    if verbose:
                        print(f"WARNING: Triplet {triplet} has insufficient grid points: {len(grid)}")
                    continue
                
                num_bins = len(grid) - 1
                max_G = max(max_G, num_bins)
                
                # Check grid is sorted
                if not np.all(grid[1:] >= grid[:-1]):
                    if verbose:
                        print(f"ERROR: Grid for triplet {triplet} is not sorted")
                    return False
            
            # Check covariance matrix dimensions
            expected_dim = len(param_triplets) * max_G
            actual_shape = self.covariance_matrix.shape
            
            if actual_shape[0] != actual_shape[1]:
                if verbose:
                    print(f"ERROR: Covariance matrix is not square: {actual_shape}")
                return False
            
            if actual_shape[0] != expected_dim:
                if verbose:
                    print(f"ERROR: Covariance matrix dimension mismatch. "
                          f"Expected: {expected_dim}, Actual: {actual_shape[0]}")
                return False
            
            if verbose:
                print(f"Validation PASSED: {len(param_triplets)} triplets, max_G={max_G}, "
                      f"matrix shape={actual_shape}")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"ERROR during union grids validation: {e}")
            return False


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

    def _lift_matrix(self, src_grid, dst_grid):
        # src_grid, dst_grid are boundary arrays (NE)
        Gs, Gd = len(src_grid)-1, len(dst_grid)-1
        A = np.zeros((Gd, Gs), dtype=float)
        j = 0
        for g in range(Gd):
            eL, eH = dst_grid[g], dst_grid[g+1]
            while j+1 < len(src_grid) and src_grid[j+1] <= eL + 1e-12:
                j += 1
            # assume dst is subset/refinement of src:
            A[g, j] = 1.0
        return A
