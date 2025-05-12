from typing import List, Dict, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import os

from mcnpy._constants import MT_TO_REACTION
from mcnpy._utils import create_repr_section



@dataclass
class ValidationIssue:
    """Single validation failure entry."""

    kind: str  # "C-S", "SYM", "PSD"
    row_isotope: int
    col_isotope: int
    row_mt: int
    col_mt: int
    g_i: int  # energy‑group index (row)   or −1 if N/A
    g_j: int  # energy‑group index (column) or −1 if N/A
    value: float
    bound: float

    def __str__(self) -> str:  # pragma: no cover
        if self.kind == "C-S":
            return (
                f"[C‑S] iso {self.row_isotope}/{self.col_isotope} "
                f"MT {self.row_mt}/{self.col_mt} g {self.g_i}/{self.g_j}: "
                f"|cov|={self.value:.3e} > bound={self.bound:.3e}"
            )
        if self.kind == "SYM":
            return (
                f"[SYM] iso {self.row_isotope} MT {self.row_mt}: "
                f"|C-Cᵀ|_∞={self.value:.3e} > tol={self.bound:.3e}"
            )
        # PSD
        return (
            f"[PSD] iso {self.row_isotope} MT {self.row_mt}: "
            f"λ_min={self.value:.3e} < −{self.bound:.3e}"
        )


@dataclass
class ValidationReport:
    ok: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    # ------------------------------------------------------------------
    def summary(self) -> str:  # pragma: no cover
        if self.ok:
            return "All covariance blocks passed validation."
        counts: Dict[str, int] = {}
        for iss in self.issues:
            counts[iss.kind] = counts.get(iss.kind, 0) + 1
        return ", ".join(f"{k}: {v}" for k, v in counts.items())

    def log(self) -> None:  # pragma: no cover
        if self.ok:
            print(self.summary())
            return
        print("Validation report – failures:")
        for iss in self.issues:
            print("  ", iss)




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
    
    def has_isotope_mt(self, isotope: int, mt: int) -> bool:
        """
        Check if covariance data is available for a specific isotope and MT number.
        
        Parameters
        ----------
        isotope : int
            Isotope ID to check
        mt : int
            MT reaction number to check
            
        Returns
        -------
        bool
            True if covariance data is available, False otherwise
        """
        isotope_reactions = self.get_isotope_reactions().get(isotope, set())
        return mt in isotope_reactions
        
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
    
    def get_isotope_covariance_matrix(
            self, isotope: int, mt_list: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Return a symmetric combined covariance matrix for the given isotope.

        Parameters
        ----------
        isotope : int
            ENDF-style isotope identifier (e.g. 92235 for 235-U)
        mt_list : List[int], optional
            List of MT numbers to include; if None, use every MT that
            actually has covariance blocks in the library.

        Returns
        -------
        np.ndarray
            2-D numpy array shaped (N*G, N*G) where N = len(mt_list)
            and G = self.num_groups.
        """
        # --- 1. Select reactions -------------------------------------------------
        all_reactions = self.get_isotope_reactions().get(isotope, set())
        if not all_reactions:
            raise ValueError(f"No covariance data for isotope {isotope}.")

        if mt_list is None:
            mt_list = sorted(all_reactions)
        else:
            mt_list = sorted(mt for mt in mt_list if mt in all_reactions)
            if not mt_list:
                raise ValueError(
                    f"None of the requested MTs have covariance data for isotope {isotope}."
                )

        # --- 2. Allocate result --------------------------------------------------
        N, G = len(mt_list), self.num_groups
        size = N * G
        combined = np.zeros((size, size))

        # --- 3. Fill only the upper triangle, then mirror -----------------------
        for i, mt_i in enumerate(mt_list):
            for j in range(i, N):
                mt_j = mt_list[j]

                block = self.get_matrix(isotope, mt_i, isotope, mt_j)
                if block is None and i != j:
                    # Try the opposite ordering
                    block = self.get_matrix(isotope, mt_j, isotope, mt_i)
                    if block is not None:
                        block = block.T  # transpose to match (mt_i, mt_j)

                if block is None:
                    continue  # leave zeros for genuinely missing data

                r0, r1 = i * G, (i + 1) * G
                c0, c1 = j * G, (j + 1) * G
                combined[r0:r1, c0:c1] = block
                if i != j:
                    combined[c0:c1, r0:r1] = block.T  # enforce symmetry

        return combined

    # ------------------------------------------------------------------
    # Validation core
    # ------------------------------------------------------------------

    def validate(
        self,
        *,
        eps_cs: float = 1.0e-10,
        eps_sym: float = 1.0e-12,
        eta_psd: float = 1.0e-8,
        verbose: bool = True,
    ) -> ValidationReport:
        """Run all validation checks.

        Parameters
        ----------
        eps_cs  : tolerance factor for the Cauchy–Schwarz inequality.
        eps_sym : relative tolerance for symmetry (∞‑norm).
        eta_psd : negative‑eigenvalue tolerance, as fraction of λ_max.
        verbose : print a human‑readable report.
        """

        issues: List[ValidationIssue] = []

        # --------------------------------------------------------------
        # 0) Build a lookup dict for diagonal blocks (iso, mt) → matrix
        # --------------------------------------------------------------
        diag_lookup: Dict[Tuple[int, int], np.ndarray] = {}
        for iso_r, mt_r, iso_c, mt_c, mat in zip(
            self.isotope_rows,
            self.reaction_rows,
            self.isotope_cols,
            self.reaction_cols,
            self.matrices,
        ):
            if iso_r == iso_c and mt_r == mt_c:
                diag_lookup[(iso_r, mt_r)] = mat

        # --------------------------------------------------------------
        # Iterate over *all* stored blocks
        # --------------------------------------------------------------
        for iso_r, mt_r, iso_c, mt_c, mat in zip(
            self.isotope_rows,
            self.reaction_rows,
            self.isotope_cols,
            self.reaction_cols,
            self.matrices,
        ):
            # ---------- 1) Cauchy–Schwarz check ------------------------
            try:
                var_i = np.diag(diag_lookup[(iso_r, mt_r)])
                var_j = np.diag(diag_lookup[(iso_c, mt_c)])
            except KeyError as err:
                raise ValueError(
                    "Missing diagonal block needed for C‑S test: "
                    f"isotope/MT {err.args[0]}"
                ) from None

            # Build bound and compare
            bound = np.sqrt(np.outer(var_i, var_j)) + eps_cs * np.sqrt(
                np.outer(var_i, var_j)
            )
            mask = np.abs(mat) > bound
            for g_i, g_j in np.argwhere(mask):
                issues.append(
                    ValidationIssue(
                        kind="C-S",
                        row_isotope=iso_r,
                        col_isotope=iso_c,
                        row_mt=mt_r,
                        col_mt=mt_c,
                        g_i=int(g_i),
                        g_j=int(g_j),
                        value=float(mat[g_i, g_j]),
                        bound=float(bound[g_i, g_j]),
                    )
                )

            # ---------- 2) Symmetry check (diagonal blocks only) -------
            if iso_r == iso_c and mt_r == mt_c:
                asym = np.max(np.abs(mat - mat.T))
                tol_sym = eps_sym * np.max(np.abs(mat))
                if asym > tol_sym:
                    issues.append(
                        ValidationIssue(
                            kind="SYM",
                            row_isotope=iso_r,
                            col_isotope=iso_c,
                            row_mt=mt_r,
                            col_mt=mt_c,
                            g_i=-1,
                            g_j=-1,
                            value=float(asym),
                            bound=float(tol_sym),
                        )
                    )

                # ---------- 3) PSD check (also only diagonal) ---------
                vals = np.linalg.eigvalsh(mat)
                lam_max = vals.max()
                lam_min = vals.min()
                if lam_min < -eta_psd * lam_max:
                    issues.append(
                        ValidationIssue(
                            kind="PSD",
                            row_isotope=iso_r,
                            col_isotope=iso_c,
                            row_mt=mt_r,
                            col_mt=mt_c,
                            g_i=-1,
                            g_j=-1,
                            value=float(lam_min),
                            bound=float(eta_psd * lam_max),
                        )
                    )

        report = ValidationReport(ok=(len(issues) == 0), issues=issues)
        if verbose:
            report.log()
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _col_iso(self, mt: int) -> int:
        """Dummy helper – in case you want to map MT to isotope."
        Currently returns the first isotope found for that MT in cols.
        """
        for iso, mt_c in zip(self.isotope_cols, self.reaction_cols):
            if mt_c == mt:
                return iso
        return -1  # unknown


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



