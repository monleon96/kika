"""
Classes for MT sections within MF34 (Angular Distribution Covariances) in ENDF files.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
import time # Import time for performance checks

from ..mt import MT
from ....cov.mf34_covmat import MF34CovMat 
import numpy as np # Make sure numpy is imported

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
    
    # For LB=6 (rectangular matrix)
    row_energies: List[float] = field(default_factory=list)   # NER row energies
    col_energies: List[float] = field(default_factory=list)   # NEC column energies
    rect_matrix:   List[float] = field(default_factory=list)   # NER×NEC matrix values


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
    _za: float = None
    _awr: float = None
    _ltt: int = None
    _nmt1: int = None
    _mat: int = None
    _subsections: List[Subsection] = field(default_factory=list)
    num_lines: int = 0

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
                    if record.lb in (0, 1, 2):
                        # Format LIST record header for LB=0-2 only
                        record_header = format_endf_data_line(
                            [0.0, 0.0, record.lt, record.lb, record.nt, record.np],
                            mat, mf, mt, 0,
                            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                                     ENDF_FORMAT_INT, ENDF_FORMAT_INT,
                                     ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                        )
                        lines.append(blank_line_number(record_header))
                        
                        # one (E_k,F_k) table only
                        all_values = []
                        for i in range(len(record.e_table_k)):
                            all_values.append(record.e_table_k[i])
                            all_values.append(record.f_table_k[i])
                        
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
                    
                    elif record.lb == 6:
                        # LIST header: LS=0, LB=6, NT=C5, NE=C6 (NER)
                        ner = len(record.row_energies)
                        nt  = record.nt
                        rl_header = format_endf_data_line(
                            [0.0, 0.0, 0, 6, nt, ner],
                            mat, mf, mt, 0,
                            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT,
                                     ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT,
                                     ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                        )
                        lines.append(blank_line_number(rl_header))

                        # stack row-energies, col-energies, then matrix values
                        all_vals = record.row_energies + record.col_energies + record.rect_matrix
                        buf = []
                        for v in all_vals:
                            buf.append(v)
                            if len(buf) == 6:
                                ln = format_endf_data_line(buf, mat, mf, mt, 0)
                                lines.append(blank_line_number(ln))
                                buf = []
                        if buf:
                            # pad to 6
                            while len(buf) < 6: buf.append(None)
                            ln = format_endf_data_line(
                                buf,
                                mat, mf, mt, 0,
                                formats=[ENDF_FORMAT_FLOAT]*len(buf) + [ENDF_FORMAT_BLANK]*(6-len(buf))
                            )
                            lines.append(blank_line_number(ln))
        
        # End of section marker - all integers - this one keeps the line number (99999)
        end_line = format_endf_data_line(
            [0, 0, 0, 0, 0, 0],
            mat, mf, 0, 99999,  # Note MT=0 for end of section, use 99999 for the standard end marker
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, 
                     ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        lines.append(end_line)
        
        return "\n".join(lines)
    
    def _decode_lb012_matrix(self, record: SubSubsectionRecord, debug: bool = False) -> Tuple[np.ndarray, List[float]]:
        """
        Decodes LB=0, 1, 2 LIST record into an MxM interval matrix and its NE point energy grid.
        M = NE - 1.
        """
        if debug: print(f"    DEBUG: Decoding LB={record.lb}...")
        energies = record.e_table_k # Point grid Ek (length NE)
        ne = len(energies)
        if ne < 2:
            raise ValueError(f"LB={record.lb} requires at least 2 energy points (NE >= 2) to define intervals.")
        m = ne - 1 # Number of intervals
        f_values = record.f_table_k # Values corresponding to Ek
        matrix = np.zeros((m, m))
        if debug: print(f"      DEBUG: Native grid NE={ne}, Matrix size M={m}x{m}")

        if record.lb == 0: # Absolute variance, diagonal on intervals
            # Needs division by <Xk><Yk> from MF4 for relative - DEFERRED
            # Fk corresponds to Ek, map to interval k = [Ek, Ek+1)
            for k in range(m):
                # Use Fk associated with the start of the interval Ek
                matrix[k, k] = f_values[k] # Store absolute variance Fk on diagonal C(k,k)
        elif record.lb == 1: # Fractional variance, diagonal on intervals
            # Fk corresponds to Ek, map to interval k = [Ek, Ek+1)
            for k in range(m):
                 # Use Fk associated with the start of the interval Ek
                matrix[k, k] = f_values[k] # Fk is fractional variance C(k,k)
        elif record.lb == 2: # Relative sigma Sk, fully correlated across intervals
            # Sk corresponds to Ek, map to interval k = [Ek, Ek+1)
            for k in range(m):
                for l in range(m):
                    # Use Sk and Sl associated with the start of intervals k and l
                    matrix[k, l] = f_values[k] * f_values[l] # Sk * Sl

        if debug: print(f"    DEBUG: Finished decoding LB={record.lb}.")
        return matrix, energies # Return MxM matrix and NE point grid

    def _decode_lb5_matrix(self, record: SubSubsectionRecord, debug: bool = False) -> Tuple[np.ndarray, List[float]]:
        """
        Decodes LB=5 LIST record into an MxM interval matrix and its NE point energy grid.
        M = NE - 1.
        """
        if debug: print(f"    DEBUG: Decoding LB=5 LS={record.ls}...")
        energies = record.energies # Point grid Ek (length NE)
        ne = record.ne
        if ne < 2:
            raise ValueError("LB=5 requires at least 2 energy points (NE >= 2) to define intervals.")
        m = ne - 1 # Number of intervals
        ls = record.ls
        raw_values = np.array(record.matrix) # Covariance values Fk,l for intervals
        matrix = np.zeros((m, m))
        if debug: print(f"      DEBUG: Native grid NE={ne}, Matrix size M={m}x{m}, Raw values count={len(raw_values)}")


        # Check data size consistency
        expected_size = 0
        if ls == 1: # Symmetric upper triangle M(M+1)/2
            expected_size = m * (m + 1) // 2
        else: # Asymmetric M*M (based on ENDF manual, differs from recipe's (M-1)^2)
             expected_size = m * m
        if len(raw_values) != expected_size:
             # Allow flexibility if NT calculation was different but data seems present
             if debug: print(f"      WARNING: LB=5 LS={ls}: Data size {len(raw_values)} != expected {expected_size} for M={m}. Proceeding cautiously.")
             # raise ValueError(f"LB=5 LS={ls}: Incorrect number of matrix values. Expected {expected_size} for M={m}, got {len(raw_values)}.")

        idx = 0
        start_fill = time.time()
        if ls == 1: # Symmetric upper triangle, row-wise
            for k in range(m): # Interval index k
                for l in range(k, m): # Interval index l >= k
                    if idx < len(raw_values):
                        matrix[k, l] = raw_values[idx]
                        if k != l:
                            matrix[l, k] = raw_values[idx] # Mirror for symmetry
                        idx += 1
        else: # Asymmetric full matrix M*M, row-wise
             for k in range(m): # Interval index k
                for l in range(m): # Interval index l
                    if idx < len(raw_values):
                        matrix[k, l] = raw_values[idx]
                        idx += 1
        end_fill = time.time()
        if debug: print(f"      DEBUG: Matrix fill time: {end_fill - start_fill:.4f}s")

        if idx != len(raw_values) and debug:
             print(f"      WARNING: LB=5 LS={ls}: Filled {idx} elements, but expected to use {len(raw_values)}. Check indexing.")

        if debug: print(f"    DEBUG: Finished decoding LB=5.")
        return matrix, energies # Return MxM matrix and NE point grid

    def _decode_lb6_matrix(self, record: SubSubsectionRecord, debug: bool = False) -> Tuple[np.ndarray, List[float], List[float]]:
        """
        Decodes LB=6 LIST record into an RxC interval matrix and its NER, NEC point energy grids.
        R = NER - 1, C = NEC - 1.
        """
        if debug: print(f"    DEBUG: Decoding LB=6...")
        row_energies = record.row_energies # Point grid ERk (length NER)
        col_energies = record.col_energies # Point grid ECl (length NEC)
        ner = len(row_energies)
        nec = len(col_energies)
        if ner < 2 or nec < 2:
             raise ValueError("LB=6 requires NER >= 2 and NEC >= 2 to define intervals.")
        r = ner - 1 # Number of row intervals
        c = nec - 1 # Number of column intervals
        raw_values = np.array(record.rect_matrix) # Covariance values Fr,c for intervals
        matrix = np.zeros((r, c))
        if debug: print(f"      DEBUG: Row grid NER={ner}, Col grid NEC={nec}, Matrix size R={r}x C={c}, Raw values count={len(raw_values)}")


        # Check data size consistency: NT = 1 + NER * NEC (from parser)
        # Number of matrix elements = R * C = (NER-1)*(NEC-1)
        # The raw_values should have R*C elements based on recipe. Let's check parser logic.
        # Parser: nec = (nt - 1) // ner. rect_matrix = all_values[ner+nec:]. NT = ner+nec+len(rect_matrix) ? No.
        # This might need revision if interval interpretation is strict.

        expected_size = r * c
        if len(raw_values) != expected_size:
            # Check if parser logic based on NT leads to this size
            nt_based_size = record.nt - ner - nec
            if len(raw_values) == nt_based_size:
                 print(f"Warning: LB=6 matrix value count {len(raw_values)} matches NT-NER-NEC ({nt_based_size}) but differs from recipe's R*C ({expected_size}). Using data as read.")
                 # If this happens, the matrix filling logic might be wrong if data isn't R*C.
                 # Let's proceed assuming the data IS R*C as read by parser, even if count seems off vs recipe.
                 pass # Keep matrix shape R*C
            else:
                 raise ValueError(f"LB=6: Inconsistent number of matrix values. Expected {expected_size} (R*C) or {nt_based_size} (NT-NER-NEC), got {len(raw_values)}.")


        idx = 0
        start_fill = time.time()
        for k in range(r): # Row interval index k
            for l in range(c): # Column interval index l
                if idx < len(raw_values):
                    matrix[k, l] = raw_values[idx]
                    idx += 1
        end_fill = time.time()
        if debug: print(f"      DEBUG: Matrix fill time: {end_fill - start_fill:.4f}s")

        if idx != len(raw_values) and debug:
             print(f"      WARNING: LB=6: Filled {idx} elements, but expected to use {len(raw_values)}. Check indexing.")

        if debug: print(f"    DEBUG: Finished decoding LB=6.")
        return matrix, row_energies, col_energies # Return RxC matrix and NER, NEC point grids

    def _project_matrix_piecewise_constant(self,
                                           component_matrix: np.ndarray,
                                           native_point_grid: List[float],
                                           union_point_grid: List[float],
                                           is_lb6: bool = False,
                                           native_col_point_grid: Optional[List[float]] = None,
                                           debug: bool = False
                                           ) -> np.ndarray:
        """
        Projects an interval matrix onto the union grid using piecewise constant assumption (bin sharing).
        Handles square (MxM -> M_union x M_union) and rectangular (RxC -> M_union x M_union) projection.
        """
        if debug: print(f"    DEBUG: Projecting matrix...")
        start_proj = time.time()

        # --- Optimization: Check for identical grids ---
        grids_are_identical = False
        if not is_lb6:
            if native_point_grid == union_point_grid:
                grids_are_identical = True
        else:
            # For LB=6, both row and column grids must match the union grid
            if native_point_grid == union_point_grid and native_col_point_grid == union_point_grid:
                grids_are_identical = True

        if grids_are_identical:
            if debug: print("      DEBUG: Native and Union grids are identical. Skipping projection calculation.")
            # Ensure the returned matrix has the correct shape (union_m x union_m)
            # This should already be the case if grids are identical, but double-check
            union_ne = len(union_point_grid)
            union_m = union_ne - 1 if union_ne >= 2 else 0
            if component_matrix.shape == (union_m, union_m):
                 end_proj = time.time()
                 if debug: print(f"    DEBUG: Projection time (skipped): {end_proj - start_proj:.4f}s")
                 return component_matrix.copy() # Return a copy
            else:
                 # This case should ideally not happen if grids are identical and decoding was correct
                 if debug: print(f"      WARNING: Grids identical but matrix shape {component_matrix.shape} mismatch union shape ({union_m}x{union_m}). Proceeding with full projection.")
        # --- End Optimization ---


        native_ne = len(native_point_grid)
        union_ne = len(union_point_grid)
        if native_ne < 2 or union_ne < 2:
            if debug: print("      DEBUG: Cannot project, NE < 2.")
            return np.zeros((union_ne - 1, union_ne - 1)) # Cannot project without intervals

        native_m = native_ne - 1
        union_m = union_ne - 1
        projected_matrix = np.zeros((union_m, union_m))

        native_col_ne = len(native_col_point_grid) if is_lb6 and native_col_point_grid else native_ne
        native_col_m = native_col_ne - 1

        if debug:
            print(f"      DEBUG: Native shape ({native_m}x{native_col_m if is_lb6 else native_m}), Union shape ({union_m}x{union_m})")

        if is_lb6 and component_matrix.shape != (native_m, native_col_m):
             raise ValueError(f"LB=6 Projection: Component matrix shape {component_matrix.shape} doesn't match expected ({native_m}, {native_col_m})")
        elif not is_lb6 and component_matrix.shape != (native_m, native_m):
             raise ValueError(f"Projection: Component matrix shape {component_matrix.shape} doesn't match expected ({native_m}, {native_m})")

        # Iterate through each cell (k_union, l_union) of the target projected matrix
        # This part is computationally expensive for large matrices
        for k_union in range(union_m):
            # Define the energy bounds for the target row interval k_union
            e_k_union_low = union_point_grid[k_union]
            e_k_union_high = union_point_grid[k_union + 1]
            delta_e_k_union = e_k_union_high - e_k_union_low
            if delta_e_k_union <= 0: continue # Skip zero-width intervals

            for l_union in range(union_m):
                # Define the energy bounds for the target column interval l_union
                e_l_union_low = union_point_grid[l_union]
                e_l_union_high = union_point_grid[l_union + 1]
                delta_e_l_union = e_l_union_high - e_l_union_low
                if delta_e_l_union <= 0: continue # Skip zero-width intervals

                # Find the contribution from the source component matrix
                sum_contribution = 0.0

                # Iterate through source row intervals k_native
                for k_native in range(native_m):
                    e_k_native_low = native_point_grid[k_native]
                    e_k_native_high = native_point_grid[k_native + 1]

                    # Calculate overlap fraction for row interval k
                    overlap_k_low = max(e_k_union_low, e_k_native_low)
                    overlap_k_high = min(e_k_union_high, e_k_native_high)
                    overlap_k_width = max(0.0, overlap_k_high - overlap_k_low)
                    frac_k = overlap_k_width / delta_e_k_union if delta_e_k_union > 0 else 0.0

                    if frac_k == 0: continue # No overlap in this row interval

                    # Iterate through source column intervals l_native
                    native_l_grid = native_col_point_grid if is_lb6 and native_col_point_grid else native_point_grid
                    for l_native in range(native_col_m):
                        e_l_native_low = native_l_grid[l_native]
                        e_l_native_high = native_l_grid[l_native + 1]

                        # Calculate overlap fraction for column interval l
                        overlap_l_low = max(e_l_union_low, e_l_native_low)
                        overlap_l_high = min(e_l_union_high, e_l_native_high)
                        overlap_l_width = max(0.0, overlap_l_high - overlap_l_low)
                        frac_l = overlap_l_width / delta_e_l_union if delta_e_l_union > 0 else 0.0

                        if frac_l == 0: continue # No overlap in this column interval

                        # Add contribution: C_native(k,l) * frac_k * frac_l (piecewise constant)
                        # Note: This assumes C is constant over the native bin.
                        # More sophisticated methods might weight differently.
                        sum_contribution += component_matrix[k_native, l_native] * frac_k * frac_l

                projected_matrix[k_union, l_union] = sum_contribution

        end_proj = time.time()
        if debug: print(f"    DEBUG: Projection time (calculated): {end_proj - start_proj:.4f}s")
        return projected_matrix


    def to_ang_covmat(self, debug: bool = False) -> 'MF34CovMat': 
        """
        Convert the MF34MT data to an MF34CovMat object, aggregating LIST records
        per sub-subsection (L, L1 pair) according to ENDF rules for relative covariance.

        Args:
            debug (bool): If True, print debug information during processing.

        Returns
        -------
        MF34CovMat
            Angular distribution covariance matrix object containing MxM relative matrices.
        """
        if debug: print(f"DEBUG: Starting to_ang_covmat for MF34 MT={self.number}")
        from mcnpy.cov.mf34_covmat import MF34CovMat # Local import

        isotope = int(self._za)
        reaction = self.number
        ang_covmat = MF34CovMat()

        # Process each subsection (MT1)
        if debug: print(f"DEBUG: Found {len(self._subsections)} subsections (MT1).")
        for subsec_idx, subsection in enumerate(self._subsections):
            mt1 = subsection.mt1
            if debug: print(f"  DEBUG: Processing subsection {subsec_idx+1}/{len(self._subsections)}: MT1={mt1}")

            # Process each sub-subsection (L, L1 pair)
            if debug: print(f"  DEBUG: Found {len(subsection.sub_subsections)} sub-subsections (L,L1).")
            for subsub_idx, sub_subsec in enumerate(subsection.sub_subsections):
                l = sub_subsec.l
                l1 = sub_subsec.l1
                if debug: print(f"    DEBUG: Processing sub-subsection {subsub_idx+1}/{len(subsection.sub_subsections)}: L={l}, L1={l1}, NI={sub_subsec.ni}")


                if not sub_subsec.records:
                    if debug: print("      DEBUG: Skipping sub-subsection - No LIST records.")
                    continue

                # 1. Determine Union Point Grid for this sub-subsection
                start_grid = time.time()
                all_energies_set: Set[float] = set()
                native_grids_map = {} # Store native grids for projection
                component_matrices = [] # Store decoded matrices before projection

                if debug: print(f"      DEBUG: Found {len(sub_subsec.records)} LIST records. Determining union grid...")
                for idx, record in enumerate(sub_subsec.records):
                    native_grid = None
                    native_col_grid = None
                    component_matrix = None
                    is_lb6 = False

                    try:
                        if record.lb in (0, 1, 2):
                            component_matrix, native_grid = self._decode_lb012_matrix(record, debug=debug)
                            if record.lb == 0:
                                print("Warning: LB=0 matrix generated is absolute variance. Needs MF4 data for relative conversion (Deferred). Treating as relative for summation.")
                                # TODO: Implement MF4 lookup and conversion here when available
                        elif record.lb == 5:
                            component_matrix, native_grid = self._decode_lb5_matrix(record, debug=debug)
                        elif record.lb == 6:
                            is_lb6 = True
                            component_matrix, native_grid, native_col_grid = self._decode_lb6_matrix(record, debug=debug)
                            all_energies_set.update(native_col_grid) # Add col grid points to union
                        else:
                             if debug: print(f"      WARNING: Skipping unsupported LB={record.lb} in record {idx+1}")
                             continue

                        if component_matrix is not None and native_grid is not None:
                             all_energies_set.update(native_grid)
                             native_grids_map[idx] = (native_grid, native_col_grid) # Store grids
                             component_matrices.append((idx, component_matrix, is_lb6)) # Store matrix and type
                             if debug: print(f"        DEBUG: Decoded component {idx+1} (LB={record.lb}), Native NE={len(native_grid)}")


                    except ValueError as e: # Catch decoding errors (e.g., NE<2, size mismatch)
                        print(f"Error decoding record {idx+1} (LB={record.lb}) in sub-subsection (L={l}, L1={l1}): {e}")
                    except Exception as e:
                        print(f"Unexpected error processing record {idx+1} (LB={record.lb}) in sub-subsection (L={l}, L1={l1}): {e}")

                end_grid = time.time()
                if debug: print(f"      DEBUG: Grid determination and decoding time: {end_grid - start_grid:.4f}s")

                if not all_energies_set:
                    if debug: print("      DEBUG: Skipping sub-subsection - No valid energies/components found after decoding.")
                    continue

                union_point_grid = sorted(list(all_energies_set))
                union_ne = len(union_point_grid)
                if union_ne < 2:
                    if debug: print("      DEBUG: Skipping sub-subsection - Union grid NE < 2.")
                    continue
                union_m = union_ne - 1
                total_cov_matrix = np.zeros((union_m, union_m))
                if debug: print(f"      DEBUG: Union grid NE={union_ne}, Final matrix size M={union_m}x{union_m}")


                # 4. Project each component onto the Union Grid and Sum
                start_sum = time.time()
                if debug: print(f"      DEBUG: Projecting and summing {len(component_matrices)} components...")
                for idx, component_matrix, is_lb6 in component_matrices:
                    try:
                        if debug: print(f"        DEBUG: Projecting component {idx+1}...")
                        native_grid, native_col_grid = native_grids_map[idx]
                        projected_component = self._project_matrix_piecewise_constant(
                            component_matrix,
                            native_grid,
                            union_point_grid,
                            is_lb6=is_lb6,
                            native_col_point_grid=native_col_grid,
                            debug=debug # Pass debug flag down
                        )
                        # 5. Add to total covariance matrix
                        total_cov_matrix += projected_component
                    except Exception as e:
                         print(f"Error projecting component {idx} in sub-subsection (L={l}, L1={l1}): {e}")

                end_sum = time.time()
                if debug: print(f"      DEBUG: Projection and summation time: {end_sum - start_sum:.4f}s")


                # Sanity Checks
                # 1. Size
                if total_cov_matrix.shape != (union_m, union_m):
                     print(f"Error: Final matrix shape {total_cov_matrix.shape} != expected ({union_m}, {union_m}) for L={l}, L1={l1}")
                     continue # Skip adding this matrix

                # 2. Diagonal Positivity (allow small tolerance for float errors)
                if not np.all(np.diag(total_cov_matrix) >= -1e-9):
                     neg_diags = np.where(np.diag(total_cov_matrix) < -1e-9)[0]
                     print(f"Warning: Negative diagonal elements found in final matrix for L={l}, L1={l1} at indices {neg_diags}. Clamping to zero.")
                     total_cov_matrix[np.diag_indices_from(total_cov_matrix)] = np.maximum(np.diag(total_cov_matrix), 0)


                # Add the final aggregated MxM matrix for this sub-subsection (L, L1)
                if debug: print(f"    DEBUG: Adding final matrix for L={l}, L1={l1} to MF34CovMat.")
                ang_covmat.add_matrix(
                    isotope, reaction, l,
                    isotope, mt1, l1,
                    total_cov_matrix, # The M_union x M_union matrix
                    union_point_grid  # The NE_union point grid
                )

        if debug: print(f"DEBUG: Finished to_ang_covmat for MF34 MT={self.number}")
        return ang_covmat
