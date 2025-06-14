from typing import List, Dict, Optional, Set, Tuple, Any, Union, Sequence
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
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
    """
    num_groups: int = 0
    energy_grid: Optional[List[float]] = None 
    isotope_rows: List[int] = field(default_factory=list)
    reaction_rows: List[int] = field(default_factory=list)
    isotope_cols: List[int] = field(default_factory=list)
    reaction_cols: List[int] = field(default_factory=list)
    matrices: List[np.ndarray] = field(default_factory=list)
    cross_sections: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)


    # ------------------------------------------------------------------
    # Basic methods
    # ------------------------------------------------------------------

    def copy(self) -> "CovMat":
        """Return a deep copy of this CovMat."""
        return copy.deepcopy(self)

    def add_matrix(
        self, 
        isotope_row: int, 
        reaction_row: int, 
        isotope_col: int, 
        reaction_col: int, 
        matrix: np.ndarray
    ) -> None:
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
    
    def remove_matrix(
        self,
        isotope: int,
        reaction_pairs: List[Tuple[int, int]],
        exceptions: Optional[List[Tuple[int, int]]] = None
    ) -> "CovMat":
        """
        Return a new CovMat without the specified matrices for a given isotope,
        but always keep any pairs listed in exceptions.
        Also removes cross section entries for diagonal or wildcard reactions that are removed.
        """
        if exceptions is None:
            exceptions = []
        exc_set = set()
        for e1, e2 in exceptions:
            exc_set.add((e1, e2))
            exc_set.add((e2, e1))

        new = CovMat(
            num_groups=self.num_groups,
            energy_grid=list(self.energy_grid) if self.energy_grid is not None else None,
        )
        # copy retained covariance matrices
        for ir, rr, ic, rc, M in zip(
            self.isotope_rows,
            self.reaction_rows,
            self.isotope_cols,
            self.reaction_cols,
            self.matrices
        ):
            remove = False
            if ir == isotope and ic == isotope:
                for r1, r2 in reaction_pairs:
                    if r1 == r2:
                        if rr == r1 or rc == r1:
                            remove = True
                    elif r1 == 0 or r2 == 0:
                        target = r2 if r1 == 0 else r1
                        if rr == target or rc == target:
                            remove = True
                    else:
                        if (rr == r1 and rc == r2) or (rr == r2 and rc == r1):
                            remove = True
                    if remove and (rr, rc) in exc_set:
                        remove = False
                    if remove:
                        break
            if not remove:
                new.isotope_rows.append(ir)
                new.reaction_rows.append(rr)
                new.isotope_cols.append(ic)
                new.reaction_cols.append(rc)
                new.matrices.append(M.copy())
        # copy retained cross sections
        for key, xs in self.cross_sections.items():
            zaid, mt = key
            remove_cs = False
            if zaid == isotope:
                for r1, r2 in reaction_pairs:
                    # diagonal removal
                    if r1 == r2 and mt == r1:
                        remove_cs = True
                    # wildcard removal (mt,0)
                    elif (r1 == 0 and mt == r2) or (r2 == 0 and mt == r1):
                        remove_cs = True
                    if remove_cs and key in exc_set:
                        remove_cs = False
                    if remove_cs:
                        break
            if not remove_cs:
                new.cross_sections[key] = xs.copy()
        return new
    




    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    
    @property
    def covariance_matrix(self) -> np.ndarray:
        """
        Return the full covariance matrix of shape (N·G) × (N·G),
        where N = number of unique (iso,rxn) blocks and G = num_groups.
        """
        param_pairs = self._get_param_pairs()
        idx_map = {p: i for i, p in enumerate(param_pairs)}

        G = self.num_groups
        N = len(param_pairs) * G
        full = np.zeros((N, N), dtype=float)

        for ir, rr, ic, rc, M in zip(
            self.isotope_rows,
            self.reaction_rows,
            self.isotope_cols,
            self.reaction_cols,
            self.matrices
        ):
            i = idx_map[(ir, rr)]
            j = idx_map[(ic, rc)]
            r0, r1 = i*G, (i+1)*G
            c0, c1 = j*G, (j+1)*G

            full[r0:r1, c0:c1] = M
            if i != j:
                full[c0:c1, r0:r1] = M.T

        return full

    @property
    def log_covariance_matrix(self) -> np.ndarray:
        cov_rel = self.covariance_matrix
        Sigma_log = np.log1p(cov_rel)
        return Sigma_log

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
    def isotopes(self) -> Set[int]:
        """
        Get the set of unique isotope IDs in the covariance matrices.
        
        Returns
        -------
        Set[int]
            Set of unique isotope IDs
        """
        return sorted(set(self.isotope_rows + self.isotope_cols))
    
    @property
    def reactions(self) -> Set[int]:
        """
        Get the set of unique reaction MT numbers in the covariance matrices.
        
        Returns
        -------
        Set[int]
            Set of unique reaction MT numbers
        """
        return sorted(set(self.reaction_rows + self.reaction_cols))

    @property
    def correlation_matrix(self) -> np.ndarray:
        """ Unclipped: diag forced to 1, undefined→nan, no off-diag correction. """
        return self._compute_correlation(clip=False, force_diagonal=True)

    @property
    def clipped_correlation_matrix(self) -> np.ndarray:
        """ Clipped into [-1,1], diag forced to 1, undefined→nan. """
        return self._compute_correlation(clip=True,  force_diagonal=True)




    # ------------------------------------------------------------------
    # General methods
    # ------------------------------------------------------------------

    def reactions_by_isotope(self, isotope: Optional[int] = None) -> Union[Dict[int, List[int]], List[int]]:
        '''
        Get a mapping of isotopes to their available reactions, or list of reactions for a specific isotope.

        Parameters
        ----------
        isotope : Optional[int]
            If provided, return reactions only for this isotope.

        Returns
        -------
        Dict[int, List[int]] or List[int]
            Mapping from isotope IDs to sorted lists of MT numbers, or list of MT numbers for the specified isotope.
        '''
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

        return sorted_dict

    def clean_cov(self, isotope: int) -> "CovMat":
        """
        Return a new CovMat containing only sub-matrices for *isotope*,
        always dropping reaction 1 and applying the mid/high-range rules:
            4 → 51-91, 103 → 600-649, 104 → 650-699,
            105 → 700-749, 106 → 750-799, 107 → 800
        """

        # ---- 1. indices with the requested isotope on both axes ----
        idxs = [
            i
            for i, (iso_r, iso_c) in enumerate(
                zip(self.isotope_rows, self.isotope_cols)
            )
            if iso_r == isotope and iso_c == isotope
        ]

        # ---- 2. reactions present in those sub-matrices + XS vectors ----
        reac_present = {
            *[self.reaction_rows[i] for i in idxs],
            *[self.reaction_cols[i] for i in idxs],
            *[
                mt
                for (iso, mt) in self.cross_sections.keys()
                if iso == isotope
            ],
        }

        # ---- 3. mid → high map ----
        mid_high = {
            4: set(range(51, 92)),
            103: set(range(600, 650)),
            104: set(range(650, 700)),
            105: set(range(700, 750)),
            106: set(range(750, 800)),
            107: {800},
        }

        # ---- 4. decide which mid codes to drop ----
        drop_mid = {
            mid: any(code in high_set for code in reac_present)
            for mid, high_set in mid_high.items()
        }

        # ---- 5. build the new object ----
        nuevo = CovMat(
            num_groups=self.num_groups,
            energy_grid=self.energy_grid.copy() if self.energy_grid is not None else None,
        )

        # copy matrices that survive the filters
        for i in idxs:
            r = self.reaction_rows[i]
            c = self.reaction_cols[i]

            if r == 1 or c == 1:
                continue
            if (r in drop_mid and drop_mid[r]) or (c in drop_mid and drop_mid[c]):
                continue

            nuevo.isotope_rows.append(isotope)
            nuevo.reaction_rows.append(r)
            nuevo.isotope_cols.append(isotope)
            nuevo.reaction_cols.append(c)
            nuevo.matrices.append(self.matrices[i])

        # copy XS vectors that survive the same filters
        for (iso, mt), xs in self.cross_sections.items():
            if iso != isotope:
                continue
            if mt == 1:
                continue
            if mt in drop_mid and drop_mid[mt]:
                continue
            nuevo.cross_sections[(iso, mt)] = xs.copy()

        return nuevo
    
    def filter_by_isotope(self, isotope: int) -> "CovMat":
        """
        Return a new CovMat containing only sub-matrices and cross-sections for the given isotope.
        All reactions for that isotope are retained.

        Similar to clean_cov, but without dropping any reactions.
        """
        # Indices where both row and column isotopes match the requested isotope
        idxs = [
            i for i, (iso_r, iso_c) in enumerate(
                zip(self.isotope_rows, self.isotope_cols)
            ) if iso_r == isotope and iso_c == isotope
        ]

        new_cov = CovMat(
            num_groups=self.num_groups,
            energy_grid=self.energy_grid.copy() if self.energy_grid is not None else None,
        )

        for i in idxs:
            new_cov.isotope_rows.append(isotope)
            new_cov.reaction_rows.append(self.reaction_rows[i])
            new_cov.isotope_cols.append(isotope)
            new_cov.reaction_cols.append(self.reaction_cols[i])
            new_cov.matrices.append(self.matrices[i])

        for (iso, mt), xs in self.cross_sections.items():
            if iso == isotope:
                new_cov.cross_sections[(iso, mt)] = xs.copy()

        return new_cov              

    def to_dataframe(self) -> pd.DataFrame:
        '''
        Convert the covariance matrix data to a pandas DataFrame.

        Includes an extra row at the beginning to store the energy grid if available.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the covariance matrix data with columns:
            ISO_H, REAC_H, ISO_V, REAC_V, STD
        '''
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
                "STD": [self.energy_grid]  # Store the list directly
            })
            # Concatenate the energy grid row at the beginning
            df = pd.concat([energy_grid_row, df], ignore_index=True)

        # Sort by ISO_H, then REAC_H, then REAC_V
        df = df.sort_values(by=["ISO_H", "REAC_H", "REAC_V"]).reset_index(drop=True)

        return df

    def fix_covariance(
        self,
        *,
        level: str = "soft",            # 'soft' | 'medium' | 'hard'
        high_val_thresh: float = 5.0,
        accept_tol: float = -1.0e-4,
        clamp_max_iter: int = 10,
        max_steps: int = 40,
        verbose: bool = True,
        logger = None,  # Optional logger for file output
    ) -> Tuple["CovMat", Dict[str, Any]]:
        """
        Clean the covariance matrix until it is positive-(semi)definite.

        Parameters
        ----------
        level
            'soft'   - clamp variances only  
            'medium' - clamp then drop the worst *block pairs*  
            'hard'   - clamp then drop the worst *reactions* (all blocks)  
        high_val_thresh
            Threshold used both by clamping and by the removal heuristic.
        accept_tol
            Minimum eigenvalue tolerated for acceptance.
        clamp_max_iter
            Maximum clamping passes before switching strategy.
        max_steps
            Maximum block-removal iterations (if used).
        verbose
            Forwarded to the removal routine.
        logger
            Optional logger instance for file output. If None, uses print().
        """

        lvl = level.lower()
        if lvl not in ("soft", "medium", "hard"):
            raise ValueError("level must be 'soft', 'medium' or 'hard'")

        # ------------------------------------------------------------------
        # 1) Always start with variance clamping
        # ------------------------------------------------------------------
        cm_after_clamp, log = self._clamp_covariance(
            high_val_thresh=high_val_thresh,
            accept_tol=accept_tol,
            max_iter=clamp_max_iter,
            verbose=verbose,
            logger=logger,
        )

        # If clamping was enough, stop here
        if log.get("converged", False):
            log.update({
                "strategy": lvl,
                "used_removal": False,
                "soft_threshold_met": True,  # Add flag for soft level success
            })
            return cm_after_clamp, log

        # For soft level, we don't do removal but we mark that threshold wasn't met
        if lvl == "soft":
            log.update({
                "strategy": lvl,
                "used_removal": False,
                "soft_threshold_met": False,  # Add flag for soft level failure
            })
            return cm_after_clamp, log

        # ------------------------------------------------------------------
        # 2) Continue with block removal (medium / hard)
        # ------------------------------------------------------------------
        remove_whole_rxn = (lvl == "hard")
        cm_final, rem_log = cm_after_clamp._autofix_covariance(
            accept_tol=accept_tol,
            max_steps=max_steps,
            verbose=verbose,
            remove_all=remove_whole_rxn,
            high_val_thresh=high_val_thresh,
            logger=logger,
        )

        # Merge the two logs for convenience - ensure final eigenvalue is used
        out_log: Dict[str, Any] = {
            **log,
            **{k: v for k, v in rem_log.items() if k not in ["min_eigenvalue"]},  # Don't overwrite final eigenvalue
            "strategy": lvl,
            "used_removal": True,
            "removal_log": rem_log,
            "min_eigenvalue": rem_log.get("min_eigenvalue", log.get("min_eigenvalue")),  # Use final eigenvalue
        }
        return cm_final, out_log
    
    def report_large_values(
        self,
        threshold: float = 1.0,
        top_n: int = 30,
        return_text: bool = False
        ) -> Optional[Tuple[str, dict]]:
        """
        Scan each block of the relative covariance matrices and generate a detailed report
        of entries that exceed the specified threshold.

        If return_text is True, returns a tuple:
        (report_text, summary_dict)
        where summary_dict contains:
        - zaid: the main isotope ZAID
        - name: human-readable symbol
        - count: number of entries > threshold
        - max_value: largest flagged value
        """
        G = self.num_groups
        if G == 0:
            if return_text:
                return None
            print("No energy groups available. Cannot generate report.")
            return None

        total_checked = 0
        total_flagged = 0
        max_value = 0.0
        large_values = []

        # map MT to reaction names
        reaction_names = {mt: MT_TO_REACTION.get(mt, f"MT={mt}")
                        for mt in self.reactions}

        if self.isotope_rows:
            raw_zaid = self.isotope_rows[0]
            try:
                zaid_int = int(raw_zaid)
            except (TypeError, ValueError):
                # if it’s not parseable, just pass it through
                zaid_int = raw_zaid
        else:
            zaid_int = 0

        from mcnpy._utils import zaid_to_symbol
        isotope_name = zaid_to_symbol(zaid_int)
        
        # scan each block
        for iso_r, rxn_r, iso_c, rxn_c, M in zip(
            self.isotope_rows,
            self.reaction_rows,
            self.isotope_cols,
            self.reaction_cols,
            self.matrices
        ):
            for ig in range(G):
                for jg in range(G):
                    total_checked += 1
                    val = M[ig, jg]
                    if val > threshold:
                        total_flagged += 1
                        if val > max_value:
                            max_value = val

                        entry = {
                            'value': val,
                            'iso_row': iso_r,
                            'rxn_row': rxn_r,
                            'rxn_name_row': reaction_names[rxn_r],
                            'iso_col': iso_c,
                            'rxn_col': rxn_c,
                            'rxn_name_col': reaction_names[rxn_c],
                            'row_idx': ig,
                            'col_idx': jg
                        }
                        # add energy ranges if available
                        if (self.energy_grid is not None
                            and len(self.energy_grid) >= G+1):
                            entry.update({
                                'e_row_low':  self.energy_grid[ig],
                                'e_row_high': self.energy_grid[ig+1],
                                'e_col_low':  self.energy_grid[jg],
                                'e_col_high': self.energy_grid[jg+1]
                            })
                        large_values.append(entry)

        # nothing found?
        if total_flagged == 0:
            if return_text:
                return None
            print(f"No entries > {threshold:.4e} for {isotope_name} (ZAID:{zaid_int}).")
            return None

        # sort & truncate
        large_values.sort(key=lambda x: x['value'], reverse=True)
        truncated = False
        if top_n is not None and len(large_values) > top_n:
            large_values = large_values[:top_n]
            truncated = True

        # build detailed report
        lines = []
        lines.append("\n" + "="*100)
        lines.append(f"LARGE VALUES REPORT FOR {isotope_name} (ZAID:{zaid_int}) > {threshold:.4e}")
        lines.append("="*100)
        lines.append("\nSUMMARY:")
        lines.append(f"  Total elements checked:     {total_checked:,}")
        lines.append(f"  Elements exceeding threshold:{total_flagged:,} "
                    f"({total_flagged/total_checked*100:.2f}%)")
        lines.append(f"  Maximum value found:        {max_value:.4e}")
        lines.append("\nDETAILED REPORT:")
        if truncated:
            lines.append(f"  (Showing top {top_n} entries)")

        # table header
        header = (
            f"{'#':>4} | {'Value':>10} | "
            f"{'Row Block':^20} | {'Col Block':^20} | "
            f"{'R#':>3} | {'C#':>3} | "
            f"{'Energy Row':^17} | {'Energy Col':^17}"
        )
        lines.append("\n" + "-"*len(header))
        lines.append(header)
        lines.append("-"*len(header))

        for i, e in enumerate(large_values, start=1):
            row_blk = f"{e['iso_row']},{e['rxn_row']}({e['rxn_name_row'][:6]})"
            col_blk = f"{e['iso_col']},{e['rxn_col']}({e['rxn_name_col'][:6]})"
            val_str = f"{e['value']:.4e}"
            energy_row = energy_col = ""
            if 'e_row_low' in e:
                energy_row = f"{e['e_row_low']:.4e}-{e['e_row_high']:.4e}"
                energy_col = f"{e['e_col_low']:.4e}-{e['e_col_high']:.4e}"
            line = (
                f"{i:>4} | {val_str:>10} | "
                f"{row_blk:<20} | {col_blk:<20} | "
                f"{e['row_idx']:>3} | {e['col_idx']:>3} | "
                f"{energy_row:<17} | {energy_col:<17}"
            )
            lines.append(line)

        lines.append("-"*len(header))
        if truncated:
            extra = total_flagged - top_n
            lines.append(f"Note: {extra:,} more entries not shown.")
        lines.append("\n" + "="*100)

        report_text = "\n".join(lines)

        if return_text:
            summary = {
                'zaid': zaid_int,
                'name': isotope_name,
                'count': total_flagged,
                'max_value': max_value
            }
            return report_text, summary
        else:
            print(report_text)
            return None

    def eigen_block_contributions(
        self,
        idx: Optional[int] = None,
        which: str = "min",      # "max", "min", or ignored when idx is not None
        top_n: int = 10,
        tol: float = 1e-12,
        symmetric: bool = True,  # if True keep only a ≤ b blocks
        relative: bool = False   # if True add weight = |c| / |eigenvalue|
    ) -> Dict[str, Any]:
        """
        Measure how each (block_i, block_j) sub-matrix of the covariance matrix
        contributes to a chosen eigenvalue λ.

        Args
        ----
        idx        : Index of the eigenvalue to inspect (overrides *which*).
        which      : If *idx* is None, choose 'max' or 'min' eigenvalue.
        top_n      : Return only the *top_n* largest |contribution| entries.
        tol        : Skip blocks with |contribution| ≤ tol.
        symmetric  : If True, include only pairs with i ≤ j (avoids duplicates).
        relative   : If True, report each block's share of |λ|.

        Returns
        -------
        A dict with:
            'index'        : eigenvalue index used.
            'eigenvalue'   : eigenvalue λ.
            'contributions': list of dicts:
                { 'block': (param_pairs[i], param_pairs[j]),
                    'contribution': c_ij,
                    'weight': |c_ij| / |λ|   # only if relative=True }
        """
        M = self.covariance_matrix               # full (n × n) matrix
        G = self.num_groups                      # rows per single block
        param_pairs = self._get_param_pairs()    # list of block labels
        n_blocks = len(param_pairs)

        # --- eigen-decomposition -------------------------------------------------
        eigvals, eigvecs = np.linalg.eigh(M)

        if idx is None:
            if which == "max":
                idx = int(np.argmax(eigvals))
            elif which == "min":
                idx = int(np.argmin(eigvals))
            else:
                raise ValueError("`which` must be 'max' or 'min' when idx is None")

        λ = float(eigvals[idx])
        v = eigvecs[:, idx]

        # Pre-slice the eigenvector into the same block structure
        v_blocks: List[np.ndarray] = [
            v[a * G:(a + 1) * G] for a in range(n_blocks)
        ]

        contribs: List[Dict[str, Any]] = []
        for i in range(n_blocks):
            vi = v_blocks[i]
            r0, r1 = i * G, (i + 1) * G
            for j in range(i if symmetric else 0, n_blocks):
                vj = v_blocks[j]
                c0, c1 = j * G, (j + 1) * G
                block = M[r0:r1, c0:c1]
                c_ij = float(vi @ block @ vj)
                if abs(c_ij) > tol:
                    entry: Dict[str, Any] = {
                        "block": (param_pairs[i], param_pairs[j]),
                        "contribution": c_ij,
                    }
                    if relative and abs(λ) > 0:
                        entry["weight"] = abs(c_ij) / abs(λ)
                    contribs.append(entry)

        # Sort and truncate
        contribs.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        contribs = contribs[:top_n]

        # --- Print nicely formatted summary ---
        print("=" * 80)
        print(f"Eigenvalue block contributions (idx={idx}, λ={λ:.4e})")
        print(f"Top {top_n} block contributions (tol={tol}, symmetric={symmetric}, relative={relative}):")
        print("-" * 80)
        header = (
            f"{'Block (iso,rxn)-(iso,rxn)':<35} {'Contribution':>15}"
            + (f" {'|c|/|λ|':>12}" if relative else "")
        )
        print(header)
        print("-" * 80)
        for entry in contribs:
            (b1, b2) = entry["block"]
            sblock = f"({b1[0]},{b1[1]})-({b2[0]},{b2[1]})"
            contrib_val = entry["contribution"]
            if relative and "weight" in entry:
                print(f"{sblock:<35} {contrib_val:15.6e} {entry['weight']:12.4e}")
            else:
                print(f"{sblock:<35} {contrib_val:15.6e}")
        print("=" * 80)

        return {
            "index": idx,
            "eigenvalue": λ,
            "contributions": contribs
        }
    
    def plot_uncertainties(
        self,
        zaid: Union[int, Sequence[int]],
        mt:   Union[int, Sequence[int]],
        ax: plt.Axes = None,
        *,
        energy_range: Optional[Tuple[float, float]] = None,
        style: str = 'default',
        figsize: Tuple[float, float] = (8, 5),
        dpi: int = 300,
        font_family: str = 'serif',
        legend_loc: str = 'best',
        **step_kwargs
    ) -> plt.Axes:
        """Delegate to the standalone plotting function without causing circular imports."""
        from mcnpy.cov.plotting import plot_uncertainties as _plot_uncertainties

        return _plot_uncertainties(
            covmat=self,
            zaid=zaid,
            mt=mt,
            ax=ax,
            energy_range=energy_range,
            style=style,
            figsize=figsize,
            dpi=dpi,
            font_family=font_family,
            legend_loc=legend_loc,
            **step_kwargs,
        )
    
    def plot_multigroup_xs(
        self,
        zaid: Union[int, Sequence[int]],
        mt: Union[int, Sequence[int]],
        ax: plt.Axes = None,
        *,
        energy_range: Optional[Tuple[float, float]] = None,
        show_uncertainties: bool = False,
        sigma: float = 1.0,
        style: str = 'default',
        figsize: Tuple[float, float] = (8, 5),
        dpi: int = 300,
        font_family: str = 'serif',
        legend_loc: str = 'best',
        **step_kwargs
    ) -> plt.Axes:
        """Delegate to the standalone plotting function without causing circular imports."""
        from mcnpy.cov.plotting import plot_multigroup_xs as _plot_multigroup_xs

        return _plot_multigroup_xs(
            covmat=self,
            zaid=zaid,
            mt=mt,
            ax=ax,
            energy_range=energy_range,
            show_uncertainties=show_uncertainties,
            sigma=sigma,
            style=style,
            figsize=figsize,
            dpi=dpi,
            font_family=font_family,
            legend_loc=legend_loc,
            **step_kwargs,
        )
    
    def plot_covariance_heatmap(
        self,
        zaid: int,
        mt: Union[int, Sequence[int], Tuple[int, int]],
        ax: plt.Axes = None,
        *,
        style: str = "default",
        figsize: Tuple[float, float] = (6, 6),
        dpi: int = 300,
        font_family: str = "serif",
        vmax: float = None,
        vmin: float = None,
        show_uncertainties: bool = True,
        show_energy_ticks: bool = True,
        **imshow_kwargs
    ) -> Union[plt.Axes, Tuple[plt.Axes, List[plt.Axes]]]:
        """
        Draw a correlation-matrix heat-map for a specified isotope and MT reaction(s).

        Parameters
        ----------
        zaid : int
            Isotope ID
        mt : int, sequence of int, or tuple of (row_mt, col_mt)
            MT reaction number(s). Can be:
            - Single int: diagonal block for that MT
            - Sequence of ints: diagonal blocks for those MTs  
            - Tuple of (row_mt, col_mt): off-diagonal block between row and column MT
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
        show_uncertainties : bool
            Whether to show uncertainty plots above the heatmap
        show_energy_ticks : bool
            Whether to show energy group ticks and labels on the heatmap axes
        **imshow_kwargs
            Additional arguments passed to imshow

        Returns
        -------
        plt.Axes or tuple
            If show_uncertainties=False: returns the heatmap axes
            If show_uncertainties=True: returns (heatmap_axes, uncertainty_axes_list)
        """
        from mcnpy.cov.heatmap import plot_covariance_heatmap as _plot_covariance_heatmap
        
        return _plot_covariance_heatmap(
            covmat=self,
            zaid=zaid,
            mt=mt,
            ax=ax,
            style=style,
            figsize=figsize,
            dpi=dpi,
            font_family=font_family,
            vmax=vmax,
            vmin=vmin,
            show_uncertainties=show_uncertainties,
            show_energy_ticks=show_energy_ticks,
            **imshow_kwargs
        )
    
    
    # ------------------------------------------------------------------
    # Decomposition methods
    # ------------------------------------------------------------------

    def cholesky_decomposition(
        self,
        *,
        space: str = "linear",        # "linear" (default) or "log"
        jitter_scale: float = 1e-10,
        max_jitter_ratio: float = 1e-3,
        verbose: bool = True,
        logger = None,  # Add logger parameter
    ) -> np.ndarray:
        """
        Robust Cholesky factor L such that  M ≈ L Lᵀ.
        """
        M = (self.covariance_matrix if space == "linear" else self.log_covariance_matrix)

        def _log_message(msg: str):
            """Helper to log message to logger or print."""
            if logger:
                logger.info(msg)
            else:
                print(msg)

        try:
            L = np.linalg.cholesky(M)
            if verbose:
                _log_message("  [INFO] No adjustment was necessary for Cholesky decomposition.")
            return L
        except np.linalg.LinAlgError:
            M_psd, _ = self._make_psd(
                M,
                jitter_scale=jitter_scale,
                max_jitter_ratio=max_jitter_ratio,
                verbose=verbose,
                logger=logger,  # Pass logger to _make_psd
            )
            return np.linalg.cholesky(M_psd)

    def eigen_decomposition(
        self,
        *,
        space: str = "linear",
        clip_negatives: bool = True,
        verbose: bool = True,
        logger = None,  # Add logger parameter
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Eigendecomposition with optional *clipping* instead of jitter.

        If ``clip_negatives`` is *True*, negative eigenvalues are set to
        zero and the user is informed of the number of clips and the minimum
        original value.
        """
        M = (self.covariance_matrix if space == "linear" else self.log_covariance_matrix)
        eigvals, eigvecs = np.linalg.eigh(M)
        
        def _log_message(msg: str):
            """Helper to log message to logger or print."""
            if logger:
                logger.info(msg)
            else:
                print(msg)
                
        if clip_negatives:
            mask = eigvals < 0
            if np.any(mask):
                if verbose:
                    _log_message(
                        f"  [INFO] Clipped {mask.sum()} negative eigenvalues (min {eigvals.min():.3e})."
                    )
                eigvals = np.where(mask, 0.0, eigvals)
            else:
                if verbose:
                    _log_message("  [INFO] No negative eigenvalues found; no clipping applied.")
        return eigvals, eigvecs

    def svd_decomposition(
        self,
        *,
        space: str = "linear",
        clip_negatives: bool = True,
        verbose: bool = True,
        full_matrices: bool = False,
        logger = None,  # Add logger parameter
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SVD with pre-clipping using eigendecomposition.

        For symmetric matrices, SVD and eigen are equivalent. If
        ``clip_negatives`` is activated, a preliminary eigendecomposition,
        clipping, and reconstruction step is performed before applying SVD,
        ensuring singular values consistent with a PSD matrix.
        """
        M = (self.covariance_matrix if space == "linear" else self.log_covariance_matrix)
        
        def _log_message(msg: str):
            """Helper to log message to logger or print."""
            if logger:
                logger.info(msg)
            else:
                print(msg)
                
        if clip_negatives:
            eigvals, eigvecs = np.linalg.eigh(M)
            mask = eigvals < 0
            if np.any(mask):
                if verbose:
                    _log_message(
                        f"  [INFO] Clipped {mask.sum()} negative eigenvalues before SVD (min {eigvals.min():.3e})."
                    )
                eigvals = np.where(mask, 0.0, eigvals)
                M = eigvecs @ np.diag(eigvals) @ eigvecs.T
            else:
                if verbose:
                    _log_message("  [INFO] No negative eigenvalues; applying SVD directly.")
        U, S, Vt = np.linalg.svd(M, full_matrices=full_matrices)
        return U, S, Vt

    

    
    

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------
    
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
        )
        
        # Create a summary table of data information
        property_col_width = 35
        value_col_width = header_width - property_col_width - 3 # -3 for spacing and formatting
        
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
            "Number of Unique Isotopes", len(self.isotopes), 
            width1=property_col_width, width2=value_col_width)
        info_table += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Unique Reactions", len(self.reactions), 
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






    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _get_param_pairs(self) -> List[Tuple[int,int]]:
        """
        Return a list of all (isotope, reaction) pairs present,
        sorted first by isotope number (ascending), then by reaction number (ascending).
        """
        pairs = set(zip(self.isotope_rows, self.reaction_rows)) \
              | set(zip(self.isotope_cols, self.reaction_cols))
        # explicit sort by isotope then reaction
        return sorted(pairs, key=lambda p: (p[0], p[1]))

    def _make_psd(
        self,
        M: np.ndarray,
        *,
        jitter_scale: float = 1e-10,
        max_jitter_ratio: float = 1e-3,
        verbose: bool = True,
        logger = None,  # Add logger parameter
    ) -> Tuple[np.ndarray, float]:
        """Devuelve una matriz PSD añadiendo el mínimo jitter admisible.

        Si **no** hace falta ajuste, se devuelve la misma matriz y jitter=0.
        Lanza ``LinAlgError`` si el desplazamiento requerido supera el umbral.
        """
        def _log_message(msg: str):
            """Helper to log message to logger or print."""
            if logger:
                logger.info(msg)
            else:
                print(msg)
                
        eigvals = np.linalg.eigvalsh(M)
        lam_min = eigvals[0]
        if lam_min >= 0:
            if verbose:
                _log_message("  [INFO] PSD check: no adjustment was necessary.")
            return M, 0.0

        avg_diag = float(np.mean(np.diag(M)))
        jitter = -lam_min + jitter_scale * avg_diag
        max_diag = float(np.max(np.diag(M)))

        if jitter / max_diag > max_jitter_ratio:
            if verbose:
                _log_message(
                    f"  [WARNING] PSD check aborted: required jitter {jitter:.3e} exceeds allowed ratio ({max_jitter_ratio})."
                )
            raise np.linalg.LinAlgError("Matrix is not positive semi‑definite and adjustment deemed too large.")

        if verbose:
            _log_message(
                f"  [INFO] Added jitter {jitter:.3e} to the diagonal to make the matrix positive semi‑definite."
            )
        return M + jitter * np.eye(M.shape[0]), jitter

    def _autofix_covariance(
        self,
        *,
        accept_tol: float = -1.0e-4,
        max_steps: int = 40,
        verbose: bool = True,
        remove_all: bool = False,
        high_val_thresh: float = 5.0,
        logger = None, 
    ) -> Tuple["CovMat", Dict[str, Any]]:
        """
        Iteratively remove block pairs or reactions until the
        matrix is positive-(semi)definite.

        New rules
        ─────────
        1. **Do not drop blocks before the eigen-analysis.**
        2. Before each eigen-analysis, *count* how many entries
        in every block exceed `high_val_thresh`, and accumulate
        that count **per reaction MT**.
        3. After eigen-analysis, quantify each negative-mode
        contribution exactly as before (`pair_scores`).
        • Boost any (ra, rb) by the *combined* high-value count
            of `ra` and `rb`.  
        • Reaction MT 2 is protected: if MT 2 ties with any
            other reaction for "worst", keep MT 2 and drop the
            other one.
        """
        if len(self.isotopes) != 1:
            raise ValueError("auto_fix_covariance works only for single-isotope matrices.")

        def _log_message(msg: str):
            """Helper to log message to logger or print."""
            if logger:
                logger.info(msg)
            else:
                print(msg)

        current: "CovMat" = self
        removed: List[Tuple[int, int]] = []
        removed_mts: List[int] = []  # Track individual MTs removed in "hard" mode
        removed_correlations: List[Tuple[int, int]] = []  # Track off-diagonal block removals
        iso = self.isotope_rows[0] if self.isotope_rows else None
        separator = "-" * 60

        if verbose:
            _log_message(f"\n[COVARIANCE] [AUTOFIX]")
            _log_message(f"  Checking covariance matrix for isotope {iso}")
            _log_message(f"{separator}")

        for step in range(1, max_steps + 1):
            M = current.covariance_matrix
            G = self.num_groups
            param_pairs = current._get_param_pairs()     # (iso, MT)

            # ──────────────────────────────────────────────────────
            # 1 Count large-magnitude entries by reaction
            # ──────────────────────────────────────────────────────
            high_cnt_per_rxn: Dict[int, int] = defaultdict(int)
            high_cnt_per_pair: Dict[Tuple[int, int], int] = {}

            for a, (_, ra) in enumerate(param_pairs):
                for b, (_, rb) in enumerate(param_pairs):
                    block = M[a*G:(a+1)*G, b*G:(b+1)*G]
                    cnt = int(np.sum(np.abs(block) > high_val_thresh))
                    if cnt:
                        high_cnt_per_pair[(ra, rb)] = cnt
                        high_cnt_per_rxn[ra] += cnt
                        high_cnt_per_rxn[rb] += cnt

            # ──────────────────────────────────────────────────────
            # 2 Eigen-analysis
            # ──────────────────────────────────────────────────────
            eigvals, eigvecs = np.linalg.eigh(M)
            min_ev = float(eigvals.min())
            if verbose:
                _log_message(f"  [STEP {step:02d}] [INFO] Smallest eigenvalue: {min_ev:.4e}")

            if min_ev >= accept_tol:
                _log_message(f"  [SUCCESS] Matrix accepted (λ_min = {min_ev:.4e} >= {accept_tol:.4e})")
                _log_message(separator)
                return current, {
                    "iterations": step,
                    "min_eigenvalue": min_ev,
                    "converged": True,  # Fix: This should be True when eigenvalue is acceptable
                    "removed_pairs": removed,
                    "removed_mts": removed_mts,
                    "removed_correlations": removed_correlations,
                }

            # ──────────────────────────────────────────────────────
            # 3 Score negative-mode contributions (unchanged core)
            # ──────────────────────────────────────────────────────
            pair_scores: Dict[Tuple[int, int], float] = defaultdict(float)
            neg_idxs = np.where(eigvals < accept_tol)[0]

            for idx in neg_idxs:
                v = eigvecs[:, idx]
                for a, (_, ra) in enumerate(param_pairs):
                    v_a = v[a*G:(a+1)*G]
                    for b, (_, rb) in enumerate(param_pairs):
                        block = M[a*G:(a+1)*G, b*G:(b+1)*G]
                        contrib = float(v_a @ block @ v[b*G:(b+1)*G])
                        if contrib < accept_tol:
                            pair_scores[(ra, rb)] += abs(contrib)

            # ──────────────────────────────────────────────────────
            # 4 Combine scores with high-value information
            #    (higher counts ⇒ stronger penalty)
            # ──────────────────────────────────────────────────────
            boosted_scores: Dict[Tuple[int, int], float] = {}
            for (ra, rb), base in pair_scores.items():
                boost = high_cnt_per_rxn.get(ra, 0) + high_cnt_per_rxn.get(rb, 0)
                boosted_scores[(ra, rb)] = base * (1.0 + boost)

            if not boosted_scores:
                # fall back to plain pair_scores – unlikely but safe
                boosted_scores = pair_scores

            worst_pair = max(boosted_scores, key=boosted_scores.get)
            ra, rb = worst_pair

            # tie-break in favour of MT 2
            if 2 in worst_pair:
                ra, rb = (rb, ra) if ra == 2 else (ra, rb)

            if remove_all:
                # choose which single reaction to drop
                r_drop = ra if boosted_scores.get((ra, ra), 0) >= boosted_scores.get((rb, rb), 0) else rb
                if r_drop == 2 and r_drop != (ra if ra != 2 else rb):
                    r_drop = rb if r_drop == ra else ra        # keep MT 2
                if verbose:
                    _log_message(f"  [STEP {step:02d}] [ACTION] Removing all blocks for reaction MT = {r_drop}")
                current = current.remove_matrix(isotope=iso,
                                                reaction_pairs=[(r_drop, 0)],
                                                exceptions=[])
                removed.append((r_drop, r_drop))
                removed_mts.append(r_drop)
            else:
                if verbose:
                    _log_message(f"  [STEP {step:02d}] [ACTION] Removing block pair = {(ra, rb)}")
                current = current.remove_matrix(isotope=iso,
                                                reaction_pairs=[(ra, rb), (rb, ra)],
                                                exceptions=[])
                removed.append((ra, rb))
                # For medium level, track when diagonal blocks are removed vs off-diagonal
                if ra == rb:
                    removed_mts.append(ra)
                else:
                    # This is an off-diagonal block removal (correlation removal)
                    removed_correlations.append((ra, rb))

        # ----------------------------------------------------------------------
        # Not converged within max_steps
        # ----------------------------------------------------------------------
        if verbose:
            _log_message(f"  [ERROR] Reached the limit of {max_steps} steps without convergence")
        # Directly compute min eigenvalue instead of using analyze_covariance
        min_eigenvalue = float(np.linalg.eigvalsh(current.covariance_matrix).min())

        logg = {
            "steps": max_steps,
            "min_eigenvalue": min_eigenvalue,
            "removed_pairs": removed,
            "removed_mts": removed_mts,
            "removed_correlations": removed_correlations,
            "converged": False,
        }

        if verbose:
            _log_message(f"  [SUMMARY]")
            _log_message(f"    Pairs removed: {logg['removed_pairs']}")
            _log_message(f"    MTs removed:   {logg['removed_mts']}")
            _log_message(f"    Correlations removed: {logg['removed_correlations']}")
            _log_message(f"    Final smallest eigenvalue: {logg['min_eigenvalue']:.4e}")
            _log_message(f"{separator}")

        return current, logg

    def _clamp_covariance(
        self,
        *,
        high_val_thresh: float = 5.0,
        accept_tol: float = -1.0e-4,
        max_iter: int = 5,
        verbose: bool = True,
        logger = None,  # Optional logger for file output
    ) -> Tuple["CovMat", Dict[str, Any]]:
        """
        Cap diagonal variances larger than `high_val_thresh`
        to **+1.0** (always positive) and rescale connected
        covariances so that correlations remain untouched.
        """
        if self.num_groups == 0:
            raise ValueError("num_groups is zero.")

        def _log_message(msg: str):
            """Helper to log message to logger or print."""
            if logger:
                logger.info(msg)
            else:
                print(msg)

        current = self.copy()
        G = self.num_groups
        param_pairs = current._get_param_pairs()
        idx_map = {p: i for i, p in enumerate(param_pairs)}
        separator = "-" * 60

        for iter_num in range(1, max_iter + 1):
            M = current.covariance_matrix
            changed_any = False
            clamped_count = 0

            # ──────────────────────────────────────────────────────────
            # 1 Clamp any |variance| > high_val_thresh → +1.0
            # ──────────────────────────────────────────────────────────
            
            _log_message(f"  [ITERATION {iter_num:02d}] Scanning variance values")
            
            for a, (iso, mt) in enumerate(param_pairs):
                r0 = a * G
                for g in range(G):
                    idx = r0 + g
                    old_var = M[idx, idx]
                    if abs(old_var) <= high_val_thresh:
                        continue

                    new_var = 1.0                        # fixed target
                    M[idx, idx] = new_var
                    changed_any = True
                    clamped_count += 1

                    _log_message(f"    [CLAMP #{clamped_count}] MT={mt:>3} G={g:>2} "
                        f"variance {old_var:.6g} → {new_var:.6g}")

                    # keep correlations
                    scale = np.sqrt(abs(new_var / old_var))
                    row_before = M[idx, :].copy()
                    col_before = M[:, idx].copy()

                    M[idx, :] *= scale
                    M[:, idx] *= scale
                    M[idx, idx] = new_var

                    diff_row = np.abs(row_before - M[idx, :]) > 0
                    diff_col = np.abs(col_before - M[:, idx]) > 0
                    diff_total = (np.count_nonzero(diff_row) +
                                np.count_nonzero(diff_col) - 1)

                    if verbose and diff_total:
                        for j in np.where(diff_row)[0]:
                            if j == idx:
                                continue
                            pp_j = param_pairs[j // G]
                            mt_j = pp_j[1]
                            g_j = j % G
                            _log_message(
                                f"      cov(MT={mt:>3}, G={g:>2}; "
                                f"MT={mt_j:>3}, G={g_j+1:>2}) "
                                f"{row_before[j]:12.4e} → {M[idx, j]:12.4e}"
                            )
                        for i in np.where(diff_col)[0]:
                            if i == idx:
                                continue
                            pp_i = param_pairs[i // G]
                            mt_i = pp_i[1]
                            g_i = i % G
                            _log_message(
                                f"      cov(MT={mt_i:>3}, G={g_i:>2}; "
                                f"MT={mt:>3}, G={g:>2}) "
                                f"{col_before[i]:12.4e} → {M[i, idx]:12.4e}"
                            )

                    _log_message(f"      {diff_total} covariances adjusted")

            if not changed_any:
                _log_message(f"  [ITERATION {iter_num:02d}] No variances above threshold; stopping clamping")
                break
            
            _log_message(f"  [ITERATION {iter_num:02d}] Clamped {clamped_count} variance values")

            # push edits back in the block structure  (unchanged)
            for ir, rr, ic, rc, mat_ref in zip(
                current.isotope_rows,
                current.reaction_rows,
                current.isotope_cols,
                current.reaction_cols,
                current.matrices,
            ):
                i = idx_map[(ir, rr)]
                j = idx_map[(ic, rc)]
                mat_ref[:, :] = M[i*G:(i+1)*G, j*G:(j+1)*G]

            min_ev = float(np.linalg.eigvalsh(M).min())
            _log_message(f"  [ITERATION {iter_num:02d}] Smallest eigenvalue: {min_ev:.4e}")

            if min_ev >= accept_tol:
                _log_message(f"  [SUCCESS] Matrix accepted (λ_min = {min_ev:.4e} >= {accept_tol:.4e})")
                _log_message(separator)
                return current, {
                    "iterations": iter_num,
                    "min_eigenvalue": min_ev,
                    "converged": True,  # Fix: This should be True when eigenvalue is acceptable
                    "clamped_values": clamped_count
                }
        
            # clamping not enough – fall back to removal strategy if autofix is True
            _log_message(f"  [WARNING] After clamping, smallest eigenvalue ({min_ev:.4e}) still below threshold ({accept_tol:.4e})")
        
        # Return the clamped matrix and log after clamping - Fix: Check final eigenvalue here too
        final_M = current.covariance_matrix
        min_ev_final = float(np.linalg.eigvalsh(final_M).min())
        
        # Check if final result is actually acceptable
        converged = min_ev_final >= accept_tol
        
        log = {
            "iterations": iter_num,
            "min_eigenvalue": min_ev_final,
            "converged": converged,  # Fix: Set based on actual final eigenvalue check
            "used_fallback": False,
            "clamp_iter": iter_num,
        }
        return current, log

    def _compute_correlation(
        self,
        clip: bool = False,
        force_diagonal: bool = True
    ) -> np.ndarray:
        """
        Core routine: builds the correlation matrix, optionally clipping
        into [-1,1] and optionally forcing diagonal==1.
        Entries where std_i * std_j == 0 become nan by default.
        """
        cov = self.covariance_matrix
        std = np.sqrt(np.diag(cov))
        denom = np.outer(std, std)

        # pure division, will give inf/nan where denom==0
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = cov / denom

        # mask all undefined entries
        corr[~np.isfinite(corr)] = np.nan

        if force_diagonal:
            # put ones on the diagonal, even if variance was zero
            np.fill_diagonal(corr, 1.0)

        if clip:
            # clip into [-1,1], leaving nan alone
            corr = np.where(np.isfinite(corr),
                            np.clip(corr, -1.0, 1.0),
                            np.nan)

        return corr


    #------------------------------------------------------------------
    # Methods and properties related to correlation matrix (Unused)
    #------------------------------------------------------------------

    def verify_correlation(self, atol: float = 1e-12, rtol: float = 1e-4) -> None:
        """Run basic checks on the correlation matrix and print any problems.

        Checks:
        1. Symmetry: ρ_ij == ρ_ji within *atol*.
        2. Diagonal consistency: 1 if variance > 0, else 0.
        3. Range: off‑diagonal in [‑1, 1].
        """
        R = self.correlation_matrix
        G = self.num_groups
        param_pairs = self._get_param_pairs()
        N = len(param_pairs)

        # Build helper lookup to turn a flat index into (ZAID, MT, group)
        def flat_to_components(idx: int) -> Tuple[int, int, int]:
            p_idx, g = divmod(idx, G)
            zaid, mt = param_pairs[p_idx]
            return zaid, mt, g + 1  # +1 for 1‑based group number

        # ------------------------------------------------------------------
        # 1. Symmetry
        # ------------------------------------------------------------------
        diff = np.abs(R - R.T)
        bad = np.argwhere(diff > atol)
        for i, j in bad:
            val_ij = R[i, j]
            val_ji = R[j, i]
            zr, mr, gr = flat_to_components(i)
            zc, mc, gc = flat_to_components(j)
            print(
                f"Asymmetry: ρ_ij={val_ij:.4e} but ρ_ji={val_ji:.4e} for "
                f"({zr},{mr}) ({zc},{mc}) groups {gr} {gc}."
            )

        # ------------------------------------------------------------------
        # 2 & 3. Diagonal and range
        # ------------------------------------------------------------------
        for i in range(R.shape[0]):
            zr, mr, gr = flat_to_components(i)
            diag_val = R[i, i]

            # Diagonal rule
            if np.isclose(diag_val, 0.0, atol=atol):
                # Variance must be zero → entire row/col should be zero
                row_nonzero = np.argwhere(~np.isclose(R[i, :], 0.0, atol=atol))
                col_nonzero = np.argwhere(~np.isclose(R[:, i], 0.0, atol=atol))
                offenders = set(row_nonzero.flatten()) | set(col_nonzero.flatten())
                offenders.discard(i)
                for j in offenders:
                    zc, mc, gc = flat_to_components(j)
                    val = R[i, j]
                    print(
                        f"Found value {val:.4e} for ({zr},{mr}) ({zc},{mc}) energy group {gr}. "
                        "Value should be 0 for correlation from a 0 variance component."
                    )
            else:
                if not np.isclose(diag_val, 1.0, atol=atol, rtol=rtol):
                    print(
                        f"Found value {diag_val:.4e} for ({zr},{mr}) ({zr},{mr}) energy group {gr}. "
                        "Value should be 1 for a diagonal element."
                    )

        # Off‑diagonal range check
        off_diag_indices = np.argwhere(~np.eye(R.shape[0], dtype=bool))
        for i, j in off_diag_indices:
            val = R[i, j]
            if val < -1 - rtol or val > 1 + rtol:
                zr, mr, gr = flat_to_components(i)
                zc, mc, gc = flat_to_components(j)
                print(
                    f"Found value {val:.4e} for ({zr},{mr}) ({zc},{mc}) energy group {gr}. "
                    "Value should be [-1,1] for an off-diagonal element."
                )

    def sanitize_by_correlation(
        self,
        *,
        max_abs_corr: float = 1.0,
        zero_threshold: float = 1.5,      # any |ρᵢⱼ| > zero_threshold → 0
        report_tol: float = 1e-6,
        eigen_floor: float = 1e-12,
        project_psd: bool = True,
        verbose: bool = True,
    ):
        """
        Clip out-of-range correlations, zero out huge outliers, inspect eigenvalues,
        and (optionally) project to PSD.

        Parameters
        ----------
        max_abs_corr
            Any correlation with |ρ| > max_abs_corr (but ≤ zero_threshold)
            is clipped to ±max_abs_corr.
        zero_threshold
            Any |ρ| > zero_threshold is set to 0.
        report_tol
            Minimum |old–new| change to actually log.
        eigen_floor
            Floor for eigenvalues when projecting to PSD.
        project_psd
            Whether to do the PSD projection here.
        verbose
            Print detailed diagnostics.
        """

        # --- build and symmetrize covariance C0 ---
        C0 = (self.covariance_matrix + self.covariance_matrix.T) * 0.5
        var = np.diag(C0)
        std = np.sqrt(var)
        D = np.outer(std, std)

        # --- form correlation R ---
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.divide(C0, D, out=np.zeros_like(C0), where=D>0)

        # --- prepare for logging ---
        p = R.shape[0]
        G = self.num_groups
        pairs = self._get_param_pairs()
        eye = np.eye(p, dtype=bool)

        # --- find zero-out entries ---
        too_big   = np.abs(R) > zero_threshold
        off_diag  = too_big & ~eye
        # only unique pairs in upper triangle
        tri_mask  = np.triu(np.ones_like(off_diag, dtype=bool), k=1)
        off_pairs = np.argwhere(off_diag & tri_mask)
        total_hits= int(too_big.sum())

        if verbose:
            print("\n[ sanitize_by_correlation ]")
            if off_pairs.size:
                print(f"  hard-zeroed {len(off_pairs)} off-diagonal pairs"
                    f" (total {total_hits} entries including symmetry):")
                for i, j in off_pairs:
                    bi, gi = divmod(i, G)
                    bj, gj = divmod(j, G)
                    iso_i, rxn_i = pairs[bi]
                    iso_j, rxn_j = pairs[bj]
                    old = R[i, j]
                    print(f"    ({iso_i},{rxn_i})-g{gi+1:02d}"
                        f" ↔ ({iso_j},{rxn_j})-g{gj+1:02d}"
                        f"   {old:+.3e} → +0.000e+00")
            else:
                print("  no entries exceeded zero_threshold")

        # --- apply the zeroing ---
        R_clipped = R.copy()
        R_clipped[too_big] = 0.0

        # --- now clip to ±max_abs_corr ---
        to_clip    = (np.abs(R_clipped) > max_abs_corr) & ~too_big
        clip_pairs = np.argwhere(to_clip & tri_mask)

        if verbose and clip_pairs.size:
            print(f"  clipped {len(clip_pairs)} off-diagonal pairs to ±{max_abs_corr}:")
            for i, j in clip_pairs:
                bi, gi = divmod(i, G)
                bj, gj = divmod(j, G)
                iso_i, rxn_i = pairs[bi]
                iso_j, rxn_j = pairs[bj]
                old = R_clipped[i, j]
                new = np.sign(old) * max_abs_corr
                if abs(old - new) > report_tol:
                    print(f"    ({iso_i},{rxn_i})-g{gi+1:02d}"
                        f" ↔ ({iso_j},{rxn_j})-g{gj+1:02d}"
                        f"   {old:+.3e} → {new:+.3e}")
        R_clipped[to_clip] = np.sign(R_clipped[to_clip]) * max_abs_corr

        # --- enforce exact 1’s on diagonal (or 0 if var==0) ---
        R_clipped[eye & (var != 0.0)] = 1.0
        R_clipped[eye & (var == 0.0)] = 0.0

        # --- rebuild covariance and symmetrize ---
        C1 = R_clipped * D
        C1[D == 0.0] = 0.0
        C1 = (C1 + C1.T) * 0.5

        # --- eigen before PSD ---
        w1, V1 = np.linalg.eigh(C1)
        if verbose:
            print("  smallest 5 eigenvalues AFTER clip :",
                " ".join(f"{x:+.3e}" for x in w1[:5]))

        # --- optional PSD projection ---
        if project_psd:
            neg_count = int((w1 < 0).sum())
            if neg_count and verbose:
                print(f"  PSD-proj: {neg_count} eigenvalues < 0, floored to {eigen_floor:.1e}")
            w2 = np.maximum(w1, eigen_floor)
            C2 = V1 @ np.diag(w2) @ V1.T
            C2 = (C2 + C2.T) * 0.5
            if verbose:
                print("  smallest 5 eigenvalues AFTER PSD  :",
                    " ".join(f"{x:+.3e}" for x in w2[:5]))
        else:
            C2 = C1

        # --- scatter back into blocks and return ---
        out = self.copy()
        out._scatter_full_into_blocks(C2)
        return out
    
    def _scatter_full_into_blocks(self, full_matrix: np.ndarray) -> None:
        """
        Scatter a full covariance matrix back into the individual block matrices.
        
        Parameters
        ----------
        full_matrix : np.ndarray
            Full covariance matrix of shape (N·G, N·G) where N is the number of 
            parameter pairs and G is num_groups.
        """
        param_pairs = self._get_param_pairs()
        idx_map = {p: i for i, p in enumerate(param_pairs)}
        G = self.num_groups

        # Update each matrix block with the corresponding section from the full matrix
        for i, (ir, rr, ic, rc, mat_ref) in enumerate(zip(
            self.isotope_rows,
            self.reaction_rows,
            self.isotope_cols,
            self.reaction_cols,
            self.matrices,
        )):
            row_idx = idx_map[(ir, rr)]
            col_idx = idx_map[(ic, rc)]
            r0, r1 = row_idx * G, (row_idx + 1) * G
            c0, c1 = col_idx * G, (col_idx + 1) * G
            
            # Update the matrix in place
            mat_ref[:, :] = full_matrix[r0:r1, c0:c1]