"""Sandwich formula for uncertainty propagation in nuclear data.

This module implements the sandwich formula σ²_R = S^T Σ S for propagating nuclear data 
uncertainties from sensitivity coefficients and covariance matrices.

The sandwich formula allows propagation of uncertainties from nuclear cross-section 
covariances to integral responses through sensitivity coefficients.
"""

from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
import numpy as np
import warnings
import logging

from mcnpy.sensitivities.sdf import SDFData, SDFReactionData
from mcnpy.cov.covmat import CovMat
from mcnpy._constants import MT_TO_REACTION, ATOMIC_NUMBER_TO_SYMBOL

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class UncertaintyContribution:
    """Container for individual uncertainty contributions from specific reactions.
    
    Attributes
    ----------
    zaid : int
        ZAID of the nuclide
    mt : int
        MT reaction number
    variance_contribution : float
        Contribution to total variance from this reaction
    uncertainty_contribution : float
        Square root of variance contribution (1-sigma)
    relative_contribution : float
        Relative contribution to total variance (fraction)
    nuclide : str
        Nuclide symbol (e.g., 'Fe-56')
    reaction_name : str
        Reaction name (e.g., 'elastic')
    """
    zaid: int
    mt: int
    variance_contribution: float
    uncertainty_contribution: float = field(init=False)
    relative_contribution: float = field(init=False)
    nuclide: str = field(init=False)
    reaction_name: str = field(init=False)
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.uncertainty_contribution = np.sqrt(abs(self.variance_contribution))
        
        # Calculate nuclide symbol
        z = self.zaid // 1000
        a = self.zaid % 1000
        
        if z in ATOMIC_NUMBER_TO_SYMBOL:
            self.nuclide = f"{ATOMIC_NUMBER_TO_SYMBOL[z]}-{a}"
        else:
            self.nuclide = f"Z{z}-{a}"
            
        # Calculate reaction name
        if self.mt in MT_TO_REACTION:
            self.reaction_name = MT_TO_REACTION[self.mt]
        else:
            self.reaction_name = f"MT{self.mt}"


@dataclass 
class UncertaintyResult:
    """Container for uncertainty propagation results.
    
    Attributes
    ----------
    total_variance : float
        Total propagated variance
    total_uncertainty : float
        Total uncertainty (1-sigma, square root of variance)
    relative_uncertainty : float
        Relative uncertainty (σ/μ)
    response_value : float
        Reference response value used for relative uncertainty
    response_error : float
        Error in the reference response value
    contributions : List[UncertaintyContribution]
        Individual reaction contributions sorted by magnitude
    n_reactions : int
        Number of reactions included in propagation
    n_energy_groups : int
        Number of energy groups used
    correlation_effects : float
        Contribution from cross-correlations between reactions
    """
    total_variance: float
    total_uncertainty: float
    relative_uncertainty: float
    response_value: float
    response_error: float
    contributions: List[UncertaintyContribution]
    n_reactions: int
    n_energy_groups: int
    correlation_effects: float = 0.0
    
    def __repr__(self) -> str:
        """Format uncertainty results for display."""
        lines = []
        lines.append("=" * 70)
        lines.append("UNCERTAINTY PROPAGATION RESULTS (Sandwich Formula)")
        lines.append("=" * 70)
        
        # Calculate absolute uncertainty
        absolute_uncertainty = self.relative_uncertainty * abs(self.response_value)
        relative_uncertainty_pct = self.relative_uncertainty * 100
        
        lines.append(f"Total variance (σ²):          {self.total_variance:.6e}")
        lines.append(f"Total uncertainty (1σ):       {self.total_uncertainty:.6e} (relative)")
        lines.append(f"Relative uncertainty:          {self.relative_uncertainty:.4%}")
        lines.append("")
        lines.append("RESPONSE VALUE WITH UNCERTAINTY:")
        lines.append(f"Response value:                {self.response_value:.6e} ± {self.response_error:.6e} (statistical)")
        lines.append(f"Nuclear data uncertainty:      ± {absolute_uncertainty:.6e} (absolute)")
        lines.append(f"Nuclear data uncertainty:      ± {relative_uncertainty_pct:.2f}% (relative)")
        lines.append(f"Final result:                  {self.response_value:.6e} ± {absolute_uncertainty:.6e}")
        lines.append("")
        lines.append(f"Reactions included:            {self.n_reactions}")
        lines.append(f"Energy groups:                 {self.n_energy_groups}")
        
        # Always show correlation effects (even if zero)
        if abs(self.correlation_effects) > 1e-15:
            corr_pct = abs(self.correlation_effects) / abs(self.total_variance) * 100 if abs(self.total_variance) > 1e-15 else 0.0
            lines.append(f"Cross-correlation effects:     {self.correlation_effects:.6e} ({corr_pct:.1f}% of total)")
        else:
            lines.append(f"Cross-correlation effects:     None (independent reactions)")
        
        lines.append("\n" + "=" * 70)
        lines.append("INDIVIDUAL REACTION CONTRIBUTIONS")
        lines.append("=" * 70)
        lines.append(f"{'Rank':<4} {'Nuclide':<12} {'Reaction':<15} {'Variance':<12} {'% of Total':<10}")
        lines.append("-" * 70)
        
        # Sort contributions by magnitude and show all (not just top 10)
        sorted_contribs = sorted(self.contributions, 
                               key=lambda x: abs(x.variance_contribution), 
                               reverse=True)
        
        for rank, contrib in enumerate(sorted_contribs, 1):
            pct = contrib.relative_contribution * 100
            lines.append(f"{rank:<4} {contrib.nuclide:<12} {contrib.reaction_name:<15} "
                        f"{contrib.variance_contribution:.4e} {pct:>8.2f}%")
        
        lines.append("=" * 70)
        
        # Add summary interpretation
        if self.relative_uncertainty > 1.0:  # > 100%
            lines.append("⚠️  WARNING: Very high relative uncertainty detected!")
            lines.append("   This may indicate incompatible sensitivity/covariance data")
            lines.append("   or issues with absolute/relative conversion.")
        elif self.relative_uncertainty > 0.5:  # > 50%
            lines.append("⚠️  High relative uncertainty - please verify results")
        elif self.relative_uncertainty > 0.1:  # > 10%
            lines.append("✓ Moderate uncertainty level")
        else:
            lines.append("✓ Low uncertainty level")
            
        return "\n".join(lines)


def sandwich_uncertainty_propagation(
    sdf_data: SDFData,
    cov_mat: CovMat,
    reaction_filter: Optional[Dict[int, List[int]]] = None,
    energy_tolerance: float = 1e-6,
    verbose: bool = True
) -> UncertaintyResult:
    """
    Apply the sandwich formula σ²_R = S^T Σ S to propagate nuclear data uncertainties.
    
    This function automatically handles:
    - Energy grid matching and validation
    - Conversion from absolute to relative sensitivities (to match relative covariances)
    - Matrix construction and sandwich formula application
    - Individual contribution analysis and cross-correlation effects
    
    Parameters
    ----------
    sdf_data : SDFData
        Sensitivity data containing sensitivity coefficients for various reactions
    cov_mat : CovMat  
        Covariance matrix data for nuclear cross sections (in relative form)
    reaction_filter : Dict[int, List[int]], optional
        Dictionary mapping ZAID to list of MT numbers to include in propagation.
        If None, all matching reactions between sensitivity and covariance data are used.
        Example: {26056: [2, 102]} includes only elastic and (n,γ) for Fe-56
    energy_tolerance : float, optional
        Tolerance for matching energy grid boundaries (default: 1e-6)
    verbose : bool, optional
        Print detailed information about the propagation process
        
    Returns
    -------
    UncertaintyResult
        Complete uncertainty propagation results including individual contributions
        and cross-correlation effects
        
    Raises
    ------
    ValueError
        If energy grids don't match or no matching reactions are found
    
    Notes
    -----
    The function assumes:
    - Sensitivity coefficients in SDF data are in relative form (dR/R)/(dσ/σ)
    - Covariance matrices are in relative form (as typically provided by SCALE/NJOY)
    - Energy grids are compatible between sensitivity and covariance data
    
    The sandwich formula σ²_R = S^T Σ S gives the relative variance of the response,
    from which the relative uncertainty is calculated as σ_R = sqrt(σ²_R).
    This relative uncertainty represents the fractional uncertainty in the response
    due to nuclear data uncertainties.
    """
    
    if verbose:
        logger.info("Starting sandwich uncertainty propagation")
        logger.info("=" * 60)
    
    # Step 1: Validate inputs
    if not sdf_data.data:
        raise ValueError("SDF data contains no sensitivity information")
    
    if not cov_mat.matrices:
        raise ValueError("Covariance matrix contains no data")
    
    if verbose:
        logger.info("✓ Input validation complete")
    
    # Step 2: Match energy grids (with automatic validation)
    energy_mapping = _match_energy_grids(
        sdf_data.pert_energies, 
        cov_mat.energy_grid,
        energy_tolerance,
        verbose
    )
    
    if not energy_mapping:
        raise ValueError("No matching energy groups found between sensitivity and covariance data")
    
    if verbose:
        logger.info("✓ Energy grid matching complete")
    
    # Step 3: Find matching reactions
    matching_reactions = _find_matching_reactions(
        sdf_data, 
        cov_mat, 
        reaction_filter,
        verbose
    )
    
    if not matching_reactions:
        raise ValueError("No matching reactions found between sensitivity and covariance data")
    
    if verbose:
        logger.info("✓ Reaction matching complete")
    
    # Step 4: Build matrices with proper relative/absolute conversion
    sensitivity_vector, covariance_matrix, reaction_indices = _build_matrices(
        sdf_data,
        cov_mat,
        matching_reactions,
        energy_mapping,
        verbose
    )
    
    if verbose:
        logger.info("✓ Matrix construction complete")
        logger.info("=" * 60)
    
    # Apply sandwich formula: σ²_R = S^T Σ S
    total_variance = float(sensitivity_vector.T @ covariance_matrix @ sensitivity_vector)
    total_uncertainty = np.sqrt(abs(total_variance))
    
    # Calculate relative uncertainty
    # Since both sensitivities and covariances are relative, the sandwich formula
    # gives relative variance directly. Therefore, sqrt(variance) IS the relative uncertainty.
    response_value = sdf_data.r0 if sdf_data.r0 and sdf_data.r0 != 0 else 1.0
    response_error = sdf_data.e0 if sdf_data.e0 else 0.0
    relative_uncertainty = total_uncertainty  # Already relative!
    
    # Calculate individual contributions
    contributions = _calculate_individual_contributions(
        sensitivity_vector,
        covariance_matrix, 
        reaction_indices,
        len(energy_mapping),
        total_variance,
        verbose
    )
    
    # Calculate correlation effects
    correlation_effects = _calculate_correlation_effects(
        sensitivity_vector,
        covariance_matrix,
        reaction_indices,
        len(energy_mapping)
    )
    
    if verbose:
        logger.info(f"Propagation complete: {relative_uncertainty:.2%} relative uncertainty")
    
    return UncertaintyResult(
        total_variance=total_variance,
        total_uncertainty=total_uncertainty,
        relative_uncertainty=relative_uncertainty,
        response_value=response_value,
        response_error=response_error,
        contributions=contributions,
        n_reactions=len(matching_reactions),
        n_energy_groups=len(energy_mapping),
        correlation_effects=correlation_effects
    )


def _match_energy_grids(
    sens_energies: List[float],
    cov_energies: Optional[List[float]],
    tolerance: float,
    verbose: bool
) -> Dict[int, int]:
    """Match energy grids between sensitivity and covariance data.
    
    Returns mapping from sensitivity group index to covariance group index.
    """
    if cov_energies is None:
        raise ValueError("Covariance matrix must have energy grid information")
    
    sens_array = np.array(sens_energies)
    cov_array = np.array(cov_energies)
    
    energy_mapping = {}
    
    # Match energy group boundaries
    # For multigroup data, we need to match group boundaries
    n_sens_groups = len(sens_energies) - 1  # Number of groups = boundaries - 1
    n_cov_groups = len(cov_energies) - 1
    
    for i in range(n_sens_groups):
        # Get sensitivity group boundaries
        sens_lower = sens_energies[i]
        sens_upper = sens_energies[i + 1]
        
        for j in range(n_cov_groups):
            # Get covariance group boundaries  
            cov_lower = cov_energies[j]
            cov_upper = cov_energies[j + 1]
            
            # Check if boundaries match within tolerance
            if (abs(sens_lower - cov_lower) < tolerance and 
                abs(sens_upper - cov_upper) < tolerance):
                energy_mapping[i] = j
                break
    
    if verbose:
        logger.info(f"Matched {len(energy_mapping)}/{n_sens_groups} energy groups")
        if len(energy_mapping) < n_sens_groups:
            logger.warning(f"Only {len(energy_mapping)} out of {n_sens_groups} energy groups matched")
    
    return energy_mapping


def _find_matching_reactions(
    sdf_data: SDFData,
    cov_mat: CovMat,
    reaction_filter: Optional[Dict[int, List[int]]],
    verbose: bool
) -> List[Tuple[int, int]]:
    """Find reactions that exist in both sensitivity and covariance data."""
    
    # Get available reactions from covariance matrix using CovMat's built-in method
    cov_reactions = set()
    cov_reactions_by_isotope = cov_mat.reactions_by_isotope()
    for isotope, reactions in cov_reactions_by_isotope.items():
        for reaction in reactions:
            cov_reactions.add((isotope, reaction))
    
    # Get available reactions from sensitivity data
    sens_reactions = {(r.zaid, r.mt) for r in sdf_data.data}
    
    # Find intersection
    matching_reactions = list(sens_reactions.intersection(cov_reactions))
    
    # Apply reaction filter if provided
    if reaction_filter:
        filtered_reactions = []
        
        # Handle special case for filtering by reaction type across all nuclides
        if "ALL_NUCLIDES" in reaction_filter:
            allowed_mts = reaction_filter["ALL_NUCLIDES"]
            for zaid, mt in matching_reactions:
                if mt in allowed_mts:
                    filtered_reactions.append((zaid, mt))
        else:
            # Normal filtering by specific nuclide and reactions
            for zaid, mt in matching_reactions:
                if zaid in reaction_filter:
                    # If empty list, include all reactions for this nuclide
                    if not reaction_filter[zaid] or mt in reaction_filter[zaid]:
                        filtered_reactions.append((zaid, mt))
        
        matching_reactions = filtered_reactions
        
        if verbose:
            logger.info(f"Applied reaction filter, {len(matching_reactions)} reactions selected")
    
    if verbose:
        logger.info(f"Found {len(matching_reactions)} matching reactions")
        for zaid, mt in matching_reactions[:5]:  # Show first 5
            z = zaid // 1000
            a = zaid % 1000
            nuclide = f"{ATOMIC_NUMBER_TO_SYMBOL.get(z, f'Z{z}')}-{a}"
            reaction = MT_TO_REACTION.get(mt, f"MT{mt}")
            logger.info(f"  {nuclide} {reaction}")
        if len(matching_reactions) > 5:
            logger.info(f"  ... and {len(matching_reactions) - 5} more")
    
    return matching_reactions


def _build_matrices(
    sdf_data: SDFData,
    cov_mat: CovMat,
    matching_reactions: List[Tuple[int, int]],
    energy_mapping: Dict[int, int],
    verbose: bool
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Tuple[int, int]]]:
    """Build sensitivity vector and covariance matrix in consistent order.
    
    Important: Both sensitivity coefficients and covariance matrices are in relative form.
    This means the sandwich formula σ²_R = S^T Σ S will give relative variance,
    and the uncertainty will be a relative uncertainty that can be directly
    compared to the relative uncertainties in the covariance matrix.
    """
    
    n_groups = len(energy_mapping)
    n_reactions = len(matching_reactions)
    total_size = n_groups * n_reactions
    
    if verbose:
        logger.info(f"Building matrices: {n_reactions} reactions × {n_groups} groups = {total_size} total elements")
    
    # Create reaction index mapping
    reaction_indices = {i: reaction for i, reaction in enumerate(matching_reactions)}
    
    # Build sensitivity vector using ONLY sensitivity coefficients (not errors)
    sensitivity_vector = np.zeros(total_size)
    
    # Create lookup for sensitivity data
    sens_lookup = {(r.zaid, r.mt): r for r in sdf_data.data}
    
    for i, (zaid, mt) in enumerate(matching_reactions):
        reaction_data = sens_lookup[(zaid, mt)]
        
        for sens_group, cov_group in energy_mapping.items():
            if sens_group < len(reaction_data.sensitivity):
                vector_idx = i * n_groups + sens_group
                # Use ONLY the sensitivity coefficient, NOT the error
                sensitivity_vector[vector_idx] = reaction_data.sensitivity[sens_group]
    
    # Build covariance matrix (in relative form)
    covariance_matrix = np.zeros((total_size, total_size))
    
    # Create lookup for covariance matrices
    for ir, rr, ic, rc, matrix in zip(cov_mat.isotope_rows, cov_mat.reaction_rows,
                                      cov_mat.isotope_cols, cov_mat.reaction_cols, 
                                      cov_mat.matrices):
        
        # Find reaction indices in our ordered list
        try:
            row_reaction_idx = matching_reactions.index((ir, rr))
            col_reaction_idx = matching_reactions.index((ic, rc))
        except ValueError:
            continue  # Skip if reaction not in our filtered list
        
        # Map the covariance matrix block
        for sens_i, cov_i in energy_mapping.items():
            for sens_j, cov_j in energy_mapping.items():
                if cov_i < matrix.shape[0] and cov_j < matrix.shape[1]:
                    matrix_row = row_reaction_idx * n_groups + sens_i
                    matrix_col = col_reaction_idx * n_groups + sens_j
                    value = matrix[cov_i, cov_j]
                    
                    # Set both (i,j) and (j,i) to ensure symmetry
                    covariance_matrix[matrix_row, matrix_col] = value
                    if matrix_row != matrix_col:  # Don't double-set diagonal elements
                        covariance_matrix[matrix_col, matrix_row] = value
    
    if verbose:
        # Check matrix properties
        nonzero_sens = np.count_nonzero(sensitivity_vector)
        nonzero_cov = np.count_nonzero(covariance_matrix)
        logger.info(f"Sensitivity vector: {nonzero_sens}/{total_size} non-zero elements")
        logger.info(f"Covariance matrix: {nonzero_cov}/{total_size**2} non-zero elements")
        
        # Report sensitivity coefficient ranges
        max_sens = np.max(np.abs(sensitivity_vector))
        logger.info(f"Max sensitivity coefficient: {max_sens:.6e}")
        
        # Report covariance matrix ranges
        max_cov = np.max(np.abs(covariance_matrix))
        typical_uncertainty = np.sqrt(np.mean(np.diag(covariance_matrix)[np.diag(covariance_matrix) > 0]))
        logger.info(f"Max covariance element: {max_cov:.6e}")
        logger.info(f"Typical relative uncertainty: {typical_uncertainty:.4f} ({typical_uncertainty*100:.2f}%)")
        
        # Check if covariance matrix is symmetric
        is_symmetric = np.allclose(covariance_matrix, covariance_matrix.T, rtol=1e-10)
        if not is_symmetric:
            logger.warning("Covariance matrix is not symmetric")
        
        # Important note about units
        logger.info("Using relative sensitivities with relative covariances")
        logger.info("Result will be in relative uncertainty units")
    
    return sensitivity_vector, covariance_matrix, reaction_indices


def _calculate_individual_contributions(
    sensitivity_vector: np.ndarray,
    covariance_matrix: np.ndarray,
    reaction_indices: Dict[int, Tuple[int, int]],
    n_groups: int,
    total_variance: float,
    verbose: bool
) -> List[UncertaintyContribution]:
    """Calculate individual reaction contributions to total uncertainty.
    
    For each reaction i, calculates its total contribution to variance including:
    1. Auto-correlation: S_i^T Σ_ii S_i (diagonal block)
    2. Cross-correlations: 2 * Σ_{j≠i} S_i^T Σ_ij S_j (off-diagonal blocks)
    
    The sum of all contributions equals the total variance.
    """
    
    contributions = []
    n_reactions = len(reaction_indices)
    
    # Calculate contribution matrix: C_ij = S_i^T Σ_ij S_j
    contribution_matrix = np.zeros((n_reactions, n_reactions))
    
    for i in range(n_reactions):
        start_i = i * n_groups
        end_i = (i + 1) * n_groups
        sens_i = sensitivity_vector[start_i:end_i]
        
        for j in range(n_reactions):
            start_j = j * n_groups
            end_j = (j + 1) * n_groups
            sens_j = sensitivity_vector[start_j:end_j]
            
            # Extract covariance block Σ_ij
            cov_ij = covariance_matrix[start_i:end_i, start_j:end_j]
            
            # Calculate contribution: S_i^T Σ_ij S_j
            contribution_matrix[i, j] = float(sens_i.T @ cov_ij @ sens_j)
    
    # For each reaction, calculate its total contribution
    for i, (zaid, mt) in reaction_indices.items():
        # Auto-correlation (diagonal)
        auto_contribution = contribution_matrix[i, i]
        
        # Cross-correlations (off-diagonal, but only count once per pair)
        cross_contribution = 0.0
        for j in range(n_reactions):
            if i != j:
                # Add half of the symmetric cross-correlation
                cross_contribution += 0.5 * contribution_matrix[i, j]
                cross_contribution += 0.5 * contribution_matrix[j, i]
        
        # Total contribution for this reaction
        total_contribution = auto_contribution + cross_contribution
        
        contribution = UncertaintyContribution(
            zaid=zaid,
            mt=mt,
            variance_contribution=total_contribution
        )
        
        contributions.append(contribution)
    
    # Calculate relative contributions as percentages
    # The sum of all contributions should equal total_variance
    total_contributions_sum = sum(c.variance_contribution for c in contributions)
    
    for contribution in contributions:
        if abs(total_contributions_sum) > 1e-15:
            contribution.relative_contribution = abs(contribution.variance_contribution) / abs(total_contributions_sum)
        else:
            contribution.relative_contribution = 0.0
    
    if verbose:
        # Show top contributors
        sorted_contribs = sorted(contributions, 
                               key=lambda x: abs(x.variance_contribution), 
                               reverse=True)
        logger.info("Top uncertainty contributors:")
        for contrib in sorted_contribs[:5]:
            pct = contrib.relative_contribution * 100
            logger.info(f"  {contrib.nuclide} {contrib.reaction_name}: {pct:.2f}%")
        
        # Verify sum
        if verbose and abs(total_contributions_sum - total_variance) > abs(total_variance) * 0.01:
            logger.warning(f"Contribution sum mismatch: {total_contributions_sum:.6e} vs {total_variance:.6e}")
    
    return contributions


def _calculate_correlation_effects(
    sensitivity_vector: np.ndarray,
    covariance_matrix: np.ndarray,
    reaction_indices: Dict[int, Tuple[int, int]],
    n_groups: int
) -> float:
    """Calculate the contribution from cross-correlations between reactions."""
    
    correlation_variance = 0.0
    
    for i in reaction_indices.keys():
        for j in reaction_indices.keys():
            if i != j:  # Only off-diagonal terms
                start_i = i * n_groups
                end_i = (i + 1) * n_groups
                start_j = j * n_groups  
                end_j = (j + 1) * n_groups
                
                sens_i = sensitivity_vector[start_i:end_i]
                sens_j = sensitivity_vector[start_j:end_j]
                cov_ij = covariance_matrix[start_i:end_i, start_j:end_j]
                
                correlation_variance += float(sens_i.T @ cov_ij @ sens_j)
    
    return correlation_variance


def filter_reactions_by_nuclide(zaid: int, mt_list: Optional[List[int]] = None) -> Dict[int, List[int]]:
    """
    Convenience function to create reaction filter for a single nuclide.
    
    Parameters
    ----------
    zaid : int
        ZAID of the nuclide
    mt_list : List[int], optional
        List of MT numbers to include. If None, includes all reactions for this nuclide.
        
    Returns
    -------
    Dict[int, List[int]]
        Reaction filter dictionary suitable for sandwich_uncertainty_propagation
        
    Examples
    --------
    >>> # Include all reactions for Fe-56
    >>> filter_dict = filter_reactions_by_nuclide(26056)
    >>> 
    >>> # Include only elastic and inelastic for Fe-56
    >>> filter_dict = filter_reactions_by_nuclide(26056, [2, 4])
    """
    return {zaid: mt_list} if mt_list else {zaid: []}


def filter_reactions_by_type(mt_numbers: List[int]) -> Dict[int, List[int]]:
    """
    Convenience function to create reaction filter by reaction type across all nuclides.
    
    Parameters
    ----------
    mt_numbers : List[int]
        List of MT numbers to include
        
    Returns
    -------
    Dict[int, List[int]]
        Reaction filter dictionary (note: this returns a special marker that the 
        main function should interpret as "these MTs for all nuclides")
        
    Examples
    --------
    >>> # Include only elastic scattering for all nuclides
    >>> filter_dict = filter_reactions_by_type([2])
    >>>
    >>> # Include elastic and (n,γ) for all nuclides  
    >>> filter_dict = filter_reactions_by_type([2, 102])
    """
    # Return special format - the main function handles this case
    return {"ALL_NUCLIDES": mt_numbers}

