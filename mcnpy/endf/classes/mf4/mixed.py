from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from .base import MF4MT
from ....endf.utils import get_interpolation_scheme_name, describe_interpolation_region

@dataclass
class MF4MTMixed(MF4MT):
    """
    MT section in MF4 with mixed representation (LTT=3).
    
    Uses Legendre expansions for low energies and tabulated data for high energies.
    """
    _ltt: int = 3
    _ne: int = None  # Number of energies
    _nm: int = None  # Maximum order of Legendre polynomials
    _nr: int = None  # Number of different interpolation intervals for Legendre part
    _ne1: int = None  # Number of energy points for Legendre coefficients
    _ne2: int = None  # Number of energy points for tabulated distributions
    _nr_tab: int = None  # Number of interpolation intervals for tabulated data energies
    _interpolation: List[Tuple[int, int]] = field(default_factory=list)  # Interpolation scheme pairs for Legendre (NBT, INT)
    _tab_interpolation: List[Tuple[int, int]] = field(default_factory=list)  # Interpolation scheme pairs for tabulated energy grid
    
    # Store energy grid and coefficients separately
    _energies: List[float] = field(default_factory=list)  # Energy grid
    _legendre_coeffs: List[List[float]] = field(default_factory=list)  # Legendre coefficients for each energy
    
    # Store tabulated data
    _tabulated_energies: List[float] = field(default_factory=list)  # Energies for tabulated data
    _tabulated_cosines: List[List[float]] = field(default_factory=list)  # Cosines for each energy
    _tabulated_probabilities: List[List[float]] = field(default_factory=list)  # Probabilities for each energy
    _angular_interpolation: List[List[Tuple[int, int]]] = field(default_factory=list)  # Interpolation schemes for angular data

    # ESSENTIAL PROPERTIES - Energy Grids
    @property
    def legendre_energies(self) -> List[float]:
        """Energy grid for Legendre coefficients"""
        return self._energies
    
    @property
    def tabulated_energies(self) -> List[float]:
        """Energy grid for tabulated angular distributions"""
        return self._tabulated_energies

    # ESSENTIAL PROPERTIES - Data Access
    @property
    def legendre_coefficients(self) -> List[List[float]]:
        """
        Legendre coefficients for each energy point.
        
        Returns a list of coefficient lists, aligned with the legendre_energies.
        Each inner list contains the coefficients for one energy point.
        """
        return self._legendre_coeffs
    
    @property
    def tabulated_cosines(self) -> List[List[float]]:
        """
        Cosine values (μ) for each tabulated energy point.
        
        Returns a list of cosine lists, aligned with tabulated_energies.
        """
        return self._tabulated_cosines
    
    @property
    def tabulated_probabilities(self) -> List[List[float]]:
        """
        Probability values f(μ,E) for each tabulated energy point.
        
        Returns a list of probability lists, aligned with tabulated_energies and tabulated_cosines.
        """
        return self._tabulated_probabilities

    # ESSENTIAL PROPERTIES - Interpolation Schemes
    @property
    def legendre_interpolation(self) -> List[Tuple[int, int]]:
        """Interpolation scheme pairs for Legendre energy grid (NBT, INT)"""
        return self._interpolation
    
    @property
    def tabulated_interpolation(self) -> List[Tuple[int, int]]:
        """Interpolation scheme pairs for tabulated energy grid (NBT, INT)"""
        return self._tab_interpolation
    
    @property
    def angular_interpolation(self) -> List[List[Tuple[int, int]]]:
        """
        Interpolation scheme pairs for angular data at each tabulated energy.
        
        Returns a list aligned with tabulated_energies, where each element
        contains the (NBT, INT) pairs for that energy's angular distribution.
        """
        return self._angular_interpolation

    # CONVENIENCE METHODS - Data Access
    def get_legendre_coefficients(self, energy: float) -> List[float]:
        """Get Legendre coefficients at a specific energy."""
        try:
            index = self._energies.index(energy)
            return self._legendre_coeffs[index]
        except (ValueError, IndexError):
            return []
    
    def get_tabulated_distribution(self, energy: float) -> Tuple[List[float], List[float]]:
        """Get tabulated angular distribution at a specific energy."""
        try:
            index = self._tabulated_energies.index(energy)
            return (self._tabulated_cosines[index], self._tabulated_probabilities[index])
        except (ValueError, IndexError):
            return ([], [])

    def get_interpolation_summary(self) -> str:
        """
        Get a comprehensive summary of all interpolation schemes used in this mixed representation.
        
        Returns:
            str: A formatted summary showing energy ranges and their interpolation schemes
        """
        summary_lines = []
        summary_lines.append("Mixed Representation (LTT=3) Interpolation Summary:")
        summary_lines.append("=" * 60)
        
        # Legendre coefficients section
        if self._energies and self._interpolation:
            summary_lines.append("\nLegendre Coefficients Section:")
            summary_lines.append("-" * 35)
            
            energy_start = 0
            for i, (nbt, int_code) in enumerate(self._interpolation):
                energy_end_idx = min(nbt - 1, len(self._energies) - 1)
                if energy_end_idx >= 0:
                    start_energy = self._energies[energy_start] if energy_start < len(self._energies) else 0.0
                    end_energy = self._energies[energy_end_idx]
                    interp_name = get_interpolation_scheme_name(int_code)
                    
                    summary_lines.append(
                        f"  Energy range: {start_energy:.3e} to {end_energy:.3e} eV"
                    )
                    summary_lines.append(
                        f"  Interpolation: {interp_name}"
                    )
                    summary_lines.append("")
                    
                    energy_start = nbt
        
        # Tabulated distributions section
        if self._tabulated_energies and self._tab_interpolation:
            summary_lines.append("Tabulated Distributions Section:")
            summary_lines.append("-" * 35)
            
            energy_start = 0
            for i, (nbt, int_code) in enumerate(self._tab_interpolation):
                energy_end_idx = min(nbt - 1, len(self._tabulated_energies) - 1)
                if energy_end_idx >= 0:
                    start_energy = self._tabulated_energies[energy_start] if energy_start < len(self._tabulated_energies) else 0.0
                    end_energy = self._tabulated_energies[energy_end_idx]
                    interp_name = get_interpolation_scheme_name(int_code)
                    
                    summary_lines.append(
                        f"  Energy range: {start_energy:.3e} to {end_energy:.3e} eV"
                    )
                    summary_lines.append(
                        f"  Interpolation: {interp_name}"
                    )
                    summary_lines.append("")
                    
                    energy_start = nbt
        
        # Angular interpolation schemes (brief overview)
        if self._angular_interpolation:
            summary_lines.append("Angular Interpolation Schemes:")
            summary_lines.append("-" * 30)
            
            unique_schemes = set()
            for ang_interp in self._angular_interpolation:
                for nbt, int_code in ang_interp:
                    unique_schemes.add(int_code)
            
            if unique_schemes:
                summary_lines.append("  Angular distributions use:")
                for scheme_code in sorted(unique_schemes):
                    scheme_name = get_interpolation_scheme_name(scheme_code)
                    summary_lines.append(f"    - {scheme_name}")
            else:
                summary_lines.append("  No angular interpolation schemes defined")
        
        return "\n".join(summary_lines)

    def __str__(self) -> str:
        """
        Convert the MF4MTMixed object back to ENDF format string.
        
        Returns:
            Multi-line string in ENDF format
        """
        # Import inside the method to avoid circular imports
        from ...utils import format_endf_data_line, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_BLANK
        
        mat = self._mat if self._mat is not None else 0
        mf = 4
        mt = self.number
        lines = []
        line_num = 1
        
        # Format first line - header - ZA, AWR as float, rest as integers with zeros printed
        line1 = format_endf_data_line(
            [self._za, self._awr, 0, self._ltt, 0, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line1)
        line_num += 1
        
        # Format second line - First value as float, rest as integers
        line2 = format_endf_data_line(
            [0.0, self._awr, self._li, self._lct, 0, self._nm],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT]
        )
        lines.append(line2)
        line_num += 1
        
        # Format third line with energy boundary and Legendre energy count
        # Use actual length instead of stored values for safety
        ne1_actual = len(self._energies)
        line3 = format_endf_data_line(
            [0.0, 0.0, 0, 0, self._nr or 0, ne1_actual],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        lines.append(line3)
        line_num += 1
        
        # Ensure interpolation scheme is set if empty
        interpolation_pairs = self._interpolation
        if not interpolation_pairs and ne1_actual > 0:
            interpolation_pairs = [(ne1_actual, 2)]  # Default to linear-linear
        
        # Format energy interpolation scheme pairs - all as integers
        if interpolation_pairs and len(interpolation_pairs) > 0:
            # Process interpolation pairs in groups of 3 (6 values per line)
            remaining_pairs = interpolation_pairs.copy()
            while remaining_pairs:
                # Take up to 3 pairs for this line
                line_pairs = remaining_pairs[:3]
                remaining_pairs = remaining_pairs[3:]
                
                # Flatten pairs into a list of values
                values = []
                for nbt, interp in line_pairs:
                    values.append(nbt)
                    values.append(interp)
                
                # All interpolation values are integers
                # Format line - pad with blanks instead of zeros
                formats = [ENDF_FORMAT_INT] * len(values)
                if len(values) < 6:
                    values.extend([None] * (6 - len(values)))
                    formats.extend([ENDF_FORMAT_BLANK] * (6 - len(formats)))
                    
                interp_line = format_endf_data_line(
                    values, mat, mf, mt, line_num, formats=formats
                )
                lines.append(interp_line)
                line_num += 1
        
        # Format Legendre coefficients section
        for i, (energy, coeffs) in enumerate(zip(self._energies, self._legendre_coeffs)):
            # For each energy point, format header and coefficients
            nl = len(coeffs)  # Number of coefficients
            
            # Format header for this energy
            energy_header = format_endf_data_line(
                [0.0, energy, 0, 0, nl, 0],
                mat, mf, mt, line_num,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO]
            )
            lines.append(energy_header)
            line_num += 1
            
            # Format coefficients (6 per line)
            coef_idx = 0
            while coef_idx < nl:
                # Get up to 6 coefficients for this line
                line_coeffs = coeffs[coef_idx:coef_idx + 6]
                coef_idx += 6
                
                # Pad with None (blank) instead of zeros if needed
                if len(line_coeffs) < 6:
                    line_coeffs.extend([None] * (6 - len(line_coeffs)))
                
                # Format line
                coef_line = format_endf_data_line(
                    line_coeffs, mat, mf, mt, line_num
                )
                lines.append(coef_line)
                line_num += 1
        
        # Format tabulated distributions section header
        if self._ne2:
            tab_header = format_endf_data_line(
                [0.0, 0.0, 0, 0, self._nr_tab or 0, self._ne2],
                mat, mf, mt, line_num,
                formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
            )
            lines.append(tab_header)
            line_num += 1
            
            # Format energy interpolation scheme pairs for tabulated data
            if self._tab_interpolation and self._nr_tab and self._nr_tab > 0:
                # Process interpolation pairs in groups of 3 (6 values per line)
                remaining_pairs = self._tab_interpolation.copy()
                while remaining_pairs:
                    # Take up to 3 pairs for this line
                    line_pairs = remaining_pairs[:3]
                    remaining_pairs = remaining_pairs[3:]
                    
                    # Flatten pairs into a list of values
                    values = []
                    for nbt, interp in line_pairs:
                        values.append(nbt)
                        values.append(interp)
                    
                    # All interpolation values are integers
                    # Format line - pad with blanks instead of zeros
                    formats = [ENDF_FORMAT_INT] * len(values)
                    if len(values) < 6:
                        values.extend([None] * (6 - len(values)))
                        formats.extend([ENDF_FORMAT_BLANK] * (6 - len(formats)))
                        
                    interp_line = format_endf_data_line(
                        values, mat, mf, mt, line_num, formats=formats
                    )
                    lines.append(interp_line)
                    line_num += 1
            
            # Format each tabulated distribution
            for i, energy in enumerate(self._tabulated_energies):
                cosines = self._tabulated_cosines[i]
                probabilities = self._tabulated_probabilities[i]
                ang_interp = self._angular_interpolation[i] if i < len(self._angular_interpolation) else []
                
                np_val = len(cosines)  # Number of angular points
                nr_ang = len(ang_interp)  # Number of angular interpolation regions
                
                # Format header for this energy
                energy_header = format_endf_data_line(
                    [0.0, energy, 0, 0, nr_ang, np_val],
                    mat, mf, mt, line_num,
                    formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
                )
                lines.append(energy_header)
                line_num += 1
                
                # Format angular interpolation scheme pairs (if any)
                if ang_interp and nr_ang > 0:
                    # Process interpolation pairs in groups of 3 (6 values per line)
                    remaining_pairs = ang_interp.copy()
                    while remaining_pairs:
                        # Take up to 3 pairs for this line
                        line_pairs = remaining_pairs[:3]
                        remaining_pairs = remaining_pairs[3:]
                        
                        # Flatten pairs into a list of values
                        values = []
                        for nbt, interp in line_pairs:
                            values.append(nbt)
                            values.append(interp)
                        
                        formats = [ENDF_FORMAT_INT] * len(values)
                        # Use None for blank fields instead of zeros
                        if len(values) < 6:
                            values.extend([None] * (6 - len(values)))
                            formats.extend([ENDF_FORMAT_BLANK] * (6 - len(formats)))
                        
                        # Format line
                        ang_interp_line = format_endf_data_line(
                            values, mat, mf, mt, line_num,
                            formats=formats
                        )
                        lines.append(ang_interp_line)
                        line_num += 1
                
                # Format cosine-probability pairs (3 pairs per line)
                pair_idx = 0
                while pair_idx < np_val:
                    # Get up to 3 pairs for this line
                    line_values = []
                    for j in range(3):
                        if pair_idx + j < np_val:
                            line_values.append(cosines[pair_idx + j])
                            line_values.append(probabilities[pair_idx + j])
                    pair_idx += 3
                    
                    # Use None for blank fields instead of zeros
                    if len(line_values) < 6:
                        line_values.extend([None] * (6 - len(line_values)))
                    
                    # Format line
                    pair_line = format_endf_data_line(
                        line_values, mat, mf, mt, line_num
                    )
                    lines.append(pair_line)
                    line_num += 1
        
        # End of section marker - all integers
        end_line = format_endf_data_line(
            [0, 0, 0, 0, 0, 0],
            mat, mf, 0, 99999,  # Note MT=0 for end of section
            formats=[ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        lines.append(end_line)
        
        return "\n".join(lines)

