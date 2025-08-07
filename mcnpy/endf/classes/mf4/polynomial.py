from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from .base import MF4MT

@dataclass
class MF4MTLegendre(MF4MT):
    """
    MT section in MF4 with Legendre expansion coefficients (LTT=1).
    
    Stores Legendre coefficients for each energy.
    """
    _ltt: int = 1
    _nr: int = None  # Number of different interpolation intervals
    _ne: int = None  # Number of energy points
    _interpolation: List[Tuple[int, int]] = field(default_factory=list)  # Interpolation scheme pairs (NBT, INT)
    
    # Store energy grid and coefficients separately
    _energies: List[float] = field(default_factory=list)  # Energy grid
    _legendre_coeffs: List[List[float]] = field(default_factory=list)  # Legendre coefficients for each energy
    
    @property
    def num_energy_points(self) -> int:
        """Number of energies in the grid"""
        return self._ne or len(self._energies)
    
    @property
    def num_interpolation_regions(self) -> int:
        """Number of different interpolation intervals"""
        return self._nr or 0
    
    @property
    def energy_interpolation(self) -> List[Tuple[int, int]]:
        """Interpolation scheme pairs (NBT, INT)"""
        return self._interpolation
    
    @property
    def legendre_energies(self) -> List[float]:
        """Energy grid for Legendre coefficients (alias for energies for compatibility)"""
        return self._energies
    
    @property
    def legendre_coefficients(self) -> List[List[float]]:
        """
        Legendre coefficients for each energy point.
        
        Returns a list of coefficient lists, aligned with the energy grid.
        Each inner list contains the coefficients for one energy point.
        """
        return self._legendre_coeffs
    
    def get_coefficients_at_energy(self, energy: float) -> List[float]:
        """
        Get Legendre coefficients at a specific energy.
        
        Args:
            energy: Energy point to retrieve coefficients for
            
        Returns:
            List of Legendre coefficients at that energy
        """
        # Find the index of this energy in the grid
        try:
            index = self._energies.index(energy)
            return self._legendre_coeffs[index]
        except (ValueError, IndexError):
            return []
    
    def get_coefficients_dict(self) -> Dict[float, List[float]]:
        """
        Get Legendre coefficients as a dictionary mapping energies to coefficients.
        
        Returns:
            Dictionary with energies as keys and coefficient lists as values
        """
        return {e: c for e, c in zip(self._energies, self._legendre_coeffs)}
    
    def __str__(self) -> str:
        """
        Convert the MF4MTLegendre object back to ENDF format string.
        
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
        
        line2 = format_endf_data_line(
            [0.0, self._awr, self._li, self._lct, 0, 0],
            mat, mf, mt, 2,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line2)
        line_num += 1
        
        # Format third line with interpolation info and energy count
        line3 = format_endf_data_line(
            [0.0, 0.0, 0, 0, self._nr or 0, self._ne or 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        lines.append(line3)
        line_num += 1
        
        # Format energy interpolation scheme pairs - all as integers
        if self._interpolation and self._nr and self._nr > 0:
            # Process interpolation pairs in groups of 3 (6 values per line)
            remaining_pairs = self._interpolation.copy()
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
        
        # End of section marker - all integers
        end_line = format_endf_data_line(
            [0, 0, 0, 0, 0, 0],
            mat, mf, 0, 99999,  # Note MT=0 for end of section
            formats=[ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT, ENDF_FORMAT_INT]
        )
        lines.append(end_line)
        
        return "\n".join(lines)
