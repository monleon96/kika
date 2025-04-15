from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from .base import MF4MT


@dataclass
class MF4MTTabulated(MF4MT):
    """
    MT section in MF4 with tabulated probability distributions (LTT=2).
    
    Stores tabulated angular distributions for each energy.
    """
    _ltt: int = 2
    _ne: int = None  # Number of energy points
    _nr: int = None  # Number of interpolation regions for energy grid
    _interpolation: List[Tuple[int, int]] = field(default_factory=list)  # Energy interpolation scheme pairs
    
    # Tabulated data storage
    _energies: List[float] = field(default_factory=list)  # Energy grid
    _cosines: List[List[float]] = field(default_factory=list)  # Cosine values for each energy
    _probabilities: List[List[float]] = field(default_factory=list)  # Probability values for each energy
    _angular_interpolation: List[List[Tuple[int, int]]] = field(default_factory=list)  # Angular interpolation schemes
    
    @property
    def num_energy_points(self) -> int:
        """Number of energy points in the grid"""
        return self._ne or len(self._energies)
    
    @property
    def num_interpolation_regions(self) -> int:
        """Number of interpolation regions for the energy grid"""
        return self._nr or 0
    
    @property
    def energy_interpolation(self) -> List[Tuple[int, int]]:
        """Interpolation scheme pairs for energy grid (NBT, INT)"""
        return self._interpolation
    
    @property
    def energies(self) -> List[float]:
        """Energy grid for angular distribution data"""
        return self._energies
    
    @property
    def cosines(self) -> List[List[float]]:
        """
        Cosine values (μ) for each energy point.
        
        Returns a list of cosine lists, aligned with the energy grid.
        Each inner list contains the cosine values for one energy point.
        """
        return self._cosines
    
    @property
    def probabilities(self) -> List[List[float]]:
        """
        Probability values f(μ,E) for each energy point and cosine.
        
        Returns a list of probability lists, aligned with the energy grid and cosines.
        Each inner list contains the probability values for one energy point.
        """
        return self._probabilities
    
    @property
    def cosine_interpolation(self) -> List[List[Tuple[int, int]]]:
        """
        Interpolation scheme pairs for angular data at each energy.
        
        Returns a list of interpolation scheme lists, aligned with the energy grid.
        Each inner list contains (NBT, INT) pairs for one energy point.
        """
        return self._angular_interpolation
    
    def get_distribution_at_energy(self, energy: float) -> Tuple[List[float], List[float]]:
        """
        Get tabulated angular distribution at a specific energy.
        
        Args:
            energy: Energy point to retrieve distribution for
            
        Returns:
            Tuple of (cosines, probabilities) at that energy
        """
        try:
            index = self._energies.index(energy)
            return (self._cosines[index], self._probabilities[index])
        except (ValueError, IndexError):
            return ([], [])
    
    def get_distribution_dict(self) -> Dict[float, Tuple[List[float], List[float]]]:
        """
        Get tabulated data as a dictionary mapping energies to (cosines, probabilities).
        
        Returns:
            Dictionary with energies as keys and (cosines, probabilities) tuples as values
        """
        return {e: (c, p) for e, c, p in zip(
            self._energies, 
            self._cosines, 
            self._probabilities
        )}
    
    def __str__(self) -> str:
        """
        Convert the MF4MTTabulated object back to ENDF format string.
        
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
            [0.0, self._awr, self._li, self._lct, 0, 0],
            mat, mf, mt, line_num,
            formats=[ENDF_FORMAT_FLOAT, ENDF_FORMAT_FLOAT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT, ENDF_FORMAT_INT_ZERO, ENDF_FORMAT_INT_ZERO]
        )
        lines.append(line2)
        line_num += 1
        
        # Format third line with number of interpolation regions and energy points
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
        
        # Format each tabulated distribution
        for i, energy in enumerate(self._energies):
            cosines = self._cosines[i]
            probabilities = self._probabilities[i]
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