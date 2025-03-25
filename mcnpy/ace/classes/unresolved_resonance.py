from mcnpy.ace.xss import XssEntry
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

@dataclass
class ProbabilityTable:
    """
    Class representing a single probability table at a given energy.
    """
    energy: float = 0.0
    cumulative_probabilities: List[XssEntry] = field(default_factory=list)
    total_xs: List[XssEntry] = field(default_factory=list)
    elastic_xs: List[XssEntry] = field(default_factory=list)
    fission_xs: List[XssEntry] = field(default_factory=list)
    capture_xs: List[XssEntry] = field(default_factory=list)
    heating_numbers: List[XssEntry] = field(default_factory=list)
    
    @property
    def num_entries(self) -> int:
        """Return the number of entries in the probability table."""
        return len(self.cumulative_probabilities)

@dataclass
class UnresolvedResonanceTables:
    """
    Container for unresolved resonance probability tables (UNR block).
    
    These tables are used to sample cross sections in the unresolved resonance range.
    """
    has_data: bool = False
    num_energies: int = 0          # N - Number of incident energies where probability tables exist
    table_length: int = 0          # M - Length of each probability table
    interpolation: int = 0         # INT - Interpolation method between tables
    inelastic_flag: int = 0        # ILF - Inelastic competition flag
    other_absorption_flag: int = 0 # IOA - Other absorption flag
    factors_flag: int = 0          # IFF - Factors flag (0=cross sections, 1=factors)
    energies: List[XssEntry] = field(default_factory=list)  # Incident energies
    tables: List[ProbabilityTable] = field(default_factory=list)  # Probability tables for each energy
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No unresolved resonance probability tables available"
        
        output = f"Unresolved Resonance Probability Tables\n"
        output += "=" * 50 + "\n"
        output += f"Number of energy points: {self.num_energies}\n"
        output += f"Table length: {self.table_length}\n"
        
        # Interpolation method
        interp_methods = {2: "linear-linear", 5: "log-log"}
        interp_method = interp_methods.get(self.interpolation, f"unknown ({self.interpolation})")
        output += f"Interpolation: {interp_method}\n"
        
        # Inelastic competition flag
        if self.inelastic_flag < 0:
            output += "Inelastic cross section: zero in the unresolved range\n"
        elif self.inelastic_flag > 0:
            output += f"Inelastic cross section: special MT={self.inelastic_flag}\n"
        else:
            output += "Inelastic cross section: calculated from balance relationship\n"
        
        # Other absorption flag
        if self.other_absorption_flag < 0:
            output += "Other absorption cross section: zero in the unresolved range\n"
        elif self.other_absorption_flag > 0:
            output += f"Other absorption cross section: special MT={self.other_absorption_flag}\n"
        else:
            output += "Other absorption cross section: calculated from balance relationship\n"
        
        # Factors flag
        if self.factors_flag == 0:
            output += "Table values represent cross sections\n"
        else:
            output += "Table values represent factors to multiply smooth cross sections\n"
        
        # Energy range
        if self.energies:
            energy_values = [e.value for e in self.energies]
            output += f"Energy range: {min(energy_values):.6e} to {max(energy_values):.6e} MeV\n"
        
        # Table summary
        if self.tables:
            total_min, total_max = float('inf'), float('-inf')
            for table in self.tables:
                if table.total_xs:
                    table_values = [xs.value for xs in table.total_xs]
                    table_min = min(table_values)
                    table_max = max(table_values)
                    total_min = min(total_min, table_min)
                    total_max = max(total_max, table_max)
            
            if total_min != float('inf') and total_max != float('-inf'):
                output += f"Total cross section range: {total_min:.6e} to {total_max:.6e}"
                if self.factors_flag == 1:
                    output += " (factors)"
                output += "\n"
        
        return output
    
    def get_probability_table(self, energy: float) -> Optional[ProbabilityTable]:
        """
        Get the probability table for a specific energy, with interpolation if needed.
        
        Parameters
        ----------
        energy : float
            The incident energy
            
        Returns
        -------
        Optional[ProbabilityTable]
            The probability table for the specified energy, or None if out of range
        """
        if not self.has_data or not self.energies or not self.tables:
            return None
        
        # Extract energy values for comparison
        energy_values = [e.value for e in self.energies]
        
        # Check if energy is outside the range
        if energy < energy_values[0] or energy > energy_values[-1]:
            return None
        
        # Find the nearest energy points for interpolation
        idx = np.searchsorted(energy_values, energy)
        if idx == 0:
            return self.tables[0]
        elif idx == len(energy_values):
            return self.tables[-1]
        
        # If the energy matches exactly, return that table
        if energy == energy_values[idx]:
            return self.tables[idx]
        
        # Perform interpolation between tables
        lower_idx = idx - 1
        upper_idx = idx
        
        lower_table = self.tables[lower_idx]
        upper_table = self.tables[upper_idx]
        
        lower_energy = energy_values[lower_idx]
        upper_energy = energy_values[upper_idx]
        
        # Create a new table for the interpolated result
        interp_table = ProbabilityTable(energy=energy)
        
        if self.interpolation == 2:  # Linear-linear
            factor = (energy - lower_energy) / (upper_energy - lower_energy)
            
            # Interpolate each array
            for attr in ['cumulative_probabilities', 'total_xs', 'elastic_xs', 
                         'fission_xs', 'capture_xs', 'heating_numbers']:
                lower_vals = getattr(lower_table, attr)
                upper_vals = getattr(upper_table, attr)
                
                if lower_vals and upper_vals:
                    # Extract values for interpolation
                    lower_values = [entry.value for entry in lower_vals]
                    upper_values = [entry.value for entry in upper_vals]
                    
                    # Create new XssEntry objects for interpolated values
                    # Note: this is creating new objects since we're interpolating
                    interp_vals = [XssEntry(-1, lower_values[i] + factor * (upper_values[i] - lower_values[i])) 
                                  for i in range(len(lower_vals))]
                    setattr(interp_table, attr, interp_vals)
                
        elif self.interpolation == 5:  # Log-log
            log_factor = np.log(energy / lower_energy) / np.log(upper_energy / lower_energy)
            
            # Interpolate each array
            for attr in ['cumulative_probabilities', 'total_xs', 'elastic_xs', 
                         'fission_xs', 'capture_xs', 'heating_numbers']:
                lower_vals = getattr(lower_table, attr)
                upper_vals = getattr(upper_table, attr)
                
                if lower_vals and upper_vals:
                    # Extract values for interpolation
                    lower_values = [entry.value for entry in lower_vals]
                    upper_values = [entry.value for entry in upper_vals]
                    
                    # Handle zero values in log-log interpolation
                    interp_vals = []
                    for i in range(len(lower_vals)):
                        if lower_values[i] <= 0 or upper_values[i] <= 0:
                            # Fall back to linear interpolation for zero/negative values
                            val = lower_values[i] + (energy - lower_energy) * (upper_values[i] - lower_values[i]) / (upper_energy - lower_energy)
                        else:
                            val = lower_values[i] * (upper_values[i] / lower_values[i]) ** log_factor
                        # Create new XssEntry with interpolated value
                        interp_vals.append(XssEntry(-1, val))
                    
                    setattr(interp_table, attr, interp_vals)
        else:
            # Unknown interpolation method, default to nearest neighbor
            return self.tables[lower_idx] if (energy - lower_energy) < (upper_energy - energy) else self.tables[upper_idx]
        
        return interp_table
    
    def sample_cross_sections(self, energy: float, rng: Optional[np.random.RandomState] = None) -> Dict[str, float]:
        """
        Sample cross sections from the probability tables.
        
        Parameters
        ----------
        energy : float
            The incident energy
        rng : np.random.RandomState, optional
            Random number generator to use, default is numpy's global RNG
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing sampled cross sections for total, elastic, fission, and capture
        """
        if rng is None:
            rng = np.random
        
        # Get the probability table for this energy
        table = self.get_probability_table(energy)
        if not table or not table.cumulative_probabilities:
            return {}
        
        # Sample a random number
        r = rng.random()
        
        # Extract probability values for searchsorted
        prob_values = [p.value for p in table.cumulative_probabilities]
        
        # Find the index in the cumulative probability distribution
        idx = np.searchsorted(prob_values, r)
        if idx == len(prob_values):
            idx = len(prob_values) - 1
        
        # Extract the cross sections
        result = {
            'total': table.total_xs[idx].value if table.total_xs else 0.0,
            'elastic': table.elastic_xs[idx].value if table.elastic_xs else 0.0,
            'fission': table.fission_xs[idx].value if table.fission_xs else 0.0,
            'capture': table.capture_xs[idx].value if table.capture_xs else 0.0,
            'heating': table.heating_numbers[idx].value if table.heating_numbers else 0.0,
        }
        
        return result
