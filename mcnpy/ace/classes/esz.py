from dataclasses import dataclass, field
from typing import List, Optional
from mcnpy.ace.xss import XssEntry

@dataclass
class EszBlock:
    """Container for ESZ block data (energy grid and cross sections)."""
    has_data: bool = False  # True if ESZ block is present
    energies: List[XssEntry] = field(default_factory=list)  # Energy grid
    total_xs: List[XssEntry] = field(default_factory=list)  # Total cross section
    absorption_xs: List[XssEntry] = field(default_factory=list)  # Absorption cross section
    elastic_xs: List[XssEntry] = field(default_factory=list)  # Elastic cross section
    heating_numbers: List[XssEntry] = field(default_factory=list)  # Average heating numbers
    
    @property
    def num_energies(self) -> int:
        """Get the number of energy points in the grid."""
        return len(self.energies)
    
    def get_energy_grid(self) -> List[float]:
        """Get the energy grid as a list of float values."""
        return [entry.value for entry in self.energies]
    
    def get_total_xs(self) -> List[float]:
        """Get the total cross section as a list of float values."""
        return [entry.value for entry in self.total_xs]
    
    def get_absorption_xs(self) -> List[float]:
        """Get the absorption cross section as a list of float values."""
        return [entry.value for entry in self.absorption_xs]
    
    def get_elastic_xs(self) -> List[float]:
        """Get the elastic cross section as a list of float values."""
        return [entry.value for entry in self.elastic_xs]
    
    def get_heating_numbers(self) -> List[float]:
        """Get the heating numbers as a list of float values."""
        return [entry.value for entry in self.heating_numbers]
    
    def print_indices(self):
        """Print the original XSS indices for debugging purposes."""
        print("ESZ Block Indices:")
        if self.energies:
            print(f"  Energies: {self.energies[0].index} to {self.energies[-1].index}")
        if self.total_xs:
            print(f"  Total XS: {self.total_xs[0].index} to {self.total_xs[-1].index}")
        if self.absorption_xs:
            print(f"  Absorption XS: {self.absorption_xs[0].index} to {self.absorption_xs[-1].index}")
        if self.elastic_xs:
            print(f"  Elastic XS: {self.elastic_xs[0].index} to {self.elastic_xs[-1].index}")
        if self.heating_numbers:
            print(f"  Heating Numbers: {self.heating_numbers[0].index} to {self.heating_numbers[-1].index}")
