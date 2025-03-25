from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

@dataclass
class ParticleProductionXSData:
    """
    Container for a single particle's production cross section and heating data (HPD block).
    
    This block contains the total production cross section for a given secondary particle
    and the associated heating numbers.
    """
    energy_grid_index: int = 0       # IE - Energy grid index
    num_energies: int = 0            # NE - Number of consecutive energies
    xs_values: List = field(default_factory=list)  # XssEntry objects for total particle production cross section
    heating_numbers: List = field(default_factory=list)  # XssEntry objects for average heating numbers
    
    def get_energies(self, full_energy_grid: List[float]) -> List[float]:
        """
        Get the energy points for this cross section data.
        
        Parameters
        ----------
        full_energy_grid : List[float]
            The full energy grid from the ACE data
            
        Returns
        -------
        List[float]
            Energy points for this cross section data
        """
        if not full_energy_grid or self.energy_grid_index <= 0 or self.energy_grid_index > len(full_energy_grid):
            return []
        
        # Adjust to 0-based indexing and get the subset of energies
        idx = self.energy_grid_index - 1
        return full_energy_grid[idx:idx + self.num_energies]
    
    def __repr__(self) -> str:
        output = f"Particle Production Cross Section Data:\n"
        output += "=" * 60 + "\n"
        output += f"Energy grid index: {self.energy_grid_index}\n"
        output += f"Number of energy points: {self.num_energies}\n"
        
        if self.xs_values:
            # Extract values from XssEntry objects for min/max calculation
            xs_float_values = [entry.value for entry in self.xs_values]
            min_xs = min(xs_float_values)
            max_xs = max(xs_float_values)
            output += f"Cross section range: {min_xs:.6e} to {max_xs:.6e} barns\n"
        
        if self.heating_numbers:
            # Extract values from XssEntry objects for min/max calculation
            heating_float_values = [entry.value for entry in self.heating_numbers]
            min_heat = min(heating_float_values)
            max_heat = max(heating_float_values)
            output += f"Heating numbers range: {min_heat:.6e} to {max_heat:.6e}\n"
        
        return output

@dataclass
class ParticleProductionXSContainer:
    """
    Container for all particles' production cross section and heating data.
    
    This container holds HPD blocks for each secondary particle type.
    """
    has_data: bool = False
    particle_data: Dict[int, ParticleProductionXSData] = field(default_factory=dict)  # Mapping of particle index to data
    
    def get_particle_data(self, particle_index: int) -> Optional[ParticleProductionXSData]:
        """
        Get the production cross section data for a specific particle type.
        
        Parameters
        ----------
        particle_index : int
            1-based index of the particle type (as defined in PTYPE block)
            
        Returns
        -------
        Optional[ParticleProductionXSData]
            The production cross section data for the specified particle, or None if not available
        """
        return self.particle_data.get(particle_index)
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No particle production cross section data available"
        
        output = f"Particle Production Cross Section Data for {len(self.particle_data)} Particle Types:\n"
        output += "=" * 70 + "\n"
        
        for particle_idx, data in sorted(self.particle_data.items()):
            output += f"Particle Type {particle_idx}:\n"
            output += f"  Energy grid index: {data.energy_grid_index}\n"
            output += f"  Number of energy points: {data.num_energies}\n"
            if data.xs_values:
                # Extract values from XssEntry objects for min/max calculation
                xs_float_values = [entry.value for entry in data.xs_values]
                min_xs = min(xs_float_values)
                max_xs = max(xs_float_values)
                output += f"  Cross section range: {min_xs:.6e} to {max_xs:.6e} barns\n"
            output += "\n"
        
        return output
