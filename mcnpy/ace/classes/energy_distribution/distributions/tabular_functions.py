# Law 22, 24: Tabular functions

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution


@dataclass
class TabularLinearFunctions(EnergyDistribution):
    """
    Law 22: Tabular Linear Functions of Incident Energy Out.
    
    From UK Law 2, this represents a tabular function form where the outgoing 
    energy is a linear function of the incident energy: E_out = C_ik * (E − T_ik).
    
    Data format (Table 41):
    - N_R: Number of interpolation regions
    - NBT, INT: Interpolation parameters
    - N_E: Number of incident energies
    - E_in(l): Tabulated incident energies
    - LOCE(l): Locators of E_out tables
    
    For each incident energy E_in(i):
    - NF_i: Number of functions for this energy
    - P_ik: Probability for each function
    - T_ik: Origin parameter for each function
    - C_ik: Slope parameter for each function
    """
    law: int = 22
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    table_locators: List[int] = field(default_factory=list)  # Locators of E_out tables
    
    # Store the function data for each incident energy
    # Each entry is a dictionary with 'nf', 'p', 't', 'c' keys
    function_data: List[Dict] = field(default_factory=list)
    
    def get_function_data(self, energy_idx: int) -> Dict:
        """
        Get the function data for a specific incident energy index.
        
        Parameters
        ----------
        energy_idx : int
            Index of the incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the function data with keys:
            - 'nf': Number of functions
            - 'p': Probability for each function
            - 't': Origin parameter for each function
            - 'c': Slope parameter for each function
            or None if index is invalid
        """
        if 0 <= energy_idx < len(self.function_data):
            return self.function_data[energy_idx]
        return None
    
    def get_interpolated_function_data(self, incident_energy: float) -> Dict:
        """
        Get interpolated function data for a specific incident energy.
        
        For Law 22, we don't interpolate between function data sets.
        Instead, we find the bracket incident energies and use the lower one's function data.
        
        Parameters
        ----------
        incident_energy : float
            The incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the function data or None if not available
        """
        # Find the bracketing incident energies
        if not self.incident_energies or incident_energy <= self.incident_energies[0]:
            # Below the minimum incident energy, return the first function data
            return self.get_function_data(0) if self.function_data else None
        
        if incident_energy >= self.incident_energies[-1]:
            # Above the maximum incident energy, return the last function data
            return self.get_function_data(len(self.incident_energies) - 1) if self.function_data else None
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(self.incident_energies, incident_energy, side='right') - 1
        
        # Return the function data for the lower incident energy
        return self.get_function_data(idx)
    
    def sample_outgoing_energy(self, incident_energy: float, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample an outgoing energy from the distribution for a given incident energy.
        
        For Law 22, we use equations:
        1. Find function index k such that: ∑(P_ij, j=1...k-1) < ξ ≤ ∑(P_ij, j=1...k)
        2. Calculate outgoing energy: E_out = C_ik * (E − T_ik)
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        float
            Sampled outgoing energy
        """
        # Get function data for this incident energy
        function_data = self.get_interpolated_function_data(incident_energy)
        if not function_data:
            return 0.0
            
        # Use numpy's random if none provided
        if rng is None:
            rng = np.random.default_rng()
            
        # Generate random number
        xi = rng.random()
        
        # Get the probability, origin, and slope arrays
        nf = function_data['nf']
        p_values = function_data['p']
        t_values = function_data['t']
        c_values = function_data['c']
        
        # Find the function index k using cumulative probability
        cum_prob = 0.0
        k = 0
        for i in range(nf):
            cum_prob += p_values[i]
            if xi <= cum_prob:
                k = i
                break
        
        # Calculate outgoing energy using selected function
        # E_out = C_ik * (E − T_ik)
        e_out = c_values[k] * (incident_energy - t_values[k])
        
        # Ensure non-negative energy
        return max(0.0, e_out)
    


@dataclass
class TabularEnergyMultipliers(EnergyDistribution):
    """
    Law 24: Tabular Energy Multipliers.
    
    From UK Law 6, this represents a tabular function where the outgoing 
    energy is a multiplier of the incident energy: E_out = T_k(l) * E.
    
    Data format (Table 42):
    - N_R: Number of interpolation regions
    - NBT, INT: Interpolation parameters
    - N_E: Number of incident energies
    - E_in(l): Tabulated incident energies
    - NET: Number of outgoing values in each table
    - T_i(l): Tables of energy multipliers for each incident energy
    """
    law: int = 24
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    n_mult_values: int = 0  # Number of multiplier values in each table (NET)
    
    # Store the multiplier tables for each incident energy
    # Each row corresponds to one incident energy
    multiplier_tables: List[List[float]] = field(default_factory=list)
    
    def get_multiplier_table(self, energy_idx: int) -> List[float]:
        """
        Get the multiplier table for a specific incident energy index.
        
        Parameters
        ----------
        energy_idx : int
            Index of the incident energy
            
        Returns
        -------
        List[float]
            List of multiplier values or None if index is invalid
        """
        if 0 <= energy_idx < len(self.multiplier_tables):
            return self.multiplier_tables[energy_idx]
        return None
    
    def get_interpolated_multiplier_table(self, incident_energy: float) -> List[float]:
        """
        Get interpolated multiplier table for a specific incident energy.
        
        For tabular energy multipliers, we interpolate between the
        multiplier tables of adjacent incident energies.
        
        Parameters
        ----------
        incident_energy : float
            The incident energy
            
        Returns
        -------
        List[float]
            Interpolated multiplier table or None if not available
        """
        # Find the bracketing incident energies
        if not self.incident_energies or incident_energy <= self.incident_energies[0]:
            # Below the minimum incident energy, return the first table
            return self.get_multiplier_table(0)
        
        if incident_energy >= self.incident_energies[-1]:
            # Above the maximum incident energy, return the last table
            return self.get_multiplier_table(len(self.incident_energies) - 1)
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(self.incident_energies, incident_energy, side='right') - 1
        
        # Get the multiplier tables for the bracketing energies
        table_low = self.get_multiplier_table(idx)
        table_high = self.get_multiplier_table(idx + 1)
        
        if not table_low or not table_high or len(table_low) != len(table_high):
            return table_low if table_low else table_high
            
        # Calculate interpolation factor
        energy_low = self.incident_energies[idx]
        energy_high = self.incident_energies[idx + 1]
        factor = (incident_energy - energy_low) / (energy_high - energy_low)
        
        # Linearly interpolate between tables
        interp_table = [table_low[i] + factor * (table_high[i] - table_low[i]) for i in range(len(table_low))]
        
        return interp_table
    
    def sample_outgoing_energy(self, incident_energy: float, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample an outgoing energy from the distribution for a given incident energy.
        
        For Law 24, the outgoing energy is: E_out = T * E
        where T is a multiplier sampled from the tables.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        float
            Sampled outgoing energy
        """
        # Get interpolated multiplier table for this incident energy
        multiplier_table = self.get_interpolated_multiplier_table(incident_energy)
        if not multiplier_table or len(multiplier_table) <= 1:
            return 0.0
            
        # Use numpy's random if none provided
        if rng is None:
            rng = np.random.default_rng()
            
        # Generate random number for equiprobable bins
        bin_idx = rng.integers(0, len(multiplier_table) - 1)
        
        # Sample multiplier value from the bin
        multiplier = multiplier_table[bin_idx]
        
        # Calculate outgoing energy: E_out = T * E
        e_out = multiplier * incident_energy
        
        # Ensure non-negative energy
        return max(0.0, e_out)
