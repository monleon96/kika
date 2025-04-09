# Law 61, 67: Angle-energy distributions

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from mcnpy.ace.classes.energy_distribution.base import EnergyDistribution
from mcnpy._utils import create_repr_section


@dataclass
class TabulatedAngleEnergyDistribution(EnergyDistribution):
    """
    Law 61: Tabulated Angle-Energy Distribution.
    
    Similar to LAW=44 but with tabular angular distributions instead of the Kalbach-Mann formalism.
    
    Data format (Table 45 and 46):
    - N_R: Number of interpolation regions
    - NBT, INT: Interpolation parameters
    - N_E: Number of incident energies
    - E(l): Incident energy grid
    - L(l): Location of distributions
    
    For each incident energy:
    - INTT': Combined interpolation parameter (10*N_D + INTT)
    - N_p: Number of points in distribution
    - E_out(l): Outgoing energy grid
    - PDF(l): Probability density function
    - CDF(l): Cumulative density function
    - LC(l): Location of angular distribution tables
    
    For each angular distribution (Table 47):
    - JJ: Interpolation flag
    - N_p: Number of points
    - CosOut(j): Cosine scattering angular grid
    - PDF(j): Probability density function
    - CDF(j): Cumulative density function
    """
    law: int = 61
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    distribution_locations: List[int] = field(default_factory=list)  # Locations of distributions
    
    # Store each energy-angle distribution
    # Each entry is a dictionary containing the distribution data
    distributions: List[Dict] = field(default_factory=list)
    
    # Store the angular distribution tables
    # Each table is a dictionary with 'jj', 'n_points', 'cosines', 'pdf', and 'cdf' keys
    angular_tables: List[Dict] = field(default_factory=list)
    
    def get_distribution(self, energy_idx: int) -> Dict:
        """
        Get the energy distribution for a specific incident energy index.
        
        Parameters
        ----------
        energy_idx : int
            Index of the incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the distribution data with keys:
            - 'intt': Interpolation type
            - 'n_discrete': Number of discrete lines
            - 'n_points': Number of points
            - 'e_out': Outgoing energy grid
            - 'pdf': Probability density function
            - 'cdf': Cumulative density function
            - 'lc': Location of angular distribution tables
            or None if index is invalid
        """
        if 0 <= energy_idx < len(self.distributions):
            return self.distributions[energy_idx]
        return None
    
    def get_angular_table(self, table_idx: int) -> Dict:
        """
        Get the angular distribution table with the specified index.
        
        Parameters
        ----------
        table_idx : int
            Index of the angular table
            
        Returns
        -------
        Dict
            Dictionary containing the angular distribution with keys:
            - 'jj': Interpolation flag
            - 'n_points': Number of points
            - 'cosines': Cosine scattering angular grid
            - 'pdf': Probability density function
            - 'cdf': Cumulative density function
            or None if index is invalid
        """
        if 0 <= table_idx < len(self.angular_tables):
            return self.angular_tables[table_idx]
        return None
    
    def get_interpolated_distribution(self, incident_energy: float) -> Dict:
        """
        Get an interpolated energy distribution for a specific incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the interpolated distribution data
        """
        # Find the bracketing incident energies
        if not self.incident_energies or incident_energy <= self.incident_energies[0]:
            # Below the minimum incident energy, return the first distribution
            return self.get_distribution(0) if self.distributions else None
        
        if incident_energy >= self.incident_energies[-1]:
            # Above the maximum incident energy, return the last distribution
            return self.get_distribution(len(self.incident_energies) - 1) if self.distributions else None
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(self.incident_energies, incident_energy, side='right') - 1
        
        # Get the distributions for the bracketing energies
        dist_low = self.get_distribution(idx)
        dist_high = self.get_distribution(idx + 1)
        
        if not dist_low or not dist_high:
            return dist_low if dist_low else dist_high
            
        # Interpolate the distributions based on the incident energy
        energy_low = self.incident_energies[idx]
        energy_high = self.incident_energies[idx + 1]
        
        # Determine the interpolation scheme from the regions
        # For simplicity, assume linear-linear interpolation
        # In a full implementation, we would determine this from the NBT and INT arrays
        
        # Interpolation fraction
        frac = (incident_energy - energy_low) / (energy_high - energy_low)
        
        # Create common grid for interpolation (use outgoing energy points from low distribution)
        e_out = dist_low['e_out']
        
        # Interpolate PDF values
        pdf_high_interp = np.interp(e_out, dist_high['e_out'], dist_high['pdf'])
        pdf_interp = (1.0 - frac) * np.array(dist_low['pdf']) + frac * pdf_high_interp
        
        # Interpolate CDF values
        cdf_high_interp = np.interp(e_out, dist_high['e_out'], dist_high['cdf'])
        cdf_interp = (1.0 - frac) * np.array(dist_low['cdf']) + frac * cdf_high_interp
        
        # LC values can't be interpolated since they're indices
        # We'll use the nearest LC value based on interpolation fraction
        lc_values = dist_low['lc'] if frac < 0.5 else dist_high['lc']
        
        # Create interpolated distribution
        interp_dist = {
            'intt': dist_low['intt'],  # Use the INTT from the lower energy
            'n_discrete': dist_low['n_discrete'],  # Use discrete count from lower energy
            'n_points': len(e_out),
            'e_out': e_out,
            'pdf': pdf_interp.tolist(),
            'cdf': cdf_interp.tolist(),
            'lc': lc_values
        }
        
        return interp_dist
    
    def sample_angular_distribution(self, table_idx: int, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample a cosine from a tabular angular distribution.
        
        Parameters
        ----------
        table_idx : int
            Index of the angular distribution table
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        float
            Sampled cosine of scattering angle
        """
        # Get the angular table
        table = self.get_angular_table(table_idx)
        if not table:
            # If table not found, return isotropic scattering
            if rng is None:
                rng = np.random.default_rng()
            return 2.0 * rng.random() - 1.0
        
        # Use numpy's random if none provided
        if rng is None:
            rng = np.random.default_rng()
        
        # Sample using CDF
        xi = rng.random()
        cosine = np.interp(xi, table['cdf'], table['cosines'])
        
        return cosine
    
    def sample_outgoing_energy_angle(self, incident_energy: float, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
        """
        Sample an outgoing energy and angle from the distribution for a given incident energy.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        Tuple[float, float]
            Tuple of (outgoing_energy, cosine)
        """
        # Use numpy's random if none provided
        if rng is None:
            rng = np.random.default_rng()
            
        # Get interpolated distribution
        dist = self.get_interpolated_distribution(incident_energy)
        if not dist:
            return 0.0, 0.0
            
        # 1. Sample outgoing energy using the CDF
        xi = rng.random()
        e_out = np.interp(xi, dist['cdf'], dist['e_out'])
        
        # 2. Find the index in the e_out array that corresponds to this energy
        idx = np.searchsorted(dist['e_out'], e_out, side='right') - 1
        idx = max(0, min(idx, len(dist['e_out']) - 1))
        
        # 3. Get the corresponding LC value (index to angular distribution table)
        lc_idx = abs(dist['lc'][idx]) - 1  # Convert to 0-indexed and handle negative values
        
        # 4. Sample cosine from the angular distribution
        cosine = self.sample_angular_distribution(lc_idx, rng)
        
        return e_out, cosine
    
    def sample_outgoing_energy(self, incident_energy: float, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample just an outgoing energy from the distribution for a given incident energy.
        
        This is a convenience method that only returns the energy part of the energy-angle pair.
        
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
        e_out, _ = self.sample_outgoing_energy_angle(incident_energy, rng)
        return e_out
    
    def __repr__(self) -> str:
        """Returns a formatted string representation of the TabulatedAngleEnergyDistribution object.
        
        Returns
        -------
        str
            Formatted string representation of the distribution
        """
        header_width = 80
        header = "=" * header_width + "\n"
        header += f"{'Tabulated Angle-Energy Distribution (Law 61)':^{header_width}}\n"
        header += "=" * header_width + "\n"
        
        # Description of the distribution
        description = (
            "This distribution provides correlated angle-energy distributions using tabular\n"
            "angular distributions. It corresponds to Law 61 in the ACE format.\n"
            "Similar to LAW=44 but uses tabular angular distributions instead of Kalbach-Mann formalism.\n\n"
        )
        
        # Basic distribution properties
        property_col_width = 40
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        properties = "Basic Properties:\n"
        properties += "-" * header_width + "\n"
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Distribution Law", f"{self.law} (Tabulated Angle-Energy)", 
            width1=property_col_width, width2=value_col_width)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Interpolation Regions", f"{self.n_interp_regions}",
            width1=property_col_width, width2=value_col_width)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Incident Energies", f"{self.n_energies}",
            width1=property_col_width, width2=value_col_width)
        
        # Show energy ranges if available
        if self.incident_energies and len(self.incident_energies) >= 2:
            properties += "{:<{width1}} {:<{width2}}\n".format(
                "Incident Energy Range", f"{self.incident_energies[0]:.4e} - {self.incident_energies[-1]:.4e} MeV",
                width1=property_col_width, width2=value_col_width)
        
        # Count distributions and angular tables
        energy_dist_count = len(self.distributions)
        angular_table_count = len(self.angular_tables)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Energy Distributions", f"{energy_dist_count}",
            width1=property_col_width, width2=value_col_width)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Angular Tables", f"{angular_table_count}",
            width1=property_col_width, width2=value_col_width)
        
        properties += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_distribution(energy_idx)": 
                "Get energy distribution for a specific incident energy index",
            ".get_angular_table(table_idx)": 
                "Get angular distribution table for a specific index",
            ".get_interpolated_distribution(incident_energy)": 
                "Get interpolated distribution for a specific incident energy",
            ".sample_angular_distribution(table_idx, rng=None)": 
                "Sample a cosine from a tabular angular distribution",
            ".sample_outgoing_energy_angle(incident_energy, rng=None)": 
                "Sample energy and angle for a given incident energy",
            ".sample_outgoing_energy(incident_energy, rng=None)": 
                "Sample just the outgoing energy for a given incident energy"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Data origin information
        data_origin = (
            "\nData Source:\n"
            "This data is parsed from the ACE-formatted nuclear data file and includes the incident\n"
            "energy grid, energy distributions, and angular distribution tables. Each energy point\n"
            "has a set of outgoing energies, and each outgoing energy has an associated angular\n"
            "distribution that describes the probability of scattering at different angles.\n"
        )
        
        return header + description + properties + methods_section + data_origin


@dataclass
class LaboratoryAngleEnergyDistribution(EnergyDistribution):
    """
    Law 67: Laboratory Angle-Energy Distribution.
    
    From ENDF-6 MF=6 LAW=7, this represents a distribution that directly 
    specifies the angular and energy distributions in the laboratory frame.
    
    Data format (Table 49, 50, 51):
    - N_R: Number of interpolation regions
    - NBT, INT: Interpolation parameters
    - N_E: Number of incident energies
    - E(l): Incident energy grid
    - L(l): Locations of distributions
    
    For each incident energy:
    - INTMU: Interpolation scheme for angles
    - NMU: Number of secondary cosines
    - XMU(l): Secondary cosines
    - LMU(l): Locations of secondary energy distributions
    
    For each cosine:
    - INTEP: Interpolation parameter for secondary energies
    - NPEP: Number of secondary energies
    - E_p(l): Secondary energy grid
    - PDF(l): Probability density function
    - CDF(l): Cumulative density function
    """
    law: int = 67
    n_interp_regions: int = 0  # Number of interpolation regions
    nbt: List[int] = field(default_factory=list)  # Interpolation region boundaries
    interp: List[int] = field(default_factory=list)  # Interpolation schemes
    n_energies: int = 0  # Number of incident energies
    incident_energies: List[float] = field(default_factory=list)  # Incident energy grid
    distribution_locations: List[int] = field(default_factory=list)  # Locations of distributions
    
    # Store each angle-energy distribution
    # Each entry corresponds to one incident energy
    # Format: List of dictionaries with keys:
    # - 'intmu': Interpolation scheme for angles
    # - 'n_cosines': Number of secondary cosines
    # - 'cosines': Array of secondary cosines
    # - 'energy_dist_locations': Locations of energy distributions for each cosine
    # - 'energy_distributions': List of energy distributions for each cosine
    angle_energy_distributions: List[Dict] = field(default_factory=list)
    
    def get_distribution(self, energy_idx: int) -> Dict:
        """
        Get the angle-energy distribution for a specific incident energy index.
        
        Parameters
        ----------
        energy_idx : int
            Index of the incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the distribution data or None if index is invalid
        """
        if 0 <= energy_idx < len(self.angle_energy_distributions):
            return self.angle_energy_distributions[energy_idx]
        return None
    
    def get_interpolated_distribution(self, incident_energy: float) -> Dict:
        """
        Get an interpolated distribution for a specific incident energy.
        
        For Law 67, we find the bracketing incident energies and use the
        distribution for the lower energy. In a full implementation,
        we would interpolate between the distributions.
        
        Parameters
        ----------
        incident_energy : float
            The incident energy
            
        Returns
        -------
        Dict
            Dictionary containing the distribution data or None if not available
        """
        # Find the bracketing incident energies
        if not self.incident_energies or incident_energy <= self.incident_energies[0]:
            # Below the minimum incident energy, return the first distribution
            return self.get_distribution(0)
        
        if incident_energy >= self.incident_energies[-1]:
            # Above the maximum incident energy, return the last distribution
            return self.get_distribution(len(self.incident_energies) - 1)
        
        # Find the energy interval containing the incident energy
        idx = np.searchsorted(self.incident_energies, incident_energy, side='right') - 1
        
        # Return the distribution for the lower incident energy
        # In a full implementation, we would interpolate between distributions
        return self.get_distribution(idx)
    
    def sample_outgoing_angle_energy(self, incident_energy: float, 
                                    rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
        """
        Sample an outgoing cosine and energy from the distribution.
        
        Parameters
        ----------
        incident_energy : float
            The incident neutron energy
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        Tuple[float, float]
            Tuple of (cosine, outgoing_energy)
        """
        # Use numpy's random if none provided
        if rng is None:
            rng = np.random.default_rng()
        
        # Get the distribution for this incident energy
        dist = self.get_interpolated_distribution(incident_energy)
        if not dist:
            return 0.0, 0.0
        
        # Sample cosine
        intmu = dist['intmu']
        cosines = dist['cosines']
        
        if not cosines or len(cosines) < 2:
            # If no valid cosines, return isotropic scattering
            return 2.0 * rng.random() - 1.0, 0.0
        
        # Choose a random cosine from the tabulated values
        # In a full implementation, we would use proper sampling according to intmu
        xi = rng.random()
        cosine_idx = int(xi * (len(cosines) - 1))
        cosine = cosines[cosine_idx]
        
        # Get the energy distribution for this cosine
        energy_dists = dist['energy_distributions']
        if not energy_dists or cosine_idx >= len(energy_dists) or not energy_dists[cosine_idx]:
            return cosine, 0.0
        
        energy_dist = energy_dists[cosine_idx]
        
        # Sample outgoing energy
        intep = energy_dist['intep']
        e_out = energy_dist['e_out']
        pdf = energy_dist['pdf']
        cdf = energy_dist['cdf']
        
        if not e_out or len(e_out) < 2:
            return cosine, 0.0
        
        # Sample from CDF
        xi = rng.random()
        outgoing_energy = np.interp(xi, cdf, e_out)
        
        return cosine, outgoing_energy
    
    def sample_outgoing_energy(self, incident_energy: float, 
                              rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample just an outgoing energy from the distribution.
        
        This is a convenience method that first samples an angle, then 
        samples an energy for that angle.
        
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
        _, energy = self.sample_outgoing_angle_energy(incident_energy, rng)
        return energy
    
    def __repr__(self) -> str:
        """Returns a formatted string representation of the LaboratoryAngleEnergyDistribution object.
        
        Returns
        -------
        str
            Formatted string representation of the distribution
        """
        header_width = 80
        header = "=" * header_width + "\n"
        header += f"{'Laboratory Angle-Energy Distribution (Law 67)':^{header_width}}\n"
        header += "=" * header_width + "\n"
        
        # Description of the distribution
        description = (
            "This distribution directly specifies the angular and energy distributions in the\n"
            "laboratory frame. It corresponds to Law 67 in the ACE format (ENDF-6 MF=6 LAW=7).\n"
            "For each incident energy, the distribution provides a set of emission angles and\n"
            "corresponding energy distributions for each angle.\n\n"
        )
        
        # Basic distribution properties
        property_col_width = 40
        value_col_width = header_width - property_col_width - 3  # -3 for spacing and formatting
        
        properties = "Basic Properties:\n"
        properties += "-" * header_width + "\n"
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Distribution Law", f"{self.law} (Laboratory Angle-Energy)", 
            width1=property_col_width, width2=value_col_width)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Interpolation Regions", f"{self.n_interp_regions}",
            width1=property_col_width, width2=value_col_width)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Incident Energies", f"{self.n_energies}",
            width1=property_col_width, width2=value_col_width)
        
        # Show energy ranges if available
        if self.incident_energies and len(self.incident_energies) >= 2:
            properties += "{:<{width1}} {:<{width2}}\n".format(
                "Incident Energy Range", f"{self.incident_energies[0]:.4e} - {self.incident_energies[-1]:.4e} MeV",
                width1=property_col_width, width2=value_col_width)
        
        # Count angle-energy distributions
        dist_count = len(self.angle_energy_distributions)
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Number of Angle-Energy Distributions", f"{dist_count}",
            width1=property_col_width, width2=value_col_width)
        
        # Count total number of cosines across all distributions
        total_cosines = 0
        for dist in self.angle_energy_distributions:
            if 'cosines' in dist:
                total_cosines += len(dist['cosines'])
        
        properties += "{:<{width1}} {:<{width2}}\n".format(
            "Total Number of Angular Points", f"{total_cosines}",
            width1=property_col_width, width2=value_col_width)
        
        properties += "-" * header_width + "\n\n"
        
        # Create a section for available methods
        methods = {
            ".get_distribution(energy_idx)": 
                "Get distribution for a specific incident energy index",
            ".get_interpolated_distribution(incident_energy)": 
                "Get interpolated distribution for a specific incident energy",
            ".sample_outgoing_angle_energy(incident_energy, rng=None)": 
                "Sample angle and energy for a given incident energy",
            ".sample_outgoing_energy(incident_energy, rng=None)": 
                "Sample just the outgoing energy for a given incident energy"
        }
        
        methods_section = create_repr_section(
            "Available Methods:", 
            methods, 
            total_width=header_width, 
            method_col_width=property_col_width
        )
        
        # Data origin information
        data_origin = (
            "\nData Source:\n"
            "This data is parsed from the ACE-formatted nuclear data file and includes the incident\n"
            "energy grid and corresponding angle-energy distributions. For each incident energy,\n"
            "there is a set of cosines, and for each cosine, there is an energy distribution that\n"
            "specifies the probability of different outgoing energies at that angle.\n"
        )
        
        return header + description + properties + methods_section + data_origin
