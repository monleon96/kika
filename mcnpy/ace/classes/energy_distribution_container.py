from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
from mcnpy.ace.classes.energy_distribution import EnergyDistribution
from mcnpy.ace.parsers.xss import XssEntry

@dataclass
class EnergyDistributionContainer:
    """Container for energy distributions from DLW, DLWP, DLWH, and DNED blocks."""
    # Neutron reaction energy distributions (MT → list of distributions)
    incident_neutron: Dict[int, List[EnergyDistribution]] = field(default_factory=dict)
    
    # Photon production energy distributions (MT → list of distributions)
    photon_production: Dict[int, List[EnergyDistribution]] = field(default_factory=dict)
    
    # Particle production energy distributions (particle index → MT → list of distributions)
    particle_production: List[Dict[int, List[EnergyDistribution]]] = field(default_factory=list)
    
    # Delayed neutron energy distributions (precursor group → distribution)
    delayed_neutron: List[EnergyDistribution] = field(default_factory=list)
    
    # Energy-dependent neutron yields (MT → yield)
    neutron_yields: Dict[int, EnergyDistribution] = field(default_factory=dict)
    
    # Energy-dependent photon yields (MT → yield)
    photon_yields: Dict[int, EnergyDistribution] = field(default_factory=dict)
    
    # Energy-dependent particle yields (particle index → MT → yield)
    particle_yields: List[Dict[int, EnergyDistribution]] = field(default_factory=list)
    
    @property
    def has_neutron_data(self) -> bool:
        """Check if neutron reaction energy distributions are available."""
        return len(self.incident_neutron) > 0
    
    @property
    def has_photon_production_data(self) -> bool:
        """Check if photon production energy distributions are available."""
        return len(self.photon_production) > 0
    
    @property
    def has_particle_production_data(self) -> bool:
        """Check if particle production energy distributions are available."""
        return len(self.particle_production) > 0
    
    @property
    def has_delayed_neutron_data(self) -> bool:
        """Check if delayed neutron energy distributions are available."""
        return len(self.delayed_neutron) > 0
    
    @property
    def has_neutron_yields(self) -> bool:
        """Check if energy-dependent neutron yields are available."""
        return len(self.neutron_yields) > 0
    
    @property
    def has_photon_yields(self) -> bool:
        """Check if energy-dependent photon yields are available."""
        return len(self.photon_yields) > 0
    
    @property
    def has_particle_yields(self) -> bool:
        """Check if energy-dependent particle yields are available."""
        return len(self.particle_yields) > 0
    
    def get_neutron_reaction_mt_numbers(self) -> List[int]:
        """Get a list of all MT numbers with neutron reaction energy distributions."""
        # Extract values from XssEntry objects if needed
        mt_numbers = []
        for mt in self.incident_neutron.keys():
            if hasattr(mt, 'value'):
                mt_numbers.append(int(mt.value))
            else:
                mt_numbers.append(int(mt))
        return sorted(mt_numbers)
    
    def get_photon_production_mt_numbers(self) -> List[int]:
        """Get a list of all MT numbers with photon production energy distributions."""
        # Extract values from XssEntry objects if needed
        mt_numbers = []
        for mt in self.photon_production.keys():
            if hasattr(mt, 'value'):
                mt_numbers.append(int(mt.value))
            else:
                mt_numbers.append(int(mt))
        return sorted(mt_numbers)
    
    def get_particle_production_mt_numbers(self, particle_idx: int = 0) -> List[int]:
        """
        Get a list of all MT numbers with particle production energy distributions for a specific particle.
        
        Parameters
        ----------
        particle_idx : int
            Index of the particle type (0-based)
            
        Returns
        -------
        List[int]
            List of MT numbers sorted in ascending order, or empty list if particle_idx is invalid
        """
        if particle_idx < 0 or particle_idx >= len(self.particle_production):
            return []
        return sorted(list(self.particle_production[particle_idx].keys()))
    
    def get_neutron_distribution(self, mt: int) -> Optional[List[EnergyDistribution]]:
        """
        Get the energy distribution for a neutron reaction with the specified MT number.
        
        Parameters
        ----------
        mt : int
            MT number of the reaction
            
        Returns
        -------
        Optional[List[EnergyDistribution]]
            List of energy distributions for the reaction, or None if not found
        """
        # First try direct lookup
        if mt in self.incident_neutron:
            return self.incident_neutron[mt]
        
        # Try looking up with XssEntry objects
        for key in self.incident_neutron:
            if hasattr(key, 'value') and int(key.value) == mt:
                return self.incident_neutron[key]
        
        # Not found
        return None
    
    def get_photon_distribution(self, mt: int) -> Optional[List[EnergyDistribution]]:
        """
        Get the energy distribution for a photon production reaction with the specified MT number.
        
        Parameters
        ----------
        mt : int
            MT number of the reaction
            
        Returns
        -------
        Optional[List[EnergyDistribution]]
            List of energy distributions for the reaction, or None if not found
        """
        # First try direct lookup
        if mt in self.photon_production:
            return self.photon_production[mt]
        
        # Try looking up with XssEntry objects
        for key in self.photon_production:
            if hasattr(key, 'value') and int(key.value) == mt:
                return self.photon_production[key]
        
        # Not found
        return None
    
    def get_particle_distribution(self, particle_idx: int, mt: int) -> Optional[List[EnergyDistribution]]:
        """
        Get the energy distribution for a particle production reaction.
        
        Parameters
        ----------
        particle_idx : int
            Index of the particle type (0-based)
        mt : int
            MT number of the reaction
            
        Returns
        -------
        Optional[List[EnergyDistribution]]
            List of energy distributions for the reaction, or None if not found
        """
        if particle_idx < 0 or particle_idx >= len(self.particle_production):
            return None
            
        # Get the particle's dictionary
        particle_dict = self.particle_production[particle_idx]
        
        # First try direct lookup
        if mt in particle_dict:
            return particle_dict[mt]
        
        # Try looking up with XssEntry objects
        for key in particle_dict:
            if hasattr(key, 'value') and int(key.value) == mt:
                return particle_dict[key]
        
        # Not found
        return None
    
    def get_delayed_neutron_distribution(self, group: int) -> Optional[EnergyDistribution]:
        """
        Get the energy distribution for a delayed neutron precursor group.
        
        Parameters
        ----------
        group : int
            Precursor group index (0-based)
            
        Returns
        -------
        Optional[EnergyDistribution]
            Energy distribution for the precursor group, or None if not found
        """
        if group < 0 or group >= len(self.delayed_neutron):
            return None
        return self.delayed_neutron[group]
    
    def __repr__(self) -> str:
        lines = ["Energy Distribution Data"]
        
        # Incident neutron distributions
        if self.incident_neutron:
            lines.append(f"  Incident Neutron Reactions: {len(self.incident_neutron)} MT numbers")
            # Sort by converting keys to integers if they're XssEntry objects
            sorted_items = sorted(self.incident_neutron.items(), 
                                 key=lambda item: int(item[0].value) if hasattr(item[0], 'value') else int(item[0]))
            for mt, dist_list in sorted_items:
                mt_value = int(mt.value) if hasattr(mt, 'value') else int(mt)
                law_names = [f"Law {d.law}" for d in dist_list]
                lines.append(f"    MT={mt_value}: {len(dist_list)} distributions ({', '.join(law_names)})")
        
        # Photon production distributions
        if self.photon_production:
            lines.append(f"  Photon Production Reactions: {len(self.photon_production)} MT numbers")
            # Sort by converting keys to integers if they're XssEntry objects
            sorted_items = sorted(self.photon_production.items(), 
                                 key=lambda item: int(item[0].value) if hasattr(item[0], 'value') else int(item[0]))
            for mt, dist_list in sorted_items:
                mt_value = int(mt.value) if hasattr(mt, 'value') else int(mt)
                law_names = [f"Law {d.law}" for d in dist_list]
                lines.append(f"    MT={mt_value}: {len(dist_list)} distributions ({', '.join(law_names)})")
        
        # Particle production distributions
        if any(self.particle_production):
            lines.append(f"  Particle Production: {len(self.particle_production)} particle types")
            for i, mt_dict in enumerate(self.particle_production):
                if mt_dict:
                    lines.append(f"    Particle Type {i+1}: {len(mt_dict)} MT numbers")
                    # Sort by converting keys to integers if they're XssEntry objects
                    sorted_items = sorted(mt_dict.items(), 
                                         key=lambda item: int(item[0].value) if hasattr(item[0], 'value') else int(item[0]))
                    for mt, dist_list in sorted_items:
                        mt_value = int(mt.value) if hasattr(mt, 'value') else int(mt)
                        law_names = [f"Law {d.law}" for d in dist_list]
                        lines.append(f"      MT={mt_value}: {len(dist_list)} distributions ({', '.join(law_names)})")
        
        # Delayed neutron distributions
        if self.delayed_neutron:
            lines.append(f"  Delayed Neutron Groups: {len(self.delayed_neutron)} groups")
            for i, dist in enumerate(self.delayed_neutron):
                lines.append(f"    Group {i+1}: Law {dist.law}")
        
        # Energy-dependent yields
        if self.neutron_yields:
            lines.append(f"  Energy-Dependent Neutron Yields: {len(self.neutron_yields)} reactions")
            # Sort by converting keys to integers if they're XssEntry objects
            sorted_keys = sorted(self.neutron_yields.keys(), 
                                key=lambda k: int(k.value) if hasattr(k, 'value') else int(k))
            for mt in sorted_keys:
                mt_value = int(mt.value) if hasattr(mt, 'value') else int(mt)
                lines.append(f"    MT={mt_value}")
                
        if self.photon_yields:
            lines.append(f"  Energy-Dependent Photon Yields: {len(self.photon_yields)} reactions")
            # Sort by converting keys to integers if they're XssEntry objects
            sorted_keys = sorted(self.photon_yields.keys(), 
                                key=lambda k: int(k.value) if hasattr(k, 'value') else int(k))
            for mt in sorted_keys:
                mt_value = int(mt.value) if hasattr(mt, 'value') else int(mt)
                lines.append(f"    MT={mt_value}")
                
        if any(self.particle_yields):
            lines.append(f"  Energy-Dependent Particle Yields: {len(self.particle_yields)} particle types")
            for i, mt_dict in enumerate(self.particle_yields):
                if mt_dict:
                    lines.append(f"    Particle Type {i+1}: {len(mt_dict)} reactions")
                    # Sort by converting keys to integers if they're XssEntry objects
                    sorted_keys = sorted(mt_dict.keys(), 
                                        key=lambda k: int(k.value) if hasattr(k, 'value') else int(k))
                    for mt in sorted_keys:
                        mt_value = int(mt.value) if hasattr(mt, 'value') else int(mt)
                        lines.append(f"      MT={mt_value}")
        
        return "\n".join(lines)
