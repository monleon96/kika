from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from mcnpy.ace.xss import XssEntry

@dataclass
class ParticleRelease:
    """Container for particle release data from TYR and TYRH blocks."""
    incident_neutron: List[XssEntry] = field(default_factory=list)  # TYR Block - neutron reaction particle release
    particle_production: List[List[XssEntry]] = field(default_factory=list)  # TYRH Block - particle production particle release
    
    @property
    def has_neutron_data(self) -> bool:
        """Check if neutron reaction particle release data is available."""
        return len(self.incident_neutron) > 0
    
    @property
    def has_particle_production_data(self) -> bool:
        """Check if particle production particle release data is available."""
        return len(self.particle_production) > 0
    
    def get_reaction_frame(self, ty_entry: XssEntry) -> str:
        """
        Get the reference frame for a given TY value.
        
        Parameters
        ----------
        ty_entry : XssEntry
            TY entry from TYR or TYRH block
            
        Returns
        -------
        str
            'center-of-mass' or 'laboratory' depending on the sign
        """
        ty_value = int(ty_entry.value)
        if ty_value < 0:
            return 'center-of-mass'
        elif ty_value > 0:
            return 'laboratory'
        else:  # ty_value == 0
            return 'none'  # Absorption, no particles released
    
    def get_num_particles(self, ty_entry: Optional[XssEntry] = None) -> str:
        """
        Get the number of particles released for a given TY entry.
        If ty_entry is not provided, returns a summary of all entries.
        
        Parameters
        ----------
        ty_entry : XssEntry, optional
            TY entry from TYR or TYRH block
            
        Returns
        -------
        str
            Description of the number of particles released or a summary
        """
        if ty_entry is None:
            # Return a summary if no specific TY entry provided
            if self.has_neutron_data:
                ty_types = {}
                for ty in self.incident_neutron:
                    ty_str = self._get_ty_description(ty)
                    ty_types[ty_str] = ty_types.get(ty_str, 0) + 1
                
                summary = "Neutron reaction particle release types:\n"
                for ty_str, count in ty_types.items():
                    summary += f"  {ty_str}: {count} reactions\n"
                return summary
            return "No particle release data available"
        
        return self._get_ty_description(ty_entry)
    
    def _get_ty_description(self, ty_entry: XssEntry) -> str:
        """
        Get a description for a given TY entry.
        
        Parameters
        ----------
        ty_entry : XssEntry
            TY entry from TYR or TYRH block
            
        Returns
        -------
        str
            Description of what the TY value means
        """
        ty_value = int(ty_entry.value)
        return self._get_ty_description_value(ty_value)

    def _get_ty_description_value(self, ty_value: int) -> str:
        """
        Get a description for a given TY value.
        
        Parameters
        ----------
        ty_value : int
            TY value from TYR or TYRH block
            
        Returns
        -------
        str
            Description of what the TY value means
        """
        if ty_value == 0:
            return "0 (absorption)"
        elif abs(ty_value) == 19:
            return "fission (see NU block)"
        elif abs(ty_value) == 5:
            return f"5 particles ({self.get_reaction_frame(type('TempEntry', (), {'value': ty_value}))} frame)"
        elif abs(ty_value) > 100:
            return f"{abs(ty_value)} (energy-dependent, see DLW/DLWH block)"
        else:
            frame = "COM" if ty_value < 0 else "LAB"
            return f"{abs(ty_value)} particles ({frame} frame)"
    
    def get_summary(self) -> str:
        """
        Get a summary of the particle release data.
        
        Returns
        -------
        str
            Summary of the particle release data
        """
        summary = []
        
        if self.has_neutron_data:
            summary.append(f"Neutron reaction particle release: {len(self.incident_neutron)} reactions")
            
            # Group and count by TY value
            ty_counts = {}
            for ty in self.incident_neutron:
                ty_value = int(ty.value)
                ty_counts[ty_value] = ty_counts.get(ty_value, 0) + 1
            
            # Show the breakdown
            summary.append("TY values distribution:")
            for ty_value, count in sorted(ty_counts.items()):
                # Use the description function directly with the ty_value instead of creating a new XssEntry
                description = self._get_ty_description_value(ty_value)
                summary.append(f"  TY={ty_value} ({description}): {count} reactions")
        
        if self.has_particle_production_data:
            summary.append(f"Particle production data: {len(self.particle_production)} particle types")
            
            for i, ty_values in enumerate(self.particle_production):
                if ty_values:
                    summary.append(f"  Particle type {i+1}: {len(ty_values)} reactions")
        
        return "\n".join(summary) if summary else "No particle release data available"
