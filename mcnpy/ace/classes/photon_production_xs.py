from mcnpy.ace.parsers.xss import XssEntry
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

@dataclass
class ProductionCrossSection:
    """Base class for production cross section data (SIGP/SIGH blocks)."""
    mt: int = 0
    mftype: int = 0
    
@dataclass
class YieldBasedCrossSection(ProductionCrossSection):
    """
    Cross section data for MF=12 or MF=16 (yield-based) format.
    
    For this format, the production cross section is calculated as:
    σ_prod(E) = Y(E) * σ_MTMULT(E)
    """
    mtmult: int = 0  # MT number of the cross section to multiply by yield
    num_regions: int = 0  # Number of interpolation regions
    interpolation_bounds: List[XssEntry] = field(default_factory=list)  # NBT array as XssEntry objects
    interpolation_schemes: List[XssEntry] = field(default_factory=list)  # INT array as XssEntry objects
    num_energies: int = 0  # Number of energy points
    energies: List[XssEntry] = field(default_factory=list)  # Energy grid as XssEntry objects
    yields: List[XssEntry] = field(default_factory=list)  # Yield values as XssEntry objects
    
@dataclass
class DirectCrossSection(ProductionCrossSection):
    """
    Cross section data for MF=13 format (direct cross section).
    
    This format is only valid for photon production.
    """
    energy_grid_index: int = 0  # IE - Starting index in the energy grid
    num_entries: int = 0        # NE - Number of consecutive entries
    cross_sections: List[XssEntry] = field(default_factory=list)  # Cross section values as XssEntry objects
    
@dataclass
class ProductionCrossSectionContainer:
    """
    Container for production cross section data.
    """
    has_data: bool = False
    cross_sections: Dict[int, Union[YieldBasedCrossSection, DirectCrossSection]] = field(default_factory=dict)
    
    def get_reaction_xs(self, mt: int) -> Optional[Union[YieldBasedCrossSection, DirectCrossSection]]:
        """Get cross section data for a specific MT number."""
        return self.cross_sections.get(mt)
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No production cross section data available"
        
        output = f"Production Cross Section Data ({len(self.cross_sections)} reactions)\n"
        output += "=" * 50 + "\n"
        
        for mt, xs in sorted(self.cross_sections.items()):
            if isinstance(xs, YieldBasedCrossSection):
                mf_type = "Yield-based (MF=12)" if xs.mftype == 12 else "Yield-based (MF=16)"
                output += f"MT={mt}: {mf_type}, MTMULT={xs.mtmult}, {xs.num_energies} energy points\n"
            else:  # DirectCrossSection
                output += f"MT={mt}: Direct cross section (MF=13), {xs.num_entries} values\n"
        
        return output

@dataclass
class PhotonProductionCrossSections(ProductionCrossSectionContainer):
    """Container for photon production cross section data (SIGP block)."""
    pass

@dataclass
class ParticleProductionCrossSections(ProductionCrossSectionContainer):
    """Container for particle production cross section data (SIGH block)."""
    particle_types: Dict[int, List[int]] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        if not self.has_data:
            return "No particle production cross section data available"
        
        output = f"Particle Production Cross Section Data\n"
        output += "=" * 50 + "\n"
        
        for particle_type, mts in self.particle_types.items():
            output += f"Particle Type {particle_type}: {len(mts)} reactions (MT={', '.join(map(str, mts))})\n"
        
        return output
