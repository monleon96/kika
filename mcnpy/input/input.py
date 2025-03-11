from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from .._constants import ATOMIC_NUMBER_TO_SYMBOL

@dataclass
class Input:
    """Main class for storing MCNP input data.

    :ivar pert: Container for all perturbation cards in the input
    :type pert: Pert
    :ivar materials: Container for all material cards in the input
    :type materials: Materials
    """
    pert: 'Pert' = None
    materials: 'Materials' = None

@dataclass
class Pert:
    """Container class for MCNP perturbation cards.

    :ivar perturbation: Dictionary mapping perturbation IDs to Perturbation objects
    :type perturbation: Dict[int, Perturbation]
    """
    perturbation: Dict[int, 'Perturbation'] = field(default_factory=dict)
    
    @property
    def reactions(self) -> List[Optional[int]]:
        """Get unique reaction numbers from all perturbations.

        :returns: Sorted list of unique reaction numbers across all perturbations
        :rtype: List[Optional[int]]
        """
        return sorted(list({pert.reaction for pert in self.perturbation.values()}))
    
    @property
    def pert_energies(self) -> List[float]:
        """Get unique energy values from all perturbation energy ranges.

        :returns: Sorted list of unique energy values from all perturbation ranges
        :rtype: List[float]
        """
        energy_values = set()
        for pert in self.perturbation.values():
            if pert.energy:
                energy_values.add(pert.energy[0])
                energy_values.add(pert.energy[1])
        return sorted(list(energy_values))
    
    def group_perts_by_reaction(self, method: int) -> Dict[Optional[int], List[int]]:
        """Groups perturbation IDs by their reaction numbers for a given method.

        :param method: The perturbation method to filter by
        :type method: int
        :returns: Dictionary mapping reaction numbers to lists of perturbation IDs
        :rtype: Dict[Optional[int], List[int]]
        :raises ValueError: If no perturbations are defined
        """
        if not self.perturbation:
            raise ValueError("No perturbations defined")
            
        # Filter perturbations by method
        filtered = {id: pert for id, pert in self.perturbation.items() if pert.method == method}
        if not filtered:
            return {}
            
        groups = {}
        for id, pert in filtered.items():
            reaction = pert.reaction
            if reaction not in groups:
                groups[reaction] = []
            groups[reaction].append(id)
        return groups

@dataclass
class Perturbation:
    """Represents a single MCNP perturbation card.

    :ivar id: Perturbation identifier number
    :type id: int
    :ivar particle: Particle type (e.g., 'n' for neutron)
    :type particle: str
    :ivar cell: List of cell numbers affected by the perturbation
    :type cell: Optional[List[int]]
    :ivar material: Material number for the perturbation
    :type material: Optional[int]
    :ivar rho: Density value for the perturbation
    :type rho: Optional[float]
    :ivar method: Method number for the perturbation calculation
    :type method: Optional[int]
    :ivar reaction: Reaction number for the perturbation
    :type reaction: Optional[int]
    :ivar energy: Energy range (min, max) for the perturbation
    :type energy: Optional[Tuple[float, float]]
    """
    id: int
    particle: str
    cell: Optional[List[int]] = None
    material: Optional[int] = None
    rho: Optional[float] = None
    method: Optional[int] = None
    reaction: Optional[int] = None
    energy: Optional[Tuple[float, float]] = None

@dataclass
class Materials:
    """Container class for MCNP material cards.

    :ivar mat: Dictionary mapping material IDs to Mat objects
    :type mat: Dict[int, Mat]
    """
    mat: Dict[int, 'Mat'] = field(default_factory=dict)

@dataclass
class Component:
    """Represents a single nuclide component within a material.
    
    :ivar zaid: Nuclide ZAID number
    :type zaid: int
    :ivar fraction: Atomic fraction of the nuclide in the material
    :type fraction: float
    :ivar library: Specific cross-section library for this nuclide
    :type library: Optional[str]
    """
    zaid: int
    fraction: float
    library: Optional[str] = None
    
    @property
    def nuclide(self) -> Optional[str]:
        """Get nuclide symbol based on atomic number.
        
        :returns: Chemical symbol of element or None if not found
        :rtype: Optional[str]
        """
        # Extract atomic number from ZAID (first 2-3 digits)
        atomic_number = self.zaid // 1000
        return ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number)

@dataclass
class Mat:
    """Represents a single MCNP material card.

    :ivar id: Material identifier number
    :type id: int
    :ivar nlib: Optional default neutron cross-section library for this material
    :type nlib: Optional[str]
    :ivar plib: Optional default photon cross-section library for this material
    :type plib: Optional[str]
    :ivar components: Dictionary mapping ZAID to component information
                     {zaid: {'fraction': float, 'nlib': str, 'plib': str}}
    :type components: Dict[int, Dict]
    """
    id: int
    nlib: Optional[str] = None
    plib: Optional[str] = None
    components: Dict[int, Dict] = field(default_factory=dict)
    
    def add_component(self, zaid: int, fraction: float, library: Optional[str] = None) -> None:
        """Add a nuclide component to this material.
        
        :param zaid: Nuclide ZAID number
        :type zaid: int
        :param fraction: Atomic fraction of the nuclide
        :type fraction: float
        :param library: Specific cross-section library for this nuclide,
                       overrides material's default if specified
        :type library: Optional[str]
        """
        nlib = None
        plib = None
        
        if library:
            # Determine library type by examining the last character
            if library.endswith('c'):
                nlib = library
            elif library.endswith('p'):
                plib = library
        
        # Check if component already exists
        if zaid in self.components:
            # Update existing component with new library information
            # Keep the existing fraction
            if nlib:
                self.components[zaid]['nlib'] = nlib
            if plib:
                self.components[zaid]['plib'] = plib
        else:
            # Create new component
            self.components[zaid] = {
                'fraction': float(fraction),
                'nlib': nlib,
                'plib': plib,
                'nuclide': self._get_nuclide_symbol(zaid)
            }
    
    def _get_nuclide_symbol(self, zaid: int) -> Optional[str]:
        """Get nuclide symbol based on atomic number.
        
        :param zaid: The ZAID number
        :type zaid: int
        :returns: Chemical symbol of element or None if not found
        :rtype: Optional[str]
        """
        # Extract atomic number from ZAID (first 2-3 digits)
        atomic_number = zaid // 1000
        return ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number)
    
    def get_effective_library(self, zaid: int, lib_type: str = 'nlib') -> Optional[str]:
        """Get effective library for a component, considering inheritance.
        
        :param zaid: The ZAID number of the component
        :type zaid: int
        :param lib_type: Library type ('nlib' or 'plib')
        :type lib_type: str
        :returns: Component's library if specified, otherwise material's default
        :rtype: Optional[str]
        """
        if zaid not in self.components:
            return getattr(self, lib_type, None)
        
        comp_lib = self.components[zaid].get(lib_type)
        return comp_lib if comp_lib else getattr(self, lib_type, None)
    
    def _format_fraction(self, fraction: float) -> str:
        """Format atomic fraction in scientific notation with 6 decimal digits.
        
        :param fraction: The fraction value to format
        :type fraction: float
        :returns: Formatted string representation
        :rtype: str
        """
        return f"{fraction:.6e}"
    
    def __str__(self) -> str:
        """String representation of the material.
        
        :returns: Formatted string representation of the material
        :rtype: str
        """
        result = [f"m{self.id}"]
        
        # Add material-level libraries if specified
        lib_parts = []
        if self.nlib:
            lib_parts.append(f"nlib={self.nlib}")
        if self.plib:
            lib_parts.append(f"plib={self.plib}")
        
        if lib_parts:
            result[0] += " " + " ".join(lib_parts)
        
        # Process each component
        for zaid, comp in self.components.items():
            # Handle neutron library if specified
            if comp['nlib']:
                lib_str = f".{comp['nlib']}"
                result.append(f"    {zaid}{lib_str} {self._format_fraction(comp['fraction'])}")
            elif self.nlib:
                # Use default neutron library at material level - no need to specify at component
                result.append(f"    {zaid} {self._format_fraction(comp['fraction'])}")
            
            # Handle photon library if specified (add a separate entry)
            if comp['plib']:
                lib_str = f".{comp['plib']}"
                result.append(f"    {zaid}{lib_str} {self._format_fraction(comp['fraction'])}")
            elif self.plib and not comp['nlib']:
                # If we didn't add an entry for neutron library and there's a default photon library,
                # we need to add an entry for photon library
                # Skip this if we already added an entry for neutron to avoid duplication
                result.append(f"    {zaid} {self._format_fraction(comp['fraction'])}")
        
        return "\n".join(result)

