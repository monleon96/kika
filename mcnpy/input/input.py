from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from .._constants import ATOMIC_NUMBER_TO_SYMBOL, ATOMIC_MASS

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
class Nuclide:
    """Represents a nuclide component in an MCNP material.
    
    :ivar zaid: Nuclide ZAID number
    :type zaid: int
    :ivar fraction: Atomic or weight fraction of the nuclide
    :type fraction: float
    :ivar nlib: Optional specific neutron cross-section library for this nuclide
    :type nlib: Optional[str]
    :ivar plib: Optional specific photon cross-section library for this nuclide
    :type plib: Optional[str]
    """
    zaid: int
    fraction: float
    nlib: Optional[str] = None
    plib: Optional[str] = None
    
    @property
    def element(self) -> Optional[str]:
        """Get the chemical symbol of the nuclide.
        
        :returns: Chemical symbol of the element or None if not found
        :rtype: Optional[str]
        """
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
    :ivar nuclides: Dictionary mapping ZAID to Nuclide objects
    :type nuclides: Dict[int, Nuclide]
    """
    id: int
    nlib: Optional[str] = None
    plib: Optional[str] = None
    nuclides: Dict[int, 'Nuclide'] = field(default_factory=dict)
    
    def add_nuclide(self, zaid: int, fraction: float, library: Optional[str] = None) -> None:
        """Add a nuclide to this material.
        
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
        
        # Check if nuclide already exists
        if zaid in self.nuclides:
            # Update existing nuclide with new library information
            # Keep the existing fraction
            if nlib:
                self.nuclides[zaid].nlib = nlib
            if plib:
                self.nuclides[zaid].plib = plib
        else:
            # Create new nuclide
            self.nuclides[zaid] = Nuclide(
                zaid=zaid,
                fraction=float(fraction),
                nlib=nlib,
                plib=plib
            )

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
        """Get effective library for a nuclide, considering inheritance.
        
        :param zaid: The ZAID number of the nuclide
        :type zaid: int
        :param lib_type: Library type ('nlib' or 'plib')
        :type lib_type: str
        :returns: Nuclide's library if specified, otherwise material's default
        :rtype: Optional[str]
        """
        if zaid not in self.nuclides:
            return getattr(self, lib_type, None)
        
        nuclide_lib = getattr(self.nuclides[zaid], lib_type, None)
        return nuclide_lib if nuclide_lib else getattr(self, lib_type, None)
    
    def to_weight_fraction(self) -> 'Mat':
        """Convert atomic fractions to weight fractions.
        
        In MCNP, weight fractions are represented as negative values.
        This method does nothing if fractions are already weight fractions.
        
        :returns: Self reference for method chaining
        :rtype: Mat
        """
        # Check if already using weight fractions (negative values)
        if any(nuclide.fraction < 0 for nuclide in self.nuclides.values()):
            return self  # Already using weight fractions
            
        # Calculate total mass
        mass_sum = 0.0
        for zaid, nuclide in self.nuclides.items():
            atomic_number = zaid // 1000
            mass_number = zaid % 1000
            
            # Construct the isotope key
            isotope_key = atomic_number * 1000 + mass_number
            
            # Get atomic mass from constants or use a default approximation
            atomic_mass = ATOMIC_MASS.get(isotope_key)
            if atomic_mass is None:
                # Fallback to approximate mass if exact isotope not found
                atomic_mass = float(mass_number)
                
            # Calculate mass contribution
            mass_sum += nuclide.fraction * atomic_mass
            
        # Convert to weight fractions (negative values)
        for zaid, nuclide in self.nuclides.items():
            atomic_number = zaid // 1000
            mass_number = zaid % 1000
            
            isotope_key = atomic_number * 1000 + mass_number
            atomic_mass = ATOMIC_MASS.get(isotope_key, float(mass_number))
            
            # Convert atomic fraction to weight fraction
            weight_fraction = nuclide.fraction * atomic_mass / mass_sum
            
            # Store as negative value (MCNP convention for weight fractions)
            nuclide.fraction = -weight_fraction
            
        return self
    
    def to_atomic_fraction(self) -> 'Mat':
        """Convert weight fractions to atomic fractions.
        
        In MCNP, weight fractions are represented as negative values.
        This method does nothing if fractions are already atomic fractions.
        
        :returns: Self reference for method chaining
        :rtype: Mat
        """
        # Check if already using atomic fractions (positive values)
        if all(nuclide.fraction > 0 for nuclide in self.nuclides.values()):
            return self  # Already using atomic fractions
            
        # First step: convert all fractions to their absolute values
        for nuclide in self.nuclides.values():
            nuclide.fraction = abs(nuclide.fraction)
            
        # Calculate sum of (weight_fraction / atomic_mass)
        atomic_sum = 0.0
        for zaid, nuclide in self.nuclides.items():
            atomic_number = zaid // 1000
            mass_number = zaid % 1000
            
            isotope_key = atomic_number * 1000 + mass_number
            atomic_mass = ATOMIC_MASS.get(isotope_key, float(mass_number))
            
            # Add contribution to atomic sum
            atomic_sum += nuclide.fraction / atomic_mass
            
        # Convert to atomic fractions (positive values)
        for zaid, nuclide in self.nuclides.items():
            atomic_number = zaid // 1000
            mass_number = zaid % 1000
            
            isotope_key = atomic_number * 1000 + mass_number
            atomic_mass = ATOMIC_MASS.get(isotope_key, float(mass_number))
            
            # Convert weight fraction to atomic fraction
            atomic_fraction = (nuclide.fraction / atomic_mass) / atomic_sum
            
            # Store as positive value (MCNP convention for atomic fractions)
            nuclide.fraction = atomic_fraction
            
        return self
    
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
        
        # Process each nuclide
        for zaid, nuclide in self.nuclides.items():
            nuclide_added = False
            
            # Handle neutron library if specified at nuclide level
            if nuclide.nlib:
                lib_str = f".{nuclide.nlib}"
                result.append(f"    {zaid}{lib_str} {self._format_fraction(nuclide.fraction)}")
                nuclide_added = True
            
            # Handle photon library if specified at nuclide level
            if nuclide.plib:
                lib_str = f".{nuclide.plib}"
                result.append(f"    {zaid}{lib_str} {self._format_fraction(nuclide.fraction)}")
                nuclide_added = True
            
            # If no nuclide-level libraries were specified, add the nuclide once
            # It will use material-level libraries if defined
            if not nuclide_added:
                result.append(f"    {zaid} {self._format_fraction(nuclide.fraction)}")
        
        return "\n".join(result)
    
    def copy(self, new_id: int) -> 'Mat':
        """Create an exact copy of this material with a new ID.
        
        All properties including the nuclides dictionary are copied.
        
        :param new_id: ID for the new material
        :type new_id: int
        :returns: New Mat instance with identical properties but different ID
        :rtype: Mat
        """
        new_material = Mat(
            id=new_id,
            nlib=self.nlib,
            plib=self.plib
        )
        
        # Deep copy all nuclides
        for zaid, nuclide in self.nuclides.items():
            new_material.nuclides[zaid] = Nuclide(
                zaid=nuclide.zaid,
                fraction=nuclide.fraction,
                nlib=nuclide.nlib,
                plib=nuclide.plib
            )
            
        return new_material

