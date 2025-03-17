from dataclasses import dataclass, field
from typing import Dict, Optional
from .._constants import ATOMIC_NUMBER_TO_SYMBOL, ATOMIC_MASS

@dataclass
class Materials:
    """Container class for MCNP material cards.

    :ivar mat: Dictionary mapping material IDs to Mat objects
    :type mat: Dict[int, Mat]
    """
    mat: Dict[int, 'Mat'] = field(default_factory=dict)
    
    def to_weight_fractions(self) -> 'Materials':
        """Convert atomic fractions to weight fractions for all materials.
        
        Calls to_weight_fraction() on each material in the collection.
        
        :returns: Self reference for method chaining
        :rtype: Materials
        """
        for material in self.mat.values():
            material.to_weight_fraction()
        return self
    
    def to_atomic_fractions(self) -> 'Materials':
        """Convert weight fractions to atomic fractions for all materials.
        
        Calls to_atomic_fraction() on each material in the collection.
        
        :returns: Self reference for method chaining
        :rtype: Materials
        """
        for material in self.mat.values():
            material.to_atomic_fraction()
        return self
    
    def __str__(self) -> str:
        """Return a detailed string representation of all materials.
        
        :returns: String representation of all materials
        :rtype: str
        """
        if not self.mat:
            return "Materials()"
    
        # Simple header
        title = f"   MCNP Materials Collection ({len(self.mat)} materials)\n"
        header = "=" * (len(title)+6) + "\n"
        
        # Concatenate material representations without empty lines
        materials_repr = "\n".join([repr(mat) for mat in self.mat.values()])
        
        return header + title + header + '\n' + materials_repr
    
    def __repr__(self) -> str:
        """Returns a formatted summary of the materials collection.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :returns: Concise representation of the materials collection
        :rtype: str
        """
        # Create a visually appealing header with a border
        header_width = 60
        header = "=" * header_width + "\n"
        header += f"{'MCNP Materials Collection':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Materials count
        materials_count = len(self.mat)
        info = f"Total Materials: {materials_count}\n\n"
        
        # Show table of materials and nuclide counts if we have materials
        table = ""
        if materials_count > 0:
            table_width = 60
            table += "-" * table_width + "\n"
            table += f"{'ID':^10}|{'Nuclides':^15}|{'Type':^15}|{'Libraries':^18}\n"
            table += "-" * table_width + "\n"
            
            # Sort material IDs for consistent display
            for mat_id in sorted(self.mat.keys()):
                material = self.mat[mat_id]
                nuclide_count = len(material.nuclide)
                
                # Determine if using weight or atomic fractions
                frac_type = "Weight" if any(nuc.fraction < 0 for nuc in material.nuclide.values()) else "Atomic"
                
                # List libraries
                libs = []
                if material.nlib:
                    libs.append(material.nlib)
                if material.plib:
                    libs.append(material.plib)
                lib_str = ", ".join(libs) if libs else "default"
                
                table += f"{mat_id:^10}|{nuclide_count:^15}|{frac_type:^15}|{lib_str:^18}\n"
            
            table += "-" * table_width + "\n\n"
        
        # Available methods section
        methods = "Available methods:\n"
        methods += "- .to_weight_fractions() - Convert all materials to weight fractions\n"
        methods += "- .to_atomic_fractions() - Convert all materials to atomic fractions\n"
        
        # Examples of accessing data section
        examples = "\nExamples of accessing data:\n"
        examples += "- .mat[material_id] - Access a specific material\n"
        
        return header + info + table + methods + examples
    
    def add_material(self, material: 'Mat') -> None:
        """Add a material to the collection.
        
        :param material: Material object to add
        :type material: Mat
        :raises ValueError: If material ID already exists in collection
        """
        if material.id in self.mat:
            raise ValueError(f"Material ID {material.id} already exists in collection")
        self.mat[material.id] = material

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
    
    def __repr__(self) -> str:
        """Return a string representation of the nuclide.
        
        :returns: String representation of the nuclide
        :rtype: str
        """
        libs = []
        if self.nlib:
            libs.append(f"nlib='{self.nlib}'")
        if self.plib:
            libs.append(f"plib='{self.plib}'")
            
        lib_str = ", ".join(libs)
        if lib_str:
            lib_str = f", {lib_str}"
            
        return f"Nuclide(zaid={self.zaid}, fraction={self.fraction:.6e}{lib_str})"

@dataclass
class Mat:
    """Represents a single MCNP material card.

    :ivar id: Material identifier number
    :type id: int
    :ivar nlib: Optional default neutron cross-section library for this material
    :type nlib: Optional[str]
    :ivar plib: Optional default photon cross-section library for this material
    :type plib: Optional[str]
    :ivar nuclide: Dictionary mapping ZAID to Nuclide objects
    :type nuclide: Dict[int, Nuclide]
    """
    id: int
    nlib: Optional[str] = None
    plib: Optional[str] = None
    nuclide: Dict[int, 'Nuclide'] = field(default_factory=dict)
    
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
        if zaid in self.nuclide:
            # Update existing nuclide with new library information
            # Keep the existing fraction
            if nlib:
                self.nuclide[zaid].nlib = nlib
            if plib:
                self.nuclide[zaid].plib = plib
        else:
            # Create new nuclide
            self.nuclide[zaid] = Nuclide(
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
        if zaid not in self.nuclide:
            return getattr(self, lib_type, None)
        
        nuclide_lib = getattr(self.nuclide[zaid], lib_type, None)
        return nuclide_lib if nuclide_lib else getattr(self, lib_type, None)
    
    def to_weight_fraction(self) -> 'Mat':
        """Convert atomic fractions to weight fractions.
        
        In MCNP, weight fractions are represented as negative values.
        This method does nothing if fractions are already weight fractions.
        
        :returns: Self reference for method chaining
        :rtype: Mat
        :raises ValueError: If the calculated mass sum is zero
        """
        # Check if already using weight fractions (negative values)
        if any(nuclide.fraction < 0 for nuclide in self.nuclide.values()):
            return self  # Already using weight fractions
            
        # Calculate total mass
        mass_sum = 0.0
        for zaid, nuclide in self.nuclide.items():
            atomic_number = zaid // 1000
            mass_number = zaid % 1000
            
            # Construct the isotope key
            isotope_key = atomic_number * 1000 + mass_number
            
            # Get atomic mass from constants or use a default approximation
            atomic_mass = ATOMIC_MASS.get(isotope_key)
            if atomic_mass is None:
                # Fallback to approximate mass if exact isotope not found
                atomic_mass = float(mass_number) if mass_number > 0 else 1.0
                
            # Calculate mass contribution
            mass_sum += nuclide.fraction * atomic_mass
        
        # Check if mass_sum is zero to avoid division by zero
        if mass_sum <= 0:
            raise ValueError(f"Cannot convert to weight fractions: total mass calculated as {mass_sum}")
            
        # Convert to weight fractions (negative values)
        for zaid, nuclide in self.nuclide.items():
            atomic_number = zaid // 1000
            mass_number = zaid % 1000
            
            isotope_key = atomic_number * 1000 + mass_number
            atomic_mass = ATOMIC_MASS.get(isotope_key, float(mass_number) if mass_number > 0 else 1.0)
            
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
        if all(nuclide.fraction > 0 for nuclide in self.nuclide.values()):
            return self  # Already using atomic fractions
            
        # First step: convert all fractions to their absolute values
        for nuclide in self.nuclide.values():
            nuclide.fraction = abs(nuclide.fraction)
            
        # Calculate sum of (weight_fraction / atomic_mass)
        atomic_sum = 0.0
        for zaid, nuclide in self.nuclide.items():
            atomic_number = zaid // 1000
            mass_number = zaid % 1000
            
            isotope_key = atomic_number * 1000 + mass_number
            atomic_mass = ATOMIC_MASS.get(isotope_key, float(mass_number))
            
            # Check if atomic_mass is valid to prevent division by zero
            if atomic_mass <= 0:
                # Use mass number as fallback if available, or use a default value
                atomic_mass = float(mass_number) if mass_number > 0 else 1.0
            
            # Add contribution to atomic sum
            atomic_sum += nuclide.fraction / atomic_mass
            
        # Convert to atomic fractions (positive values)
        for zaid, nuclide in self.nuclide.items():
            atomic_number = zaid // 1000
            mass_number = zaid % 1000
            
            isotope_key = atomic_number * 1000 + mass_number
            atomic_mass = ATOMIC_MASS.get(isotope_key, float(mass_number))
            
            # Check if atomic_mass is valid
            if atomic_mass <= 0:
                atomic_mass = float(mass_number) if mass_number > 0 else 1.0
            
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
        for zaid, nuclide in self.nuclide.items():
            nuclide_added = False
            
            # Handle neutron library if specified at nuclide level
            if nuclide.nlib:
                lib_str = f".{nuclide.nlib}"
                result.append(f"\t{zaid}{lib_str} {self._format_fraction(nuclide.fraction)}")
                nuclide_added = True
            
            # Handle photon library if specified at nuclide level
            if nuclide.plib:
                lib_str = f".{nuclide.plib}"
                result.append(f"\t{zaid}{lib_str} {self._format_fraction(nuclide.fraction)}")
                nuclide_added = True
            
            # If no nuclide-level libraries were specified, add the nuclide once
            # It will use material-level libraries if defined
            if not nuclide_added:
                result.append(f"\t{zaid} {self._format_fraction(nuclide.fraction)}")
        
        return "\n".join(result)
    
    def __repr__(self) -> str:
        """Return a string representation of the material.
        
        :returns: String representation of the material in MCNP format
        :rtype: str
        """
        return str(self)
    
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
        for zaid, nuclide in self.nuclide.items():
            new_material.nuclide[zaid] = Nuclide(
                zaid=nuclide.zaid,
                fraction=nuclide.fraction,
                nlib=nuclide.nlib,
                plib=nuclide.plib
            )
            
        return new_material

