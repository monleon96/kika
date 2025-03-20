from dataclasses import dataclass, field
from typing import Dict, Optional, Set, List, Union
from copy import deepcopy
from .._constants import ATOMIC_NUMBER_TO_SYMBOL, ATOMIC_MASS, NATURAL_ABUNDANCE

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
                
                # Gather all libraries (material-level and nuclide-level)
                libs_set: Set[str] = set()
                # Material level libraries
                if material.nlib:
                    libs_set.add(material.nlib)
                if material.plib:
                    libs_set.add(material.plib)
                if material.ylib:
                    libs_set.add(material.ylib)
                
                # Nuclide level libraries
                for nuclide in material.nuclide.values():
                    if nuclide.nlib:
                        libs_set.add(nuclide.nlib)
                    if nuclide.plib:
                        libs_set.add(nuclide.plib)
                    if nuclide.ylib:
                        libs_set.add(nuclide.ylib)
                
                lib_str = ", ".join(sorted(libs_set)) if libs_set else "default"
                
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
class Mat:
    """Represents a single MCNP material card.

    :ivar id: Material identifier number
    :type id: int
    :ivar nlib: Optional default neutron cross-section library for this material
    :type nlib: Optional[str]
    :ivar plib: Optional default photon cross-section library for this material
    :type plib: Optional[str]
    :ivar ylib: Optional default dosimetry library for this material
    :type ylib: Optional[str]
    :ivar nuclide: Dictionary mapping ZAID to Nuclide objects
    :type nuclide: Dict[int, Nuclide]
    """
    id: int
    nlib: Optional[str] = None
    plib: Optional[str] = None
    ylib: Optional[str] = None
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
        ylib = None
        
        if library:
            # Determine library type by examining the last character
            if library.endswith('c'):
                nlib = library
            elif library.endswith('p'):
                plib = library
            elif library.endswith('y'):
                ylib = library
        
        # Check if nuclide already exists
        if zaid in self.nuclide:
            # Update existing nuclide with new library information
            # Keep the existing fraction
            if nlib:
                self.nuclide[zaid].nlib = nlib
            if plib:
                self.nuclide[zaid].plib = plib
            if ylib:
                self.nuclide[zaid].ylib = ylib
        else:
            # Create new nuclide
            self.nuclide[zaid] = Nuclide(
                zaid=zaid,
                fraction=float(fraction),
                nlib=nlib,
                plib=plib,
                ylib=ylib
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
        :param lib_type: Library type ('nlib', 'plib', or 'ylib')
        :type lib_type: str
        :returns: Nuclide's library if specified, otherwise material's default
        :rtype: Optional[str]
        """
        if zaid not in self.nuclide:
            return getattr(self, lib_type, None)
        
        nuclide_lib = getattr(self.nuclide[zaid], lib_type, None)
        return nuclide_lib if nuclide_lib else getattr(self, lib_type, None)
    
    def _get_effective_atomic_mass(self, zaid: int) -> float:
        """Calculate the effective atomic mass of a nuclide, handling natural elements.
        
        For natural elements (ZAID ending in '00'), calculates a weighted average
        of isotope masses based on natural abundances.
        
        :param zaid: The ZAID number of the nuclide
        :type zaid: int
        :returns: Effective atomic mass
        :rtype: float
        """
        # Check if this is a natural element (ZAID ending in '00')
        if zaid % 1000 == 0 and zaid > 1000:
            atomic_number = zaid // 1000
            atomic_key = atomic_number * 1000  # e.g., 6000 for carbon
            
            # Handle natural elements using abundance data
            if atomic_key in NATURAL_ABUNDANCE:
                natural_isotopes = NATURAL_ABUNDANCE[atomic_key]
                weighted_mass = 0.0
                
                for iso_zaid, abundance in natural_isotopes.items():
                    # Get isotope mass from constants or approximate
                    iso_mass = ATOMIC_MASS.get(iso_zaid, iso_zaid % 1000)
                    weighted_mass += abundance * iso_mass
                    
                return weighted_mass
        
        # For specific isotopes or fallback
        atomic_number = zaid // 1000
        mass_number = zaid % 1000
        
        # Construct the isotope key
        isotope_key = atomic_number * 1000 + mass_number
        
        # Get atomic mass from constants or use a default approximation
        atomic_mass = ATOMIC_MASS.get(isotope_key)
        if atomic_mass is None:
            # Fallback to approximate mass if exact isotope not found
            atomic_mass = float(mass_number) if mass_number > 0 else 1.0
            
        return atomic_mass
    
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
            return self
        
        # Calculate total mass
        mass_sum = 0.0
        for zaid, nuclide in self.nuclide.items():
            # Calculate mass contribution using helper method for accurate atomic mass
            atomic_mass = self._get_effective_atomic_mass(zaid)
            mass_sum += nuclide.fraction * atomic_mass
        
        # Check if mass_sum is zero to avoid division by zero
        if mass_sum <= 0:
            raise ValueError(f"Cannot convert to weight fractions: total mass calculated as {mass_sum}")
            
        # Convert to weight fractions (negative values)
        for zaid, nuclide in self.nuclide.items():
            # Get accurate atomic mass using helper method
            atomic_mass = self._get_effective_atomic_mass(zaid)
            
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
            return self
            
        # First step: convert all fractions to their absolute values
        for nuclide in self.nuclide.values():
            nuclide.fraction = abs(nuclide.fraction)
            
        # Calculate sum of (weight_fraction / atomic_mass)
        atomic_sum = 0.0
        for zaid, nuclide in self.nuclide.items():
            # Get accurate atomic mass using helper method
            atomic_mass = self._get_effective_atomic_mass(zaid)
            
            # Check if atomic_mass is valid to prevent division by zero
            if atomic_mass <= 0:
                # Use mass number as fallback
                mass_number = zaid % 1000
                atomic_mass = float(mass_number) if mass_number > 0 else 1.0
            
            # Add contribution to atomic sum
            atomic_sum += nuclide.fraction / atomic_mass
            
        # Convert to atomic fractions (positive values)
        for zaid, nuclide in self.nuclide.items():
            # Get accurate atomic mass using helper method
            atomic_mass = self._get_effective_atomic_mass(zaid)
            
            # Check if atomic_mass is valid
            if atomic_mass <= 0:
                mass_number = zaid % 1000
                atomic_mass = float(mass_number) if mass_number > 0 else 1.0
            
            # Convert weight fraction to atomic fraction
            atomic_fraction = (nuclide.fraction / atomic_mass) / atomic_sum
            
            # Store as positive value (MCNP convention for atomic fractions)
            nuclide.fraction = atomic_fraction
            
        return self

    
    def convert_natural_elements(self, zaid_to_expand: Optional[Union[int, List[int]]] = None) -> 'Mat':
        """Convert natural elements into their constituent isotopes based on natural abundances.

        This method expands natural elements (ZAIDs ending in '00') into their isotopic composition
        using natural abundance data. It can either convert all natural elements in the material,
        a specific one, or a list of specific ZAIDs.

        :param zaid_to_expand: Optional specific ZAID or list of ZAIDs to expand. If None, expands all natural elements.
        :type zaid_to_expand: Optional[Union[int, List[int]]]
        :returns: Self reference for method chaining
        :rtype: Mat
        :raises ValueError: If ZAID is not found in material, is not a natural element, or lacks abundance data
        """
        # Remember if the original fractions are in weight form.
        original_weight = any(nuc.fraction < 0 for nuc in self.nuclide.values())
        
        # If using weight fractions, convert to atomic fractions first.
        if original_weight:
            self.to_atomic_fraction()
        
        # Determine which natural nuclides to expand.
        if zaid_to_expand is not None:
            # Handle both single ZAID and list of ZAIDs
            zaids_list = [zaid_to_expand] if isinstance(zaid_to_expand, int) else zaid_to_expand
            
            # Validate all ZAIDs in the list
            natural_zaids = []
            for zaid in zaids_list:
                if zaid not in self.nuclide:
                    raise ValueError(f"ZAID {zaid} not found in material {self.id}")
                if not (zaid % 1000 == 0 and zaid > 1000):
                    raise ValueError(f"ZAID {zaid} is not a natural element (does not end in '00')")
                natural_zaids.append(zaid)
        else:
            natural_zaids = [zaid for zaid in list(self.nuclide.keys()) 
                            if zaid % 1000 == 0 and zaid > 1000]
        
        for zaid in natural_zaids:
            nuclide = self.nuclide[zaid]
            atomic_number = zaid // 1000
            
            atomic_key = atomic_number * 1000  # e.g., 6000 for carbon
            if atomic_key not in NATURAL_ABUNDANCE:
                raise ValueError(f"No natural abundance data available for element ZAID {zaid}")
            
            original_fraction = nuclide.fraction  # Now in atomic fractions
            natural_isotopes = NATURAL_ABUNDANCE[atomic_key]
            new_isotopes = {}
            
            # Expand natural element using atomic fractions.
            for iso_zaid, atomic_abundance in sorted(natural_isotopes.items()):
                fraction = original_fraction * atomic_abundance
                new_isotopes[iso_zaid] = Nuclide(
                    zaid=iso_zaid,
                    fraction=fraction,
                    nlib=nuclide.nlib,
                    plib=nuclide.plib,
                    ylib=nuclide.ylib
                )
            
            # Insert new isotopes and remove the original natural element.
            for iso_zaid, isotope in new_isotopes.items():
                self.nuclide[iso_zaid] = isotope
            del self.nuclide[zaid]
        
        # Convert back to weight fractions if that was the original format.
        if original_weight:
            self.to_weight_fraction()
        
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
        """Return a string representation of the material in MCNP format.
        
        :returns: String representation of the material in MCNP format
        :rtype: str
        """
        result = [f"m{self.id}"]
        
        # Add material-level libraries if specified
        lib_parts = []
        if self.nlib:
            lib_parts.append(f"nlib={self.nlib}")
        if self.plib:
            lib_parts.append(f"plib={self.plib}")
        if self.ylib:
            lib_parts.append(f"ylib={self.ylib}")
        
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
            
            # Handle dosimetry library if specified at nuclide level
            if nuclide.ylib:
                lib_str = f".{nuclide.ylib}"
                result.append(f"\t{zaid}{lib_str} {self._format_fraction(nuclide.fraction)}")
                nuclide_added = True
            
            # If no nuclide-level libraries were specified, add the nuclide once
            # It will use material-level libraries if defined
            if not nuclide_added:
                result.append(f"\t{zaid} {self._format_fraction(nuclide.fraction)}")
        
        return "\n".join(result)
    
    def __repr__(self) -> str:
        """Return a concise summary representation of the material.
        
        This method is called when the object is evaluated in interactive environments
        like Jupyter notebooks or the Python interpreter.
        
        :returns: Concise representation of the material
        :rtype: str
        """
        # Create a visually appealing header with a border
        header_width = 50
        header = "=" * header_width + "\n"
        header += f"{'MCNP Material (ID: ' + str(self.id) + ')':^{header_width}}\n"
        header += "=" * header_width + "\n\n"
        
        # Material properties
        props = []
        if self.nlib:
            props.append(f"nlib={self.nlib}")
        if self.plib:
            props.append(f"plib={self.plib}")
        if self.ylib:
            props.append(f"ylib={self.ylib}")
        
        properties = f"Properties: {', '.join(props) if props else 'None'}\n"
        
        # Nuclide count and fraction type
        nuclide_count = len(self.nuclide)
        fraction_type = "Weight" if any(nuc.fraction < 0 for nuc in self.nuclide.values()) else "Atomic"
        info = f"Total Nuclides: {nuclide_count}\n\n"
        
        # Show table of nuclides if we have any
        table = ""
        if nuclide_count > 0:
            table_width = 50
            table += "-" * table_width + "\n"
            table += f"{'ZAID':^10}|{'Element':^10}|{'Fraction':^15}|{'Libraries':^13}\n"
            table += "-" * table_width + "\n"
            
            # Sort ZAIDs for consistent display
            for zaid in sorted(self.nuclide.keys()):
                nuclide = self.nuclide[zaid]
                elem = nuclide.element if nuclide.element else "?"
                
                # Format the fraction with sign indicating if it's weight or atomic
                frac_str = f"{nuclide.fraction:.4e}"
                
                # Get nuclide-specific libraries
                libs = []
                if nuclide.nlib:
                    libs.append(nuclide.nlib)
                if nuclide.plib:
                    libs.append(nuclide.plib)
                if nuclide.ylib:
                    libs.append(nuclide.ylib)
                
                lib_str = ", ".join(libs) if libs else "-"
                
                table += f"{zaid:^10}|{elem:^10}|{frac_str:^15}|{lib_str:^13}\n"
            
            table += "-" * table_width + "\n"
            
            # If there are many nuclides, show a summary instead
            if nuclide_count > 10:
                # Only show first 5 and last 5 nuclides
                table_lines = table.split('\n')
                header_lines = table_lines[:3]  # Header and separator lines
                first_nuclides = table_lines[3:8]  # First 5 nuclides
                ellipsis_line = [f"{'···':^10}|{'···':^10}|{'···':^15}|{'···':^13}"]  # Ellipsis line
                last_nuclides = table_lines[-6:-1]  # Last 5 nuclides
                footer_line = [table_lines[-1]]  # Footer separator line
                
                table = "\n".join(header_lines + first_nuclides + ellipsis_line + last_nuclides + footer_line)
        
        # Available methods section
        methods = "\nAvailable methods:\n"
        methods += "- .to_weight_fraction() - Convert material to weight fractions\n"
        methods += "- .to_atomic_fraction() - Convert material to atomic fractions\n"
        methods += "- .convert_natural_elements() - Convert natural elements into isotopic composition\n"
        methods += "- .copy(new_id) - Create a copy with a new material ID\n"
        
        # Examples of accessing data section
        examples = "\nExamples of accessing data:\n"
        examples += "- .nuclide[zaid] - Access a specific nuclide\n"
        examples += "- print(material) - Print material in MCNP format\n"
        
        return header + properties + info + table + methods + examples
    
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
            plib=self.plib,
            ylib=self.ylib
        )
        
        # Deep copy all nuclides
        for zaid, nuclide in self.nuclide.items():
            new_material.nuclide[zaid] = Nuclide(
                zaid=nuclide.zaid,
                fraction=nuclide.fraction,
                nlib=nuclide.nlib,
                plib=nuclide.plib,
                ylib=nuclide.ylib
            )
            
        return new_material


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
    :ivar ylib: Optional specific dosimetry library for this nuclide
    :type ylib: Optional[str]
    """
    zaid: int
    fraction: float
    nlib: Optional[str] = None
    plib: Optional[str] = None
    ylib: Optional[str] = None
    
    @property
    def element(self) -> Optional[str]:
        """Get the chemical symbol of the nuclide.
        
        :returns: Chemical symbol of the element or None if not found
        :rtype: Optional[str]
        """
        atomic_number = self.zaid // 1000
        return ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number)
    
    @property
    def is_natural(self) -> bool:
        """Check if this is a natural element (ZAID ending in '00').
        
        :returns: True if this is a natural element, False otherwise
        :rtype: bool
        """
        return self.zaid % 1000 == 0 and self.zaid > 1000
        
    def convert_natural_element(self) -> None:
        if not self.is_natural:
            raise ValueError(f"Nuclide with ZAID {self.zaid} is not a natural element (ZAID does not end in '00')")
        
        atomic_number = self.zaid // 1000
        atomic_key = atomic_number * 1000  # e.g., 6000 for carbon
        
        if atomic_key not in NATURAL_ABUNDANCE:
            raise ValueError(f"No natural abundance data available for element ZAID {self.zaid}")
        
        original_fraction = abs(self.fraction)
        natural_isotopes = NATURAL_ABUNDANCE[atomic_key]
        
        if self.fraction < 0:
            # Weight density provided: compute isotopic mass densities
            total_mass_per_atom = 0.0
            isotope_masses = {}
            # Compute the average atomic mass
            for iso_zaid, atomic_abundance in natural_isotopes.items():
                iso_mass = ATOMIC_MASS.get(iso_zaid, iso_zaid % 1000)
                isotope_masses[iso_zaid] = iso_mass
                total_mass_per_atom += atomic_abundance * iso_mass
            
            # Compute and print isotopic mass densities
            for iso_zaid, atomic_abundance in sorted(natural_isotopes.items()):
                iso_mass = isotope_masses[iso_zaid]
                iso_mass_fraction = (atomic_abundance * iso_mass) / total_mass_per_atom
                iso_mass_density = -(original_fraction * iso_mass_fraction)
                print(f"{iso_zaid} {iso_mass_density:.6e}")
        else:
            # Atomic fractions provided: expand directly using atomic abundance
            for iso_zaid, atomic_abundance in sorted(natural_isotopes.items()):
                fraction = original_fraction * atomic_abundance
                print(f"{iso_zaid} {fraction:.6e}")
    
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
        if self.ylib:
            libs.append(f"ylib='{self.ylib}'")
            
        lib_str = ", ".join(libs)
        if lib_str:
            lib_str = f", {lib_str}"
            
        return f"Nuclide(zaid={self.zaid}, fraction={self.fraction:.6e}{lib_str})"