from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union

from .._constants import ATOMIC_MASS, ATOMIC_NUMBER_TO_SYMBOL, NATURAL_ABUNDANCE
from .._utils import zaid_to_symbol, symbol_to_zaid

# Density unit normalisation helpers.
_DENSITY_UNIT_ALIASES: Dict[str, str] = {
    "g/cc": "g/cc",
    "g/cm3": "g/cc",
    "g/cm^3": "g/cc",
    "g/cm³": "g/cc",
    "kg/m3": "kg/m3",
    "kg/m^3": "kg/m3",
    "kg/m³": "kg/m3",
}

_DENSITY_CONVERSIONS: Dict[tuple[str, str], float] = {
    ("g/cc", "g/cc"): 1.0,
    ("g/cc", "kg/m3"): 1000.0,
    ("kg/m3", "kg/m3"): 1.0,
    ("kg/m3", "g/cc"): 0.001,
}


def _normalize_density_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    key = unit.strip().lower().replace(" ", "")
    return _DENSITY_UNIT_ALIASES.get(key, key)


def _validate_fraction_type(fraction_type: str) -> None:
    """Validate that fraction_type is exactly 'ao' or 'wo'."""
    if fraction_type not in ('ao', 'wo'):
        raise ValueError(f"Fraction type must be exactly 'ao' or 'wo', got '{fraction_type}'")


def _is_natural_zaid(zaid: int) -> bool:
    """Return True for natural elements (mass number exactly 0 → ZAID ends with '000')."""
    return zaid > 1000 and zaid % 1000 == 0


@dataclass
class Nuclide:
    """Lightweight nuclide entry for a material.

    Attributes
    ----------
    zaid : int
        ZAID identifier (e.g. 92235 for U-235).
    fraction : float
        Composition fraction stored as a positive value; interpretation is driven by
        the parent ``Material.fraction_type``.
    libs : dict
        Per-nuclide library overrides keyed by library keyword (e.g. ``'nlib'``, ``'plib'``).
    """

    zaid: int
    fraction: float
    libs: Dict[str, str] = field(default_factory=dict)

    @property
    def element(self) -> Optional[str]:
        atomic_number = self.zaid // 1000
        return ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number)

    @property
    def symbol(self) -> str:
        """Return the element-mass symbol (e.g., 'Fe56', 'U235')."""
        return zaid_to_symbol(self.zaid)

    @property
    def is_natural(self) -> bool:
        return _is_natural_zaid(self.zaid)

    def convert_natural_element(self) -> None:
        """Print an isotopic breakdown of a natural element.

        Raises
        ------
        ValueError
            If ``zaid`` does not represent a natural element or if abundance data
            are missing.
        """
        if not self.is_natural:
            raise ValueError(
                f"Nuclide with ZAID {self.zaid} is not a natural element (mass number must be 0, e.g. ZAID ends with '000')"
            )

        atomic_number = self.zaid // 1000
        atomic_key = atomic_number * 1000

        if atomic_key not in NATURAL_ABUNDANCE:
            raise ValueError(f"No natural abundance data available for element ZAID {self.zaid}")

        original_fraction = abs(self.fraction)
        natural_isotopes = NATURAL_ABUNDANCE[atomic_key]

        if self.fraction < 0:
            total_mass_per_atom = 0.0
            isotope_masses = {}
            for iso_zaid, atomic_abundance in natural_isotopes.items():
                iso_mass = ATOMIC_MASS.get(iso_zaid, iso_zaid % 1000)
                isotope_masses[iso_zaid] = iso_mass
                total_mass_per_atom += atomic_abundance * iso_mass

            for iso_zaid, atomic_abundance in sorted(natural_isotopes.items()):
                iso_mass = isotope_masses[iso_zaid]
                iso_mass_fraction = (atomic_abundance * iso_mass) / total_mass_per_atom
                iso_mass_density = -(original_fraction * iso_mass_fraction)
                print(f"{iso_zaid} {iso_mass_density:.6e}")
        else:
            for iso_zaid, atomic_abundance in sorted(natural_isotopes.items()):
                fraction = original_fraction * atomic_abundance
                print(f"{iso_zaid} {fraction:.6e}")

    def __str__(self) -> str:
        """Return a user-friendly string representation of the nuclide."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"{'Nuclide: ' + self.symbol:^70}")
        lines.append("=" * 70)
        
        # Format attributes in two columns
        lines.append(f"{'ZAID:':<25} {self.zaid}")
        lines.append(f"{'Element:':<25} {self.element or 'Unknown'}")
        lines.append(f"{'Mass Number:':<25} {self.zaid % 1000}")
        lines.append(f"{'Fraction:':<25} {self.fraction:.6e}")
        lines.append(f"{'Is Natural:':<25} {self.is_natural}")
        
        if self.libs:
            lines.append("")
            lines.append(f"{'Library Overrides:'}")
            for key, value in sorted(self.libs.items()):
                lines.append(f"  {key:<23} {value}")
        else:
            lines.append("")
            lines.append(f"{'Library Overrides:':<25} None")
        
        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return the same formatted representation as __str__."""
        return self.__str__()


@dataclass
class Material:
    """General-purpose material representation.

    Fractions are stored as non-negative numbers; ``fraction_type`` records whether
    they represent atomic (``'ao'``) or weight (``'wo'``) fractions. This class provides
    methods for managing nuclide composition, density, and metadata for a material.

    Parameters
    ----------
    id : int
        Material identifier number.
    nuclide : dict, optional
        Mapping of ZAID to :class:`Nuclide` objects.
    libs : dict, optional
        Default library keywords (e.g. ``'nlib'``). Nuclide libraries override these defaults.
    name : str, optional
        Optional descriptive name.
    fraction_type : str, optional
        ``'ao'`` for atomic fractions or ``'wo'`` for weight fractions.
    density : float, optional
        Density magnitude paired with ``density_unit``.
    density_unit : str, optional
        Density unit (``'g/cc'`` or ``'kg/m3'``).
    temperature : float, optional
        Material temperature if available in Kelvin.
    metadata : dict, optional
        Arbitrary metadata tags.
    """

    id: int
    nuclide: Dict[str, Nuclide] = field(default_factory=dict)  # Now keyed by symbol, not ZAID
    libs: Dict[str, str] = field(default_factory=dict)
    name: Optional[str] = None
    fraction_type: str = "ao"
    density: Optional[float] = None
    density_unit: Optional[str] = None
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_fraction_type(self.fraction_type)
        self.libs = dict(self.libs)
        for nuc in self.nuclide.values():
            nuc.fraction = float(abs(nuc.fraction))
            nuc.libs = dict(nuc.libs)

        if self.density is not None:
            self.density = float(self.density)
        self.density_unit = _normalize_density_unit(self.density_unit)

    @property
    def is_atomic(self) -> bool:
        return self.fraction_type == "ao"

    @property
    def is_weight(self) -> bool:
        return self.fraction_type == "wo"

    def _infer_lib_key_from_suffix(self, library: str) -> Optional[str]:
        """Infer MCNP library keyword from a library suffix.
        
        Parameters
        ----------
        library : str
            Library suffix to analyze.
            
        Returns
        -------
        str or None
            Library keyword ('nlib', 'plib', or 'ylib') or None if unrecognized.
        """
        if not library:
            return None
        last = library[-1].lower()
        if last == "c":
            return "nlib"
        if last == "p":
            return "plib"
        if last == "y":
            return "ylib"
        return None

    def add_nuclide(self, nuclide: Union[int, str], fraction: float, fraction_type: str, 
                    library: Optional[str] = None) -> None:
        """Add or update a nuclide in the material.

        Parameters
        ----------
        nuclide : int or str
            ZAID identifier (int, e.g. 92235) or element-mass symbol (str, e.g. 'U235', 'Fe56').
            For natural elements, use symbol without mass number (e.g. 'Fe' -> 26000).
        fraction : float
            Fraction value; the absolute value is stored and interpreted according
            to the provided ``fraction_type``.
        fraction_type : str
            Fraction type for this addition: ``'ao'`` for atomic fractions or ``'wo'`` for 
            weight fractions. This updates the material's fraction_type.
        library : str, optional
            MCNP library suffix (e.g. ``'80c'``, ``'70c'``, ``'12p'``). The library
            type (neutron, photon, or thermal) is automatically inferred from the
            suffix: 'c' → neutron (nlib), 'p' → photon (plib), 'y' → thermal (ylib).
            This provides a nuclide-level library override.

        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If nuclide is a string and does not represent a valid element symbol.
        
        Examples
        --------
        >>> mat = Material(id=1)
        >>> mat.add_nuclide('U235', 1.0, 'ao')  # Add U-235 as atomic fraction
        >>> mat.add_nuclide(92238, 0.5, 'ao')   # Add U-238 using ZAID
        >>> mat.add_nuclide('Fe', 1.0, 'ao')    # Add natural Fe (26000)
        """
        # Convert symbol to zaid if necessary
        if isinstance(nuclide, str):
            zaid = symbol_to_zaid(nuclide)
        else:
            zaid = nuclide
        
        # Validate fraction type (must be exactly 'ao' or 'wo')
        _validate_fraction_type(fraction_type)
        self.fraction_type = fraction_type
        
        # Get symbol for dictionary key
        symbol = zaid_to_symbol(zaid)
        
        # Add or update the nuclide
        lib_key = self._infer_lib_key_from_suffix(library) if library else None

        if symbol in self.nuclide:
            nuc = self.nuclide[symbol]
            nuc.fraction = float(abs(fraction))
        else:
            nuc = Nuclide(zaid=zaid, fraction=float(abs(fraction)))
            self.nuclide[symbol] = nuc

        if lib_key is not None and library is not None:
            nuc.libs[lib_key] = library

    def add_element(self, element: str, fraction: float, fraction_type: str = "ao",
                    library: Optional[str] = None) -> None:
        """Add a natural element to the material.
        
        This method is a convenience for adding natural elements by symbol only (without
        mass number). For specific isotopes, use ``add_nuclide()``.

        Parameters
        ----------
        element : str
            Element symbol (e.g., 'Fe', 'U', 'O'). Will be converted to natural ZAID
            (mass number = 0, e.g., 'Fe' -> 26000).
        fraction : float
            Fraction value; the absolute value is stored and interpreted according
            to the provided ``fraction_type``.
        fraction_type : str, optional
            Fraction type: ``'ao'`` for atomic fractions or ``'wo'`` for weight fractions.
            Default is ``'ao'``.
        library : str, optional
            MCNP library suffix (e.g. ``'80c'``, ``'70c'``, ``'12p'``).

        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If element is not a recognized element symbol.
        
        Examples
        --------
        >>> mat = Material(id=1)
        >>> mat.add_element('Fe', 1.0, 'ao')      # Add natural Fe with atomic fraction
        >>> mat.add_element('O', 2.0, 'wo')       # Add natural O with weight fraction
        """
        self.add_nuclide(element, fraction, fraction_type, library)

    def _get_nuclide_symbol(self, zaid: int) -> Optional[str]:
        atomic_number = zaid // 1000
        return ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number)

    def get_effective_library(self, nuclide: Union[int, str], lib_key: str = "nlib") -> Optional[str]:
        """Return the effective MCNP library for a given nuclide.

        Parameters
        ----------
        nuclide : int or str
            ZAID identifier or symbol.
        lib_key : str, optional
            Library keyword to query (e.g. ``'nlib'``, ``'plib'``, ``'ylib'``).

        Returns
        -------
        str or None
            Nuclide-level override if present, otherwise the material default.
        """
        # Convert to symbol if ZAID provided
        if isinstance(nuclide, int):
            symbol = zaid_to_symbol(nuclide)
        else:
            symbol = nuclide
        
        nuc = self.nuclide.get(symbol)
        if nuc is not None and lib_key in nuc.libs:
            return nuc.libs[lib_key]
        return self.libs.get(lib_key)

    def set_density(self, density: float, unit: str) -> None:
        """Set the material density with explicit units.

        Parameters
        ----------
        density : float
            Density magnitude.
        unit : str
            Unit string (``'g/cc'`` or ``'kg/m3'``); case-insensitive.

        Returns
        -------
        None
        """
        normalized_unit = _normalize_density_unit(unit)
        if normalized_unit not in {"g/cc", "kg/m3"}:
            raise ValueError(f"Unsupported density unit '{unit}'. Use 'g/cc' or 'kg/m3'.")
        self.density = float(density)
        self.density_unit = normalized_unit

    def set_temperature(self, temperature: float) -> None:
        """Set the material temperature in Kelvin.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        None
        """
        self.temperature = float(temperature)

    def density_in(self, unit: str) -> float:
        """Return the current density expressed in a different unit.
        
        Parameters
        ----------
        unit : str
            Target unit string ('g/cc' or 'kg/m3').
            
        Returns
        -------
        float
            Density value in the requested unit.
            
        Raises
        ------
        ValueError
            If density is not set or unit is unsupported.
        """
        if self.density is None or self.density_unit is None:
            raise ValueError("Density or density_unit not set.")
        target_unit = _normalize_density_unit(unit)
        if target_unit not in {"g/cc", "kg/m3"}:
            raise ValueError(f"Unsupported density unit '{unit}'. Use 'g/cc' or 'kg/m3'.")
        factor = _DENSITY_CONVERSIONS[(self.density_unit, target_unit)]
        return self.density * factor

    def convert_density(self, unit: str) -> None:
        """Convert density in-place to the requested unit.
        
        Parameters
        ----------
        unit : str
            Target unit string ('g/cc' or 'kg/m3').
            
        Returns
        -------
        None
        """
        converted = self.density_in(unit)
        self.density = converted
        self.density_unit = _normalize_density_unit(unit)

    def _get_effective_atomic_mass(self, zaid: int) -> float:
        """Return the effective atomic mass, handling natural elements.

        Parameters
        ----------
        zaid : int
            ZAID identifier for which to retrieve mass.

        Returns
        -------
        float
            Atomic mass; for natural elements the abundance-weighted mean is used.
        """
        # Check if this is a natural element (ZAID mass number equals 0)
        if _is_natural_zaid(zaid):
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

    def to_weight_fraction(self) -> Material:
        """Convert atomic fractions to weight fractions (stored as positives).

        Returns
        -------
        Material
            Self reference for chaining.
        """
        if self.is_weight:
            return self

        mass_sum = 0.0
        for symbol, nuclide in self.nuclide.items():
            atomic_mass = self._get_effective_atomic_mass(nuclide.zaid)
            mass_sum += nuclide.fraction * atomic_mass

        if mass_sum <= 0:
            raise ValueError(f"Cannot convert to weight fractions: total mass calculated as {mass_sum}")

        for symbol, nuclide in self.nuclide.items():
            atomic_mass = self._get_effective_atomic_mass(nuclide.zaid)
            nuclide.fraction = nuclide.fraction * atomic_mass / mass_sum

        self.fraction_type = "wo"
        return self

    def to_atomic_fraction(self) -> Material:
        """Convert weight fractions to atomic fractions (stored as positives).

        Returns
        -------
        Material
            Self reference for chaining.
        """
        if self.is_atomic:
            return self

        atomic_sum = 0.0
        for symbol, nuclide in self.nuclide.items():
            atomic_mass = self._get_effective_atomic_mass(nuclide.zaid)
            if atomic_mass <= 0:
                mass_number = nuclide.zaid % 1000
                atomic_mass = float(mass_number) if mass_number > 0 else 1.0
            atomic_sum += nuclide.fraction / atomic_mass

        if atomic_sum <= 0:
            raise ValueError("Cannot convert to atomic fractions: atomic_sum calculated as 0 or negative.")

        for symbol, nuclide in self.nuclide.items():
            atomic_mass = self._get_effective_atomic_mass(nuclide.zaid)
            if atomic_mass <= 0:
                mass_number = nuclide.zaid % 1000
                atomic_mass = float(mass_number) if mass_number > 0 else 1.0
            nuclide.fraction = (nuclide.fraction / atomic_mass) / atomic_sum

        self.fraction_type = "ao"
        return self

    def normalize(self, fraction_type: str = "ao") -> Material:
        """Normalize all fractions so they sum to 1.0.
        
        This method first converts all fractions to the specified fraction_type (atomic or weight),
        then rescales them proportionally so that the total sum equals 1.0. This is useful after
        manually adding nuclides with arbitrary fractional values, such as defining water
        as H with fraction 2.0 and O with fraction 1.0, then normalizing.

        Parameters
        ----------
        fraction_type : str, optional
            Target fraction type for normalization: ``'ao'`` for atomic fractions or 
            ``'wo'`` for weight fractions. Default is ``'ao'``.

        Returns
        -------
        Material
            Self reference for chaining.
            
        Raises
        ------
        ValueError
            If total fraction sum is zero or negative, or if fraction_type is invalid.
        
        Examples
        --------
        >>> mat = Material(id=1)
        >>> mat.add_nuclide('H', 2.0, 'ao')
        >>> mat.add_nuclide('O', 1.0, 'ao')
        >>> mat.normalize('ao')  # Normalize to atomic fractions (default)
        >>> mat.normalize('wo')  # Or normalize to weight fractions
        """
        # Validate fraction type
        _validate_fraction_type(fraction_type)
        
        # Convert to target fraction type if needed
        if fraction_type == "wo" and self.is_atomic:
            self.to_weight_fraction()
        elif fraction_type == "ao" and self.is_weight:
            self.to_atomic_fraction()
        
        # Now normalize
        total_fraction = sum(nuc.fraction for nuc in self.nuclide.values())
        
        if total_fraction <= 0:
            raise ValueError(f"Cannot normalize: total fraction sum is {total_fraction} (must be positive)")
        
        if total_fraction != 1.0:
            for nuclide in self.nuclide.values():
                nuclide.fraction = nuclide.fraction / total_fraction
        
        return self

    def expand_natural_elements(self, elements: Optional[Union[str, List[str]]] = None) -> Material:
        """Expand natural elements into isotopic composition using abundance data.

        Parameters
        ----------
        elements : str or list[str], optional
            Specific natural element symbol(s) to expand (e.g., 'Fe', 'C'). 
            If ``None``, all natural elements in the material are expanded.

        Returns
        -------
        Material
            Self reference for chaining.

        Raises
        ------
        ValueError
            If a requested element is missing, not natural, or lacks abundance data.
        """
        original_fraction_type = self.fraction_type

        if self.is_weight:
            self.to_atomic_fraction()

        if elements is not None:
            # Convert to list of symbols
            symbols_list = [elements] if isinstance(elements, str) else elements
            natural_symbols: List[str] = []
            
            for elem in symbols_list:
                # Convert element symbol to natural symbol (e.g., 'Fe' -> 'Fe0')
                zaid = symbol_to_zaid(elem)
                symbol = zaid_to_symbol(zaid)
                
                if symbol not in self.nuclide:
                    raise ValueError(f"Element '{elem}' (symbol '{symbol}') not found in material {self.id}")
                if not _is_natural_zaid(zaid):
                    raise ValueError(
                        f"Element '{elem}' is not a natural element (must have mass number 0)"
                    )
                natural_symbols.append(symbol)
        else:
            # Find all natural elements
            natural_symbols = [sym for sym, nuc in self.nuclide.items() if nuc.is_natural]

        for symbol in natural_symbols:
            nuclide = self.nuclide[symbol]
            zaid = nuclide.zaid
            atomic_number = zaid // 1000

            atomic_key = atomic_number * 1000
            if atomic_key not in NATURAL_ABUNDANCE:
                raise ValueError(f"No natural abundance data available for element {symbol}")

            original_fraction = nuclide.fraction
            natural_isotopes = NATURAL_ABUNDANCE[atomic_key]
            new_isotopes: Dict[str, Nuclide] = {}

            for iso_zaid, atomic_abundance in sorted(natural_isotopes.items()):
                fraction = original_fraction * atomic_abundance
                iso_symbol = zaid_to_symbol(iso_zaid)
                new_isotopes[iso_symbol] = Nuclide(
                    zaid=iso_zaid,
                    fraction=fraction,
                    libs=dict(nuclide.libs),
                )

            for iso_symbol, isotope in new_isotopes.items():
                self.nuclide[iso_symbol] = isotope
            del self.nuclide[symbol]

        if original_fraction_type == "wo":
            self.to_weight_fraction()

        return self

    def _format_fraction(self, fraction: float) -> str:
        """Format a fraction value for MCNP output.
        
        Parameters
        ----------
        fraction : float
            Fraction value to format.
            
        Returns
        -------
        str
            Formatted fraction in scientific notation.
        """
        return f"{fraction:.6e}"

    def to_mcnp(self) -> str:
        """Serialise the material as an MCNP ``m`` card.
        
        Returns
        -------
        str
            MCNP material card formatted as a multi-line string.
        """
        lines = [f"m{self.id}"]

        lib_parts = [f"{k}={v}" for k, v in sorted(self.libs.items())]
        if lib_parts:
            lines[0] += " " + " ".join(lib_parts)

        sign = -1.0 if self.is_weight else 1.0
        for symbol, nuclide in sorted(self.nuclide.items()):
            zaid = nuclide.zaid
            if nuclide.libs:
                for key, lib in sorted(nuclide.libs.items()):
                    lines.append(f"\t{zaid}.{lib} {self._format_fraction(sign * nuclide.fraction)}")
            else:
                lines.append(f"\t{zaid} {self._format_fraction(sign * nuclide.fraction)}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return a user-friendly string representation of the material.
        
        Returns
        -------
        str
            Formatted string showing material properties and composition.
        """
        lines = []
        lines.append("=" * 80)
        title = f"Material {self.id}"
        if self.name:
            title += f": {self.name}"
        lines.append(f"{title:^80}")
        lines.append("=" * 80)
        
        # Format all material attributes in two columns
        lines.append(f"{'Name:':<30} {self.name if self.name else 'None'}")
        lines.append(f"{'Fraction Type:':<30} {self.fraction_type}")
        lines.append(f"{'Number of Nuclides:':<30} {len(self.nuclide)}")
        
        # Density
        if self.density is not None:
            density_str = f"{self.density} {self.density_unit if self.density_unit else ''}"
        else:
            density_str = "None"
        lines.append(f"{'Density:':<30} {density_str}")
        
        # Temperature in Kelvin
        if self.temperature is not None:
            temp_str = f"{self.temperature} K"
        else:
            temp_str = "None"
        lines.append(f"{'Temperature:':<30} {temp_str}")
        
        # Material-level libraries
        if self.libs:
            lib_str = ", ".join(f"{k}={v}" for k, v in sorted(self.libs.items()))
        else:
            lib_str = "None"
        lines.append(f"{'Libraries:':<30} {lib_str}")
        
        # Metadata
        if self.metadata:
            meta_str = str(self.metadata)
            if len(meta_str) > 48:
                meta_str = meta_str[:45] + "..."
            lines.append(f"{'Metadata:':<30} {meta_str}")
        else:
            lines.append(f"{'Metadata:':<30} None")
        
        # Nuclide composition table (without Libraries column)
        if self.nuclide:
            lines.append("\n" + "-" * 80)
            lines.append(f"{'Symbol':^15} | {'ZAID':^12} | {'Fraction':^18} | {'Type':^8}")
            lines.append("-" * 80)
            
            for symbol in sorted(self.nuclide.keys()):
                nuc = self.nuclide[symbol]
                lines.append(f"{symbol:^15} | {nuc.zaid:^12} | {nuc.fraction:^18.6e} | {self.fraction_type:^8}")
            
            lines.append("-" * 80)
        
        # Usage information
        lines.append("\nAvailable methods:")
        lines.append("  .add_nuclide()           - Add a nuclide by symbol or ZAID")
        lines.append("  .add_element()           - Add a natural element by symbol")
        lines.append("  .normalize()             - Normalize fractions to sum to 1.0")
        lines.append("  .to_weight_fraction()    - Convert to weight fractions")
        lines.append("  .to_atomic_fraction()    - Convert to atomic fractions")
        lines.append("  .expand_natural_elements() - Expand natural elements to isotopes")
        lines.append("  .set_density()           - Set material density")
        lines.append("  .set_temperature()       - Set material temperature (Kelvin)")
        lines.append("  .to_mcnp()               - Export as MCNP material card")
        lines.append("  .copy(new_id)            - Create a copy with a new ID")
        
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return the same formatted representation as __str__.
        
        Returns
        -------
        str
            Formatted material representation.
        """
        return self.__str__()

    def copy(self, new_id: int) -> Material:
        """Create an exact copy of this material with a new ID.
        
        Parameters
        ----------
        new_id : int
            Material ID for the new copy.
            
        Returns
        -------
        Material
            New material instance with all properties copied.
        """
        new_material = Material(
            id=new_id,
            nuclide={sym: Nuclide(zaid=nuc.zaid, fraction=nuc.fraction, libs=dict(nuc.libs)) for sym, nuc in self.nuclide.items()},
            libs=dict(self.libs),
            name=self.name,
            fraction_type=self.fraction_type,
            density=self.density,
            density_unit=self.density_unit,
            temperature=self.temperature,
            metadata=deepcopy(self.metadata),
        )
        return new_material


@dataclass
class MaterialCollection:
    """Container class for material objects."""

    by_id: Dict[int, Material] = field(default_factory=dict)

    @property
    def mat(self) -> Dict[int, Material]:
        """Dictionary of materials keyed by ID.
        
        Returns
        -------
        dict of int to Material
            Materials dictionary.
        """
        return self.by_id

    def __iter__(self) -> Iterable[Material]:
        return iter(self.by_id.values())

    def __len__(self) -> int:
        return len(self.by_id)

    def add_material(self, material: Material) -> None:
        """Add a material to the collection.
        
        Parameters
        ----------
        material : Material
            Material to add.
            
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If a material with the same ID already exists.
        """
        if material.id in self.by_id:
            raise ValueError(f"Material ID {material.id} already exists in collection")
        self.by_id[material.id] = material

    def to_mcnp(self) -> str:
        """Serialise all materials as MCNP material cards.
        
        Returns
        -------
        str
            All material cards concatenated with newlines.
        """
        return "\n".join(material.to_mcnp() for material in self.by_id.values())

    @classmethod
    def from_mcnp(cls, path: str) -> MaterialCollection:
        """Create a MaterialCollection from an MCNP input file.
        
        Parameters
        ----------
        path : str
            Path to MCNP input file.
            
        Returns
        -------
        MaterialCollection
            Collection of materials parsed from the file.
        """
        from .parse_input import read_mcnp

        input_obj = read_mcnp(path)
        return input_obj.materials

    def __str__(self) -> str:
        """Return a user-friendly string representation of the collection.
        
        Returns
        -------
        str
            Formatted string showing materials summary and table.
        """
        header_width = 80
        lines = []
        lines.append("=" * header_width)
        lines.append(f"{'MCNP Materials Collection':^{header_width}}")
        lines.append("=" * header_width)
        
        materials_count = len(self.by_id)
        lines.append(f"\nTotal Materials: {materials_count}\n")
        
        if materials_count > 0:
            # Table header
            table_width = 80
            lines.append("-" * table_width)
            lines.append(f"{'ID':^12} | {'Nuclides':^10} | {'Type':^10} | {'Libraries':^38}")
            lines.append("-" * table_width)
            
            # Table rows
            for mat_id in sorted(self.by_id.keys()):
                material = self.by_id[mat_id]
                nuclide_count = len(material.nuclide)
                
                frac_type = "Atomic" if material.is_atomic else "Weight"
                
                # Gather all libraries (material-level and nuclide-level)
                libs_set = set()
                if material.libs:
                    for k, v in material.libs.items():
                        libs_set.add(v)
                
                for nuclide in material.nuclide.values():
                    if nuclide.libs:
                        for k, v in nuclide.libs.items():
                            libs_set.add(v)
                
                lib_str = ", ".join(sorted(libs_set)) if libs_set else "default"
                if len(lib_str) > 36:
                    lib_str = lib_str[:33] + "..."
                
                lines.append(f"{mat_id:^12} | {nuclide_count:^10} | {frac_type:^10} | {lib_str:^38}")
            
            lines.append("-" * table_width)
        
        # Available methods
        lines.append("\nAvailable methods:")
        lines.append("  .add_material(material)  - Add a material to the collection")
        lines.append("  .to_mcnp()               - Export all materials as MCNP cards")
        lines.append("  .mat[material_id]        - Access a specific material by ID")
        
        # Examples
        lines.append("\nExamples of accessing data:")
        lines.append("  collection.mat[1]        - Access material with ID 1")
        lines.append("  for mat in collection:   - Iterate over all materials")
        
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return the same formatted representation as __str__.
        
        Returns
        -------
        str
            Formatted collection representation.
        """
        return self.__str__()
