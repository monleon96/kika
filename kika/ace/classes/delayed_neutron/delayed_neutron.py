from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from kika.ace.classes.xss import XssEntry
from kika.ace.classes.delayed_neutron.delayed_neutron_repr import precursor_repr, delayed_neutron_data_repr

@dataclass
class DelayedNeutronPrecursor:
    """Data for a single delayed neutron precursor group."""
    decay_constant: Optional[XssEntry] = None  # Decay constant for the group
    interpolation_regions: List[Tuple[int, int]] = field(default_factory=list)  # (NBT, INT) pairs
    energies: List[XssEntry] = field(default_factory=list)  # Energy points
    probabilities: List[XssEntry] = field(default_factory=list)  # Corresponding probabilities
    
    def evaluate(self, energy: float) -> float:
        """
        Evaluate the precursor probability at the given energy using interpolation.
        
        Parameters
        ----------
        energy : float
            Energy in MeV
            
        Returns
        -------
        float
            The probability value at the given energy
        """
        # Simple linear interpolation for now
        if not self.energies or not self.probabilities:
            return 0.0
            
        if energy <= self.energies[0].value:
            return self.probabilities[0].value
        
        if energy >= self.energies[-1].value:
            return self.probabilities[-1].value
        
        # Find the bracketing energy points
        for i in range(len(self.energies) - 1):
            if self.energies[i].value <= energy <= self.energies[i + 1].value:
                # Linear interpolation
                x1, x2 = self.energies[i].value, self.energies[i + 1].value
                y1, y2 = self.probabilities[i].value, self.probabilities[i + 1].value
                return y1 + (y2 - y1) * (energy - x1) / (x2 - x1)
        
        # Shouldn't reach here, but just in case
        return self.probabilities[-1].value
        
    # Define repr explicitly as a method to ensure it's picked up correctly
    def __repr__(self):
        return precursor_repr(self)

@dataclass
class DelayedNeutronData:
    """Container for all delayed neutron precursor groups."""
    has_delayed_neutron_data: bool = False  # True if BDD block is present
    precursors: List[DelayedNeutronPrecursor] = field(default_factory=list)
    
    def get_precursor_probability(self, group_idx: int, energy: float) -> Optional[float]:
        """
        Get the probability for a specific precursor group at the given energy.
        
        Parameters
        ----------
        group_idx : int
            Index of the precursor group (0-based)
        energy : float
            Energy in MeV
            
        Returns
        -------
        float or None
            The probability value, or None if the group doesn't exist
        """
        if not self.has_delayed_neutron_data or group_idx < 0 or group_idx >= len(self.precursors):
            return None
        
        return self.precursors[group_idx].evaluate(energy)
    
    def get_decay_constant(self, group_idx: int) -> Optional[float]:
        """
        Get the decay constant for a specific precursor group.
        
        Parameters
        ----------
        group_idx : int
            Index of the precursor group (0-based)
            
        Returns
        -------
        float or None
            The decay constant, or None if the group doesn't exist
        """
        if not self.has_delayed_neutron_data or group_idx < 0 or group_idx >= len(self.precursors):
            return None
        
        precursor = self.precursors[group_idx]
        return precursor.decay_constant.value if precursor.decay_constant else None
        
    # Define repr explicitly as a method to ensure it's picked up correctly
    def __repr__(self):
        return delayed_neutron_data_repr(self)
