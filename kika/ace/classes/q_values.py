from dataclasses import dataclass, field
from typing import List, Optional
from kika.ace.classes.xss import XssEntry

@dataclass
class QValues:
    """Container for reaction Q-values from the LQR block."""
    q_values: List[XssEntry] = field(default_factory=list)  # List of Q-values for each reaction
    
    @property
    def has_q_values(self) -> bool:
        """Check if Q-values are available."""
        return len(self.q_values) > 0
    
    def get_q_value(self, reaction_index: int) -> Optional[float]:
        """
        Get the Q-value for a specific reaction.
        
        Parameters
        ----------
        reaction_index : int
            Index of the reaction (0-based)
            
        Returns
        -------
        float or None
            The Q-value, or None if the reaction index is invalid
        """
        if 0 <= reaction_index < len(self.q_values):
            return self.q_values[reaction_index].value
        return None
    
    def get_all_q_values(self) -> List[float]:
        """
        Get all Q-values as a list of floats.
        
        Returns
        -------
        List[float]
            List of all Q-values
        """
        return [q.value for q in self.q_values]
