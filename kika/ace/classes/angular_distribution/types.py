from enum import Enum

class AngularDistributionType(Enum):
    """Enumeration of angular distribution types."""
    ISOTROPIC = 0
    EQUIPROBABLE = 1
    TABULATED = 2
    KALBACH_MANN = 3  # Law=44 distributions