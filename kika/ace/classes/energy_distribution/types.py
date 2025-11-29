from enum import Enum, auto

class EnergyDistributionType(Enum):
    """Enumeration of energy distribution types."""
    TABULAR = auto()
    DISCRETE = auto()
    LEVEL_SCATTERING = auto()
    EVAPORATION = auto()
    MAXWELL = auto()
    WATT = auto()
    KALBACH_MANN = auto()
    ANGLE_ENERGY = auto()
    N_BODY = auto()
    
    @classmethod
    def from_law(cls, law: int) -> 'EnergyDistributionType':
        """Get distribution type from law number."""
        law_to_type = {
            1: cls.TABULAR,
            2: cls.DISCRETE,
            3: cls.LEVEL_SCATTERING,
            4: cls.TABULAR,
            5: cls.EVAPORATION,
            7: cls.MAXWELL,
            9: cls.EVAPORATION,
            11: cls.WATT,
            22: cls.TABULAR,
            24: cls.TABULAR,
            44: cls.KALBACH_MANN,
            61: cls.ANGLE_ENERGY,
            66: cls.N_BODY,
            67: cls.ANGLE_ENERGY
        }
        return law_to_type.get(law, cls.TABULAR)