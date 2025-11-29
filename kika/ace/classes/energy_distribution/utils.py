class EnergyDistributionError(Exception):
    """Base exception for energy distribution errors."""
    pass

class LawNotFoundError(EnergyDistributionError):
    """Raised when a requested law is not found."""
    pass

class EnergyOutOfRangeError(EnergyDistributionError):
    """Raised when energy is outside the valid range."""
    pass