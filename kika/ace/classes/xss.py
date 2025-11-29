class XssEntry:
    def __init__(self, index: int, value: float):
        self.index = index
        self.value = value

    def __repr__(self):
        return f"XssEntry(Index={self.index}, Value={self.value})"
    
    def __eq__(self, other):
        """Compare two XssEntry objects based on their values, not their identity."""
        if not isinstance(other, XssEntry):
            return False
        return (self.index == other.index) and (self.value == other.value)
    
    def __hash__(self):
        """Generate a hash based on the index and value, allowing XssEntry to be used in sets and as dict keys."""
        return hash((self.index, self.value))
