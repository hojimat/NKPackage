""" nkpack custom exceptions """

class InvalidParameterError(Exception):
    """User entered an invalid parameter"""

class InvalidBitstringError(Exception):
    """Bitstring size does not match the function"""

class UninitializedError(Exception):
    """Initialize the object before refering to it"""