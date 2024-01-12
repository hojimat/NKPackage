"""
`nkpack` is library for agent-based modeling using NK framework.
It provides tools to handle binary variables, generate landscapes, find maxima.
The library heavily uses `numpy` for its data types, and is annotated with type hints.
The library also uses `numba` to speed up global maximum search.

"""

from .bitstrings import *
from .exceptions import *
from .helpers import *
from .interactions import *
from .landscapes import *
from .metrics import *
