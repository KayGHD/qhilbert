"""
qhilbert - Parallel-prefix Hilbert 2D encoding/decoding for Python.

All functions take an `order` parameter (1..16, default 16) controlling
the curve resolution. Coordinates range 0..2^order-1, distances 0..4^order-1.

Scalar functions (single values, C++ parallel-prefix):
    hilbert2D_encode(x, y, order=16)   -> int
    hilbert2D_decode(dist, order=16)   -> (x, y)

Vectorized functions (numpy/cupy arrays, parallel-prefix):
    hilbert2D_encode_array(x, y, order=16)   -> array
    hilbert2D_decode_array(dist, order=16)    -> (x_array, y_array)
"""

from qhilbert._hilbert import (
    hilbert2D_decode,
    hilbert2D_encode,
)
from qhilbert._vectorized import (
    hilbert2D_decode as hilbert2D_decode_array,
    hilbert2D_encode as hilbert2D_encode_array,
)

__all__ = [
    "hilbert2D_decode",
    "hilbert2D_decode_array",
    "hilbert2D_encode",
    "hilbert2D_encode_array",
]

__version__ = "0.1.0"
