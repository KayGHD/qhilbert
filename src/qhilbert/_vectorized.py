"""
Vectorized Hilbert 2D encoding/decoding using parallel-prefix method.

O(1) per element, branchless, no loops. Works with both numpy and cupy
arrays. The array module is detected automatically from the input type.

Encode based on threadlocalmutex.com logarithmic algorithm.
Decode based on qHilbert / Hacker's Delight (pg. 365).
"""

import numpy as np


def _get_xp(*arrays):
    """Return the array module (numpy or cupy) for the given arrays."""
    for arr in arrays:
        mod = type(arr).__module__.split(".")[0]
        if mod == "cupy":
            import cupy
            return cupy
    return np


def _spread_bits_32(v, xp):
    """Spread bits: interleave value into even bit positions (32-bit)."""
    v = v.astype(xp.uint32)
    v = (v | (v << xp.uint32(8))) & xp.uint32(0x00FF00FF)
    v = (v | (v << xp.uint32(4))) & xp.uint32(0x0F0F0F0F)
    v = (v | (v << xp.uint32(2))) & xp.uint32(0x33333333)
    v = (v | (v << xp.uint32(1))) & xp.uint32(0x55555555)
    return v


# ---------- Encode (xy -> distance) ----------

def hilbert2D_encode(x, y, order=16):
    """Encode arrays of 2D coordinates into Hilbert curve distances.

    Args:
        x: Array of x coordinates (0..2^order-1).
        y: Array of y coordinates (0..2^order-1).
        order: Curve order (1..16), default 16.

    Returns:
        Array of uint32 Hilbert distances.

    Works with numpy and cupy arrays.
    """
    xp = _get_xp(x, y)
    u32 = xp.uint32

    mask = u32((1 << order) - 1)
    x = x.astype(u32) & mask
    y = y.astype(u32) & mask

    # Initial prefix scan round
    a = x ^ y
    b = mask ^ a
    c = mask ^ (x | y)
    d = x & (y ^ mask)

    A = a | (b >> u32(1))
    B = (a >> u32(1)) ^ a
    C = ((c >> u32(1)) ^ (b & (d >> u32(1)))) ^ c
    D = ((a & (c >> u32(1))) ^ (d >> u32(1))) ^ d

    # Prefix scan round 2 (shift 2)
    a, b, c, d = A.copy(), B.copy(), C.copy(), D.copy()
    A = (a & (a >> u32(2))) ^ (b & (b >> u32(2)))
    B = (a & (b >> u32(2))) ^ (b & ((a ^ b) >> u32(2)))
    C ^= (a & (c >> u32(2))) ^ (b & (d >> u32(2)))
    D ^= (b & (c >> u32(2))) ^ ((a ^ b) & (d >> u32(2)))

    # Prefix scan round 3 (shift 4)
    a, b, c, d = A.copy(), B.copy(), C.copy(), D.copy()
    A = (a & (a >> u32(4))) ^ (b & (b >> u32(4)))
    B = (a & (b >> u32(4))) ^ (b & ((a ^ b) >> u32(4)))
    C ^= (a & (c >> u32(4))) ^ (b & (d >> u32(4)))
    D ^= (b & (c >> u32(4))) ^ ((a ^ b) & (d >> u32(4)))

    # Prefix scan round 4 (shift 8)
    a, b = A, B
    c, d = C.copy(), D.copy()
    C ^= (a & (c >> u32(8))) ^ (b & (d >> u32(8)))
    D ^= (b & (c >> u32(8))) ^ ((a ^ b) & (d >> u32(8)))

    # Undo transformation prefix scan
    a = C ^ (C >> u32(1))
    b = D ^ (D >> u32(1))

    # Recover index bits
    i0 = x ^ y
    i1 = b | (mask ^ (i0 | a))

    result = (_spread_bits_32(i1, xp) << u32(1)) | _spread_bits_32(i0, xp)
    return result & u32((1 << (2 * order)) - 1)


# ---------- Decode (distance -> xy) ----------

def hilbert2D_decode(distances, order=16):
    """Decode arrays of Hilbert curve distances into 2D coordinates.

    Args:
        distances: Array of uint32 Hilbert distances (0..4^order-1).
        order: Curve order (1..16), default 16.

    Returns:
        Tuple of (x, y) arrays with uint16 dtype.

    Works with numpy and cupy arrays.
    """
    xp = _get_xp(distances)
    u32 = xp.uint32

    s = distances.astype(u32)

    # Sentinel bits above valid range
    if order < 16:
        s = s | (u32(0x55555555) << u32(2 * order))

    sr = (s >> u32(1)) & u32(0x55555555)
    cs = ((s & u32(0x55555555)) + sr) ^ u32(0x55555555)

    # Parallel-prefix XOR propagation
    cs = cs ^ (cs >> u32(2))
    cs = cs ^ (cs >> u32(4))
    cs = cs ^ (cs >> u32(8))
    cs = cs ^ (cs >> u32(16))

    swap = cs & u32(0x55555555)
    comp = (cs >> u32(1)) & u32(0x55555555)

    t = (s & swap) ^ comp
    s = s ^ sr ^ t ^ (t << u32(1))
    s = s & u32((1 << (2 * order)) - 1)

    # Unshuffle (deinterleave): x into high 16 bits, y into low 16
    t = (s ^ (s >> u32(1))) & u32(0x22222222); s = s ^ t ^ (t << u32(1))
    t = (s ^ (s >> u32(2))) & u32(0x0C0C0C0C); s = s ^ t ^ (t << u32(2))
    t = (s ^ (s >> u32(4))) & u32(0x00F000F0); s = s ^ t ^ (t << u32(4))
    t = (s ^ (s >> u32(8))) & u32(0x0000FF00); s = s ^ t ^ (t << u32(8))

    x = (s >> u32(16)).astype(xp.uint16)
    y = (s & u32(0xFFFF)).astype(xp.uint16)
    return x, y
