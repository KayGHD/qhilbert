#include <cstdint>
#include <tuple>

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;
using namespace nb::literals;

// Bit-interleave: spread 16-bit value into even bits of 32-bit result
static inline uint32_t interleave(uint32_t x) {
    x = (x | (x << 8)) & 0x00FF00FFu;
    x = (x | (x << 4)) & 0x0F0F0F0Fu;
    x = (x | (x << 2)) & 0x33333333u;
    x = (x | (x << 1)) & 0x55555555u;
    return x;
}

// Parallel-prefix encode: (x, y) -> Hilbert distance
// Based on threadlocalmutex.com logarithmic algorithm
static uint32_t hilbert2D_encode_impl(uint32_t x, uint32_t y, uint32_t order) {
    const uint32_t mask = (1u << order) - 1u;
    x &= mask;
    y &= mask;

    uint32_t A, B, C, D;

    // Initial prefix scan round
    {
        const uint32_t a = x ^ y;
        const uint32_t b = mask ^ a;
        const uint32_t c = mask ^ (x | y);
        const uint32_t d = x & (y ^ mask);

        A = a | (b >> 1);
        B = (a >> 1) ^ a;
        C = ((c >> 1) ^ (b & (d >> 1))) ^ c;
        D = ((a & (c >> 1)) ^ (d >> 1)) ^ d;
    }

    // Prefix scan round 2 (shift 2)
    {
        const uint32_t a = A, b = B, c = C, d = D;
        A = (a & (a >> 2)) ^ (b & (b >> 2));
        B = (a & (b >> 2)) ^ (b & ((a ^ b) >> 2));
        C ^= (a & (c >> 2)) ^ (b & (d >> 2));
        D ^= (b & (c >> 2)) ^ ((a ^ b) & (d >> 2));
    }

    // Prefix scan round 3 (shift 4)
    {
        const uint32_t a = A, b = B, c = C, d = D;
        A = (a & (a >> 4)) ^ (b & (b >> 4));
        B = (a & (b >> 4)) ^ (b & ((a ^ b) >> 4));
        C ^= (a & (c >> 4)) ^ (b & (d >> 4));
        D ^= (b & (c >> 4)) ^ ((a ^ b) & (d >> 4));
    }

    // Prefix scan round 4 (shift 8)
    {
        const uint32_t a = A, b = B, c = C, d = D;
        C ^= (a & (c >> 8)) ^ (b & (d >> 8));
        D ^= (b & (c >> 8)) ^ ((a ^ b) & (d >> 8));
    }

    const uint32_t a = C ^ (C >> 1);
    const uint32_t b = D ^ (D >> 1);
    const uint32_t i0 = x ^ y;
    const uint32_t i1 = b | (mask ^ (i0 | a));

    return ((interleave(i1) << 1) | interleave(i0)) & ((1u << (2u * order)) - 1u);
}

// Parallel-prefix decode: Hilbert distance -> (x, y)
// Based on qHilbert / Hacker's Delight (pg. 365)
static std::tuple<uint32_t, uint32_t> hilbert2D_decode_impl(uint32_t dist, uint32_t order) {
    uint32_t s = dist;

    // Sentinel bits above valid range (safe for order < 16; wraps to 0 at order=16)
    if (order < 16) {
        s |= 0x55555555u << (2u * order);
    }

    const uint32_t sr = (s >> 1) & 0x55555555u;
    uint32_t cs = ((s & 0x55555555u) + sr) ^ 0x55555555u;

    // Parallel-prefix XOR propagation
    cs ^= (cs >> 2);
    cs ^= (cs >> 4);
    cs ^= (cs >> 8);
    cs ^= (cs >> 16);

    const uint32_t swap = cs & 0x55555555u;
    const uint32_t comp = (cs >> 1) & 0x55555555u;

    uint32_t t = (s & swap) ^ comp;
    s = s ^ sr ^ t ^ (t << 1);
    s &= ((1u << (2u * order)) - 1u);

    // Unshuffle (deinterleave) x into high 16, y into low 16
    t = (s ^ (s >> 1)) & 0x22222222u; s ^= t ^ (t << 1);
    t = (s ^ (s >> 2)) & 0x0C0C0C0Cu; s ^= t ^ (t << 2);
    t = (s ^ (s >> 4)) & 0x00F000F0u; s ^= t ^ (t << 4);
    t = (s ^ (s >> 8)) & 0x0000FF00u; s ^= t ^ (t << 8);

    return {s >> 16, s & 0xFFFFu};
}

NB_MODULE(_hilbert, m) {
    m.doc() = "Scalar Hilbert 2D encoding/decoding via parallel-prefix method. "
              "O(1) per element, branchless, no loops. "
              "Based on qHilbert/Hacker's Delight (decode) and "
              "threadlocalmutex.com (encode).";

    m.def("hilbert2D_encode",
        [](uint32_t x, uint32_t y, uint32_t order) -> uint32_t {
            return hilbert2D_encode_impl(x, y, order);
        },
        "x"_a, "y"_a, "order"_a = 16,
        "Encode 2D coordinates into a Hilbert curve distance.\n"
        "order: curve order (1..16), default 16. "
        "Coordinates must be in range 0..2^order-1.\n"
        "Returns distance in range 0..4^order-1.");

    m.def("hilbert2D_decode",
        [](uint32_t dist, uint32_t order) -> std::tuple<uint32_t, uint32_t> {
            return hilbert2D_decode_impl(dist, order);
        },
        "dist"_a, "order"_a = 16,
        "Decode a Hilbert curve distance into (x, y) coordinates.\n"
        "order: curve order (1..16), default 16. "
        "Distance must be in range 0..4^order-1.\n"
        "Returns (x, y) tuple.");
}
