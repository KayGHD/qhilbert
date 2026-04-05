# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python bindings for parallel-prefix Hilbert 2D encoding/decoding. Provides both scalar C++ bindings (via nanobind) and vectorized array operations (numpy/cupy) using the O(1) branchless parallel-prefix method from qHilbert/Hacker's Delight (decode) and threadlocalmutex.com (encode). 32-bit only, order 1-16.

## Build

```powershell
pip install .                        # standard build
pip install . --no-build-isolation   # if nanobind/scikit-build-core already installed
pip install . --force-reinstall      # rebuild after changes
```

Build requires: `scikit-build-core>=0.9.2`, `nanobind>=2.0.0`. No external C++ dependencies -- algorithms are implemented inline.

## Architecture

Two layers, both 2D-only (32-bit, parameterized order 1-16):

- **`_hilbert` (C++ extension)** - `ext/hilbert_ext.cpp` compiled via nanobind. Scalar encode/decode using parallel-prefix method. Built with `STABLE_ABI` + `NB_STATIC` (single `.pyd`, Python 3.12+).

- **`_vectorized.py`** - Pure Python array operations using the same parallel-prefix algorithms. Auto-detects numpy vs cupy via `_get_xp()` module inspection. Works on CPU or GPU transparently.

- **`__init__.py`** - Wires both layers. Scalar C++ functions keep original names (`hilbert2D_encode`). Vectorized functions get `_array` suffix (`hilbert2D_encode_array`).

## Src Layout

Uses `src/` layout to prevent local package shadowing installed package. Python package lives at `src/qhilbert/`, C++ extension source at `ext/`.
