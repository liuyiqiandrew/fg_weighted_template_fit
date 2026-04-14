# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Weighted foreground template fitting for Healpix Q/U polarization maps. A lightweight alternative to a full inverse-covariance pipeline: smooth maps to a common beam, optionally apply harmonic (`ell`/`m`) filtering, build difference templates (e.g. `353 - 217`), and solve for template amplitudes via diagonal-weight normal equations. Includes Monte Carlo bootstrap for amplitude uncertainty.

## Commands

```bash
# Run full test suite
python -m pytest -q

# Run a single test
python -m pytest -q tests/test_fg_weighted_template_fit.py::test_name

# Run only tests that don't require healpy (fast, pure-numpy tests)
python -m pytest -q -m "not skipif"
```

There is no build step, linter, or formatter configured. The package is imported directly from the source tree (`import fg_weighted_template_fit as ftf`).

## Dependencies

- `numpy` (required)
- `healpy` (required for smoothing/harmonic filtering; algebraic fitting works without it)
- `tqdm` (optional, for `show_progress=True` in bootstrap)
- `pytest` (test suite)

## Architecture

The package is a single flat module under `fg_weighted_template_fit/` with no subpackages. All public symbols are re-exported through `__init__.py`.

### Module data flow

The pipeline has two layers: a **harmonic preprocessing** layer and a **pixel-space fitting** layer.

1. **`_types.py`** - Frozen dataclasses: `HarmonicFilter`, `DifferenceTemplateInput`, `WeightedFitResult`, `BootstrapFitResult`. `FloatArray` type alias lives here.

2. **`_arrays.py`** - Shape normalization (`as_qu_map`, `as_template_stack`, `as_weight_map`, `as_covariance`) that accepts `(2, npix)` or `(npix, 2)` and always returns `(2, npix)`. Also contains `weighted_inner_product` and `coerce_rng`. No healpy dependency.

3. **`_filters.py`** - Healpy-dependent harmonic operations. `smooth_and_filter_qu_map` is the single-map workhorse: it prepends a zero-T row, does one `map2alm`/`almxfl`/`alm2map` round-trip combining beam matching and `ell`-filter, then applies `m`-filter. `construct_difference_template` and `build_template_stack` call it for each input map. `_build_apodized_highpass` implements the NaMaster-style C1/C2 taper shared by both `ell` and `m` cutoffs.

4. **`_fit.py`** - Pixel-space fitting. `weighted_template_gls` builds the cross normal matrix `T_left^T W T_right` and solves via `np.linalg.solve` (falls back to `pinv` if singular). `fit_foreground_templates` is the high-level entry: preprocess target + templates, then call `weighted_template_gls`.

5. **`_noise.py`** - `realize_qu_noise` does per-pixel 2x2 Cholesky-style draws from `(QQ, UU, QU)` covariance. `bootstrap_template_amplitudes` runs `n_mc` noisy refits through the full pipeline, storing every amplitude draw.

### Key design patterns

- **Cross-estimator**: Separate left/right template stacks (`template_inputs` vs `template_inputs_rhs`) avoid noise bias in the normal matrix auto-term. When `template_inputs_rhs` is omitted, both sides use the same stack.
- **Pre-harmonic masking**: When `mask` is supplied, it is applied in pixel space *before* the harmonic transform (to apodize edges), then again in the pixel-space fit weights.
- **healpy is optional at import**: `_filters.py` catches `ImportError` so the fitting algebra can be used without healpy installed. Tests that need healpy use `@pytest.mark.skipif`.

## Data Conventions

- Q/U maps: `(2, npix)` canonical, `(npix, 2)` accepted and transposed automatically.
- Per-pixel covariance order: `QQ, UU, QU` with shape `(3, npix)`.
- All beam FWHM values are in **radians**.
- Healpix default ordering is RING; pass `nest=True` for NEST.

# Code style
- Use type hinting
- Write numpy-style docstrings for functions and comment the code
- Use black to format code
- Provide unit test for functions