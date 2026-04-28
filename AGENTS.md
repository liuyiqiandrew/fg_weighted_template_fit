# Repository Guidelines

## Project Structure & Module Organization
This is a small Python package for weighted foreground template fitting of Healpix Q/U maps. Source code lives in `fg_weighted_template_fit/`, with public exports collected in `__init__.py`. Core modules are split by responsibility: `_types.py` for dataclasses, `_arrays.py` for shape and RNG helpers, `_filters.py` for Healpix smoothing and harmonic filters, `_fit.py` for weighted solves, and `_noise.py` for Monte Carlo uncertainty. Tests live in `tests/`, and API notes live in `docs/api.md`.

## Build, Test, and Development Commands
There is no configured build step; import the package directly from the repository root.

- `python -m pytest -q`: run the full test suite.
- `python -m pytest -q tests/test_fg_weighted_template_fit.py::test_name`: run one test.
- `python -m pytest -q -m "not skipif"`: run the fast tests that avoid optional Healpix-dependent paths.

Required development dependencies are `numpy` and `pytest`; `healpy` is needed for smoothing/filtering tests, and `tqdm` enables bootstrap progress bars.

## Coding Style & Naming Conventions
Use Python 3 style with `from __future__ import annotations`, type hints, and NumPy arrays coerced to `float64` where numerical routines expect them. Keep modules flat and focused. Use snake_case for functions, variables, and test names; use PascalCase for dataclasses such as `HarmonicFilter`. Prefer concise docstrings that document shapes, units, and failure modes.

## Testing Guidelines
Tests use `pytest` and `numpy.testing`. Name tests `test_<behavior>` and keep fixtures local unless reuse becomes necessary. Cover numerical behavior with deterministic arrays or seeded RNGs. Mark Healpy-dependent tests with `@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")`.

## Commit & Pull Request Guidelines
Recent commits use short, imperative, lowercase summaries such as `fix filter bug, lowpass to highpass`. Keep commits focused on one behavior change. Pull requests should describe the scientific or API impact, list the tests run, and call out changes to array shapes, units, mask handling, or optional dependencies.
