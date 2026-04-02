# fg_weighted_template_fit

Weighted foreground template fitting for Healpix Q/U maps.

This repository provides a lightweight alternative to a full inverse-covariance
pipeline when the fitting problem is already expressed through a user-supplied
weight map. The core workflow is:

1. Smooth target and template-construction maps to a common beam.
2. Optionally apply harmonic filtering in `ell` and `m`.
3. Build foreground templates from map differences such as `353 - 217`.
4. Estimate template amplitudes with a weighted normal equation.
5. Bootstrap amplitude uncertainty with Monte Carlo noise realizations built
   from per-pixel `QQ`, `UU`, and `QU` covariances.

The main package is [`fg_weighted_template_fit/`](./fg_weighted_template_fit).

## Method

For a target polarization map `m` and left/right template stacks
`T_left`, `T_right`, the fitted amplitudes are

```text
a_hat = (T_left^T W T_right)^(-1) T_left^T W m
```

where:

- `m` is the target map with Q and U pixels stacked into one data vector
- `T_left` and `T_right` are matrices of stacked template maps
- `W` is a diagonal weight map supplied by the user

Using separate left and right template realizations is useful when you want to
avoid noise bias from a template auto-term in the normal matrix. If no separate
right-hand stack is supplied, the package falls back to the same-template solve

```text
a_hat = (T^T W T)^(-1) T^T W m
```

This is still not the fully optimal generalized least-squares estimator you
would get from a full inverse covariance matrix, but it is often a useful fast
estimator when a scalar or diagonal weight definition is already available and
you want a swift amplitude estimate.

## Repository Layout

```text
fg_weighted_template_fit/
├── README.md
├── docs/
│   └── api.md
├── fg_weighted_template_fit/
│   ├── __init__.py
│   ├── _arrays.py
│   ├── _filters.py
│   ├── _fit.py
│   ├── _noise.py
│   └── _types.py
└── tests/
    └── test_fg_weighted_template_fit.py
```

Module responsibilities:

- `_types.py`: dataclasses for filters, template definitions, and fit results
- `_filters.py`: Healpix smoothing, `ell`/`m` filtering, and template building
- `_fit.py`: weighted template solving
- `_noise.py`: Q/U noise realizations and Monte Carlo uncertainty estimation
- `_arrays.py`: array-shape normalization helpers

## Main Features

- Difference-template construction for split maps such as dust or synchrotron
- Common-beam matching from input `fwhm_in` to output `fwhm_out`
- Optional harmonic filtering in both `ell` and `m`
- Smooth `ell` and `m` cutoffs with `C1` or `C2` apodized edges
- Weighted diagonal GLS-like solve for template amplitudes
- Bootstrap uncertainty estimation from per-pixel `QQ`, `UU`, `QU` covariance
- Storage of the recovered amplitude from every Monte Carlo realization

## Key Assumptions

- Maps are Healpix Q/U polarization maps.
- Beam widths are given in radians.
- Template maps are built as a difference of two Q/U maps after smoothing to a
  common resolution.
- Noise covariance is provided per pixel in the order `QQ`, `UU`, `QU`.
- The fit weight is diagonal in pixel space. This package does not implement a
  dense inverse covariance solve.

## Dependencies

- `numpy`
- `healpy` for smoothing and harmonic filtering
- `pytest` for the test suite

The purely algebraic fitting functions can still be imported without `healpy`,
but smoothing and harmonic filtering require it.

## Quick Start

```python
import numpy as np
import fg_weighted_template_fit as ftf

# Example target map and diagonal weights.
target_qu = np.random.standard_normal((2, 12 * 8**2))
weight_map = np.ones(target_qu.shape[1])

dust_split_a = ftf.DifferenceTemplateInput(
    map_a_qu=planck_353_qu,
    map_b_qu=planck_217_qu,
    fwhm_in_a=fwhm_353_rad,
    fwhm_in_b=fwhm_217_rad,
    noise_cov_a=planck_353_cov,
    noise_cov_b=planck_217_cov,
    name="dust",
)

dust_split_b = ftf.DifferenceTemplateInput(
    map_a_qu=planck_353_split_b_qu,
    map_b_qu=planck_217_split_b_qu,
    fwhm_in_a=fwhm_353_rad,
    fwhm_in_b=fwhm_217_rad,
    noise_cov_a=planck_353_split_b_cov,
    noise_cov_b=planck_217_split_b_cov,
    name="dust",
)

sync_split_a = ftf.DifferenceTemplateInput(
    map_a_qu=wm23_qu,
    map_b_qu=ka23_qu,
    fwhm_in_a=fwhm_w_rad,
    fwhm_in_b=fwhm_ka_rad,
    noise_cov_a=wm23_cov,
    noise_cov_b=ka23_cov,
    name="sync",
)

sync_split_b = ftf.DifferenceTemplateInput(
    map_a_qu=wm23_split_b_qu,
    map_b_qu=ka23_split_b_qu,
    fwhm_in_a=fwhm_w_rad,
    fwhm_in_b=fwhm_ka_rad,
    noise_cov_a=wm23_split_b_cov,
    noise_cov_b=ka23_split_b_cov,
    name="sync",
)

filter_config = ftf.HarmonicFilter(
    ell_cutoff=180.0,
    ell_halfwidth=20.0,
    m_cutoff=64.0,
    m_halfwidth=8.0,
    transition_type="C2",
)

result = ftf.fit_foreground_templates(
    target_qu=target_qu,
    target_fwhm_in=target_fwhm_rad,
    template_inputs=[dust_split_a, sync_split_a],
    template_inputs_rhs=[dust_split_b, sync_split_b],
    weight_map=weight_map,
    fwhm_out=common_fwhm_rad,
    target_filter=filter_config,
)

print(result.template_names)
print(result.amplitudes)
```

## Monte Carlo Uncertainty Example

The Monte Carlo routine can propagate uncertainty from both the target map and
the maps used to build the templates. Template uncertainty is included when
each `DifferenceTemplateInput` carries `noise_cov_a` and `noise_cov_b`.

```python
bootstrap = ftf.bootstrap_template_amplitudes(
    target_qu=target_qu,
    target_noise_cov=target_noise_cov,
    target_fwhm_in=target_fwhm_rad,
    template_inputs=[dust_split_a, sync_split_a],
    template_inputs_rhs=[dust_split_b, sync_split_b],
    weight_map=weight_map,
    fwhm_out=common_fwhm_rad,
    n_mc=200,
    target_filter=filter_config,
    rng=1234,
    show_progress=True,
)

print(bootstrap.amplitude_mean)
print(bootstrap.amplitude_std)
print(bootstrap.amplitude_samples.shape)
```

Pass `show_progress=True` to show a standard `tqdm` progress bar in notebooks
or terminals while the Monte Carlo draws are running, without relying on
ipywidgets.

In the example above, the reported `bootstrap.amplitude_std` includes:

- target-map noise from `target_noise_cov`
- template-map noise from `noise_cov_a` and `noise_cov_b` on each template input
- the effect of rebuilding the templates after adding those noise realizations

If template noise covariances are omitted, the Monte Carlo spread will only
reflect target-map noise and will therefore underestimate the total uncertainty
associated with noisy templates.

## Filtering Options

`HarmonicFilter` supports two styles of harmonic filtering:

- Explicit transfer arrays via `ell_filter` and `m_filter`
- Smooth high-pass cutoffs via `ell_cutoff`, `ell_halfwidth`, `m_cutoff`,
  `m_halfwidth`, and `transition_type`

For cutoff-based filters:

- modes below `cutoff - halfwidth` are set to zero
- modes above `cutoff + halfwidth` pass unchanged
- the transition band uses a NaMaster-style `C1` or `C2` edge
- the default transition type is `C2`

## Public API

Most users will interact with:

- `HarmonicFilter`
- `DifferenceTemplateInput`
- `fit_foreground_templates`
- `bootstrap_template_amplitudes`
- `construct_difference_template`
- `smooth_and_filter_qu_map`

If you want the cross-template estimator specifically, pass the left-hand split
through `template_inputs` and the independent right-hand split through
`template_inputs_rhs`.

A more detailed API reference is available in [`docs/api.md`](./docs/api.md).

## Testing

Run the tests from the repository root:

```bash
python -m pytest -q
```

The current test suite covers:

- recovery of known template amplitudes
- Q/U noise realization from requested covariance
- Monte Carlo sample storage and nonzero uncertainty
- smoothing and `m`-filter integration
- `C1` and `C2` taper behavior for smooth cutoffs

## Notes

- Importing `healpy` may trigger local Matplotlib cache warnings in restricted
  environments. Those do not affect the numerical routines.
- The repository currently focuses on the fitting utilities themselves rather
  than file I/O. You are expected to load maps and noise covariances upstream.
