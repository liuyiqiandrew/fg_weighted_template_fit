# API Reference

This document summarizes the main public objects exposed by
`fg_weighted_template_fit`.

## Core Dataclasses

### `HarmonicFilter`

Configuration for optional harmonic-domain filtering.

Fields:

- `ell_filter`: explicit multiplicative transfer function indexed by `ell`
- `m_filter`: explicit multiplicative transfer function indexed by `m`
- `ell_cutoff`: optional high-pass multipole cutoff
- `ell_halfwidth`: half-width of the smooth `ell` transition
- `m_cutoff`: optional high-pass azimuthal cutoff
- `m_halfwidth`: half-width of the smooth `m` transition
- `transition_type`: `"C1"` or `"C2"` smooth edge, default `"C2"`
- `lmax`: optional explicit harmonic truncation
- `iter`: number of `healpy.map2alm` iterations

Usage notes:

- If both an explicit filter and a cutoff are provided, both are applied.
- `build_ell_filter` and `build_m_filter` return explicit windows that can be
  passed to `ell_filter` and `m_filter`.
- `ell_cutoff` and `m_cutoff` suppress low modes but do not lower the transform
  `lmax`.
- All beam widths used with filtering are in radians.

### `DifferenceTemplateInput`

Definition of one template built from a difference of two Q/U maps.

Fields:

- `map_a_qu`
- `map_b_qu`
- `fwhm_in_a`
- `fwhm_in_b`
- `noise_cov_a`
- `noise_cov_b`
- `filter_config`
- `name`

Usage notes:

- The template is built as `processed(map_a_qu) - processed(map_b_qu)`.
- Noise covariances are optional, but they are needed for Monte Carlo
  uncertainty propagation through template construction.

### `WeightedFitResult`

Container returned by the weighted fit.

Fields:

- `amplitudes`
- `normal_matrix`
- `normal_matrix_inverse`
- `rhs`
- `residual_qu`
- `processed_target_qu`
- `processed_templates_qu`
- `processed_templates_rhs_qu`
- `template_names`
- `solver`

### `BootstrapFitResult`

Container returned by the Monte Carlo uncertainty routine.

Fields:

- `reference_fit`
- `amplitude_samples`
- `amplitude_mean`
- `amplitude_std`
- `template_names`

## Main Functions

### `build_ell_filter`

```python
build_ell_filter(
    lmax,
    *,
    cutoff,
    halfwidth=0.0,
    transition_type="C2",
)
```

Builds a reusable high-pass `ell`-space transfer function with shape
`(lmax + 1,)`.

Important behavior:

- returns the same taper shape used by `HarmonicFilter.ell_cutoff`
- uses a hard cutoff when `halfwidth == 0`
- supports `C1` and `C2` smooth edges
- is intended for `HarmonicFilter(ell_filter=...)`

### `build_m_filter`

```python
build_m_filter(
    lmax,
    *,
    cutoff,
    halfwidth=0.0,
    transition_type="C2",
)
```

Builds a reusable high-pass `m`-space transfer function with shape
`(lmax + 1,)`.

Important behavior:

- returns the same taper shape used by `HarmonicFilter.m_cutoff`
- uses a hard cutoff when `halfwidth == 0`
- supports `C1` and `C2` smooth edges
- is intended for `HarmonicFilter(m_filter=...)`

### `smooth_and_filter_qu_map`

```python
smooth_and_filter_qu_map(
    qu_map,
    fwhm_in,
    fwhm_out,
    *,
    filter_config=None,
    mask=None,
    nest=False,
)
```

Applies beam matching and optional harmonic filtering to a Q/U Healpix map.

Important behavior:

- accepts shape `(2, npix)` or `(npix, 2)`
- raises if `fwhm_out < fwhm_in`
- when a mask is supplied, applies it in pixel space before the harmonic
  transform used for smoothing/filtering
- uses a single alm pass to combine smoothing and filtering
- supports both RING and NEST ordering

### `construct_difference_template`

```python
construct_difference_template(
    map_a_qu,
    map_b_qu,
    fwhm_in_a,
    fwhm_in_b,
    fwhm_out,
    *,
    filter_config=None,
    mask=None,
    nest=False,
)
```

Builds a foreground template from two Q/U maps after matching both to the same
output resolution and filter definition.

### `build_template_stack`

```python
build_template_stack(
    template_inputs,
    *,
    fwhm_out,
    default_filter=None,
    mask=None,
    nest=False,
)
```

Constructs a stack of processed difference templates and returns
`(templates, template_names)`.

### `weighted_template_gls`

```python
weighted_template_gls(
    target_qu,
    templates_qu,
    weight_map,
    *,
    templates_rhs_qu=None,
    mask=None,
    template_names=None,
)
```

Solves the weighted normal equations

```text
(T_left^T W T_right) a = T_left^T W m
```

with Q/U pixels stacked into one data vector.

Accepted weight shapes:

- `(npix,)`
- `(2, npix)`
- `(npix, 2)`
- scalar

Important behavior:

- non-finite target, template, and weight entries are automatically removed
- if a mask is supplied, the solver converts it to binary support instead of
  using apodized mask values as extra weights
- `templates_qu` is the left-hand template stack
- `templates_rhs_qu` is the right-hand template stack
- if `templates_rhs_qu` is omitted, the routine falls back to the
  same-template normal matrix
- if the normal matrix is singular, the routine falls back to a pseudoinverse

### `fit_foreground_templates`

```python
fit_foreground_templates(
    target_qu,
    target_fwhm_in,
    template_inputs,
    weight_map,
    fwhm_out,
    *,
    template_inputs_rhs=None,
    target_filter=None,
    mask=None,
    nest=False,
)
```

High-level entry point that:

- smooths and filters the target map
- constructs the left-hand template stack
- optionally constructs an independent right-hand template stack
- solves for the weighted template amplitudes

Procedure overview:

1. The target map is processed first with `smooth_and_filter_qu_map`.
2. Inside that per-map processing step, any NEST input is reordered to RING in
   pixel space before harmonic transforms.
3. If `mask` is supplied, that same mask is first applied in pixel space to
   the target map before any harmonic transform so an apodized edge can reduce
   truncation ringing.
4. Beam matching from `target_fwhm_in` to `fwhm_out` and any `ell`-space filter
   are combined into one multiplicative transfer function in harmonic space.
5. Any `m`-space filter is then applied in harmonic space after the `ell`-space
   transfer and before transforming back to map space.
6. The filtered target is transformed back to Q/U pixel space, and converted
   back to NEST ordering if requested.
7. The left-hand template stack is built next. For each
   `DifferenceTemplateInput`, `map_a_qu` and `map_b_qu` are each processed
   through the same masked smooth-and-filter pipeline, using the entry's own
   `filter_config` when present or `target_filter` otherwise.
8. Each template difference `processed(map_a_qu) - processed(map_b_qu)` is
   formed in pixel space after both maps have already been beam-matched and
   harmonically filtered.
9. If `template_inputs_rhs` is omitted, the right-hand stack reuses the
   already-built left-hand templates. If it is supplied, the right-hand stack is
   built afterward with the same per-template ordering.
10. Only after all target and template preprocessing is complete does the
   routine enter `weighted_template_gls`, where the user-supplied weights are
   applied in pixel space, non-finite samples are zero-weighted, the weighted
   normal matrix and right-hand side are accumulated, and the amplitudes are
   solved.

Space/order notes:

- Harmonic-space work is limited to the per-map preprocessing inside
  `smooth_and_filter_qu_map`.
- Pixel-space work covers NEST reordering, template differencing, pre-harmonic
  apodization, weight application, normal-matrix accumulation, model
  construction, and residual formation.
- The code never subtracts `map_a_qu - map_b_qu` in harmonic space, and it does
  not apply the fit weights in harmonic space.

### `realize_qu_noise`

```python
realize_qu_noise(
    pixel_cov_qu,
    *,
    rng=None,
)
```

Draws a Q/U noise realization from per-pixel covariance in the order
`QQ`, `UU`, `QU`.

Accepted covariance shapes:

- `(3, npix)`
- `(npix, 3)`

### `bootstrap_template_amplitudes`

```python
bootstrap_template_amplitudes(
    target_qu,
    target_noise_cov,
    target_fwhm_in,
    template_inputs,
    weight_map,
    fwhm_out,
    *,
    n_mc,
    template_inputs_rhs=None,
    target_filter=None,
    mask=None,
    nest=False,
    rng=None,
    show_progress=False,
    n_jobs=1,
)
```

Runs Monte Carlo amplitude estimation by:

1. realizing noise for the target map
2. realizing noise for the maps used to construct each left-hand template
3. optionally realizing noise for the independent right-hand template stack
4. rebuilding templates at the target output resolution
5. reapplying harmonic filtering
6. refitting amplitudes
7. storing the recovered amplitudes from every draw

The output spread is reported through `amplitude_std`, and every draw is kept in
`amplitude_samples`.

Usage note:

- `amplitude_std` includes template-map uncertainty when the corresponding
  `DifferenceTemplateInput` entries provide `noise_cov_a` and `noise_cov_b`
- if those template noise covariances are omitted, the Monte Carlo spread only
  reflects target-map noise and any uncertainty induced by the supplied target
  covariance
- `show_progress=True` displays a standard `tqdm` progress bar over the Monte
  Carlo draws and does not depend on ipywidgets
- `n_jobs > 1` runs independent Monte Carlo draws with worker threads, which can
  be used from JupyterLab without multiprocessing setup
- threaded runs are reproducible for the same `rng` and `n_jobs`, but are not
  required to match the exact serial sample sequence

The parallel execution design is documented in
[`parallel_bootstrap.md`](./parallel_bootstrap.md).

## Data Conventions

- Q/U maps may be passed as `(2, npix)` or `(npix, 2)`.
- Template stacks may be passed as `(n_template, 2, npix)` or
  `(n_template, npix, 2)`.
- Per-pixel covariance must be ordered as `QQ`, `UU`, `QU`.
- FWHM values are always in radians.

## Internal Helpers

The package re-exports `_build_apodized_highpass` for testing and inspection.
It is not intended to be the main user-facing entry point, but it can be useful
when validating filter shapes.
