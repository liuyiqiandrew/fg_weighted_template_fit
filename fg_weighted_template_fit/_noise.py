from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import numpy.typing as npt

from ._arrays import as_covariance, as_qu_map, coerce_rng
from ._fit import fit_foreground_templates
from ._types import (
    BootstrapFitResult,
    DifferenceTemplateInput,
    FloatArray,
    HarmonicFilter,
)


def bootstrap_template_amplitudes(
    target_qu: npt.ArrayLike,
    target_noise_cov: npt.ArrayLike,
    target_fwhm_in: float,
    template_inputs: Sequence[DifferenceTemplateInput],
    weight_map: npt.ArrayLike,
    fwhm_out: float,
    *,
    n_mc: int,
    template_inputs_rhs: Sequence[DifferenceTemplateInput] | None = None,
    target_filter: HarmonicFilter | None = None,
    mask: npt.ArrayLike | None = None,
    nest: bool = False,
    rng: np.random.Generator | int | None = None,
) -> BootstrapFitResult:
    """Estimate template amplitude uncertainty with Monte Carlo noise draws.

    Parameters
    ----------
    target_qu
        Target Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    target_noise_cov
        Per-pixel target-map covariance in the order ``QQ, UU, QU`` with shape
        ``(3, npix)`` or ``(npix, 3)``.
    target_fwhm_in
        Beam FWHM of the target map in radians.
    template_inputs
        Sequence of left-hand template definitions used in the fit.
    weight_map
        Diagonal pixel weight map used in the fit.
    fwhm_out
        Common output beam FWHM in radians.
    n_mc
        Number of Monte Carlo realizations.
    template_inputs_rhs
        Optional sequence of right-hand template definitions for the cross
        normal matrix.
    target_filter
        Optional harmonic filter applied to the target map. Template entries may
        override this with their own ``filter_config`` values.
    mask
        Optional binary or floating fit mask.
    nest
        If ``True``, maps are treated as NEST ordered during harmonic
        transforms.
    rng
        Existing random generator, integer seed, or ``None``.

    Returns
    -------
    BootstrapFitResult
        Reference fit together with the amplitude recovered from each Monte
        Carlo realization.

    Raises
    ------
    ValueError
        If ``n_mc`` is not positive.
    """

    if n_mc <= 0:
        raise ValueError("n_mc must be a positive integer.")

    reference_fit = fit_foreground_templates(
        target_qu=target_qu,
        target_fwhm_in=target_fwhm_in,
        template_inputs=template_inputs,
        weight_map=weight_map,
        fwhm_out=fwhm_out,
        template_inputs_rhs=template_inputs_rhs,
        target_filter=target_filter,
        mask=mask,
        nest=nest,
    )

    rng_obj = coerce_rng(rng)
    target_noise_cov_qu = as_covariance(
        target_noise_cov,
        npix=reference_fit.processed_target_qu.shape[1],
        name="target_noise_cov",
    )

    samples = np.zeros((n_mc, len(reference_fit.template_names)), dtype=np.float64)
    for draw_index in range(n_mc):
        # Realize noise on the native-resolution inputs first so every draw goes
        # through the same smoothing/filtering/template-construction pipeline.
        noisy_target = as_qu_map(target_qu, name="target_qu") + realize_qu_noise(
            target_noise_cov_qu,
            rng=rng_obj,
        )
        noisy_templates = tuple(
            _realize_noisy_template_input(template_input, rng_obj)
            for template_input in template_inputs
        )
        if template_inputs_rhs is None:
            noisy_templates_rhs = None
        else:
            noisy_templates_rhs = tuple(
                _realize_noisy_template_input(template_input, rng_obj)
                for template_input in template_inputs_rhs
            )
        draw_fit = fit_foreground_templates(
            target_qu=noisy_target,
            target_fwhm_in=target_fwhm_in,
            template_inputs=noisy_templates,
            weight_map=weight_map,
            fwhm_out=fwhm_out,
            template_inputs_rhs=noisy_templates_rhs,
            target_filter=target_filter,
            mask=mask,
            nest=nest,
        )
        samples[draw_index] = draw_fit.amplitudes

    ddof = 1 if n_mc > 1 else 0
    return BootstrapFitResult(
        reference_fit=reference_fit,
        amplitude_samples=samples,
        amplitude_mean=np.mean(samples, axis=0),
        amplitude_std=np.std(samples, axis=0, ddof=ddof),
        template_names=reference_fit.template_names,
    )


def realize_qu_noise(
    pixel_cov_qu: npt.ArrayLike,
    *,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Realize a Q/U noise map from per-pixel ``QQ, UU, QU`` covariance.

    Parameters
    ----------
    pixel_cov_qu
        Covariance array with shape ``(3, npix)`` or ``(npix, 3)`` in the
        order ``QQ, UU, QU``.
    rng
        Existing random generator, integer seed, or ``None``.

    Returns
    -------
    numpy.ndarray
        Noise realization with shape ``(2, npix)``.

    Raises
    ------
    ValueError
        If the covariance is not positive semi-definite on a per-pixel basis.
    """

    covariance = as_covariance(pixel_cov_qu, name="pixel_cov_qu")
    qq, uu, qu = covariance
    npix = qq.size

    cov_matrices = np.zeros((npix, 2, 2), dtype=np.float64)
    cov_matrices[:, 0, 0] = qq
    cov_matrices[:, 1, 1] = uu
    cov_matrices[:, 0, 1] = qu
    cov_matrices[:, 1, 0] = qu

    eigvals, eigvecs = np.linalg.eigh(cov_matrices)
    scale = max(np.max(np.abs(cov_matrices)), 1.0)
    tolerance = 1e-12 * scale
    if np.min(eigvals) < -tolerance:
        raise ValueError("pixel_cov_qu must be positive semi-definite per pixel.")

    # Each 2x2 pixel covariance is diagonalized, sampled with independent
    # standard normals, and rotated back into the native Q/U basis.
    eigvals = np.clip(eigvals, 0.0, None)
    rng_obj = coerce_rng(rng)
    standard_draw = rng_obj.standard_normal((npix, 2))
    scaled_draw = np.sqrt(eigvals) * standard_draw
    realized = np.einsum("pij,pj->pi", eigvecs, scaled_draw)
    return realized.T


def _realize_noisy_template_input(
    template_input: DifferenceTemplateInput,
    rng: np.random.Generator,
) -> DifferenceTemplateInput:
    """Add one noise realization to the two maps defining a template input.

    Parameters
    ----------
    template_input
        Template definition containing the input maps and optional per-pixel
        noise covariances.
    rng
        Random generator used for the noise draw.

    Returns
    -------
    DifferenceTemplateInput
        Copy of ``template_input`` with realized noise added to ``map_a_qu`` and
        ``map_b_qu`` when the corresponding covariance arrays are available.
    """

    map_a_qu = as_qu_map(template_input.map_a_qu, name="template_input.map_a_qu")
    map_b_qu = as_qu_map(template_input.map_b_qu, name="template_input.map_b_qu")

    if template_input.noise_cov_a is not None:
        map_a_qu = map_a_qu + realize_qu_noise(template_input.noise_cov_a, rng=rng)
    if template_input.noise_cov_b is not None:
        map_b_qu = map_b_qu + realize_qu_noise(template_input.noise_cov_b, rng=rng)

    return replace(
        template_input,
        map_a_qu=map_a_qu,
        map_b_qu=map_b_qu,
    )
