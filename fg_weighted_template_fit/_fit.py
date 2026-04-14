from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from ._arrays import as_qu_map, as_template_stack, as_weight_map, weighted_inner_product
from ._filters import build_template_stack, smooth_and_filter_qu_map
from ._types import DifferenceTemplateInput, HarmonicFilter, WeightedFitResult


def weighted_template_gls(
    target_qu: npt.ArrayLike,
    templates_qu: npt.ArrayLike,
    weight_map: npt.ArrayLike,
    *,
    templates_rhs_qu: npt.ArrayLike | None = None,
    mask: npt.ArrayLike | None = None,
    template_names: Sequence[str] | None = None,
) -> WeightedFitResult:
    """Fit template amplitudes with a weighted normal equation.

    Parameters
    ----------
    target_qu
        Target Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    templates_qu
        Left-hand template stack with shape ``(n_template, 2, npix)`` or
        ``(n_template, npix, 2)``.
    weight_map
        Diagonal pixel weight map. Accepted shapes are scalar, ``(npix,)``,
        ``(2, npix)``, and ``(npix, 2)``.
    templates_rhs_qu
        Optional right-hand template stack. When supplied, the estimator
        becomes ``(T_left^T W T_right)^-1 T_left^T W m``. When omitted, the
        routine falls back to the same-template solve
        ``(T^T W T)^-1 T^T W m``.
    mask
        Optional pixel-domain exclusion mask with the same accepted shapes as
        ``weight_map``. The mask is converted to binary support before entering
        the weighted solve, so any finite nonzero value keeps a sample and zero
        removes it.
    template_names
        Optional names associated with the templates.

    Returns
    -------
    WeightedFitResult
        Fitted amplitudes, normal matrix, processed maps, and residual map.

    Raises
    ------
    ValueError
        If the target map and template stacks do not have compatible shapes.

    Notes
    -----
    Q and U pixels are stacked into a single data vector, but the computation is
    carried out in map form for clarity.
    """

    target = as_qu_map(target_qu, name="target_qu")
    templates_lhs = as_template_stack(templates_qu, name="templates_qu")
    if templates_rhs_qu is None:
        templates_rhs = templates_lhs.copy()
    else:
        templates_rhs = as_template_stack(
            templates_rhs_qu,
            name="templates_rhs_qu",
        )

    if templates_lhs.shape[1:] != target.shape:
        raise ValueError(
            "Each left-hand template must have the same Q/U shape as the target map."
        )
    if templates_rhs.shape[1:] != target.shape:
        raise ValueError(
            "Each right-hand template must have the same Q/U shape as the target map."
        )
    if templates_lhs.shape[0] != templates_rhs.shape[0]:
        raise ValueError(
            "Left and right template stacks must contain the same number of templates."
        )

    weights = as_weight_map(weight_map, npix=target.shape[1], name="weight_map")
    if mask is not None:
        weights = weights * _as_binary_fit_mask(mask, npix=target.shape[1])

    # Non-finite samples are dropped by zeroing their weights and the matching
    # left/right template and target entries, so the normal equations only see
    # valid samples on every side of the cross-estimator.
    valid = np.isfinite(target)
    valid &= np.isfinite(weights)
    valid &= np.isfinite(templates_lhs).all(axis=0)
    valid &= np.isfinite(templates_rhs).all(axis=0)

    weights = np.where(valid, weights, 0.0)
    target = np.where(valid, target, 0.0)
    templates_lhs = np.where(valid[None, :, :], templates_lhs, 0.0)
    templates_rhs = np.where(valid[None, :, :], templates_rhs, 0.0)

    n_template = templates_lhs.shape[0]
    normal_matrix = np.zeros((n_template, n_template), dtype=np.float64)
    rhs = np.zeros(n_template, dtype=np.float64)

    # Build the cross normal matrix one template pair at a time so the algebra
    # stays close to the estimator definition T_left^T W T_right and
    # T_left^T W m.
    for i in range(n_template):
        rhs[i] = weighted_inner_product(templates_lhs[i], target, weights)
        for j in range(n_template):
            value = weighted_inner_product(
                templates_lhs[i],
                templates_rhs[j],
                weights,
            )
            normal_matrix[i, j] = value

    try:
        amplitudes = np.linalg.solve(normal_matrix, rhs)
        solver = "solve"
    except np.linalg.LinAlgError:
        amplitudes = np.linalg.pinv(normal_matrix) @ rhs
        solver = "pinv"

    normal_matrix_inverse = np.linalg.pinv(normal_matrix)
    model_qu = np.tensordot(amplitudes, templates_rhs, axes=(0, 0))
    residual_qu = target - model_qu

    if template_names is None:
        names = tuple(f"template_{index}" for index in range(n_template))
    else:
        if len(template_names) != n_template:
            raise ValueError(
                "template_names must match the number of templates in templates_qu."
            )
        names = tuple(template_names)

    return WeightedFitResult(
        amplitudes=amplitudes,
        normal_matrix=normal_matrix,
        normal_matrix_inverse=normal_matrix_inverse,
        rhs=rhs,
        residual_qu=residual_qu,
        processed_target_qu=target,
        processed_templates_qu=templates_lhs,
        processed_templates_rhs_qu=templates_rhs,
        template_names=names,
        solver=solver,
    )


def fit_foreground_templates(
    target_qu: npt.ArrayLike,
    target_fwhm_in: float,
    template_inputs: Sequence[DifferenceTemplateInput],
    weight_map: npt.ArrayLike,
    fwhm_out: float,
    *,
    template_inputs_rhs: Sequence[DifferenceTemplateInput] | None = None,
    target_filter: HarmonicFilter | None = None,
    mask: npt.ArrayLike | None = None,
    nest: bool = False,
) -> WeightedFitResult:
    """Construct difference templates and fit their amplitudes to a target map.

    Parameters
    ----------
    target_qu
        Target Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    target_fwhm_in
        Beam FWHM of the target map in radians.
    template_inputs
        Sequence of left-hand template definitions.
    weight_map
        Diagonal pixel weight map used in the fit.
    fwhm_out
        Common output beam FWHM in radians applied to target and templates.
    template_inputs_rhs
        Optional sequence of right-hand template definitions. When supplied,
        the fit uses separate left- and right-hand template stacks in the
        normal matrix, ``(T_left^T W T_right)^-1 T_left^T W m``.
    target_filter
        Optional harmonic filter applied to the target map. Template entries may
        override this with their own ``filter_config`` values.
    mask
        Optional binary or floating mask applied once to the target and
        template input maps before any harmonic smoothing or filtering. This is
        intended for pre-apodizing map edges before the harmonic operations,
        not for modifying the final GLS weights.
    nest
        If ``True``, maps are treated as NEST ordered during harmonic
        transforms.

    Returns
    -------
    WeightedFitResult
        Weighted template-fit result for the processed target and template
        stacks.

    Raises
    ------
    ValueError
        If the left- and right-hand template lists have different lengths.
    """

    processed_target = smooth_and_filter_qu_map(
        target_qu,
        fwhm_in=target_fwhm_in,
        fwhm_out=fwhm_out,
        filter_config=target_filter,
        mask=mask,
        nest=nest,
    )
    processed_templates, template_names = build_template_stack(
        template_inputs=template_inputs,
        fwhm_out=fwhm_out,
        default_filter=target_filter,
        mask=mask,
        nest=nest,
    )
    if template_inputs_rhs is None:
        processed_templates_rhs = processed_templates
    else:
        processed_templates_rhs, template_names_rhs = build_template_stack(
            template_inputs=template_inputs_rhs,
            fwhm_out=fwhm_out,
            default_filter=target_filter,
            mask=mask,
            nest=nest,
        )
        if len(template_names_rhs) != len(template_names):
            raise ValueError(
                "template_inputs_rhs must contain the same number of templates as template_inputs."
            )
    return weighted_template_gls(
        target_qu=processed_target,
        templates_qu=processed_templates,
        templates_rhs_qu=processed_templates_rhs,
        weight_map=weight_map,
        template_names=template_names,
    )


def _as_binary_fit_mask(mask: npt.ArrayLike, *, npix: int) -> npt.NDArray[np.float64]:
    """Convert a general mask definition into binary support for the GLS step."""

    mask_map = as_weight_map(mask, npix=npix, name="mask")
    return np.where(np.isfinite(mask_map) & (mask_map != 0.0), 1.0, 0.0)
