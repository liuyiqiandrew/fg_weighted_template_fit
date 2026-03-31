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
    mask: npt.ArrayLike | None = None,
    template_names: Sequence[str] | None = None,
) -> WeightedFitResult:
    """Fit template amplitudes with a weighted normal equation."""

    target = as_qu_map(target_qu, name="target_qu")
    templates = as_template_stack(templates_qu, name="templates_qu")

    if templates.shape[1:] != target.shape:
        raise ValueError(
            "Each template must have the same Q/U shape as the target map."
        )

    weights = as_weight_map(weight_map, npix=target.shape[1], name="weight_map")
    if mask is not None:
        weights = weights * as_weight_map(mask, npix=target.shape[1], name="mask")

    # Non-finite samples are dropped by zeroing their weights and the matching
    # template/target entries, so the normal equations only see valid pixels.
    valid = np.isfinite(target)
    valid &= np.isfinite(weights)
    valid &= np.isfinite(templates).all(axis=0)

    weights = np.where(valid, weights, 0.0)
    target = np.where(valid, target, 0.0)
    templates = np.where(valid[None, :, :], templates, 0.0)

    n_template = templates.shape[0]
    normal_matrix = np.zeros((n_template, n_template), dtype=np.float64)
    rhs = np.zeros(n_template, dtype=np.float64)

    # Build the weighted Gram matrix one template pair at a time so the algebra
    # stays close to the estimator definition T^T W T and T^T W m.
    for i in range(n_template):
        rhs[i] = weighted_inner_product(templates[i], target, weights)
        for j in range(i, n_template):
            value = weighted_inner_product(templates[i], templates[j], weights)
            normal_matrix[i, j] = value
            normal_matrix[j, i] = value

    try:
        amplitudes = np.linalg.solve(normal_matrix, rhs)
        solver = "solve"
    except np.linalg.LinAlgError:
        amplitudes = np.linalg.pinv(normal_matrix) @ rhs
        solver = "pinv"

    normal_matrix_inverse = np.linalg.pinv(normal_matrix)
    model_qu = np.tensordot(amplitudes, templates, axes=(0, 0))
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
        processed_templates_qu=templates,
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
    target_filter: HarmonicFilter | None = None,
    mask: npt.ArrayLike | None = None,
    nest: bool = False,
) -> WeightedFitResult:
    """Construct difference templates and fit their amplitudes to a target map."""

    processed_target = smooth_and_filter_qu_map(
        target_qu,
        fwhm_in=target_fwhm_in,
        fwhm_out=fwhm_out,
        filter_config=target_filter,
        nest=nest,
    )
    processed_templates, template_names = build_template_stack(
        template_inputs=template_inputs,
        fwhm_out=fwhm_out,
        default_filter=target_filter,
        nest=nest,
    )
    return weighted_template_gls(
        target_qu=processed_target,
        templates_qu=processed_templates,
        weight_map=weight_map,
        mask=mask,
        template_names=template_names,
    )
