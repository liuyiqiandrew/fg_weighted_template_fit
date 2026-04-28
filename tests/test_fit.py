from __future__ import annotations

import numpy as np
import pytest

import fg_weighted_template_fit as ftf
import fg_weighted_template_fit._fit as fit_mod


def test_weighted_template_gls_recovers_known_amplitudes() -> None:
    """Recover exact amplitudes for the standard weighted template solve."""

    npix = 6
    dust = np.array(
        [
            [1.0, 0.5, -0.2, 0.3, 0.8, -0.4],
            [0.2, -0.3, 0.6, 1.2, -0.5, 0.1],
        ]
    )
    sync = np.array(
        [
            [0.4, -0.6, 0.1, 0.9, -0.1, 0.3],
            [-0.2, 0.7, 0.5, -0.4, 0.6, -0.8],
        ]
    )
    amplitudes_true = np.array([1.7, -0.35])
    target = amplitudes_true[0] * dust + amplitudes_true[1] * sync
    weight_map = np.array([1.0, 3.0, 0.5, 2.0, 4.0, 1.5])

    result = ftf.weighted_template_gls(
        target_qu=target,
        templates_qu=np.stack([dust, sync], axis=0),
        weight_map=weight_map,
        template_names=("dust", "sync"),
    )

    np.testing.assert_allclose(result.amplitudes, amplitudes_true, atol=1e-12)
    np.testing.assert_allclose(result.residual_qu, 0.0, atol=1e-12)
    assert result.template_names == ("dust", "sync")


def test_weighted_template_gls_supports_cross_normal_matrix() -> None:
    """Recover amplitudes when left- and right-hand template stacks differ."""

    npix = 6
    dust_lhs = np.array(
        [
            [1.0, 0.3, -0.2, 0.6, 0.1, -0.4],
            [0.4, -0.2, 0.5, 0.1, -0.3, 0.7],
        ]
    )
    sync_lhs = np.array(
        [
            [0.2, -0.4, 0.8, -0.1, 0.5, 0.3],
            [-0.5, 0.6, 0.1, -0.2, 0.4, -0.3],
        ]
    )
    dust_rhs = np.array(
        [
            [0.8, 0.2, -0.1, 0.5, 0.0, -0.2],
            [0.3, -0.1, 0.4, 0.2, -0.2, 0.5],
        ]
    )
    sync_rhs = np.array(
        [
            [0.1, -0.2, 0.7, 0.0, 0.3, 0.2],
            [-0.4, 0.5, 0.2, -0.1, 0.2, -0.2],
        ]
    )
    amplitudes_true = np.array([1.25, -0.55])
    target = amplitudes_true[0] * dust_rhs + amplitudes_true[1] * sync_rhs

    result = ftf.weighted_template_gls(
        target_qu=target,
        templates_qu=np.stack([dust_lhs, sync_lhs], axis=0),
        templates_rhs_qu=np.stack([dust_rhs, sync_rhs], axis=0),
        weight_map=np.ones(npix),
        template_names=("dust", "sync"),
    )

    np.testing.assert_allclose(result.amplitudes, amplitudes_true, atol=1e-12)
    np.testing.assert_allclose(
        result.processed_templates_rhs_qu,
        np.stack([dust_rhs, sync_rhs], axis=0),
        atol=1e-12,
    )


def test_weighted_template_gls_treats_mask_as_binary_support() -> None:
    """Convert any nonzero fit mask value into unit support inside the solver."""

    template = np.ones((1, 2, 3), dtype=np.float64)
    target = np.ones((2, 3), dtype=np.float64)
    mask = np.array([1.0, 0.5, 0.0])

    result = ftf.weighted_template_gls(
        target_qu=target,
        templates_qu=template,
        weight_map=np.ones(3),
        mask=mask,
    )

    np.testing.assert_allclose(result.normal_matrix, [[4.0]])
    np.testing.assert_allclose(result.rhs, [4.0])


def test_fit_foreground_templates_passes_mask_to_preprocessing_helpers(
    monkeypatch,
) -> None:
    """Use the fit mask in preprocessing only, not as a second GLS weight."""

    target = np.array(
        [
            [1.0, 0.5, -0.2, 0.3, 0.8, -0.4],
            [0.2, -0.3, 0.6, 1.2, -0.5, 0.1],
        ]
    )
    zero = np.zeros_like(target)
    mask = np.array([1.0, 0.9, 0.6, 0.4, 0.2, 0.0])
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=target,
        map_b_qu=zero,
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    target_masks: list[np.ndarray | None] = []
    template_masks: list[np.ndarray | None] = []
    gls_masks: list[np.ndarray | None] = []

    def fake_smooth_and_filter_qu_map(
        qu_map,
        fwhm_in,
        fwhm_out,
        *,
        filter_config=None,
        mask=None,
        nest=False,
    ):
        del fwhm_in, fwhm_out, filter_config, nest
        target_masks.append(
            None if mask is None else np.asarray(mask, dtype=np.float64)
        )
        return np.asarray(qu_map, dtype=np.float64)

    def fake_build_template_stack(
        *,
        template_inputs,
        fwhm_out,
        default_filter=None,
        mask=None,
        nest=False,
    ):
        del fwhm_out, default_filter, nest
        template_masks.append(
            None if mask is None else np.asarray(mask, dtype=np.float64)
        )
        templates = np.stack(
            [
                np.asarray(template_input.map_a_qu, dtype=np.float64)
                - np.asarray(template_input.map_b_qu, dtype=np.float64)
                for template_input in template_inputs
            ],
            axis=0,
        )
        template_names = tuple(
            template_input.name or f"template_{index}"
            for index, template_input in enumerate(template_inputs)
        )
        return templates, template_names

    original_weighted_template_gls = fit_mod.weighted_template_gls

    def fake_weighted_template_gls(
        target_qu,
        templates_qu,
        weight_map,
        *,
        templates_rhs_qu=None,
        mask=None,
        template_names=None,
    ):
        gls_masks.append(None if mask is None else np.asarray(mask, dtype=np.float64))
        return original_weighted_template_gls(
            target_qu=target_qu,
            templates_qu=templates_qu,
            templates_rhs_qu=templates_rhs_qu,
            weight_map=weight_map,
            mask=mask,
            template_names=template_names,
        )

    monkeypatch.setattr(
        fit_mod, "smooth_and_filter_qu_map", fake_smooth_and_filter_qu_map
    )
    monkeypatch.setattr(fit_mod, "build_template_stack", fake_build_template_stack)
    monkeypatch.setattr(fit_mod, "weighted_template_gls", fake_weighted_template_gls)

    result = ftf.fit_foreground_templates(
        target_qu=target,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        template_inputs_rhs=(template_input,),
        weight_map=np.ones(target.shape[1]),
        fwhm_out=0.0,
        mask=mask,
    )

    np.testing.assert_allclose(result.amplitudes, [1.0], atol=1e-12)
    assert len(target_masks) == 1
    np.testing.assert_allclose(target_masks[0], mask)
    assert len(template_masks) == 2
    np.testing.assert_allclose(template_masks[0], mask)
    np.testing.assert_allclose(template_masks[1], mask)
    assert gls_masks == [None]


def test_fit_foreground_templates_multi_mask_applies_binary_master_support() -> None:
    """Post-filter with binary support instead of applying apodization twice."""

    template = np.array(
        [
            [1.0, 2.0, -1.0, 0.5],
            [0.5, -0.3, 0.7, -0.2],
        ],
        dtype=np.float64,
    )
    target = 2.5 * template
    master_mask = np.array([1.0, 0.25, 0.0, np.nan])
    support = np.array([1.0, 1.0, 0.0, 0.0])
    support_qu = np.vstack([support, support])
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template,
        map_b_qu=np.zeros_like(template),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    result = ftf.fit_foreground_templates_multi_mask(
        target_qu=target,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        weight_maps={
            "m1": np.ones(template.shape[1]),
            "m2": np.array([1.0, 3.0, 1.0, 1.0]),
        },
        fwhm_out=0.0,
        master_mask=master_mask,
    )

    assert result.fit_names == ("m1", "m2")
    np.testing.assert_allclose(result.processed_target_qu, target * support_qu)
    np.testing.assert_allclose(
        result.processed_templates_qu[0],
        template * support_qu,
    )
    np.testing.assert_allclose(result.fit_results["m1"].amplitudes, [2.5])
    np.testing.assert_allclose(result.fit_results["m2"].amplitudes, [2.5])


def test_fit_foreground_templates_multi_mask_accepts_explicit_master_support() -> None:
    """Let callers override threshold-derived post-filter support."""

    template = np.array(
        [
            [1.0, 2.0, -1.0, 0.5],
            [0.5, -0.3, 0.7, -0.2],
        ],
        dtype=np.float64,
    )
    target = 1.75 * template
    support = np.array([0.0, 1.0, 0.0, 1.0])
    support_qu = np.vstack([support, support])
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template,
        map_b_qu=np.zeros_like(template),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    result = ftf.fit_foreground_templates_multi_mask(
        target_qu=target,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        weight_maps={"m1": np.ones(template.shape[1])},
        fwhm_out=0.0,
        master_mask=np.ones(template.shape[1]),
        master_support_mask=support,
        master_support_threshold=2.0,
    )

    np.testing.assert_allclose(result.processed_target_qu, target * support_qu)
    np.testing.assert_allclose(result.fit_results["m1"].amplitudes, [1.75])


def test_fit_foreground_templates_multi_mask_validates_named_masks() -> None:
    """Reject empty or shape-incompatible multi-mask inputs early."""

    npix = 4
    template = np.ones((2, npix), dtype=np.float64)
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template,
        map_b_qu=np.zeros_like(template),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    with pytest.raises(ValueError, match="weight_maps"):
        ftf.fit_foreground_templates_multi_mask(
            target_qu=template,
            target_fwhm_in=0.0,
            template_inputs=(template_input,),
            weight_maps={},
            fwhm_out=0.0,
            master_mask=np.ones(npix),
        )

    with pytest.raises(ValueError, match="weight_maps"):
        ftf.fit_foreground_templates_multi_mask(
            target_qu=template,
            target_fwhm_in=0.0,
            template_inputs=(template_input,),
            weight_maps={"m1": np.ones(npix + 1)},
            fwhm_out=0.0,
            master_mask=np.ones(npix),
        )

    with pytest.raises(ValueError, match="master_support_mask"):
        ftf.fit_foreground_templates_multi_mask(
            target_qu=template,
            target_fwhm_in=0.0,
            template_inputs=(template_input,),
            weight_maps={"m1": np.ones(npix)},
            fwhm_out=0.0,
            master_mask=np.ones(npix),
            master_support_mask=np.ones(npix + 1),
        )


def test_fit_foreground_templates_multi_mask_uses_master_mask_for_preprocessing(
    monkeypatch,
) -> None:
    """Keep fitting weights out of the harmonic preprocessing path."""

    target = np.array(
        [
            [1.0, 0.5, -0.2, 0.3],
            [0.2, -0.3, 0.6, 1.2],
        ],
        dtype=np.float64,
    )
    zero = np.zeros_like(target)
    master_mask = np.array([1.0, 0.9, 0.6, 0.4])
    weight_m1 = np.array([1.0, 0.0, 1.0, 0.0])
    weight_m2 = np.array([0.0, 1.0, 0.0, 1.0])
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=target,
        map_b_qu=zero,
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    target_masks: list[np.ndarray | None] = []
    template_masks: list[np.ndarray | None] = []
    gls_masks: list[np.ndarray | None] = []
    gls_weights: list[np.ndarray] = []

    def fake_smooth_and_filter_qu_map(
        qu_map,
        fwhm_in,
        fwhm_out,
        *,
        filter_config=None,
        mask=None,
        nest=False,
    ):
        del fwhm_in, fwhm_out, filter_config, nest
        target_masks.append(
            None if mask is None else np.asarray(mask, dtype=np.float64)
        )
        return np.asarray(qu_map, dtype=np.float64)

    def fake_build_template_stack(
        *,
        template_inputs,
        fwhm_out,
        default_filter=None,
        mask=None,
        nest=False,
    ):
        del fwhm_out, default_filter, nest
        template_masks.append(
            None if mask is None else np.asarray(mask, dtype=np.float64)
        )
        templates = np.stack(
            [
                np.asarray(template_input.map_a_qu, dtype=np.float64)
                - np.asarray(template_input.map_b_qu, dtype=np.float64)
                for template_input in template_inputs
            ],
            axis=0,
        )
        template_names = tuple(
            template_input.name or f"template_{index}"
            for index, template_input in enumerate(template_inputs)
        )
        return templates, template_names

    original_weighted_template_gls = fit_mod.weighted_template_gls

    def fake_weighted_template_gls(
        target_qu,
        templates_qu,
        weight_map,
        *,
        templates_rhs_qu=None,
        mask=None,
        template_names=None,
    ):
        gls_masks.append(None if mask is None else np.asarray(mask, dtype=np.float64))
        gls_weights.append(np.asarray(weight_map, dtype=np.float64))
        return original_weighted_template_gls(
            target_qu=target_qu,
            templates_qu=templates_qu,
            templates_rhs_qu=templates_rhs_qu,
            weight_map=weight_map,
            mask=mask,
            template_names=template_names,
        )

    monkeypatch.setattr(
        fit_mod, "smooth_and_filter_qu_map", fake_smooth_and_filter_qu_map
    )
    monkeypatch.setattr(fit_mod, "build_template_stack", fake_build_template_stack)
    monkeypatch.setattr(fit_mod, "weighted_template_gls", fake_weighted_template_gls)

    result = ftf.fit_foreground_templates_multi_mask(
        target_qu=target,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        template_inputs_rhs=(template_input,),
        weight_maps={"m1": weight_m1, "m2": weight_m2},
        fwhm_out=0.0,
        master_mask=master_mask,
    )

    expected_master = np.vstack([master_mask, master_mask])
    assert result.fit_names == ("m1", "m2")
    assert len(target_masks) == 1
    np.testing.assert_allclose(target_masks[0], expected_master)
    assert len(template_masks) == 2
    np.testing.assert_allclose(template_masks[0], expected_master)
    np.testing.assert_allclose(template_masks[1], expected_master)
    assert gls_masks == [None, None]
    np.testing.assert_allclose(gls_weights[0], np.vstack([weight_m1, weight_m1]))
    np.testing.assert_allclose(gls_weights[1], np.vstack([weight_m2, weight_m2]))
