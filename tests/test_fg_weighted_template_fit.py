from __future__ import annotations

import numpy as np
import pytest

import fg_weighted_template_fit as ftf
import fg_weighted_template_fit._filters as filters_mod


def test_weighted_template_gls_recovers_known_amplitudes() -> None:
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


def test_realize_qu_noise_matches_requested_covariance() -> None:
    npix = 20_000
    covariance = np.array([[1.5], [0.9], [0.3]])
    covariance = np.repeat(covariance, npix, axis=1)

    noise = ftf.realize_qu_noise(covariance, rng=0)
    sample_cov = np.cov(noise)
    target_cov = np.array([[1.5, 0.3], [0.3, 0.9]])

    np.testing.assert_allclose(sample_cov, target_cov, atol=0.035)


def test_fit_and_bootstrap_store_mc_amplitudes() -> None:
    npix = 12
    dust_map_a = np.array(
        [
            [1.0, 1.2, 0.8, 0.5, -0.2, 0.4, 1.1, 0.7, -0.1, 0.2, 0.9, 0.6],
            [0.1, 0.2, -0.3, 0.4, 0.5, -0.4, 0.2, -0.2, 0.3, 0.8, 0.1, -0.1],
        ]
    )
    dust_map_b = np.array(
        [
            [0.2, 0.3, 0.1, 0.0, -0.1, 0.2, 0.4, 0.2, -0.1, 0.1, 0.3, 0.1],
            [0.0, 0.1, -0.1, 0.2, 0.1, -0.2, 0.0, -0.1, 0.1, 0.3, 0.0, -0.1],
        ]
    )
    sync_map_a = np.array(
        [
            [0.5, -0.2, 0.1, -0.4, 0.2, 0.3, -0.1, 0.4, 0.6, -0.3, 0.2, 0.1],
            [-0.3, 0.6, 0.2, -0.1, 0.4, -0.2, 0.5, -0.4, 0.1, 0.3, -0.5, 0.2],
        ]
    )
    sync_map_b = np.array(
        [
            [0.1, -0.1, 0.0, -0.1, 0.1, 0.1, -0.1, 0.1, 0.2, -0.1, 0.0, 0.0],
            [-0.1, 0.2, 0.1, 0.0, 0.1, -0.1, 0.2, -0.1, 0.0, 0.1, -0.2, 0.1],
        ]
    )

    dust_template = dust_map_a - dust_map_b
    sync_template = sync_map_a - sync_map_b
    amplitudes_true = np.array([1.4, -0.6])
    target = amplitudes_true[0] * dust_template + amplitudes_true[1] * sync_template

    noise_cov = np.repeat(np.array([[0.01], [0.01], [0.002]]), npix, axis=1)
    template_cov = np.repeat(np.array([[0.005], [0.005], [0.001]]), npix, axis=1)

    templates = (
        ftf.DifferenceTemplateInput(
            map_a_qu=dust_map_a,
            map_b_qu=dust_map_b,
            fwhm_in_a=0.0,
            fwhm_in_b=0.0,
            noise_cov_a=template_cov,
            noise_cov_b=template_cov,
            name="dust",
        ),
        ftf.DifferenceTemplateInput(
            map_a_qu=sync_map_a,
            map_b_qu=sync_map_b,
            fwhm_in_a=0.0,
            fwhm_in_b=0.0,
            noise_cov_a=template_cov,
            noise_cov_b=template_cov,
            name="sync",
        ),
    )

    reference = ftf.fit_foreground_templates(
        target_qu=target,
        target_fwhm_in=0.0,
        template_inputs=templates,
        weight_map=np.ones(npix),
        fwhm_out=0.0,
        template_inputs_rhs=templates,
    )
    np.testing.assert_allclose(reference.amplitudes, amplitudes_true, atol=1e-12)

    bootstrap = ftf.bootstrap_template_amplitudes(
        target_qu=target,
        target_noise_cov=noise_cov,
        target_fwhm_in=0.0,
        template_inputs=templates,
        weight_map=np.ones(npix),
        fwhm_out=0.0,
        n_mc=8,
        template_inputs_rhs=templates,
        rng=1234,
    )

    assert bootstrap.template_names == ("dust", "sync")
    assert bootstrap.amplitude_samples.shape == (8, 2)
    assert np.all(np.isfinite(bootstrap.amplitude_samples))
    assert np.all(bootstrap.amplitude_std > 0.0)


def test_apodized_highpass_matches_c1_and_c2_profiles() -> None:
    c2_window = ftf._build_apodized_highpass(
        num_modes=10,
        cutoff=5.0,
        halfwidth=2.0,
        transition_type="C2",
    )
    c1_window = ftf._build_apodized_highpass(
        num_modes=10,
        cutoff=5.0,
        halfwidth=2.0,
        transition_type="C1",
    )

    np.testing.assert_allclose(c2_window[:4], 0.0)
    np.testing.assert_allclose(c1_window[:4], 0.0)
    np.testing.assert_allclose(c2_window[7:], 1.0)
    np.testing.assert_allclose(c1_window[7:], 1.0)

    transition_x = np.array([0.25, 0.5, 0.75])
    expected_c2 = 0.5 * (1.0 - np.cos(np.pi * transition_x))
    expected_c1 = transition_x - np.sin(2.0 * np.pi * transition_x) / (2.0 * np.pi)

    np.testing.assert_allclose(c2_window[4:7], expected_c2)
    np.testing.assert_allclose(c1_window[4:7], expected_c1)


def test_resolve_lmax_does_not_truncate_highpass_ell_cutoff() -> None:
    filter_config = ftf.HarmonicFilter(
        ell_cutoff=3.0,
        ell_halfwidth=1.0,
    )

    assert filters_mod._resolve_lmax(nside=8, filter_config=filter_config) == 23


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_smooth_and_filter_qu_map_smoothing_reduces_variance() -> None:
    nside = 8
    npix = 12 * nside**2
    rng = np.random.default_rng(42)
    qu_map = rng.standard_normal((2, npix))

    smoothed = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=0.0,
        fwhm_out=np.radians(1.0),
    )

    assert smoothed.shape == qu_map.shape
    assert np.var(smoothed) < np.var(qu_map)


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_smooth_and_filter_qu_map_accepts_m_cutoff_with_smooth_edge() -> None:
    nside = 8
    npix = 12 * nside**2
    rng = np.random.default_rng(123)
    qu_map = rng.standard_normal((2, npix))

    filtered = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=0.0,
        fwhm_out=0.0,
        filter_config=ftf.HarmonicFilter(
            m_cutoff=3.0,
            m_halfwidth=2.0,
            transition_type="C1",
        ),
    )

    assert filtered.shape == qu_map.shape
    assert np.all(np.isfinite(filtered))
    assert np.var(filtered) < np.var(qu_map)
