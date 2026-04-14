from __future__ import annotations

import numpy as np
import pytest

import fg_weighted_template_fit as ftf
import fg_weighted_template_fit._fit as fit_mod
import fg_weighted_template_fit._filters as filters_mod
import fg_weighted_template_fit._noise as noise_mod


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


def test_realize_qu_noise_matches_requested_covariance() -> None:
    """Draw Q/U noise realizations with the requested pixel covariance."""

    npix = 20_000
    covariance = np.array([[1.5], [0.9], [0.3]])
    covariance = np.repeat(covariance, npix, axis=1)

    noise = ftf.realize_qu_noise(covariance, rng=0)
    sample_cov = np.cov(noise)
    target_cov = np.array([[1.5, 0.3], [0.3, 0.9]])

    np.testing.assert_allclose(sample_cov, target_cov, atol=0.035)


def test_fit_and_bootstrap_store_mc_amplitudes() -> None:
    """Keep Monte Carlo amplitude draws and summarize their spread."""

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


def test_fit_foreground_templates_passes_mask_to_preprocessing_helpers(
    monkeypatch,
) -> None:
    """Pass the fit mask into both target and template preprocessing helpers."""

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

    monkeypatch.setattr(
        fit_mod, "smooth_and_filter_qu_map", fake_smooth_and_filter_qu_map
    )
    monkeypatch.setattr(fit_mod, "build_template_stack", fake_build_template_stack)

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


def test_bootstrap_template_amplitudes_show_progress_uses_tqdm(monkeypatch) -> None:
    """Wrap bootstrap draws in tqdm when progress reporting is enabled."""

    npix = 6
    template = np.array(
        [
            [1.0, 0.5, -0.2, 0.3, 0.8, -0.4],
            [0.2, -0.3, 0.6, 1.2, -0.5, 0.1],
        ]
    )
    target = 1.7 * template
    target_noise_cov = np.zeros((3, npix), dtype=np.float64)

    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template,
        map_b_qu=np.zeros_like(template),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    calls: list[dict[str, object]] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    monkeypatch.setattr(noise_mod, "_tqdm", fake_tqdm)

    result = ftf.bootstrap_template_amplitudes(
        target_qu=target,
        target_noise_cov=target_noise_cov,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        weight_map=np.ones(npix),
        fwhm_out=0.0,
        n_mc=3,
        rng=1234,
        show_progress=True,
    )

    assert result.amplitude_samples.shape == (3, 1)
    assert calls == [{"total": 3, "desc": "Bootstrap MC", "unit": "draw"}]


def test_bootstrap_template_amplitudes_show_progress_requires_tqdm(
    monkeypatch,
) -> None:
    """Raise an informative error when tqdm-backed progress is unavailable."""

    npix = 6
    template = np.array(
        [
            [1.0, 0.5, -0.2, 0.3, 0.8, -0.4],
            [0.2, -0.3, 0.6, 1.2, -0.5, 0.1],
        ]
    )
    target_noise_cov = np.zeros((3, npix), dtype=np.float64)

    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template,
        map_b_qu=np.zeros_like(template),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )

    monkeypatch.setattr(noise_mod, "_tqdm", None)

    with pytest.raises(ImportError, match="requires tqdm"):
        ftf.bootstrap_template_amplitudes(
            target_qu=template,
            target_noise_cov=target_noise_cov,
            target_fwhm_in=0.0,
            template_inputs=(template_input,),
            weight_map=np.ones(npix),
            fwhm_out=0.0,
            n_mc=2,
            show_progress=True,
        )


def test_build_ell_filter_matches_c1_and_c2_profiles() -> None:
    """Build ell-space tapers that match the expected C1 and C2 profiles."""

    c2_window = ftf.build_ell_filter(
        lmax=9,
        cutoff=5.0,
        halfwidth=2.0,
        transition_type="C2",
    )
    c1_window = ftf.build_ell_filter(
        lmax=9,
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


def test_build_m_filter_matches_ell_filter_profile() -> None:
    """Match the m-space helper output to the ell-space taper profile."""

    ell_window = ftf.build_ell_filter(
        lmax=9,
        cutoff=4.0,
        halfwidth=1.0,
        transition_type="C2",
    )
    m_window = ftf.build_m_filter(
        lmax=9,
        cutoff=4.0,
        halfwidth=1.0,
        transition_type="C2",
    )

    np.testing.assert_allclose(m_window, ell_window)


def test_build_ell_filter_rejects_negative_lmax() -> None:
    """Reject negative harmonic truncation when building ell filters."""

    with pytest.raises(ValueError, match="lmax must be non-negative"):
        ftf.build_ell_filter(
            lmax=-1,
            cutoff=3.0,
        )


def test_resolve_lmax_does_not_truncate_highpass_ell_cutoff() -> None:
    """Keep the native transform support when only an ell cutoff is requested."""

    filter_config = ftf.HarmonicFilter(
        ell_cutoff=3.0,
        ell_halfwidth=1.0,
    )

    assert filters_mod._resolve_lmax(nside=8, filter_config=filter_config) == 23


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_smooth_and_filter_qu_map_smoothing_reduces_variance() -> None:
    """Lower map variance when Gaussian smoothing is applied."""

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


def test_smooth_and_filter_qu_map_applies_mask_before_transform(
    monkeypatch,
) -> None:
    """Apply the preprocessing mask before calling the harmonic transform."""

    class FakeHealpy:
        class Alm:
            @staticmethod
            def getlm(lmax):
                nalm = (lmax + 1) * (lmax + 2) // 2
                return np.zeros(nalm, dtype=np.int64), np.zeros(nalm, dtype=np.int64)

        def __init__(self) -> None:
            self.last_tqu: np.ndarray | None = None

        def npix2nside(self, npix: int) -> int:
            assert npix == 12
            return 1

        def map2alm(self, tqu, lmax, iter, pol):
            del iter, pol
            self.last_tqu = np.asarray(tqu, dtype=np.float64).copy()
            nalm = (lmax + 1) * (lmax + 2) // 2
            zeros = np.zeros(nalm, dtype=np.complex128)
            return zeros.copy(), zeros.copy(), zeros.copy()

        def almxfl(self, alm, transfer, inplace=False):
            del transfer, inplace
            return np.asarray(alm, dtype=np.complex128).copy()

        def alm2map(self, alms, nside, lmax, pol):
            del alms, nside, lmax, pol
            assert self.last_tqu is not None
            return self.last_tqu.copy()

    fake_hp = FakeHealpy()
    monkeypatch.setattr(filters_mod, "hp", fake_hp)

    qu_map = np.arange(24, dtype=np.float64).reshape(2, 12)
    mask = np.linspace(0.0, 1.0, 12)

    filtered = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=0.0,
        fwhm_out=np.radians(1.0),
        mask=mask,
    )

    expected = qu_map * np.vstack([mask, mask])
    assert fake_hp.last_tqu is not None
    np.testing.assert_allclose(fake_hp.last_tqu[0], 0.0)
    np.testing.assert_allclose(fake_hp.last_tqu[1:], expected)
    np.testing.assert_allclose(filtered, expected)


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_smooth_and_filter_qu_map_accepts_m_cutoff_with_smooth_edge() -> None:
    """Accept smooth-edge m cutoffs and return a finite filtered map."""

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


@pytest.mark.skipif(filters_mod.hp is None, reason="healpy not installed")
def test_smooth_and_filter_qu_map_accepts_public_explicit_filters() -> None:
    """Accept explicit public ell and m filter arrays."""

    nside = 8
    npix = 12 * nside**2
    rng = np.random.default_rng(456)
    qu_map = rng.standard_normal((2, npix))

    filtered = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=0.0,
        fwhm_out=0.0,
        filter_config=ftf.HarmonicFilter(
            ell_filter=ftf.build_ell_filter(
                lmax=3 * nside - 1,
                cutoff=6.0,
                halfwidth=2.0,
                transition_type="C2",
            ),
            m_filter=ftf.build_m_filter(
                lmax=3 * nside - 1,
                cutoff=4.0,
                halfwidth=1.0,
                transition_type="C1",
            ),
        ),
    )

    assert filtered.shape == qu_map.shape
    assert np.all(np.isfinite(filtered))
    assert not np.allclose(filtered, qu_map)
