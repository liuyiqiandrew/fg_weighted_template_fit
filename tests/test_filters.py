from __future__ import annotations

import numpy as np
import pytest

import fg_weighted_template_fit as ftf
import fg_weighted_template_fit._filters as filters_mod


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
