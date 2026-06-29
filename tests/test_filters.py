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


def test_resolve_lmax_accounts_for_custom_beam_window() -> None:
    """Use custom input beam support when resolving the harmonic truncation."""

    beam_window = np.ones(11, dtype=np.float64)

    assert (
        filters_mod._resolve_lmax(
            nside=8,
            filter_config=ftf.HarmonicFilter(),
            beam_window_in=beam_window,
        )
        == 10
    )


def test_build_ell_transfer_uses_custom_input_beam() -> None:
    """Use B_out / B_in when a custom input beam window is supplied."""

    lmax = 6
    fwhm_out = 0.05
    beam_window = np.linspace(1.0, 1.6, lmax + 1)

    transfer = filters_mod._build_ell_transfer(
        lmax=lmax,
        fwhm_in=10.0,
        fwhm_out=fwhm_out,
        filter_config=ftf.HarmonicFilter(),
        beam_window_in=beam_window,
    )

    ells = np.arange(lmax + 1, dtype=np.float64)
    sigma_out = filters_mod._fwhm_to_sigma(fwhm_out)
    expected = np.exp(-0.5 * ells * (ells + 1.0) * sigma_out**2) / beam_window
    np.testing.assert_allclose(transfer, expected)


def test_build_ell_transfer_custom_gaussian_matches_default() -> None:
    """Match the legacy Gaussian transfer when B_in is the same Gaussian beam."""

    lmax = 8
    fwhm_in = 0.03
    fwhm_out = 0.08
    ells = np.arange(lmax + 1, dtype=np.float64)
    sigma_in = filters_mod._fwhm_to_sigma(fwhm_in)
    beam_window = np.exp(-0.5 * ells * (ells + 1.0) * sigma_in**2)

    default = filters_mod._build_ell_transfer(
        lmax=lmax,
        fwhm_in=fwhm_in,
        fwhm_out=fwhm_out,
        filter_config=ftf.HarmonicFilter(),
    )
    custom = filters_mod._build_ell_transfer(
        lmax=lmax,
        fwhm_in=10.0,
        fwhm_out=fwhm_out,
        filter_config=ftf.HarmonicFilter(),
        beam_window_in=beam_window,
    )

    np.testing.assert_allclose(custom, default)


def test_build_ell_transfer_custom_beam_composes_with_ell_filters() -> None:
    """Apply custom beam matching and ell filters in one transfer."""

    lmax = 8
    fwhm_out = 0.04
    beam_window = np.linspace(1.0, 1.4, lmax + 1)
    ell_filter = np.linspace(0.5, 1.0, lmax + 1)
    filter_config = ftf.HarmonicFilter(
        ell_filter=ell_filter,
        ell_cutoff=4.0,
        ell_halfwidth=1.0,
        transition_type="C2",
    )

    transfer = filters_mod._build_ell_transfer(
        lmax=lmax,
        fwhm_in=10.0,
        fwhm_out=fwhm_out,
        filter_config=filter_config,
        beam_window_in=beam_window,
    )

    ells = np.arange(lmax + 1, dtype=np.float64)
    sigma_out = filters_mod._fwhm_to_sigma(fwhm_out)
    output_beam = np.exp(-0.5 * ells * (ells + 1.0) * sigma_out**2)
    highpass = filters_mod._build_apodized_highpass(
        num_modes=lmax + 1,
        cutoff=4.0,
        halfwidth=1.0,
        transition_type="C2",
    )
    expected = output_beam / beam_window * ell_filter * highpass

    np.testing.assert_allclose(transfer, expected)


@pytest.mark.parametrize(
    ("beam_window", "match"),
    [
        (np.ones((2, 4)), "1D"),
        (np.array([1.0, 1.0]), "length"),
        (np.array([1.0, 0.0, 1.0]), "strictly positive"),
        (np.array([1.0, -0.5, 1.0]), "strictly positive"),
        (np.array([1.0, np.nan, 1.0]), "finite"),
        (np.array([1.0, np.inf, 1.0]), "finite"),
    ],
)
def test_build_ell_transfer_rejects_invalid_custom_beam_window(
    beam_window,
    match,
) -> None:
    """Reject custom input beams that would make deconvolution ill-defined."""

    with pytest.raises(ValueError, match=match):
        filters_mod._build_ell_transfer(
            lmax=2,
            fwhm_in=0.0,
            fwhm_out=0.0,
            filter_config=ftf.HarmonicFilter(),
            beam_window_in=beam_window,
        )


def test_smooth_and_filter_qu_map_custom_beam_overrides_fwhm_validation(
    monkeypatch,
) -> None:
    """Let a custom input beam replace the Gaussian input FWHM."""

    class FakeHealpy:
        class Alm:
            @staticmethod
            def getlm(lmax):
                nalm = (lmax + 1) * (lmax + 2) // 2
                return np.zeros(nalm, dtype=np.int64), np.zeros(nalm, dtype=np.int64)

        def __init__(self) -> None:
            self.transfers: list[np.ndarray] = []
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
            del inplace
            self.transfers.append(np.asarray(transfer, dtype=np.float64).copy())
            return np.asarray(alm, dtype=np.complex128).copy()

        def alm2map(self, alms, nside, lmax, pol):
            del alms, nside, lmax, pol
            assert self.last_tqu is not None
            return self.last_tqu.copy()

    fake_hp = FakeHealpy()
    monkeypatch.setattr(filters_mod, "hp", fake_hp)

    beam_window = np.array([1.0, 1.1, 1.2])
    qu_map = np.ones((2, 12), dtype=np.float64)
    filtered = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=10.0,
        fwhm_out=0.0,
        beam_window_in=beam_window,
    )

    np.testing.assert_allclose(filtered, qu_map)
    assert len(fake_hp.transfers) == 2
    np.testing.assert_allclose(fake_hp.transfers[0], 1.0 / beam_window)


def test_smooth_and_filter_qu_map_gaussian_beam_window_matches_fwhm(
    monkeypatch,
) -> None:
    """Match public preprocessing when B_ell is the same Gaussian input beam."""

    class FakeHealpy:
        class Alm:
            @staticmethod
            def getlm(lmax):
                nalm = (lmax + 1) * (lmax + 2) // 2
                return np.zeros(nalm, dtype=np.int64), np.zeros(nalm, dtype=np.int64)

        def __init__(self) -> None:
            self.transfers: list[np.ndarray] = []

        def npix2nside(self, npix: int) -> int:
            assert npix == 12
            return 1

        def map2alm(self, tqu, lmax, iter, pol):
            del tqu, iter, pol
            nalm = (lmax + 1) * (lmax + 2) // 2
            zeros = np.zeros(nalm, dtype=np.complex128)
            return zeros.copy(), zeros.copy(), zeros.copy()

        def almxfl(self, alm, transfer, inplace=False):
            del alm, inplace
            self.transfers.append(np.asarray(transfer, dtype=np.float64).copy())
            return np.asarray(transfer, dtype=np.complex128)

        def alm2map(self, alms, nside, lmax, pol):
            del lmax, pol
            npix = 12 * nside**2
            q_value = float(np.sum(alms[1].real))
            u_value = float(np.sum(alms[2].real))
            return np.vstack(
                [
                    np.zeros(npix, dtype=np.float64),
                    np.full(npix, q_value, dtype=np.float64),
                    np.full(npix, u_value, dtype=np.float64),
                ]
            )

    fake_hp = FakeHealpy()
    monkeypatch.setattr(filters_mod, "hp", fake_hp)

    lmax = 2
    fwhm_in = 0.03
    fwhm_out = 0.08
    ells = np.arange(lmax + 1, dtype=np.float64)
    sigma_in = filters_mod._fwhm_to_sigma(fwhm_in)
    gaussian_input_beam = np.exp(-0.5 * ells * (ells + 1.0) * sigma_in**2)
    qu_map = np.arange(24, dtype=np.float64).reshape(2, 12)

    gaussian_fwhm_result = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=fwhm_in,
        fwhm_out=fwhm_out,
    )
    custom_beam_result = ftf.smooth_and_filter_qu_map(
        qu_map=qu_map,
        fwhm_in=10.0,
        fwhm_out=fwhm_out,
        beam_window_in=gaussian_input_beam,
    )

    np.testing.assert_allclose(custom_beam_result, gaussian_fwhm_result)
    assert len(fake_hp.transfers) == 4
    np.testing.assert_allclose(fake_hp.transfers[2], fake_hp.transfers[0])
    np.testing.assert_allclose(fake_hp.transfers[3], fake_hp.transfers[1])


def test_build_template_stack_threads_custom_beam_windows(monkeypatch) -> None:
    """Pass per-map custom beam windows from DifferenceTemplateInput."""

    calls: list[dict[str, object]] = []

    def fake_construct_difference_template(**kwargs):
        calls.append(kwargs)
        return np.ones((2, 4), dtype=np.float64)

    monkeypatch.setattr(
        filters_mod,
        "construct_difference_template",
        fake_construct_difference_template,
    )

    beam_a = np.ones(5, dtype=np.float64)
    beam_b = np.linspace(1.0, 1.2, 5)
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=np.ones((2, 4), dtype=np.float64),
        map_b_qu=np.zeros((2, 4), dtype=np.float64),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
        beam_window_a=beam_a,
        beam_window_b=beam_b,
    )

    templates, names = ftf.build_template_stack(
        (template_input,),
        fwhm_out=0.0,
    )

    assert names == ("dust",)
    assert templates.shape == (1, 2, 4)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0]["beam_window_a"], beam_a)
    np.testing.assert_allclose(calls[0]["beam_window_b"], beam_b)


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
