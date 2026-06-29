from __future__ import annotations

import numpy as np
import pytest

import fg_weighted_template_fit as ftf
import fg_weighted_template_fit._noise as noise_mod


def _make_bootstrap_concurrency_problem() -> dict[str, object]:
    """Build a tiny bootstrap problem that avoids healpy-dependent paths.

    Returns
    -------
    dict
        Keyword arguments accepted by ``bootstrap_template_amplitudes``. The
        target and templates are native-resolution Q/U maps with zero beam
        widths, so the fit stays on the pure NumPy path.
    """

    npix = 8
    dust = np.array(
        [
            [1.0, 0.4, -0.3, 0.8, -0.1, 0.6, 0.2, -0.5],
            [0.2, -0.6, 0.7, 0.1, 0.5, -0.4, 0.3, 0.9],
        ],
        dtype=np.float64,
    )
    sync = np.array(
        [
            [-0.2, 0.7, 0.5, -0.4, 0.9, 0.1, -0.6, 0.3],
            [0.8, 0.1, -0.5, 0.6, -0.3, 0.4, 0.2, -0.7],
        ],
        dtype=np.float64,
    )
    target = 1.35 * dust - 0.45 * sync
    target_noise_cov = np.repeat(
        np.array([[0.01], [0.015], [0.002]], dtype=np.float64),
        npix,
        axis=1,
    )
    template_noise_cov = np.repeat(
        np.array([[0.004], [0.006], [0.001]], dtype=np.float64),
        npix,
        axis=1,
    )
    template_inputs = (
        ftf.DifferenceTemplateInput(
            map_a_qu=dust,
            map_b_qu=np.zeros_like(dust),
            fwhm_in_a=0.0,
            fwhm_in_b=0.0,
            noise_cov_a=template_noise_cov,
            noise_cov_b=template_noise_cov,
            name="dust",
        ),
        ftf.DifferenceTemplateInput(
            map_a_qu=sync,
            map_b_qu=np.zeros_like(sync),
            fwhm_in_a=0.0,
            fwhm_in_b=0.0,
            noise_cov_a=template_noise_cov,
            noise_cov_b=template_noise_cov,
            name="sync",
        ),
    )

    return {
        "target_qu": target,
        "target_noise_cov": target_noise_cov,
        "target_fwhm_in": 0.0,
        "template_inputs": template_inputs,
        "weight_map": np.ones(npix),
        "fwhm_out": 0.0,
    }


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


def test_bootstrap_template_amplitudes_multi_mask_uses_shared_draws() -> None:
    """Reuse each noisy realization across every named fitting mask."""

    npix = 6
    template = np.array(
        [
            [1.0, 0.5, -0.2, 0.3, 0.8, -0.4],
            [0.2, -0.3, 0.6, 1.2, -0.5, 0.1],
        ],
        dtype=np.float64,
    )
    target = 1.7 * template
    target_noise_cov = np.repeat(
        np.array([[0.01], [0.015], [0.002]], dtype=np.float64),
        npix,
        axis=1,
    )
    template_input = ftf.DifferenceTemplateInput(
        map_a_qu=template,
        map_b_qu=np.zeros_like(template),
        fwhm_in_a=0.0,
        fwhm_in_b=0.0,
        name="dust",
    )
    weight_maps = {
        "m1": np.ones(npix),
        "m2": np.ones(npix),
    }

    first = ftf.bootstrap_template_amplitudes_multi_mask(
        target_qu=target,
        target_noise_cov=target_noise_cov,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        weight_maps=weight_maps,
        fwhm_out=0.0,
        n_mc=5,
        master_mask=np.array([1.0, 0.8, 1.0, 0.0, 1.0, 1.0]),
        rng=2468,
        n_jobs=2,
    )
    second = ftf.bootstrap_template_amplitudes_multi_mask(
        target_qu=target,
        target_noise_cov=target_noise_cov,
        target_fwhm_in=0.0,
        template_inputs=(template_input,),
        weight_maps=weight_maps,
        fwhm_out=0.0,
        n_mc=5,
        master_mask=np.array([1.0, 0.8, 1.0, 0.0, 1.0, 1.0]),
        rng=2468,
        n_jobs=2,
    )

    assert first.fit_names == ("m1", "m2")
    assert first.template_names == ("dust",)
    assert first.amplitude_samples.shape == (5, 2, 1)
    assert np.all(np.isfinite(first.amplitude_samples))
    np.testing.assert_allclose(
        first.amplitude_samples[:, 0, :],
        first.amplitude_samples[:, 1, :],
    )
    np.testing.assert_allclose(first.amplitude_samples, second.amplitude_samples)
    np.testing.assert_allclose(first.amplitude_mean, second.amplitude_mean)
    np.testing.assert_allclose(first.amplitude_std, second.amplitude_std)


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


def test_bootstrap_template_amplitudes_n_jobs_one_matches_default_serial() -> None:
    """Keep explicit single-worker bootstrap equivalent to the default path."""

    problem = _make_bootstrap_concurrency_problem()

    default = ftf.bootstrap_template_amplitudes(
        **problem,
        n_mc=5,
        rng=9876,
    )
    explicit = ftf.bootstrap_template_amplitudes(
        **problem,
        n_mc=5,
        rng=9876,
        n_jobs=1,
    )

    np.testing.assert_allclose(explicit.amplitude_samples, default.amplitude_samples)
    np.testing.assert_allclose(explicit.amplitude_mean, default.amplitude_mean)
    np.testing.assert_allclose(explicit.amplitude_std, default.amplitude_std)


def test_bootstrap_template_amplitudes_threaded_is_reproducible_for_seed() -> None:
    """Use independent per-draw RNG streams so threaded output is reproducible."""

    problem = _make_bootstrap_concurrency_problem()

    first = ftf.bootstrap_template_amplitudes(
        **problem,
        n_mc=6,
        rng=12345,
        n_jobs=2,
    )
    second = ftf.bootstrap_template_amplitudes(
        **problem,
        n_mc=6,
        rng=12345,
        n_jobs=2,
    )

    np.testing.assert_allclose(first.amplitude_samples, second.amplitude_samples)
    np.testing.assert_allclose(first.amplitude_mean, second.amplitude_mean)
    np.testing.assert_allclose(first.amplitude_std, second.amplitude_std)


def test_bootstrap_template_amplitudes_threaded_returns_valid_samples() -> None:
    """Return finite MC samples with the expected threaded output shape."""

    problem = _make_bootstrap_concurrency_problem()

    result = ftf.bootstrap_template_amplitudes(
        **problem,
        n_mc=6,
        rng=4321,
        n_jobs=2,
    )

    assert result.amplitude_samples.shape == (6, 2)
    assert np.all(np.isfinite(result.amplitude_samples))
    assert np.all(np.isfinite(result.amplitude_mean))
    assert np.all(np.isfinite(result.amplitude_std))


def test_bootstrap_template_amplitudes_rejects_nonpositive_n_jobs() -> None:
    """Reject invalid worker counts before starting a bootstrap run."""

    problem = _make_bootstrap_concurrency_problem()

    for n_jobs in (0, -1):
        with pytest.raises(ValueError, match="n_jobs"):
            ftf.bootstrap_template_amplitudes(
                **problem,
                n_mc=1,
                rng=2468,
                n_jobs=n_jobs,
            )


def test_bootstrap_template_amplitudes_threaded_show_progress_uses_tqdm(
    monkeypatch,
) -> None:
    """Wrap completed threaded draws in tqdm when progress reporting is enabled."""

    problem = _make_bootstrap_concurrency_problem()
    calls: list[dict[str, object]] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    monkeypatch.setattr(noise_mod, "_tqdm", fake_tqdm)

    result = ftf.bootstrap_template_amplitudes(
        **problem,
        n_mc=3,
        rng=1357,
        show_progress=True,
        n_jobs=2,
    )

    assert result.amplitude_samples.shape == (3, 2)
    assert calls == [{"total": 3, "desc": "Bootstrap MC", "unit": "draw"}]


def test_bootstrap_template_amplitudes_threaded_show_progress_requires_tqdm(
    monkeypatch,
) -> None:
    """Raise the standard progress error before threaded progress reporting."""

    problem = _make_bootstrap_concurrency_problem()
    monkeypatch.setattr(noise_mod, "_tqdm", None)

    with pytest.raises(ImportError, match="requires tqdm"):
        ftf.bootstrap_template_amplitudes(
            **problem,
            n_mc=2,
            rng=9753,
            show_progress=True,
            n_jobs=2,
        )


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
