from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from ._arrays import as_qu_map
from ._types import DifferenceTemplateInput, FloatArray, HarmonicFilter

try:
    import healpy as hp
except ImportError:  # pragma: no cover - exercised only when healpy is unavailable.
    hp = None


def smooth_and_filter_qu_map(
    qu_map: npt.ArrayLike,
    fwhm_in: float,
    fwhm_out: float,
    *,
    filter_config: HarmonicFilter | None = None,
    nest: bool = False,
) -> FloatArray:
    """Smooth and optionally filter a Healpix Q/U map.

    Parameters
    ----------
    qu_map
        Input Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    fwhm_in
        Beam FWHM of the input map in radians.
    fwhm_out
        Target beam FWHM in radians. The routine applies additional Gaussian
        smoothing, so ``fwhm_out`` must be at least ``fwhm_in``.
    filter_config
        Optional harmonic filter configuration. Both the beam matching and the
        harmonic filters are applied in a single alm-domain pass.
    nest
        If ``True``, the map is assumed to be in NEST ordering and is converted
        to RING before harmonic transforms, then converted back on output.

    Returns
    -------
    numpy.ndarray
        Smoothed and filtered Q/U map with shape ``(2, npix)``.

    Notes
    -----
    All beam widths are expressed in radians.
    """

    qu = as_qu_map(qu_map, name="qu_map")
    filter_config = filter_config or HarmonicFilter()

    if fwhm_out < fwhm_in and not np.isclose(fwhm_out, fwhm_in):
        raise ValueError("fwhm_out must be greater than or equal to fwhm_in.")

    if _is_identity_harmonic_operation(fwhm_in, fwhm_out, filter_config):
        return qu.copy()

    _require_healpy()

    map_for_transform = qu
    if nest:
        map_for_transform = np.asarray(
            [hp.reorder(component, n2r=True) for component in qu],
            dtype=np.float64,
        )

    npix = map_for_transform.shape[1]
    nside = hp.npix2nside(npix)
    lmax = _resolve_lmax(nside=nside, filter_config=filter_config)

    # Healpy's polarized transform works on T/Q/U. We prepend a zero-temperature
    # map, transform once, and only keep the polarization alms.
    tqu = np.vstack([np.zeros(npix, dtype=np.float64), map_for_transform])
    alm_t, alm_e, alm_b = hp.map2alm(
        tqu,
        lmax=lmax,
        iter=filter_config.iter,
        pol=True,
    )
    alm_t[...] = 0.0

    ell_transfer = _build_ell_transfer(
        lmax=lmax,
        fwhm_in=fwhm_in,
        fwhm_out=fwhm_out,
        filter_config=filter_config,
    )
    alm_e = hp.almxfl(alm_e, ell_transfer, inplace=False)
    alm_b = hp.almxfl(alm_b, ell_transfer, inplace=False)

    if filter_config.m_filter is not None or filter_config.m_cutoff is not None:
        _apply_m_filter_inplace(
            alm=alm_e,
            filter_config=filter_config,
            lmax=lmax,
        )
        _apply_m_filter_inplace(
            alm=alm_b,
            filter_config=filter_config,
            lmax=lmax,
        )

    filtered_tqu = hp.alm2map(
        [alm_t, alm_e, alm_b],
        nside=nside,
        lmax=lmax,
        pol=True,
    )
    filtered_qu = np.asarray(filtered_tqu[1:], dtype=np.float64)

    if nest:
        filtered_qu = np.asarray(
            [hp.reorder(component, r2n=True) for component in filtered_qu],
            dtype=np.float64,
        )

    return filtered_qu


def construct_difference_template(
    map_a_qu: npt.ArrayLike,
    map_b_qu: npt.ArrayLike,
    fwhm_in_a: float,
    fwhm_in_b: float,
    fwhm_out: float,
    *,
    filter_config: HarmonicFilter | None = None,
    nest: bool = False,
) -> FloatArray:
    """Construct a foreground template from the difference of two Q/U maps.

    Parameters
    ----------
    map_a_qu
        First Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    map_b_qu
        Second Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    fwhm_in_a
        Beam FWHM of ``map_a_qu`` in radians.
    fwhm_in_b
        Beam FWHM of ``map_b_qu`` in radians.
    fwhm_out
        Common output beam FWHM in radians.
    filter_config
        Optional harmonic filter applied after beam matching.
    nest
        If ``True``, input maps are interpreted as NEST ordered during the
        harmonic transform stage.

    Returns
    -------
    numpy.ndarray
        Difference template ``processed(map_a_qu) - processed(map_b_qu)`` with
        shape ``(2, npix)``.
    """

    map_a_processed = smooth_and_filter_qu_map(
        map_a_qu,
        fwhm_in=fwhm_in_a,
        fwhm_out=fwhm_out,
        filter_config=filter_config,
        nest=nest,
    )
    map_b_processed = smooth_and_filter_qu_map(
        map_b_qu,
        fwhm_in=fwhm_in_b,
        fwhm_out=fwhm_out,
        filter_config=filter_config,
        nest=nest,
    )
    return map_a_processed - map_b_processed


def build_template_stack(
    template_inputs: Sequence[DifferenceTemplateInput],
    *,
    fwhm_out: float,
    default_filter: HarmonicFilter | None = None,
    nest: bool = False,
) -> tuple[FloatArray, tuple[str, ...]]:
    """Construct a stack of difference templates ready for fitting.

    Parameters
    ----------
    template_inputs
        Sequence of template definitions. Each entry specifies the two maps
        used to build one difference template.
    fwhm_out
        Common output beam FWHM in radians for all templates.
    default_filter
        Harmonic filter used for template entries that do not define their own
        ``filter_config``.
    nest
        If ``True``, input maps are interpreted as NEST ordered during the
        harmonic transform stage.

    Returns
    -------
    tuple
        Tuple ``(templates, template_names)`` where ``templates`` has shape
        ``(n_template, 2, npix)``.

    Raises
    ------
    ValueError
        If ``template_inputs`` is empty.
    """

    template_maps: list[FloatArray] = []
    template_names: list[str] = []

    for index, template_input in enumerate(template_inputs):
        template_filter = template_input.filter_config or default_filter
        template_maps.append(
            construct_difference_template(
                map_a_qu=template_input.map_a_qu,
                map_b_qu=template_input.map_b_qu,
                fwhm_in_a=template_input.fwhm_in_a,
                fwhm_in_b=template_input.fwhm_in_b,
                fwhm_out=fwhm_out,
                filter_config=template_filter,
                nest=nest,
            )
        )
        template_names.append(template_input.name or f"template_{index}")

    if not template_maps:
        raise ValueError("template_inputs must contain at least one template.")

    return np.stack(template_maps, axis=0), tuple(template_names)


def _is_identity_harmonic_operation(
    fwhm_in: float,
    fwhm_out: float,
    filter_config: HarmonicFilter,
) -> bool:
    """Check whether smoothing/filtering would leave the map unchanged.

    Parameters
    ----------
    fwhm_in
        Input beam FWHM in radians.
    fwhm_out
        Requested output beam FWHM in radians.
    filter_config
        Harmonic filter configuration.

    Returns
    -------
    bool
        ``True`` when no beam change, no explicit filters, and no cutoff-based
        filters are requested.
    """

    no_beam_change = np.isclose(fwhm_in, fwhm_out)
    no_ell_filter = filter_config.ell_filter is None
    no_m_filter = filter_config.m_filter is None
    no_ell_cutoff = filter_config.ell_cutoff is None
    no_m_cutoff = filter_config.m_cutoff is None
    return (
        no_beam_change
        and no_ell_filter
        and no_m_filter
        and no_ell_cutoff
        and no_m_cutoff
    )


def _require_healpy() -> None:
    """Raise an informative error when Healpy-dependent paths are requested.

    Raises
    ------
    ImportError
        If ``healpy`` is not available in the current Python environment.
    """

    if hp is None:
        raise ImportError(
            "healpy is required for beam smoothing or harmonic filtering. "
            "Install healpy, or run with matching fwhm and no l/m filter."
        )


def _resolve_lmax(nside: int, filter_config: HarmonicFilter) -> int:
    """Resolve the harmonic truncation consistent with map and filter support.

    Parameters
    ----------
    nside
        Healpix ``nside`` of the working map.
    filter_config
        Harmonic filter configuration.

    Returns
    -------
    int
        Maximum multipole used in the harmonic transform.

    Raises
    ------
    ValueError
        If the requested ``lmax`` exceeds the supported range or resolves to a
        value smaller than 2.
    """

    native_lmax = 3 * nside - 1
    max_supported = native_lmax

    if filter_config.ell_filter is not None:
        max_supported = min(
            max_supported,
            len(np.asarray(filter_config.ell_filter)) - 1,
        )
    if filter_config.m_filter is not None:
        max_supported = min(
            max_supported,
            len(np.asarray(filter_config.m_filter)) - 1,
        )
    if filter_config.ell_cutoff is not None:
        max_supported = min(
            max_supported,
            int(np.ceil(filter_config.ell_cutoff + filter_config.ell_halfwidth)),
        )

    if filter_config.lmax is None:
        lmax = max_supported
    else:
        lmax = int(filter_config.lmax)
        if lmax > max_supported:
            raise ValueError(
                "filter_config.lmax exceeds the supported range of the map or "
                "supplied filters."
            )

    if lmax < 2:
        raise ValueError("Resolved lmax must be at least 2.")
    return lmax


def _build_ell_transfer(
    *,
    lmax: int,
    fwhm_in: float,
    fwhm_out: float,
    filter_config: HarmonicFilter,
) -> FloatArray:
    """Assemble the full multipole transfer function for beam/filter matching.

    Parameters
    ----------
    lmax
        Maximum multipole included in the transfer function.
    fwhm_in
        Input beam FWHM in radians.
    fwhm_out
        Output beam FWHM in radians.
    filter_config
        Harmonic filter configuration.

    Returns
    -------
    numpy.ndarray
        Multiplicative transfer function indexed by ``ell`` with length
        ``lmax + 1``.
    """

    ells = np.arange(lmax + 1, dtype=np.float64)
    sigma_in = _fwhm_to_sigma(fwhm_in)
    sigma_out = _fwhm_to_sigma(fwhm_out)
    sigma_extra_sq = np.maximum(sigma_out**2 - sigma_in**2, 0.0)
    transfer = np.exp(-0.5 * ells * (ells + 1.0) * sigma_extra_sq)

    if filter_config.ell_filter is not None:
        ell_filter_array = np.asarray(filter_config.ell_filter, dtype=np.float64)
        if ell_filter_array.shape[0] < lmax + 1:
            raise ValueError(
                "ell_filter must have length at least lmax + 1 when lmax is explicit."
            )
        transfer *= ell_filter_array[: lmax + 1]

    if filter_config.ell_cutoff is not None:
        transfer *= _build_apodized_lowpass(
            num_modes=lmax + 1,
            cutoff=filter_config.ell_cutoff,
            halfwidth=filter_config.ell_halfwidth,
            transition_type=filter_config.transition_type,
        )

    return transfer


def _apply_m_filter_inplace(
    *,
    alm: npt.NDArray[np.complex128],
    filter_config: HarmonicFilter,
    lmax: int,
) -> None:
    """Apply the configured explicit and cutoff-based ``m`` filters in place.

    Parameters
    ----------
    alm
        Harmonic coefficients modified in place.
    filter_config
        Harmonic filter configuration.
    lmax
        Maximum multipole of ``alm``.
    """

    m_transfer = np.ones(lmax + 1, dtype=np.float64)

    if filter_config.m_filter is not None:
        m_filter_array = np.asarray(filter_config.m_filter, dtype=np.float64)
        if m_filter_array.shape[0] < lmax + 1:
            raise ValueError("m_filter must have length at least lmax + 1.")
        m_transfer *= m_filter_array[: lmax + 1]

    if filter_config.m_cutoff is not None:
        m_transfer *= _build_apodized_lowpass(
            num_modes=lmax + 1,
            cutoff=filter_config.m_cutoff,
            halfwidth=filter_config.m_halfwidth,
            transition_type=filter_config.transition_type,
        )

    ell, emm = hp.Alm.getlm(lmax)
    del ell
    alm *= m_transfer[emm]


def _build_apodized_lowpass(
    *,
    num_modes: int,
    cutoff: float,
    halfwidth: float,
    transition_type: str,
) -> FloatArray:
    """Build a low-pass taper with a NaMaster-style smooth edge.

    Parameters
    ----------
    num_modes
        Number of discrete modes in the output transfer function.
    cutoff
        Center of the low-pass transition band.
    halfwidth
        Half-width of the transition band. A value of zero gives a hard cutoff.
    transition_type
        Smooth edge type. Supported values are ``"C1"`` and ``"C2"``.

    Returns
    -------
    numpy.ndarray
        Low-pass transfer function with length ``num_modes``.

    Raises
    ------
    ValueError
        If ``num_modes`` is not positive, or if ``cutoff``/``halfwidth`` are
        negative.
    """

    if num_modes <= 0:
        raise ValueError("num_modes must be positive.")
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative.")
    if halfwidth < 0:
        raise ValueError("halfwidth must be non-negative.")

    modes = np.arange(num_modes, dtype=np.float64)
    if halfwidth == 0:
        return np.where(modes <= cutoff, 1.0, 0.0)

    # Treat the cutoff as the center of the apodized band so the edge is fully
    # specified by the user-provided half-width.
    left = cutoff - halfwidth
    right = cutoff + halfwidth

    response = np.ones(num_modes, dtype=np.float64)
    response[modes >= right] = 0.0

    transition = (modes > left) & (modes < right)
    if np.any(transition):
        x = (modes[transition] - left) / (right - left)
        response[transition] = 1.0 - _namaster_transition_profile(
            x=x,
            transition_type=transition_type,
        )

    return response


def _namaster_transition_profile(
    *,
    x: FloatArray,
    transition_type: str,
) -> FloatArray:
    """Evaluate the normalized NaMaster-style edge profile.

    Parameters
    ----------
    x
        Normalized transition coordinate in the interval ``[0, 1]``.
    transition_type
        Smooth edge type. Supported values are ``"C1"`` and ``"C2"``.

    Returns
    -------
    numpy.ndarray
        Edge profile evaluated at ``x``.

    Raises
    ------
    ValueError
        If ``transition_type`` is not one of the supported options.
    """

    transition = transition_type.upper()
    if transition == "C1":
        return x - np.sin(2.0 * np.pi * x) / (2.0 * np.pi)
    if transition == "C2":
        return 0.5 * (1.0 - np.cos(np.pi * x))
    raise ValueError("transition_type must be either 'C1' or 'C2'.")


def _fwhm_to_sigma(fwhm: float) -> float:
    """Convert a Gaussian FWHM to its standard deviation.

    Parameters
    ----------
    fwhm
        Full width at half maximum in radians.

    Returns
    -------
    float
        Corresponding Gaussian standard deviation in radians.
    """

    return float(fwhm) / np.sqrt(8.0 * np.log(2.0))
