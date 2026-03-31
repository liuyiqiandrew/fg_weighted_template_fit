from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HarmonicFilter:
    """Configuration for optional harmonic-domain filtering.

    Parameters
    ----------
    ell_filter
        Optional multiplicative transfer function indexed by multipole ``ell``.
    m_filter
        Optional multiplicative transfer function indexed by azimuthal mode
        ``m``.
    ell_cutoff
        Optional low-pass cutoff in multipole. The response is unity below
        ``ell_cutoff - ell_halfwidth`` and zero above
        ``ell_cutoff + ell_halfwidth``.
    ell_halfwidth
        Half-width of the smooth transition around ``ell_cutoff``.
    m_cutoff
        Optional low-pass cutoff in azimuthal mode. The response is unity below
        ``m_cutoff - m_halfwidth`` and zero above ``m_cutoff + m_halfwidth``.
    m_halfwidth
        Half-width of the smooth transition around ``m_cutoff``.
    transition_type
        Edge shape used for the optional ``ell`` and ``m`` cutoffs. Supported
        values are ``"C1"`` and ``"C2"``. The default ``"C2"`` follows the
        NaMaster convention for a smooth edge on a normalized transition
        coordinate.
    lmax
        Maximum multipole used for the spherical-harmonic transform. If not
        provided, the implementation uses ``3 * nside - 1`` or the shortest
        supplied filter length minus one, whichever is smaller.
    iter
        Number of map-to-alm Jacobi iterations used by ``healpy.map2alm``.
    """

    ell_filter: npt.ArrayLike | None = None
    m_filter: npt.ArrayLike | None = None
    ell_cutoff: float | None = None
    ell_halfwidth: float = 0.0
    m_cutoff: float | None = None
    m_halfwidth: float = 0.0
    transition_type: str = "C2"
    lmax: int | None = None
    iter: int = 3


@dataclass(frozen=True)
class DifferenceTemplateInput:
    """Inputs needed to build a foreground template from two Q/U maps.

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
    noise_cov_a
        Optional per-pixel ``QQ, UU, QU`` covariance for ``map_a_qu`` with
        shape ``(3, npix)`` or ``(npix, 3)``.
    noise_cov_b
        Optional per-pixel ``QQ, UU, QU`` covariance for ``map_b_qu`` with
        shape ``(3, npix)`` or ``(npix, 3)``.
    filter_config
        Optional harmonic filter applied while constructing the template. If
        omitted, the default filter passed to the fit routine is used.
    name
        Human-readable name for the template.
    """

    map_a_qu: npt.ArrayLike
    map_b_qu: npt.ArrayLike
    fwhm_in_a: float
    fwhm_in_b: float
    noise_cov_a: npt.ArrayLike | None = None
    noise_cov_b: npt.ArrayLike | None = None
    filter_config: HarmonicFilter | None = None
    name: str = "template"


@dataclass(frozen=True)
class WeightedFitResult:
    """Result of the weighted template fit."""

    amplitudes: FloatArray
    normal_matrix: FloatArray
    normal_matrix_inverse: FloatArray
    rhs: FloatArray
    residual_qu: FloatArray
    processed_target_qu: FloatArray
    processed_templates_qu: FloatArray
    template_names: tuple[str, ...]
    solver: str


@dataclass(frozen=True)
class BootstrapFitResult:
    """Monte Carlo uncertainty estimate for weighted template amplitudes."""

    reference_fit: WeightedFitResult
    amplitude_samples: FloatArray
    amplitude_mean: FloatArray
    amplitude_std: FloatArray
    template_names: tuple[str, ...]
