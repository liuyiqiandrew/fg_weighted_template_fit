from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._types import FloatArray


def as_qu_map(qu_map: npt.ArrayLike, *, name: str) -> FloatArray:
    """Normalize a Q/U map into shape ``(2, npix)``.

    Parameters
    ----------
    qu_map
        Input Q/U map with shape ``(2, npix)`` or ``(npix, 2)``.
    name
        Human-readable array name used in validation errors.

    Returns
    -------
    numpy.ndarray
        Q/U map with shape ``(2, npix)`` and ``float64`` dtype.

    Raises
    ------
    ValueError
        If ``qu_map`` is not a 2D array with one dimension of length 2.
    """

    array = np.asarray(qu_map, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of Q/U maps.")
    if array.shape[0] == 2:
        return array
    if array.shape[1] == 2:
        return array.T
    raise ValueError(f"{name} must have shape (2, npix) or (npix, 2).")


def as_template_stack(templates_qu: npt.ArrayLike, *, name: str) -> FloatArray:
    """Normalize a template stack into shape ``(n_template, 2, npix)``.

    Parameters
    ----------
    templates_qu
        Template stack with shape ``(n_template, 2, npix)`` or
        ``(n_template, npix, 2)``.
    name
        Human-readable array name used in validation errors.

    Returns
    -------
    numpy.ndarray
        Template stack with shape ``(n_template, 2, npix)`` and ``float64``
        dtype.

    Raises
    ------
    ValueError
        If ``templates_qu`` does not have one of the supported shapes.
    """

    array = np.asarray(templates_qu, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(
            f"{name} must have shape (n_template, 2, npix) or (n_template, npix, 2)."
        )
    if array.shape[1] == 2:
        return array
    if array.shape[2] == 2:
        return np.swapaxes(array, 1, 2)
    raise ValueError(
        f"{name} must have shape (n_template, 2, npix) or (n_template, npix, 2)."
    )


def as_weight_map(
    weight_map: npt.ArrayLike,
    *,
    npix: int,
    name: str,
) -> FloatArray:
    """Broadcast a weight definition onto the stacked Q/U layout.

    Parameters
    ----------
    weight_map
        Scalar, pixel-domain weight vector, or explicit Q/U weight map.
        Accepted shapes are scalar, ``(npix,)``, ``(2, npix)``, and
        ``(npix, 2)``.
    npix
        Number of pixels expected in the output map.
    name
        Human-readable array name used in validation errors.

    Returns
    -------
    numpy.ndarray
        Weight map with shape ``(2, npix)`` and ``float64`` dtype.

    Raises
    ------
    ValueError
        If ``weight_map`` does not have one of the supported shapes.
    """

    array = np.asarray(weight_map, dtype=np.float64)

    if array.ndim == 0:
        return np.full((2, npix), float(array), dtype=np.float64)
    if array.ndim != 2 and array.ndim != 1:
        raise ValueError(f"{name} must have shape (npix,), (2, npix), or (npix, 2).")
    if array.ndim == 1:
        if array.shape[0] != npix:
            raise ValueError(f"{name} has length {array.shape[0]}, expected {npix}.")
        return np.vstack([array, array])
    if array.shape == (2, npix):
        return array
    if array.shape == (npix, 2):
        return array.T

    raise ValueError(f"{name} must have shape (npix,), (2, npix), or (npix, 2).")


def as_covariance(
    covariance: npt.ArrayLike,
    *,
    name: str,
    npix: int | None = None,
) -> FloatArray:
    """Normalize a per-pixel ``QQ, UU, QU`` covariance array.

    Parameters
    ----------
    covariance
        Covariance array with shape ``(3, npix)`` or ``(npix, 3)`` in the
        order ``QQ, UU, QU``.
    name
        Human-readable array name used in validation errors.
    npix
        Optional number of pixels expected in the output covariance.

    Returns
    -------
    numpy.ndarray
        Covariance array with shape ``(3, npix)`` and ``float64`` dtype.

    Raises
    ------
    ValueError
        If ``covariance`` does not have one of the supported shapes, or if its
        pixel count does not match ``npix`` when that argument is supplied.
    """

    array = np.asarray(covariance, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (3, npix) or (npix, 3).")
    if array.shape[0] == 3:
        cov = array
    elif array.shape[1] == 3:
        cov = array.T
    else:
        raise ValueError(f"{name} must have shape (3, npix) or (npix, 3).")

    if npix is not None and cov.shape[1] != npix:
        raise ValueError(f"{name} has npix={cov.shape[1]}, expected {npix}.")
    return cov


def weighted_inner_product(
    lhs_qu: FloatArray,
    rhs_qu: FloatArray,
    weight_map: FloatArray,
) -> float:
    """Evaluate the diagonal-weight inner product on stacked Q/U pixels.

    Parameters
    ----------
    lhs_qu
        Left-hand Q/U map with shape ``(2, npix)``.
    rhs_qu
        Right-hand Q/U map with shape ``(2, npix)``.
    weight_map
        Diagonal pixel weight map with shape ``(2, npix)``.

    Returns
    -------
    float
        Weighted inner product ``sum(lhs_qu * weight_map * rhs_qu)``.
    """

    return float(np.sum(lhs_qu * weight_map * rhs_qu))


def coerce_rng(
    rng: np.random.Generator | int | None,
) -> np.random.Generator:
    """Normalize seeds and generators to a ``Generator`` instance.

    Parameters
    ----------
    rng
        Existing generator, integer seed, or ``None``.

    Returns
    -------
    numpy.random.Generator
        Generator instance derived from ``rng``.
    """

    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
