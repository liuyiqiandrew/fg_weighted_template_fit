from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._types import FloatArray


def as_qu_map(qu_map: npt.ArrayLike, *, name: str) -> FloatArray:
    """Normalize a Q/U map into shape ``(2, npix)``."""

    array = np.asarray(qu_map, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of Q/U maps.")
    if array.shape[0] == 2:
        return array
    if array.shape[1] == 2:
        return array.T
    raise ValueError(f"{name} must have shape (2, npix) or (npix, 2).")


def as_template_stack(templates_qu: npt.ArrayLike, *, name: str) -> FloatArray:
    """Normalize a template stack into shape ``(n_template, 2, npix)``."""

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
    """Broadcast a scalar or 1D weight map onto the stacked Q/U layout."""

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
    """Normalize a per-pixel ``QQ, UU, QU`` covariance array."""

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
    """Evaluate the diagonal-weight inner product on stacked Q/U pixels."""

    return float(np.sum(lhs_qu * weight_map * rhs_qu))


def coerce_rng(
    rng: np.random.Generator | int | None,
) -> np.random.Generator:
    """Normalize seeds and generators to a ``Generator`` instance."""

    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
