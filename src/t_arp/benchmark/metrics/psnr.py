# modification of metrax psnr implementation. from modularized implementation to pure functions. Source: https://github.com/google/metrax

r"""PSNR (Peak Signal-to-Noise Ratio)  Metric.

This class calculates the Peak Signal-to-Noise Ratio (PSNR) between two images
to measure the quality of a reconstructed image compared to a reference.

.. math::

    \text{PSNR}(I, J) = 10 \cdot \log_{10} \left(
    \frac{\max(I)^2}{\text{MSE}(I, J)} \right)

Where:
  - :math:`\max(I)` is the maximum possible pixel value of the input image.
  - :math:`\text{MSE}(I, J)` is the mean squared error between images
  :math:`I` and :math:`J`.
"""

import jax
import chex
from jax import numpy as jnp

from t_arp.benchmark.metrics.utils import validate_shapes


def _calculate_psnr(
    img1: chex.Array,
    img2: chex.Array,
    max_val: float,
    eps: float = 0,
) -> chex.Array:
    """Computes PSNR (Peak Signal-to-Noise Ratio) values.

    Args:
            img1: Predicted images, shape ``(batch, H, W, C)``.
            img2: Ground‑truth images, same shape as ``img1``.
            max_val: Dynamic range of the images (e.g. ``1.0`` or ``255``).
            eps: Small constant to avoid ``log(0)`` when images are identical.

        Returns:
            A 1D JAX array of shape ``(batch,)`` containing PSNR in dB.
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Input images must have the same shape, got {img1.shape} and {img2.shape}."
        )
    # if img1.ndim != 4:  # (batch, H, W, C)
    #     raise ValueError(
    #         f"Inputs must be 4‑D (batch, height, width, channels), got {img1.ndim}‑D."
    #     )

    if jax.config.read("jax_enable_x64"):
        img1 = img1.astype(jnp.float64)
        img2 = img2.astype(jnp.float64)
    else:
        img1 = img1.astype(jnp.float32)
        img2 = img2.astype(jnp.float32)

    # Mean‑squared error per image.
    mse = jnp.mean(jnp.square(img1 - img2), axis=(1, 2, 3))
    mse = jnp.maximum(mse, eps)

    psnr = 20.0 * jnp.log10(max_val) - 10.0 * jnp.log10(mse)
    return psnr


def compute_psnr(
    predictions: chex.Array,
    targets: chex.Array,
    max_val: float,
) -> chex.Array:
    """Computes PSNR for a batch of images.

    Args:
        predictions: A JAX array of predicted images, with shape ``(batch, H, W,
            C)``.
        targets: A JAX array of ground truth images, with shape ``(batch, H, W,
            C)``.
        max_val: The maximum possible pixel value (dynamic range) of the images
            (e.g., 1.0 for float images in [0,1], 255 for uint8 images).

    Returns:
        An array of PSNR values across the batch.
    """
    predictions, targets = validate_shapes(predictions, targets)

    batch_psnr = _calculate_psnr(predictions, targets, max_val=max_val)

    return batch_psnr
