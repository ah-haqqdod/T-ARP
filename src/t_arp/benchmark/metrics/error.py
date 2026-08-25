import chex
import jax
from jax import numpy as jnp

from t_arp.benchmark.metrics.utils import validate_shapes


def _relative_error(
    x: chex.Array, x_hat: chex.Array, eps: float = 1e-8
) -> chex.Array:
    x_norm = jnp.linalg.norm(x)
    return jax.lax.cond(
        x_norm > jnp.finfo(x_norm.dtype).eps,
        lambda x, x_hat: jnp.linalg.norm(x - x_hat) / (x_norm),
        lambda x, x_hat: jnp.linalg.norm(x_hat),
        x,
        x_hat,
    )


def compute_relative_error(
    predictions: chex.Array, targets: chex.Array, eps: float = 1e-8
) -> chex.Array:
    predictions, targets = validate_shapes(predictions, targets)
    batch_relative_error = jax.vmap(_relative_error, in_axes=(0, 0))(
        targets, predictions
    )
    return batch_relative_error
