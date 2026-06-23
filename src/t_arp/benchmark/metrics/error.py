import chex
import jax
from jax import numpy as jnp

from t_arp.benchmark.metrics.utils import validate_shapes


def _relative_error(
    tensor_org: chex.Array, tensor_rec: chex.Array, eps: float = 1e-8
) -> chex.Array:
    return jnp.linalg.norm(tensor_org - tensor_rec) / (
        jnp.linalg.norm(tensor_org) + eps
    )


def compute_relative_error(
    predictions: chex.Array, targets: chex.Array, eps: float = 1e-8
) -> chex.Array:
    predictions, targets = validate_shapes(predictions, targets)
    batch_relative_error = jax.vmap(_relative_error, in_axes=(0, 0))(
        predictions, targets
    )
    return batch_relative_error
