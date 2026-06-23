from typing import Tuple

import chex
import jax.numpy as jnp


def _add_batch_axis(array: chex.Array) -> chex.Array:
    """Adds a new axis to the array."""
    return jnp.expand_dims(array, axis=0)


def validate_shapes(
    predictions: chex.Array, targets: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    predictions = predictions if predictions.ndim == 4 else _add_batch_axis(predictions)
    targets = targets if targets.ndim == 4 else _add_batch_axis(targets)
    if predictions.ndim != 4:
        raise ValueError(
            f"Predictions must have 4 dimensions (Batch, Height, Width, Channels), got {predictions.shape}."
        )
    if targets.ndim != 4:
        raise ValueError(
            f"Targets must have 4 dimensions (Batch, Height, Width, Channels), got {targets.shape}."
        )

    return predictions, targets
