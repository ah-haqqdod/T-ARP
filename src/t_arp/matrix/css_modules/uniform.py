from dataclasses import dataclass
from typing import Optional

import chex
import jax
import jax.numpy as jnp

from t_arp.matrix.css_modules.abc import CSS_module


@dataclass
class UniformSampling_module(CSS_module):
    # css_params: Optional[CSS_params] = eqx.field(default=None, static=True)

    def __call__(self, y: chex.Array, key: Optional[chex.PRNGKey] = None) -> chex.Array:
        if key is None:
            raise ValueError("key must be provided for randomized methods")
        chex.assert_rank(y, 2)

        return UniformSampling_module._uniform(key, y, self.r)

    def sample_row_from_orthonormal_basis(
        self, Q: chex.Array, key: Optional[chex.PRNGKey] = None
    ) -> chex.Array:
        if key is None:
            raise ValueError("key must be provided for randomized methods")
        chex.assert_rank(Q, 2)
        return UniformSampling_module._uniform_column_sample(key, Q.T, self.r)

    @staticmethod
    def _uniform(key: chex.PRNGKey, y: chex.Array, r: int) -> chex.Array:
        key, subkey = jax.random.split(key)
        J = UniformSampling_module._uniform_column_sample(subkey, y, r)
        return J

    @staticmethod
    def _uniform_column_sample(key: chex.PRNGKey, y: chex.Array, r: int) -> chex.Array:
        max_rank = min(y.shape)
        r = min(r, max_rank)
        indices = jnp.arange(y.shape[1])
        J = jax.random.choice(key, indices, shape=(r,), replace=False)
        return J
