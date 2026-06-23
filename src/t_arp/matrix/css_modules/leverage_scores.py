from dataclasses import dataclass
from typing import Optional

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from t_arp.matrix.css_modules.abc import CSS_module, CSS_params
from t_arp.matrix.utils import RSVD


@dataclass
class LeverageScoresSampling_params(CSS_params):
    rsvd_r: int = eqx.field(static=True)
    n_oversamples: Optional[int] = eqx.field(default=None, static=True)
    n_subspace_iters: Optional[int] = eqx.field(default=None, static=True)


@dataclass
class LeverageScoresSampling_module(CSS_module):
    # css_params: LeverageScoresSampling_params = eqx.field(static=True)

    def __call__(self, y: chex.Array, key: Optional[chex.PRNGKey] = None) -> chex.Array:
        if key is None:
            raise ValueError("key must be provided for randomized methods")
        if not isinstance(self.css_params, LeverageScoresSampling_params):
            raise ValueError(
                "css_params must be an instance of LeverageScoresSampling_params"
            )

        chex.assert_rank(y, 2)

        return LeverageScoresSampling_module._leverage_scores(
            key,
            y,
            self.r,
            self.css_params,
        )

    def sample_row_from_orthonormal_basis(
        self, Q: chex.Array, key: Optional[chex.PRNGKey] = None
    ) -> chex.Array:
        if key is None:
            raise ValueError("key must be provided for randomized methods")
        if not isinstance(self.css_params, LeverageScoresSampling_params):
            raise ValueError(
                "css_params must be an instance of LeverageScoresSampling_params"
            )

        chex.assert_rank(Q, 2)

        return LeverageScoresSampling_module._leverage_score_row_sample(key, Q, self.r)

    @staticmethod
    def _leverage_score_row_sample(
        key: chex.PRNGKey, Q: chex.Array, r: int
    ) -> chex.Array:
        max_rank = min(Q.shape)
        r = min(r, max_rank)
        p = jnp.linalg.norm(Q, axis=1) ** 2  # column norms, shape (n_cols,)
        p = p / jnp.sum(p)
        key, subkey = jax.random.split(key)
        indices = jnp.arange(Q.shape[0])
        J = jax.random.choice(subkey, indices, shape=(r,), replace=False, p=p)
        return J

    @staticmethod
    def _leverage_scores(
        key: chex.PRNGKey,
        y: chex.Array,
        r: int,
        leverage_scores_params: LeverageScoresSampling_params,
    ):
        # RSVD then leverage‑score sampling (column norms)
        key, subkey = jax.random.split(key)
        rsvd = RSVD(
            rank=leverage_scores_params.rsvd_r,
            n_oversamples=leverage_scores_params.n_oversamples,
            n_subspace_iters=leverage_scores_params.n_subspace_iters,
        )
        _, _, Vt = rsvd(subkey, y)  # Vt shape (rank, n_cols)

        key, subkey = jax.random.split(key)
        J = LeverageScoresSampling_module._leverage_score_row_sample(subkey, Vt.T, r)
        # p = jnp.linalg.norm(Vt, axis=0) ** 2  # column norms, shape (n_cols,)
        # p = p / jnp.sum(p)
        # key, subkey = jax.random.split(key)
        # indices = jnp.arange(y.shape[1])
        # J = jax.random.choice(subkey, indices, shape=(r,), replace=False, p=p)
        return J
