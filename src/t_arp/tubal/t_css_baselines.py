# Tubal column subset selection baselines
from dataclasses import dataclass
from typing import Literal

import chex
import equinox as eqx
import jax
from jax import numpy as jnp

# from t_arp.tubal import TMatrix, TMatrixTOnly
from t_arp.tubal.t_matrix import TMatrix, TMatrixAbstract, TMatrixTOnly


@dataclass
class TCSSBaselines(eqx.Module):
    n_slices: int = eqx.field(static=True)
    use_householder: bool = eqx.field(default=True, static=True)
    method: Literal["uniform", "leverage_scores", "length_squared"] = eqx.field(
        default="uniform", static=True
    )

    def __init__(
        self,
        n_slices,
        method: Literal["uniform", "leverage_scores", "length_squared"] = "uniform",
    ):
        self.n_slices = n_slices
        self.method = method

    def __call__(self, key, V: TMatrix):
        """
        Returns the indices of the slices to be used for the CSS baseline.
        REMARK: length_squared expects the original matrix for which V was computed, not the right singular vectors V.

        Args:
            key: A JAX random key.
            V: The TMatrix to sample from.

        Returns:
            J: The indices of the slices to be used for the CSS baseline.
        """
        # ARP method returns only J
        if self.method == "uniform":
            J = self.__uniform(key=key, V=V, n_slices=self.n_slices)
        elif self.method == "length_squared":
            J = self.__length_squared(key=key, M=V, n_slices=self.n_slices)
        elif self.method == "leverage_scores":
            J = self.__leverage_scores(key=key, V=V, n_slices=self.n_slices)
        else:
            return NotImplemented

        return J

    @staticmethod
    def __uniform(key, V: TMatrix, n_slices: int):
        p = jnp.ones(V.shape[0]) / V.shape[0]
        idcs = jnp.arange(0, V.shape[0])

        # print(f"Uniform args: V.shape={V.shape}, n_slices={n_slices}")

        key, subkey = jax.random.split(key)
        J = jax.random.choice(subkey, idcs, shape=(n_slices,), replace=False, p=p)

        return J

    @staticmethod
    def __length_squared(key, M: TMatrix, n_slices: int):
        shape = M.shape
        M_ = jnp.reshape(M.data, (shape[0], shape[1], -1))

        len_sqr = jnp.linalg.norm(M_, axis=(0, 2)) ** 2
        p = len_sqr / jnp.sum(len_sqr)
        idcs = jnp.arange(0, M.shape[1])

        key, subkey = jax.random.split(key)
        J = jax.random.choice(subkey, idcs, shape=(n_slices,), replace=False, p=p)

        return J

    @staticmethod
    def __leverage_scores(key, V: TMatrix, n_slices: int):
        # NOTE: this is the same norm function used in ARP methods
        norm_sq_fn = jax.vmap(lambda M: jnp.linalg.norm(M) ** 2, in_axes=(0))

        p = norm_sq_fn(V.data)
        denom = jnp.sum(p)
        p = p / denom
        idcs = jnp.arange(0, V.shape[0])

        key, subkey = jax.random.split(key)
        J = jax.random.choice(subkey, idcs, shape=(n_slices,), replace=False, p=p)

        return J
