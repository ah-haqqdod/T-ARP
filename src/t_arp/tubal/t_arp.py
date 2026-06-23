from dataclasses import dataclass
from functools import partial, reduce
from typing import Callable, Literal

import chex
import equinox as eqx
import jax
from jax import numpy as jnp

from t_arp.matrix import HouseholderReflection
from t_arp.tubal.t_matrix import TMatrix, TMatrixTOnly

# NOTE: this T-ARP module uses two new tubal algebra operations: t-multiplication and t-householder reflection.


@dataclass
class TARP(eqx.Module):
    n_slices: int = eqx.field(static=True)
    method: Literal["householder", "orth_proj_pinv", "orth_proj_normalized"] = (
        eqx.field(default="householder", static=True)
    )
    use_derandomized: bool = eqx.field(default=False, static=True)

    def __call__(self, key, V: TMatrix):
        # cannot sample more columns than the number of rows or columns of V
        n_slices = min(self.n_slices, *V.shape[:2])

        if self.method == "householder":
            return self._arp_householder(
                key=key,
                V=V,
                n_slices=n_slices,
                use_derandomized=self.use_derandomized,
            )

        if self.method == "orth_proj_pinv":
            return self._arp_orth_proj(
                key=key,
                V=V,
                n_slices=n_slices,
                use_pinv=True,
                use_derandomized=self.use_derandomized,
            )

        return self._arp_orth_proj(
            key=key,
            V=V,
            n_slices=n_slices,
            use_pinv=False,
            use_derandomized=self.use_derandomized,
        )

    @staticmethod
    def _arp_orth_proj(
        key,
        V: TMatrix,
        n_slices: int,
        use_pinv=False,
        use_derandomized: bool = False,
    ):
        max_n_slices = min(V.shape[0], V.shape[1])
        indices = jnp.arange(0, V.shape[0])

        def loop_body(carry, x):
            """V is of shape (n, r, *C), s.t. n >= r.
            We need to select subset of indices in the axis of size n (rows).

            Args:
                carry (_type_): _description_
                x (_type_): _description_

            Returns:
                _type_: _description_
            """
            key, V, i = carry

            key, subkey = jax.random.split(key)
            j_k, v = TARP._sample_row(
                key=subkey,
                V=V,
                indices=indices,
                use_derandomized=use_derandomized,
                max_n_slices=max_n_slices,
                i=i,
            )

            v = TMatrix(v)
            if use_pinv:
                v_t = v.facewise_operation(jnp.linalg.pinv)
            else:
                v_norm_inv = v.facewise_operation(
                    lambda M: jnp.reshape(1 / (jnp.linalg.norm(M) + 1e-8), (1, 1))
                )
                # print("v t-vector norm inv shape", v_norm_inv.shape)
                # NOTE: this method uses new t-multiplication
                v = v * v_norm_inv
                v_t = v.T

            # Pivot V
            # v in (1, r, *C) and v_t in (r, 1, *C)
            V = V - ((V @ v_t) @ v).create_t_matrix()

            return (key, V, i + 1), j_k

        key, subkey = jax.random.split(key)
        init_carry = (subkey, V, 0)
        _, (J) = jax.lax.scan(loop_body, init_carry, length=n_slices)

        return J

    @staticmethod
    def _arp_householder(
        key,
        V: TMatrix,
        n_slices: int,
        use_derandomized: bool = False,
    ):
        # used to zero out a vertical slice
        zero_hypercolumn = jnp.zeros(
            shape=(V.shape[0], 1, reduce(lambda a, b: a * b, V.shape[2:])),
            dtype=V.t_data.dtype,
        )
        max_n_slices = min(V.shape[0], V.shape[1])
        indices = jnp.arange(0, V.shape[0])

        def loop_body(carry, xs):
            key, V, i = carry

            key, subkey = jax.random.split(key)
            j_k, v = TARP._sample_row(
                key=subkey,
                V=V,
                indices=indices,
                use_derandomized=use_derandomized,
                max_n_slices=max_n_slices,
                i=i,
            )
            v_tubal = TMatrix(v)

            # Apply t-Householder
            V_t_data = jax.vmap(
                HouseholderReflection.reflect_row_vector,
                in_axes=(
                    -1,
                    -1,
                    None,
                ),
                out_axes=(-1),
            )(V.t_data, v_tubal.t_data, i)

            # Hide columns. Experiments show that hiding the columns in source domain and frequency domain produce similar results.
            # Here we use the frequency domain mask to hide the columns since it will omit an unnecessary fourier transformation from source to frequency.
            V_t_data = jax.lax.dynamic_update_index_in_dim(
                V_t_data, zero_hypercolumn, index=i, axis=1
            )

            # Transform back to source domain after reflection and hiding columns
            V = TMatrixTOnly.asmatrix(V_t_data, V).create_t_matrix()

            return (key, V, i + 1), j_k

        key, subkey = jax.random.split(key)
        init_carry = (subkey, V, 0)
        _, (J) = jax.lax.scan(loop_body, init_carry, length=n_slices)

        return J

    @staticmethod
    # @jax.jit(static_argnames=("max_n_slices", "indices"))
    def _sample_row(
        key: chex.PRNGKey,
        V: TMatrix,
        indices: chex.Array,
        use_derandomized: bool = False,
        max_n_slices: int = 0,
        i: int = 0,
    ):
        # Computes the tensor norm for every horizontal slice
        norm_sq_fn = jax.vmap(lambda M: jnp.linalg.norm(M) ** 2, in_axes=(0))
        # Compute leverage scores
        p = norm_sq_fn(V.data)
        # denom_theor = jnp.maximum(1, max_n_slices - i)
        # jax.block_until_ready(p)
        # jax.debug.print(
        #     "theoretical denom {x}, actual denom {y}",
        #     x=denom_theor,
        #     y=jnp.sum(p),
        # )
        # p = p / denom_theor
        p = p / jnp.sum(p)

        # sample index
        if use_derandomized:
            j_k = jnp.argmax(p)
        else:
            key, subkey = jax.random.split(key)
            j_k = jax.random.choice(subkey, indices, p=p)

        # index the row
        v = jax.lax.dynamic_index_in_dim(V.data, index=j_k, axis=0)

        return j_k, v
