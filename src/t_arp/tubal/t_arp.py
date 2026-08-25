from dataclasses import dataclass
from functools import reduce
from typing import Literal

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
    use_fast: bool = eqx.field(default=True, static=True)

    def __call__(self, key, V: TMatrix):
        # cannot sample more columns than the number of rows or columns of V
        n_slices = min(self.n_slices, *V.shape[:2])

        if self.method == "householder":
            return self._arp_householder(
                key=key,
                V=V,
                n_slices=n_slices,
                use_fast=self.use_fast,
                use_derandomized=self.use_derandomized,
            )

        if self.method == "orth_proj_pinv":
            return self._arp_orth_proj(
                key=key,
                V=V,
                n_slices=n_slices,
                use_pinv=True,
                use_fast=self.use_fast,
                use_derandomized=self.use_derandomized,
            )

        return self._arp_orth_proj(
            key=key,
            V=V,
            n_slices=n_slices,
            use_pinv=False,
            use_fast=self.use_fast,
            use_derandomized=self.use_derandomized,
        )

    @staticmethod
    def _arp_orth_proj(
        key,
        V: TMatrix,
        n_slices: int,
        use_pinv=False,
        use_fast: bool = True,
        use_derandomized: bool = False,
    ):
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
            j_k, v_t_data = TARP._sample_row(
                key=subkey,
                V=V,
                use_fast=use_fast,
                use_derandomized=use_derandomized,
            )
            v = TMatrixTOnly(v_t_data, data_domain_shape=(1, *V.shape[1:]), data_dtype=V.dtype)

            if use_pinv:
                v_t = v.facewise_operation(jnp.linalg.pinv)
            else:
                # v_norm_inv = v.facewise_operation(
                #     # lambda M: jnp.reshape(1 / (jnp.linalg.norm(M) + 1e-8), (1, 1))
                #     lambda M: 1 / (jnp.linalg.norm(M) + 1e-8)
                # )
                v_norm_inv = v.facewise_operation(
                    lambda M: jnp.where(
                        jnp.linalg.norm(M) > jnp.finfo(M.dtype).eps,
                        1.0 / jnp.linalg.norm(M),
                        0.0,
                    )
                )
                # print("v t-vector norm inv shape", v_norm_inv.shape)
                # NOTE: this method uses new t-multiplication
                v = v * v_norm_inv
                v_t = v.T

            # Pivot V
            # v in (1, r, *C) and v_t in (r, 1, *C)
            V = V - ((V @ v_t) @ v)

            return (key, V, i + 1), j_k

        key, subkey = jax.random.split(key)

        # We only need the t_data information (for comp. efficiency)
        V = TMatrixTOnly.asmatrix(V.t_data, V)

        init_carry = (subkey, V, 0)
        _, (J) = jax.lax.scan(loop_body, init_carry, length=n_slices)

        return J

    @staticmethod
    def _arp_householder(
        key,
        V: TMatrix,
        n_slices: int,
        use_fast: bool = True,
        use_derandomized: bool = False,
    ):
        # used to zero out a lateral slice
        zero_hypercolumn = jnp.zeros(
            shape=(V.shape[0], 1, reduce(lambda a, b: a * b, V.shape[2:])),
            dtype=V.t_data.dtype,
        )

        def loop_body(carry, xs):
            key, V, i = carry

            # Sample an index and the corresponding row
            key, subkey = jax.random.split(key)
            j_k, v = TARP._sample_row(
                key=subkey,
                V=V,
                use_fast=use_fast,
                use_derandomized=use_derandomized,
            )
            # Apply t-Householder
            V_t_data = jax.vmap(
                HouseholderReflection.reflect_row_vector,
                in_axes=(
                    -1,
                    -1,
                    None,
                ),
                out_axes=(-1),
            )(V.t_data, v, i)

            # Hide "columns".
            V_t_data = jax.lax.dynamic_update_index_in_dim(
                V_t_data, zero_hypercolumn, index=i, axis=1
            )

            # Create the object for next iteration
            V = TMatrixTOnly.asmatrix(V_t_data, V)

            return (key, V, i + 1), j_k

        key, subkey = jax.random.split(key)

        # We only need the t_data information (for comp. efficiency)
        V = TMatrixTOnly.asmatrix(V.t_data, V)

        init_carry = (subkey, V, 0)
        _, (J) = jax.lax.scan(loop_body, init_carry, length=n_slices)

        return J

    @staticmethod
    # @jax.jit(static_argnames=("max_n_slices", "indices"))
    def _sample_row(
        key: chex.PRNGKey,
        V: TMatrixTOnly,
        use_fast: bool = True,
        use_derandomized: bool = False,
    ):
        # Computes the tensor norm for every horizontal slice
        # norm_sq_fn = jax.vmap(lambda M: jnp.linalg.norm(M) ** 2, in_axes=(0))
        norm_sq_fn = jax.vmap(
            lambda M: jnp.real(jnp.vdot(M.ravel(), M.ravel())),
            in_axes=(0)
        )


        # Compute leverage scores
        if use_fast:
            # here we use Parseval's identity.
            p = norm_sq_fn(V.t_data)
        else:
            # here .data is computed on the fly using ifft of t_data.
            p = norm_sq_fn(V.data)

        p = jnp.where(p > jnp.finfo(p.dtype).eps, p, 0)
        p = p / jnp.sum(p)

        # sample index
        if use_derandomized:
            j_k = jnp.argmax(p)
        else:
            key, subkey = jax.random.split(key)
            j_k = jax.random.choice(subkey, V.shape[0], p=p)

        # index the row
        v_t_data = jax.lax.dynamic_index_in_dim(V.t_data, index=j_k, axis=0)

        return j_k, v_t_data
