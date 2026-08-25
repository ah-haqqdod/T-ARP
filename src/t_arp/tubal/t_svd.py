from typing import Optional, Tuple

import chex
import jax
from jax import numpy as jnp

from t_arp.matrix import RSVD
from t_arp.tubal.t_matrix import TMatrix, TMatrixTOnly

@jax.jit
def t_svd(
    M: TMatrix,
) -> Tuple[
    TMatrixTOnly,
    TMatrixTOnly,
    TMatrixTOnly,
]:
    return M.facewise_operation(lambda A: jnp.linalg.svd(A, full_matrices=False))


@jax.jit(static_argnames=["n_slices"])
def t_tsvd(
    M: TMatrix, n_slices: int
) -> Tuple[
    TMatrixTOnly,
    TMatrixTOnly,
    TMatrixTOnly,
]:
    U, S, Vt = t_svd(M)

    # Truncate to n_slices
    indices = jnp.arange(n_slices)
    shape = M.shape
    tube_shape = shape[2:]
    U = TMatrixTOnly(
        t_data=jnp.take(U.t_data, indices=indices, axis=1),
        data_domain_shape=(shape[0], n_slices) + tube_shape,
        data_dtype=M.dtype,
    )
    S = TMatrixTOnly(
        t_data=jnp.take(S.t_data, indices=indices, axis=0),
        data_domain_shape=(n_slices,) + tube_shape,
        data_dtype=M.dtype,
    )
    Vt = TMatrixTOnly(
        t_data=jnp.take(Vt.t_data, indices=indices, axis=0),
        data_domain_shape=(n_slices, shape[1]) + tube_shape,
        data_dtype=M.dtype,
    )

    return U, S, Vt


@jax.jit(static_argnames=["n_slices", "n_oversamples", "n_subspace_iters"])
def t_rsvd(
    key: chex.PRNGKey,
    M: TMatrix,
    n_slices: int,
    n_oversamples: Optional[int] = None,
    n_subspace_iters: Optional[int] = None,
) -> Tuple[
    TMatrixTOnly,
    TMatrixTOnly,
    TMatrixTOnly,
]:
    shape = M.shape
    rsvd_rank = min(n_slices, shape[0], shape[1])

    rsvd = RSVD(
        rank=rsvd_rank,
        n_oversamples=n_oversamples,
        n_subspace_iters=n_subspace_iters,
        return_range=False
    )

    U, S, Vt = M.facewise_operation(lambda key, A: rsvd(key=key, A=A), key=key)

    U = U.conjugate_symmetrize()
    S = S.conjugate_symmetrize()
    Vt = Vt.conjugate_symmetrize()

    return U, S, Vt
