from functools import partial
from typing import Callable, Optional, Tuple, Union

import chex
import equinox as eqx
import jax
from jax import numpy as jnp

from t_arp.matrix import RSVD, CSS_module
from t_arp.tubal import TMatrix, TMatrixTOnly


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
    n_oversamples: int,
    n_subspace_iters: int,
) -> Tuple[
    TMatrixTOnly,
    TMatrixTOnly,
    TMatrixTOnly,
]:
    rsvd = RSVD(
        rank=min(n_slices, M.shape[0], M.shape[1]),
        n_oversamples=n_oversamples,
        n_subspace_iters=n_subspace_iters,
    )

    def rsvd_fn(key: chex.PRNGKey, A: chex.Array):
        return rsvd(key=key, A=A)

    return M.facewise_operation(rsvd_fn, key=key)
    # _, _, t_Vt = M.facewise_operation(rsvd_fn, key=key)

    # V = t_Vt.T.create_t_matrix()
    # return V


@jax.jit
def reconstruct_t_svd(U: TMatrix, S: TMatrix, Vt: TMatrix) -> jnp.ndarray:
    S = S.facewise_operation(lambda x: jnp.diag(x.flatten()))
    recon = (U @ S @ Vt).create_t_matrix()
    return recon.data


def t_cross(
    key,
    A: TMatrix,
    V: TMatrix,
    partial_tcss_module_constructor: Callable,
    n_vert_slices: int,
    n_horiz_slices: Optional[int] = None,
    use_intersection: bool = False,
):
    """
    Perform TARP cross decomposition on the given TMatrixAbstract A and V.

    key: random key for tubal column selection
    A: original tubal matrix
    V: orthonormal basis of the tubal matrix A (in source domain, hence TMatrix is required)
    partial_tcssp_module_constructor: a partial constructor for the tubal column selection module where the only exposed argument is number of slices (e.g. partial(TARP, use_householder=True))
    n_vert_slices: number of vertical slices
    n_horiz_slices: number of horizontal slices
    """

    def __compute_W(
        A_J: TMatrix, A_I: TMatrix, I: chex.Array
    ) -> Union[TMatrix, TMatrixTOnly]:
        # Intersection
        if use_intersection:
            W = TMatrix(jnp.take(A_J.data, I, axis=0)).facewise_operation(
                partial(jnp.linalg.pinv)
            )
        else:
            W = (
                A_J.facewise_operation(jnp.linalg.pinv)
                @ A
                @ A_I.facewise_operation(jnp.linalg.pinv)
            )
        return W

    tcss_fn_J = partial_tcss_module_constructor(n_vert_slices)

    if n_horiz_slices is None:
        tcss_fn_I = tcss_fn_J
    else:
        tcss_fn_I = partial_tcss_module_constructor(n_horiz_slices)

    subkey_J, subkey_I = jax.random.split(key)

    if tcss_fn_J.method == "length_squared":
        J = tcss_fn_J(subkey_J, A)
        I = tcss_fn_I(subkey_I, A.T)
        A_J = TMatrix(jnp.take(A.data, J, axis=1))
        A_I = TMatrix(jnp.take(A.data, I, axis=0))
        W = __compute_W(A_J, A_I, I)
        return A_J, W, A_I

    # Find columns J based on orthonormal basis V
    J = tcss_fn_J(subkey_J, V)

    # NOTE: we can save some computations by constructing TMatrixTOnly from the column subset since Q_J is computed in the frequency domain and does not require the source domain.

    # t-QR
    A_J = TMatrix(jnp.take(A.data, J, axis=1))

    Q_J, _ = A_J.facewise_operation(partial(jnp.linalg.qr, mode="reduced"))
    Q_J = Q_J.create_t_matrix(dtype=V.dtype)

    # Find rows I based on orthonormal basis Q_J
    I = tcss_fn_I(subkey_I, Q_J)
    A_I = TMatrix(jnp.take(A.data, I, axis=0))

    W = __compute_W(A_J, A_I, I)

    return A_J, W, A_I


def _cross2D(
    key: Optional[chex.PRNGKey],
    t_data: chex.Array,
    css_jit: Union[
        Callable[[chex.Array, chex.PRNGKey], chex.Array],
        Callable[[chex.Array], chex.Array],
    ],
):
    if key is not None:
        key_J, key_I = jax.random.split(key)
        J = css_jit(t_data, key=key_J)
    else:
        J = css_jit(t_data)
    C = jnp.take(t_data, indices=J, axis=1)

    Q, _ = jnp.linalg.qr(C)

    if key is not None:
        I = css_jit(Q, key=key_I)
    else:
        I = css_jit(Q)
    R = jnp.take(t_data, indices=I, axis=0)
    W = jnp.linalg.pinv(C) @ t_data @ jnp.linalg.pinv(R)

    return C, W, R


def t_cur(
    M: TMatrix, css_method: CSS_module, key: Optional[chex.PRNGKey] = None
) -> Tuple[TMatrixTOnly, TMatrixTOnly, TMatrixTOnly]:
    n_slices = css_method.r
    if key is not None:
        keys = jax.random.split(key, M.t_data.shape[-1])

    # TODO: properly handle key for deterministic css method!
    css_jit = eqx.filter_jit(lambda x, key: css_method(x, key=key))
    C, W, R = jax.vmap(_cross2D, in_axes=(0, -1, None), out_axes=(-1, -1, -1))(
        keys, M.t_data, css_jit
    )

    shape = (M.shape[0], n_slices, *M.shape[2:])
    C = TMatrixTOnly(C, data_domain_shape=shape, data_dtype=M.dtype)
    shape = (n_slices, n_slices, *M.shape[2:])
    W = TMatrixTOnly(W, data_domain_shape=shape, data_dtype=M.dtype)
    shape = (n_slices, *M.shape[1:])
    R = TMatrixTOnly(R, data_domain_shape=shape, data_dtype=M.dtype)

    return C, W, R


@jax.jit
def reconstruct_t_cross(C: TMatrix, W: TMatrix, R: TMatrix) -> jnp.ndarray:
    recon = (C @ W @ R).create_t_matrix()
    return recon.data
