from functools import partial
from typing import Callable, Optional, Tuple, Union

import chex
import equinox as eqx
import jax
from jax import numpy as jnp

from t_arp.matrix import CSS_module
from t_arp.tubal import TMatrix, TMatrixTOnly


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
