from functools import partial
from typing import Callable, Optional, Union

import chex
import jax
from jax import numpy as jnp

from t_arp.tubal.t_matrix import TMatrix, TMatrixTOnly


def _compute_W(
    A: TMatrix, A_J: TMatrix, A_I: TMatrix, I: chex.Array, use_intersection: bool
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


def t_cross(
    key,
    A: TMatrix,
    V: TMatrix,
    partial_tcss_module_constructor: Callable,  # TODO: define the types here
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
        W = _compute_W(A, A_J, A_I, I, use_intersection=use_intersection)
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

    W = _compute_W(A, A_J, A_I, I, use_intersection=use_intersection)

    return A_J, W, A_I
