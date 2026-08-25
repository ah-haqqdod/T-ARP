from typing import Callable, Optional, Tuple, Union

import chex
import equinox as eqx
import jax
from jax import numpy as jnp

from t_arp.matrix import CSS_module
from t_arp.tubal.t_matrix import TMatrix, TMatrixTOnly

def _cross2D(
    key: Optional[chex.PRNGKey],
    t_data: chex.Array,
    css_jit: Union[
        Callable[[chex.Array, chex.PRNGKey], chex.Array],
        Callable[[chex.Array], chex.Array],
    ],
    css_sample_row_jit: Union[
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
        I = css_sample_row_jit(Q, key=key_I)
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
    sample_row_from_orthonormal_basis_jit = eqx.filter_jit(lambda x, key: css_method.sample_row_from_orthonormal_basis(Q=x, key=key))

    C_t_data, W_t_data, R_t_data = jax.vmap(_cross2D, in_axes=(0, -1, None, None), out_axes=(-1, -1, -1))(
        keys, M.t_data, css_jit, sample_row_from_orthonormal_basis_jit
    )
    shape = (M.shape[0], n_slices, *M.shape[2:])
    C = TMatrixTOnly(C_t_data, data_domain_shape=shape, data_dtype=M.dtype)
    shape = (n_slices, n_slices, *M.shape[2:])
    W = TMatrixTOnly(W_t_data, data_domain_shape=shape, data_dtype=M.dtype)
    shape = (n_slices, *M.shape[1:])
    R = TMatrixTOnly(R_t_data, data_domain_shape=shape, data_dtype=M.dtype)

    return C, W, R
