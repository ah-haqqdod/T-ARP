import jax
from jax import numpy as jnp
from typing import Union

from t_arp.tubal.t_matrix import TMatrix, TMatrixTOnly


@jax.jit
def reconstruct_t_svd(
    U: Union[TMatrix, TMatrixTOnly],
    S: Union[TMatrix, TMatrixTOnly],
    Vt: Union[TMatrix, TMatrixTOnly],
) -> jnp.ndarray:
    S = S.facewise_operation(lambda x: jnp.diag(x.flatten()))
    recon = (U @ S @ Vt).conjugate_symmetrize().create_t_matrix()
    return recon.data

@jax.jit
def reconstruct_t_cross(
    C: Union[TMatrix, TMatrixTOnly],
    W: Union[TMatrix, TMatrixTOnly],
    R: Union[TMatrix, TMatrixTOnly],
) -> jnp.ndarray:
    recon = (C @ W @ R).conjugate_symmetrize().create_t_matrix()
    return recon.data
