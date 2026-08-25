import jax
from jax import numpy as jnp

from t_arp.matrix import RSVD
from t_arp.tubal import t_svd, t_tsvd, t_rsvd, reconstruct_t_svd
from t_arp.tubal import TMatrix

def check_is_real(x_idft, eps=1e-6):
    print("Biggest imaginary part:", jnp.max(jnp.abs(jnp.imag(x_idft))))
    return jnp.all(jnp.abs(jnp.imag(x_idft)) < eps)

def true_reconstruct_t_svd(U, S, Vt, conjugate_symmetrize=False):
    S = S.facewise_operation(lambda x: jnp.diag(x.flatten()))
    recon_t_matrix = (U @ S @ Vt)

    if conjugate_symmetrize:
        recon_t_matrix = recon_t_matrix.conjugate_symmetrize()
    M = recon_t_matrix.t_data.reshape(recon_t_matrix.shape)
    axes = tuple(range(2, M.ndim))
    print(axes)
    M = jnp.fft.ifftn(M, axes=axes)
    return M


key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

X = jax.random.uniform(subkey, (30, 30, 30))
M = TMatrix(X)

# T-TSVD
U, S, Vt = t_tsvd(M, n_slices=10)

print("accuracy")
M_hat = U @ S.facewise_operation(lambda x: jnp.diag(x.flatten())) @ Vt
print("With conjugate symmetrization",
    jnp.linalg.norm((M - M_hat.conjugate_symmetrize()).data) / jnp.linalg.norm(M.data)
)
print("Without conjugate symmetrization",
    jnp.linalg.norm((M - M_hat).data) / jnp.linalg.norm(M.data)
)

# T-RSVD
U, S, Vt = t_rsvd(key=key, M=M, n_slices=10)

print("accuracy")
M_hat = U @ S.facewise_operation(lambda x: jnp.diag(x.flatten())) @ Vt
print("With conjugate symmetrization",
    jnp.linalg.norm((M - M_hat.conjugate_symmetrize()).data) / jnp.linalg.norm(M.data)
)
print("Without conjugate symmetrization",
    jnp.linalg.norm((M - M_hat).data) / jnp.linalg.norm(M.data)
)

# ---

# 0
print("T-SVD with conjugate symmetrization of factors")

U, S, Vt = t_svd(M)

X_rec_0 = true_reconstruct_t_svd(U, S, Vt)

cond = check_is_real(X_rec_0)

print("Is real:", cond)

# 1
print("T-RSVD without conjugate symmetrization of factors")
U, S, Vt = t_rsvd(key=key, M=M, n_slices=10)

X_rec_1 = true_reconstruct_t_svd(U, S, Vt)

cond = check_is_real(X_rec_1)

print("Is real:", cond)
print("Difference 1:", jnp.linalg.norm(X_rec_1 - X) / jnp.linalg.norm(X))
# print(X_rec_1)


# 2
print("T-RSVD with conjugate symmetrization of factors")
U, S, Vt = t_rsvd(key=key, M=M, n_slices=10)

U = U.conjugate_symmetrize()
S = S.conjugate_symmetrize()
Vt = Vt.conjugate_symmetrize()


X_rec_2 = true_reconstruct_t_svd(U, S, Vt)

cond = check_is_real(X_rec_2)

print("Is real:", cond)
print("Difference 2:", jnp.linalg.norm(X_rec_2 - X) / jnp.linalg.norm(X))


# 3
print("T-RSVD with conjugate symmetrization of reconstruction")
U, S, Vt = t_rsvd(key=key, M=M, n_slices=10)

U = U.conjugate_symmetrize()
S = S.conjugate_symmetrize()
Vt = Vt.conjugate_symmetrize()


X_rec_3 = true_reconstruct_t_svd(U, S, Vt, conjugate_symmetrize=True)

cond = check_is_real(X_rec_3)

print("Is real:", cond)
print("Difference 3:", jnp.linalg.norm(X_rec_3 - X) / jnp.linalg.norm(X))
