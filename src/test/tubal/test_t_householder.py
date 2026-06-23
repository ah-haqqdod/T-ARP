# %%
from functools import reduce

import jax
from jax import numpy as jnp

from t_arp.matrix import HouseholderReflection
from t_arp.tubal import TMatrix
from t_arp.tubal.t_matrix import TMatrixTOnly

# %%
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

A = jax.random.normal(subkey, (3, 3, 3))
A = TMatrix(A)

# %%
# Test regular T-QR
Q, R = A.facewise_operation(jnp.linalg.qr)

for i in range(R.shape[-1]):
    print(R.data[:, :, i])


# %%
# Test T-QR implementation using the householder module
def householder_qr(A):
    """Compute reduced QR decomposition using Householder reflections."""
    m, n = A.shape
    R = A
    Q = jnp.eye(m, dtype=R.dtype)

    # h = HouseholderReflection(is_rowwise=False)

    for k in range(min(m, n)):
        v = R[:, k]
        v = v.at[:k].set(0.0)  # caller's responsibility
        Q, R = HouseholderReflection.QR_step(Q, R, v, k)

    Q = Q[:, :n]
    R = R[:n, :]
    # R = jnp.triu(R)
    return Q, R


Q, R = A.facewise_operation(householder_qr)

for i in range(R.shape[-1]):
    print(R.data[:, :, i])


# %%
# Test T-RQ implementation using the householder module
def householder_lq(A):
    """Compute reduced RQ decomposition using Householder reflections."""
    m, n = A.shape
    L = A
    Q = jnp.eye(n, dtype=L.dtype)

    # h = HouseholderReflection(is_rowwise=True)

    for k in range(min(m, n)):
        v = L[k, :]
        v = v.at[:k].set(0.0)  # caller's responsibility
        L, Q = HouseholderReflection.LQ_step(L, Q, v, k)

    return L, Q


print("Real matrix")
# B = jax.random.normal(subkey, (3, 3), dtype=jnp.complex64)
B = jax.random.normal(subkey, (3, 3), dtype=jnp.float32)

Q, R = householder_qr(B)
print("QR decomposition\n", R)

L, Q = householder_lq(B)
print("LQ decomposition\n", L)
print("Complex matrix")
B = jax.random.normal(subkey, (3, 3), dtype=jnp.complex64)

Q, R = householder_qr(B)
print("QR decomposition\n", R)

L, Q = householder_lq(B)
print("LQ decomposition\n", L)

# %%

U, S, V = jnp.linalg.svd(B)

# Q, R = householder_qr(V)
# print("QR decomposition\n", R)

L, Q = householder_lq(V)
print("LQ decomposition of complex orthonormal V\n", L)
print("L^H L is expected to be the identity matrix")
print(jnp.conj(L.T) @ L)


# %%
print("LQ decomposition of TMatrix A\n")
L, Q = A.facewise_operation(householder_lq)

for i in range(L.shape[-1]):
    print(L.data[:, :, i])

# %%
# Test householder reflection source domain to understand t-ARP
# use greedy sampling


def test_householder_reflection(M: TMatrix):
    # orthonormal basis of t-matrix M
    # hh_reflect = HouseholderReflection(is_rowwise=True)

    print("Initial V")
    for i in range(M.shape[-1]):
        print(M.data[:, :, i])

    Js = []

    zero_hypercolumn = jnp.zeros(
        shape=(M.shape[0], 1, reduce(lambda a, b: a * b, M.shape[2:])),
        dtype=M.t_data.dtype,
    )

    for i in range(M.shape[0]):
        j_k = i

        # NOTE: it does not matter in which transformation domain to zero out tails
        # sample hyperrow (1, m, ...)
        v = jax.lax.dynamic_index_in_dim(M.data, index=j_k, axis=0)
        v = v.at[0, :i, ...].set(0.0)  # caller's responsibility
        # print("sampled vector")
        # print(v[:, :, 0])
        # print(v)

        v_tubal = TMatrix(v)
        # v_t_data = v_tubal.t_data.at[0, :i, ...].set(0.0)  # caller's responsibility
        # v_tubal = TMatrixTOnly.asmatrix(v_t_data, v_tubal)
        print("v_tubal shape", v_tubal.shape)

        V_t_data = jax.vmap(
            HouseholderReflection.reflect_row_vector,
            in_axes=(
                -1,
                -1,
                None,
            ),
            out_axes=(-1),
        )(M.t_data, v_tubal.t_data, i)

        # print(f"at i={i}")
        # V_temp = TMatrixTOnly.asmatrix(V_t_data, V)
        # for i in range(V_temp.shape[-1]):
        #     print(V_temp.t_data[:, :, 1])

        # V_t_data = jax.lax.dynamic_update_index_in_dim(
        #     V_t_data, zero_hypercolumn, index=i, axis=1
        # )

        M = TMatrixTOnly.asmatrix(V_t_data, M).create_t_matrix()
        Js.append(j_k)

    return M, Js


V, Js = test_householder_reflection(A)
print("!!!")
for i in range(V.shape[-1]):
    print(V.data[:, :, i])
# %%
U, S, V = A.facewise_operation(jnp.linalg.svd)
V, Js = test_householder_reflection(V)
print(
    "The new pivoted V is supposed to be a lower triangular matrix L such that L^H L = I. L^H L is"
)
V_H = V.facewise_operation(jnp.conj)
print((V_H @ V).data)
t_eye = TMatrix.eye((V.shape[0], V.shape[1], *V.shape[2:]), dtype=V.dtype)

print()
print("|| V_H @ V - t_eye || =", jnp.linalg.norm((V_H @ V).data - t_eye.data))

# %%


# def test_t_arp(key, M: TMatrix):
#     # orthonormal basis of t-matrix M
#     U, S, V = M.facewise_operation(jnp.linalg.svd)
#     hh_reflect = HouseholderReflection(is_rowwise=True)

#     norm_sq_fn = jax.vmap(lambda M: jnp.linalg.norm(M) ** 2, in_axes=(0))

#     Js = []

#     for i in range(M.shape[0]):
#         p = norm_sq_fn(V.data)
#         p = p / jnp.sum(p)

#         # sample index
#         key, subkey = jax.random.split(key)
#         indices = jnp.arange(0, p.shape[0])
#         j_k = jax.random.choice(subkey, indices, p=p)
#         # j_k = i

#         # sample hyperrow (1, m, ...)
#         v = jax.lax.dynamic_index_in_dim(V.data, index=j_k, axis=0)
#         # print("sampled vector")
#         # print(v[:, :, 0])
#         # print(v)

#         v_tubal = TMatrix(v)
#         v_t_data = v_tubal.t_data.at[0, :i, ...].set(0.0)  # caller's responsibility
#         v_tubal = TMatrixTOnly.asmatrix(v_t_data, v_tubal)
#         print("v_tubal shape", v_tubal.shape)

#         V_t_data = jax.vmap(
#             hh_reflect.reflect_without_Q,
#             in_axes=(
#                 -1,
#                 -1,
#                 None,
#             ),
#             out_axes=(-1),
#         )(V.t_data, v_tubal.t_data, i)

#         # print(f"at i={i}")
#         # V_temp = TMatrixTOnly.asmatrix(V_t_data, V)
#         # for i in range(V_temp.shape[-1]):
#         #     print(V_temp.t_data[:, :, 1])

#         # V_t_data = jax.lax.dynamic_update_index_in_dim(
#         #     V_t_data, zero_hypercolumn, index=i, axis=1
#         # )

#         V = TMatrixTOnly.asmatrix(V_t_data, V).create_t_matrix()
#         Js.append(j_k)

#     return V, Js


# key, subkey = jax.random.split(key)
# V, Js = test_t_arp(subkey, A)

# for i in range(V.shape[-1]):
#     print(V.t_data[:, :, i])
# for i in range(V.shape[-1]):
#     print(V.data[:, :, i])
