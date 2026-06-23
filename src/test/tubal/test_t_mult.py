from functools import partial

import jax
from jax import numpy as jnp

from t_arp.tubal import TMatrix, TMatrixAbstract, TMatrixTOnly


def test_commutativity(a, b, M):
    print("COMMUTATIVITY")
    "a * b = b * a"
    a_b = a * b
    b_a = b * a
    print("a * b - b * a =", (a_b - b_a).data)

    # "a * M = M * a"
    a_M = a * M
    M_a = M * a
    print("a * M - M * a =", (a_M - M_a).data)
    pass


def test_associativity(a, b, M):
    print("ASSOCIATIVITY")
    "a * (b * c) = (a * b) * c"
    c = a + b
    b_c = b * c
    a_b = a * b
    print("a * (b * c) - (a * b) * c =", (a * b_c - a_b * c).data)

    # "a * (b * M) = (a * b) * M"
    b_M = b * M
    a_b = a * b
    print("a * (b * M) - (a * b) * M =", (a * b_M - a_b * M).data)
    pass


def test_identity(a, b, M):
    print("IDENTITY")
    "a * a^{-1} = I"
    a_inv = TMatrixTOnly(
        t_data=1 / a.t_data, data_domain_shape=a.shape, data_dtype=a.dtype
    )
    # a_inv = TMatrix(1 / a.data)
    print("a * a^{-1} =", (a * a_inv).data)

    "(a * M) * (a * M)^{-1} = I"
    a_M = a * M
    a_M_inv = a_M.facewise_operation(jnp.linalg.inv)
    print("(a * M) * (a * M)^{-1} =", (a_M @ a_M_inv).data)
    pass


key = jax.random.PRNGKey(1)
key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

a = jax.random.normal(subkey1, (1, 1, 3), dtype=jnp.float32)
b = jax.random.normal(subkey2, (1, 1, 3), dtype=jnp.float32)
M = jax.random.normal(subkey3, (3, 3, 3), dtype=jnp.float32)

print("a", a.shape, a)
print("b", b.shape, b)
print("M", M.shape, M)


a = TMatrix(a)
b = TMatrix(b)
M = TMatrix(M)
print("M * M^{-1}", (M @ M.facewise_operation(jnp.linalg.inv)).data)

test_commutativity(a, b, M)
test_associativity(a, b, M)
test_identity(a, b, M)


# eye = t_eye(3, 3, 3, dtype=jnp.float32)
# print(eye)
# print(TMatrix(eye).t_data)


# E = eye[:, 0, :] * 5
# E = E.reshape((3, 1, 3))

# print(E)
# print(E.shape)
# E_t = TMatrix(E).t_data
# print(E_t.shape, E_t)
# # print(E_t[:, :, 0])
# # print(E_t[:, :, 1])
# # print(E_t[:, :, 2])

# s = jnp.array([-1, 1, -1]).reshape(1, 1, 3)

# print(s.shape, s)
# print(TMatrix(s).t_data)


# # a_sqrt = TMatrixTOnly(jnp.sqrt(a.t_data), a.shape, a.dtype)
# a_sqrt = TMatrix(jnp.sqrt(jnp.abs(a.data)))
# a_abs = TMatrix(jnp.abs(a.data))
# # print(a.data)
# # print(a_sqrt.data)
# print((a_abs - a_sqrt * a_sqrt).data)

# Q_, R_ = M.facewise_operation(partial(jnp.linalg.qr, mode="complete"))

# print("Q * Q.T", (Q_ @ Q_.facewise_operation(jnp.transpose)).data)
# print("Q * Q.H", (Q_ @ Q_.facewise_operation(jnp.transpose).facewise_operation(jnp.conjugate)).data)

# print("Q.T * Q", (Q_.facewise_operation(jnp.transpose) @ Q_).data)
# print("Q * Q.H", (Q_.facewise_operation(jnp.transpose).facewise_operation(jnp.conjugate) @ Q_).data)
