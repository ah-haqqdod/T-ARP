from functools import reduce
import math

import chex
import jax
from jax import numpy as jnp

from t_arp.matrix import RSVD
from t_arp.matrix.utils import HouseholderReflection
from t_arp.tubal import t_svd, t_rsvd, reconstruct_t_svd
from t_arp.tubal import TMatrix
from t_arp.tubal.t_matrix import TMatrixTOnly, conjugate_symmetrize

N_SLICES = 20

def compare_distributions(p_1, p_2, eps=1e-12):
    # Ensure they sum to 1 (they already do, but safe)
    p_1 = p_1 / jnp.sum(p_1)
    p_2 = p_2 / jnp.sum(p_2)

    # 1. Total Variation
    tv = 0.5 * jnp.sum(jnp.abs(p_1 - p_2))

    # 2. Hellinger
    bc = jnp.sum(jnp.sqrt(p_1 * p_2 + eps))  # eps avoids NaN if zeros
    hellinger = jnp.sqrt(jnp.clip(1 - bc, 0.0, 1.0))

    # 3. Jensen-Shannon
    m = (p_1 + p_2) / 2
    kl_p1_m = jnp.sum(p_1 * (jnp.log(p_1 + eps) - jnp.log(m + eps)))
    kl_p2_m = jnp.sum(p_2 * (jnp.log(p_2 + eps) - jnp.log(m + eps)))
    jsd = 0.5 * (kl_p1_m + kl_p2_m)
    # Convert to base-2 so range is [0,1]
    jsd_base2 = jsd / jnp.log(2)

    # 4. Histogram Intersection (similarity)
    intersection = jnp.sum(jnp.minimum(p_1, p_2))

    return {
        "Total_Variation_Distance": tv,
        "Hellinger_Distance": hellinger,
        "Jensen_Shannon_Divergence (base2)": jsd_base2,
        "Histogram_Intersection (Similarity)": intersection
    }

def print_metrics(results):
    print("\n" + "=" * 50)
    print(" DISTRIBUTION COMPARISON RESULTS")
    print("=" * 50)
    for key, value in results.items():
        # Convert JAX/JNP floats to Python float for clean formatting
        print(f"  {key:>35} :  {float(value):.8f}")
    print("=" * 50)


# # 0
# print("T-SVD without conjugate symmetrization of factors")

# U, S, Vt = t_svd(M)

# X_rec_0 = true_reconstruct_t_svd(U, S, Vt)

# cond = check_is_real(X_rec_0)

# print("Is real:", cond)
# print("X", X_rec_0[0, 0, 0, 0, :])
# print("Difference 0:", jnp.linalg.norm(X_rec_0 - X) / jnp.linalg.norm(X))
def sample_row(
    key: chex.PRNGKey,
    V: TMatrix,
    use_derandomized: bool = False,
):
    # Computes the tensor norm for every horizontal slice
    norm_sq_fn = jax.vmap(lambda M: jnp.linalg.norm(M) ** 2, in_axes=(0))
    # Compute leverage scores
    p = norm_sq_fn(V.data)
    jax.debug.print("STD denom {x}", x=jnp.sum(p))
    # denom_theor = jnp.maximum(1, max_n_slices - i)
    # jax.block_until_ready(p)
    # jax.debug.print(
    #     "theoretical denom {x}, actual denom {y}",
    #     x=denom_theor,
    #     y=jnp.sum(p),
    # )
    # p = p / denom_theor
    p = p / jnp.sum(p)

    # sample index
    if use_derandomized:
        j_k = jnp.argmax(p)
    else:
        key, subkey = jax.random.split(key)
        j_k = jax.random.choice(subkey, V.shape[0], p=p)

    # index the row
    v = jax.lax.dynamic_index_in_dim(V.data, index=j_k, axis=0)

    return j_k, v

def sample_row_fast(key, V_t_data, use_derandomized: bool = False):
    # By Parseval's identity we can say that ||X||_F^2 = ||\hat X||_F^2 / Prod(tube_shape)
    # Computes the tensor norm for every horizontal slice
    norm_sq_fn = jax.vmap(lambda M: jnp.linalg.norm(M) ** 2, in_axes=(0))
    # Compute leverage scores
    tube_size = V_t_data.shape[-1]
    p = norm_sq_fn(V_t_data) / tube_size
    jax.debug.print("FST denom {x}", x=jnp.sum(p))
    p = p / jnp.sum(p)

    # sample index
    if use_derandomized:
        j_k = jnp.argmax(p)
    else:
        key, subkey = jax.random.split(key)
        j_k = jax.random.choice(subkey, V.shape[0], p=p)

    # index the row
    v = jax.lax.dynamic_index_in_dim(V_t_data, index=j_k, axis=0)

    return j_k, v


def test_t_arp(key, V, V_t_data, n_slices, use_conjugate_symmetrization):

    J_1 = []
    J_2 = []
    zero_hypercolumn = jnp.zeros(
        shape=(V.shape[0], 1, reduce(lambda a, b: a * b, V.shape[2:])),
        dtype=V.t_data.dtype,
    )
    for i in range(n_slices):
        if use_conjugate_symmetrization:
            V = V.conjugate_symmetrize().create_t_matrix()
            V_t_data = conjugate_symmetrize(V_t_data, V.shape, V.dtype)

        key, subkey = jax.random.split(key)

        # STANDARD T-ARP
        j_1, v = sample_row(subkey, V)

        v_tubal = TMatrix(v)

        # Apply t-Householder
        t_data = jax.vmap(
            HouseholderReflection.reflect_row_vector,
            in_axes=(
                -1,
                -1,
                None,
            ),
            out_axes=(-1),
        )(V.t_data, v_tubal.t_data, i)

        # Hide columns. Experiments show that hiding the columns in source domain and frequency domain produce similar results.
        # Here we use the frequency domain mask to hide the columns since it will omit an unnecessary fourier transformation from source to frequency.
        t_data = jax.lax.dynamic_update_index_in_dim(
            t_data, zero_hypercolumn, index=i, axis=1
        )

        # Transform back to source domain after reflection and hiding columns
        V = TMatrixTOnly.asmatrix(t_data, V).create_t_matrix()

        # FAST T-ARP
        j_2, v_t_data = sample_row_fast(key=subkey, V_t_data=V_t_data,)

        # Apply t-Householder
        V_t_data = jax.vmap(
            HouseholderReflection.reflect_row_vector,
            in_axes=(
                -1,
                -1,
                None,
            ),
            out_axes=(-1),
        )(V_t_data, v_t_data, i)

        # Hide columns. Experiments show that hiding the columns in source domain and frequency domain produce similar results.
        # Here we use the frequency domain mask to hide the columns since it will omit an unnecessary fourier transformation from source to frequency.
        V_t_data = jax.lax.dynamic_update_index_in_dim(
            V_t_data, zero_hypercolumn, index=i, axis=1
        )

        print(f"{i}: T-ARP selected row {j_1}")
        print(f"{i}: Fast T-ARP selected row {j_2}")

        J_1.append(int(j_1))
        J_2.append(int(j_2))

    return J_1, J_2

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

X = jax.random.uniform(subkey, (512, 768, 3))
M = TMatrix(X)

key, subkey = jax.random.split(key)
U, S, Vt = t_rsvd(key=key, M=M, n_slices=N_SLICES)

superkey = key


# 1
print("\n" + "=" * 50)
print("Without conjugate symmetrization")
print("=" * 50)

V = Vt.T.create_t_matrix()
# V = Vt.T.conjugate_symmetrize().create_t_matrix()
norm_sq_fn = jax.vmap(lambda M: jnp.linalg.norm(M) ** 2, in_axes=(0))

# Leverage scores in original domain
p_1 = norm_sq_fn(V.data)
p_1 = p_1 / jnp.sum(p_1)


# Leverage scores in frequency domain
p_2 = norm_sq_fn(V.t_data) # / math.prod(M.shape[2:])
p_2 = p_2 / jnp.sum(p_2)

# Compare
key, subkey = jax.random.split(superkey)

idx_1 = jax.random.choice(subkey, len(p_1))
idx_2 = jax.random.choice(subkey, len(p_2))
print(f"Selected indices: {idx_1}, {idx_2}")
results = compare_distributions(p_1, p_2)
print_metrics(results)


# Inspect T-ARP
n_slices = N_SLICES
use_conjugate_symmetrization = False
J_1, J_2 = test_t_arp(key, V, V.t_data, n_slices, use_conjugate_symmetrization)
print("STD list", J_1)
print("FST list", J_2)

# 2
print("\n" + "=" * 50)
print("With conjugate symmetrization")
print("=" * 50)
# U, S, Vt = t_rsvd(key=key, M=M, n_slices=N_SLICES)

# V = Vt.T.create_t_matrix()
V = Vt.T.conjugate_symmetrize().create_t_matrix()
norm_sq_fn = jax.vmap(lambda M: jnp.linalg.norm(M) ** 2, in_axes=(0))

# Leverage scores in original domain
p_1 = norm_sq_fn(V.data)
p_1 = p_1 / jnp.sum(p_1)


# Leverage scores in frequency domain
p_2 = norm_sq_fn(V.t_data) # / math.prod(M.shape[2:])
p_2 = p_2 / jnp.sum(p_2)

# Compare
key, subkey = jax.random.split(superkey)

idx_1 = jax.random.choice(subkey, len(p_1))
idx_2 = jax.random.choice(subkey, len(p_2))
print(f"Selected indices: {idx_1}, {idx_2}")
results = compare_distributions(p_1, p_2)
print_metrics(results)


# Inspect T-ARP
n_slices = N_SLICES
use_conjugate_symmetrization = True
J_1, J_2 = test_t_arp(key, V, V.t_data, n_slices, use_conjugate_symmetrization)
print("STD list", J_1)
print("FST list", J_2)

# 3
print("\n" + "=" * 50)
print("With conjugate symmetrization (only first step)")
print("=" * 50)
# U, S, Vt = t_rsvd(key=key, M=M, n_slices=N_SLICES)

# V = Vt.T.create_t_matrix()
V = Vt.T.conjugate_symmetrize().create_t_matrix()
norm_sq_fn = jax.vmap(lambda M: jnp.linalg.norm(M) ** 2, in_axes=(0))

# Leverage scores in original domain
p_1 = norm_sq_fn(V.data)
p_1 = p_1 / jnp.sum(p_1)


# Leverage scores in frequency domain
p_2 = norm_sq_fn(V.t_data) # / math.prod(M.shape[2:])
p_2 = p_2 / jnp.sum(p_2)

# Compare
key, subkey = jax.random.split(superkey)

idx_1 = jax.random.choice(subkey, len(p_1))
idx_2 = jax.random.choice(subkey, len(p_2))
print(f"Selected indices: {idx_1}, {idx_2}")
results = compare_distributions(p_1, p_2)
print_metrics(results)


# Inspect T-ARP
n_slices = N_SLICES
use_conjugate_symmetrization = False
J_1, J_2 = test_t_arp(key, V, V.t_data, n_slices, use_conjugate_symmetrization)
print("STD list", J_1)
print("FST list", J_2)
