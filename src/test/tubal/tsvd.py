from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from PIL import Image

from t_arp.tubal import TMatrix, TMatrixTOnly, t_eye

IMAGE_PATH = "datasets/kodak/kodim19.png"
SAVE_PATH = "src/t_arp/tubal/"


def png_to_tensor(image_path: str) -> jnp.ndarray:
    """
    Load a PNG image and convert it to a JAX tensor.

    Args:
        image_path: Path to the PNG image file

    Returns:
        JAX tensor of shape (height, width, channels) with float32 values in [0, 1]
    """
    # Load image using PIL
    img = Image.open(image_path)

    # Convert to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Convert to numpy array and then to JAX array
    img_array = jnp.array(img, dtype=jnp.float32)

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    return img_array


def t_svd(M: TMatrix) -> TMatrixTOnly:
    svd_fn = partial(jnp.linalg.svd, full_matrices=False)
    t_U, t_S, t_V = M.facewise_operation(svd_fn)

    return t_V


M = TMatrix(png_to_tensor(IMAGE_PATH))
print(M._shape)

t_V = t_svd(M)

print(t_V._data_domain_shape, t_V)

V = t_V.create_t_matrix()

print("Tensor norm")
print(jnp.linalg.norm(V._data))

print("Channel-wise norm")
print(jnp.linalg.norm(V._data[:, :, 0]))
print(jnp.linalg.norm(V._data[:, :, 1]))
print(jnp.linalg.norm(V._data[:, :, 2]))

print("Tube norm")
print(jnp.linalg.norm(V._data[120, 1, :]))


def loop_body(carry, x):
    """V is of shape (n, r, *C), s.t. n >= r.
    We need to select subset of indices in the axis of size n (rows).

    Args:
        carry (_type_): _description_
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    key, V, denom = carry
    # V = Vt.facewise_operation(jnp.transpose).create_t_matrix()
    # define leverage scores
    vecnorm_fn = jax.vmap(lambda M: jnp.linalg.norm(M), in_axes=(0) ** 2)
    p = vecnorm_fn(V)
    # p = p / jnp.sum(p)
    denom = jnp.sum(p)
    jax.debug.print("denom = {x}", x=denom)
    p = p / denom

    # sample a column of A (row of V)
    key, subkey = jax.random.split(key)
    indices = jnp.arange(0, p.shape[0])
    j_k = jax.random.choice(subkey, indices, p=p)

    # remove column from V (using pseudoinverse vector)
    x = jax.lax.dynamic_index_in_dim(V, j_k, axis=0)
    x_norm = jnp.linalg.norm(x) + 1e-8

    x = TMatrix(x / x_norm)
    # x_t = x.facewise_operation(jnp.transpose)
    x_t = x.facewise_operation(jnp.linalg.pinv)
    # x in (1, r, *C) and x_t in (r, 1, *C)

    V = V - ((V @ x_t) @ x).create_t_matrix().data

    return (key, V, denom - 1), j_k


rank = 50

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
Vt = V
V = Vt.facewise_operation(jnp.transpose).create_t_matrix()

print("Finding columns J")
denom = min(V._shape[0], V._shape[1])
init_carry = (subkey, V._data, denom)
# print("init carry", init_carry)
_, (J) = jax.lax.scan(loop_body, init_carry, length=rank)

print("QR decomposition of A(J, :, ...)")
A_J = TMatrix(M._data[:, J, :])
Q_J, _ = A_J.facewise_operation(partial(jnp.linalg.qr, mode="complete"))

Q_J = Q_J.create_t_matrix()

print("Finding columns J")
denom = min(V._shape[0], V._shape[1])
init_carry = (subkey, Q_J.data, denom)
# print("init carry", init_carry)
_, (I) = jax.lax.scan(loop_body, init_carry, length=rank)
A_I = TMatrix(M._data[I, :, :])

# Proj = A_J @ A_J.facewise_operation(jnp.linalg.pinv)

# print(Proj.data_domain_shape, M.shape)

# M_proj = (Proj @ M).create_t_matrix()

# print(jnp.linalg.norm(M.data - M_proj.data) / jnp.linalg.norm(M.data))


# img_pil = Image.fromarray(np.asarray(M_proj.data * 255).astype(np.uint8))
# img_pil.save(SAVE_PATH + f"t_arp_rank={rank}.png")

print("M shape", M._data.shape)
print("I len", len(I), "J len", len(J))
# W = TMatrix(M.data[I, :, :][:, J, :]).facewise_operation(jnp.linalg.pinv)
W = (
    A_J.facewise_operation(jnp.linalg.pinv)
    @ M
    @ A_I.facewise_operation(jnp.linalg.pinv)
)
print(A_J._shape, W._data_domain_shape, A_I._shape)
M_recon = (A_J @ W @ A_I).create_t_matrix()

print(jnp.linalg.norm(M._data - M_recon._data) / jnp.linalg.norm(M._data))

img_pil = Image.fromarray(np.asarray(M_recon._data * 255).astype(np.uint8))
img_pil.save(SAVE_PATH + f"t_arp_rank={rank}.png")


# # Select top 100 rows and project original M onto them

# row_norms = jax.vmap(jnp.linalg.norm, in_axes=(0))(V.data)
# print(row_norms.shape)
# idcs_argsort = jnp.argsort(row_norms)
# best_idcs = idcs_argsort[:100]
# print(best_idcs)

# A_J = TMatrix(M.data[:, best_idcs, :])


# Proj = A_J @ A_J.facewise_operation(jnp.linalg.pinv)

# print(Proj.data_domain_shape, M.shape)

# M_proj = (Proj @ M).create_t_matrix()

# print(jnp.linalg.norm(M.data - M_proj.data) / jnp.linalg.norm(M.data))

# import numpy as np

# img_pil = Image.fromarray(np.asarray(M_proj.data * 255).astype(np.uint8))
# img_pil.save(SAVE_PATH + "123.png")

# # V_J = TMatrix(V_J)
# # V_J_T = V_J.facewise_operation(jnp.transpose)

# # # M_proj = M @ V_J @ V_J_T
# # M_proj = (M @ V_J_T @ V_J).create_t_matrix()

# # print(M_proj.shape)


# # save M_proj to verify
