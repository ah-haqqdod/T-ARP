# Usage

## TMatrix

A `TMatrix` wraps a JAX/NumPy array of shape `(rows, cols, *tube_dims)`. The tube dimensions (the third axis onward) carry the multi-channel or higher-order structure of each element. The structure allows tensor of any order, e.g., a 4-order tensor of video frames (height, width, channels, frames).

### Creating a `TMatrix`

```python
import jax
import jax.numpy as jnp
from t_arp.tubal import TMatrix

# A 3-order tensor: 128 rows × 256 cols × 3 colour channels
data = jax.random.normal(jax.random.PRNGKey(0), (128, 256, 3))
A = TMatrix(data)

print(A)          # TMatrix(shape=(128, 256, 3), dtype=float32)
print(A.shape)    # (128, 256, 3)
print(A.dtype)    # float32

# Access the raw data tensor
print(A.data.shape)    # (128, 256, 3)  – spatial domain
print(A.t_data.shape)  # (128, 256, 3)  – Fourier (transform) domain
```

### `TMatrix` vs `TMatrixTOnly`

Both classes implement the same `TMatrixAbstract` interface and support the same operations (`@`, `.T`, `facewise_operation`, arithmetic). The difference is in what each one stores internally.

| | `TMatrix` | `TMatrixTOnly` |
|---|---|---|
| Spatial-domain data (`_data`) | stored | **not stored** |
| Fourier-domain data (`_t_data`) | stored (computed on construction) | stored |
| `.data` access cost | O(1) read | iFFT computed on demand |
| Typical use | input tensors, final outputs | intermediate computation results |

**`TMatrix`** stores both representations simultaneously. When you construct it from a raw array, the forward FFT along the tube dimensions is computed immediately and cached. Reading `.data` and `.t_data` are both O(1).

**`TMatrixTOnly`** stores only the Fourier-domain array. It is returned by all operations that work entirely in the transform domain — `@`, `.T`, `facewise_operation`, arithmetic — because spatial-domain data is unnecessary during those intermediate steps. This avoids redundant iFFT/FFT round-trips and reduces memory pressure in long computation chains. Accessing `.data` on a `TMatrixTOnly` triggers an iFFT; calling `.create_t_matrix()` materialises a full `TMatrix` when both domains are needed.

```python
from t_arp.tubal import TMatrix, TMatrixTOnly

# TMatrix: both domains available immediately
A = TMatrix(data)           # FFT computed on construction
A.data                      # O(1) — stored
A.t_data                    # O(1) — stored

# Operations return TMatrixTOnly (Fourier domain only)
C = A @ B                   # TMatrixTOnly — no iFFT performed
C.t_data                    # O(1) — stored
C.data                      # iFFT computed on access

# Materialise back to TMatrix when spatial data is needed
C_full = C.create_t_matrix()   # TMatrix — iFFT applied once, result cached
C_full.data                    # O(1)
```

As a rule of thumb: use `TMatrix` for inputs and final outputs; let intermediate results stay as `TMatrixTOnly` and only call `.create_t_matrix()` when you actually need the spatial values.

### Identity tensor

```python
from t_arp.tubal import TMatrix

# Tubal identity of shape (64, 64, 3)
I = TMatrix.eye(shape=(64, 64, 3), dtype=jnp.float32)
```

### t-product (`@`)

The `@` operator computes the **t-product** defined via face-wise multiplication of the Fourier-domain frontal slices.

```python
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)

A = TMatrix(jax.random.normal(k1, (4, 6, 3)))   # (4 × 6 × 3)
B = TMatrix(jax.random.normal(k2, (6, 5, 3)))   # (6 × 5 × 3)

C = A @ B          # t-product → TMatrixTOnly of shape (4, 5, 3)
print(C.shape)     # (4, 5, 3)

# Convert back to spatial domain
C_spatial = C.create_t_matrix()
print(C_spatial.data.shape)   # (4, 5, 3)
```

### Tubal transpose

```python
A_T = A.T           
print(A_T.shape)    #  (4, 6, 3) --> (6, 4, 3)
```

### Face-wise operations

`facewise_operation` applies any function (or JAX-compatible callable) **independently to each frontal slice** in the Fourier domain, enabling tubal analogues of SVD, QR, inverse, etc.

```python
from functools import partial

# Tubal QR decomposition
Q, R = A.facewise_operation(partial(jnp.linalg.qr, mode="reduced"))
print(Q.shape, R.shape)   # (4, 4, 3)  (4, 6, 3)

# Tubal pseudo-inverse
A_pinv = A.facewise_operation(jnp.linalg.pinv)
print(A_pinv.shape)   # (6, 4, 3)

# Tubal matrix inverse (square tensors only)
M_sq = TMatrix(jax.random.normal(k1, (4, 4, 3)) + 10 * jnp.eye(4)[..., None])
M_inv = M_sq.facewise_operation(jnp.linalg.inv)
```

---

## T-ARP Algorithms and Other "Column" Selectors

All "column" selectors take a JAX random key and a `TMatrix` `V` (typically the right-singular-vector matrix from a t-SVD), and return an integer index array `J` of length `n_slices`.

### `TARP` — T Adaptive Randomized Pivoting

```python
from functools import partial
import jax
from t_arp.tubal import TARP, TMatrix
from t_arp.tubal.utils import t_tsvd

key = jax.random.PRNGKey(0)
data = jax.random.normal(key, (128, 256, 3))
A = TMatrix(data)

# 1. Compute low-rank right singular vectors
_, _, Vt = t_tsvd(A, n_slices=32)
V = Vt.T.create_t_matrix()   # shape (256, 32, 3)

# 2. Run T-ARP with T Householder reflection (recommended)
tarp = TARP(n_slices=32, method="householder")
key, subkey = jax.random.split(key)
J = tarp(subkey, V)   # integer array of shape (32,)
print(J)              # selected column indices into A

# 3. Reconstruct via the selected columns (CUR-style)
A_J = TMatrix(jnp.take(A.data, J, axis=1))   # (128, 32, 3)
```

**Available methods:**

| `method` | Description |
|---|---|
| `"householder"` | Decomposition via tubal Householder reflections (default) |
| `"orth_proj_pinv"` | Decomposition via orthogonal projection using the pseudo-inverse |
| `"orth_proj_normalized"` | Decomposition via normalised orthogonal projection |

**Derandomized variant** — pass `use_derandomized=True` to replace random sampling with deterministic argmax selection:

```python
tarp_det = TARP(n_slices=32, method="householder", use_derandomized=True)
J = tarp_det(subkey, V)
```

### `TCSSBaselines` — Common-Index "Column" Subset Selection Baselines

Three non-adaptive baselines for comparison.

```python
from t_arp.tubal import TCSSBaselines

# Uniform random sampling
baseline = TCSSBaselines(n_slices=32, method="uniform")
J = baseline(subkey, V)

# Leverage-score sampling (uses row norms of V)
baseline_lev = TCSSBaselines(n_slices=32, method="leverage_scores")
J = baseline_lev(subkey, V)

# Length-squared (column-norm) sampling — pass the original matrix A, not V
baseline_ls = TCSSBaselines(n_slices=32, method="length_squared")
J = baseline_ls(subkey, A)
```

### `t_cross` — Common-Index Tubal Cross-Approximation

`t_cross` selects both a lateral slice index subset `J` and a horizontal slice index subset `I` and returns the common-index CUR decomposition `A ≈ A_J @ W @ A_I`.

By passing `use_intersection=True`, `t_cross` will select the intersection `A(I, J, :)^+` as the middle tensor `W`; otherwise `W = A(I, :, :)^+ @ A @ A(:, J, :)^+`.

```python
from functools import partial
from t_arp.tubal import TARP
from t_arp.tubal.utils import t_cross, reconstruct_t_cross, t_tsvd

key = jax.random.PRNGKey(0)
data = jax.random.normal(key, (128, 256, 3))
A = TMatrix(data)

_, _, Vt = t_tsvd(A, n_slices=32)
V = Vt.T.create_t_matrix()

# partial constructor that only exposes n_slices
tarp_constructor = partial(TARP, method="householder")

key, subkey = jax.random.split(key)
A_J, W, A_I = t_cross(
    key=subkey,
    A=A,
    V=V,
    partial_tcss_module_constructor=tarp_constructor,
    n_vert_slices=32,   # number of selected columns
    n_horiz_slices=32,  # number of selected rows (defaults to n_vert_slices)
)
print(A_J.shape, W.shape, A_I.shape)  # (128, 32, 3)  (32, 32, 3)  (32, 256, 3)

# Reconstruct
M_recon = reconstruct_t_cross(A_J, W, A_I)   # JAX array
rel_err = jnp.linalg.norm(A.data - M_recon) / jnp.linalg.norm(A.data)
print(f"Relative error: {rel_err:.2e}")
```

### `t_cur` — Tubal Cross-Approximation without Common-Index Guarantee

`t_cur` runs a CUR cross-approximation **independently on every frontal slice** in the Fourier domain; each slice selects its own row and column indices (no common-index guarantee).
This is in contrast to `t_cross`, which selects horizontal and lateral slice indices of the tensor.

The method accepts any `CSS_module` from `t_arp.matrix`, so the per-slice selector (ARP, leverage scores, uniform) can be swapped freely.

```python
import jax
import jax.numpy as jnp
from t_arp.tubal import TMatrix
from t_arp.tubal.utils import t_cur, reconstruct_t_cross
from t_arp.matrix import (
    ARP_module, ARP_params,
    LeverageScoresSampling_module,
    UniformSampling_module, CSS_params,
)

key = jax.random.PRNGKey(0)
data = jax.random.normal(key, (128, 256, 3))
M = TMatrix(data)

rank = 32

# --- Option 1: ARP per frontal slice ---
# rsvd_r is the rank used for the internal randomised SVD inside ARP
css = ARP_module(
    r=rank,
    css_params=ARP_params(rsvd_r=rank, n_oversamples=8, n_subspace_iters=2, use_householder=False),
)

key, subkey = jax.random.split(key)
C, W, R = t_cur(M, css_method=css, key=subkey)
print(C.shape, W.shape, R.shape)
# (128, 32, 3)  (32, 32, 3)  (32, 256, 3)

# --- Option 2: Leverage-score sampling per frontal slice ---
css_lev = LeverageScoresSampling_module(r=rank, css_params=LeverageScoresSampling_params(rsvd_r=rank, n_oversamples=5, n_subspace_iters=1))
C, W, R = t_cur(M, css_method=css_lev, key=subkey)

# --- Option 3: Uniform sampling per frontal slice ---
css_uni = UniformSampling_module(r=rank, css_params=CSS_params())
C, W, R = t_cur(M, css_method=css_uni, key=subkey)
```

### `reconstruct_t_cross` — Reconstruct a `TMatrix` after `t_cross` or `t_cur`

> **Note:** `C`, `W`, and `R` are returned as `TMatrixTOnly` objects (Fourier-domain representation). Pass them directly to `reconstruct_t_cross`; calling `.create_t_matrix()` beforehand is not needed.

`reconstruct_t_cross` computes the t-product `C @ W @ R`, maps the result back to the spatial domain, and returns the raw JAX array. It accepts the three-factor output of both `t_cross` and `t_cur`.

```python
from t_arp.tubal.utils import reconstruct_t_cross

# Works with the output of t_cross or t_cur
M_recon = reconstruct_t_cross(C, W, R)   # jnp.ndarray, same shape as M.data

rel_err = jnp.linalg.norm(M.data - M_recon) / jnp.linalg.norm(M.data)
print(f"Relative error: {rel_err:.2e}")
```

---

## Tubal SVD utilities

All SVD helpers are importable directly from `t_arp.tubal`.

### Full t-SVD

```python
from t_arp.tubal import TMatrix
from t_arp.tubal.utils import t_svd, reconstruct_t_svd

data = jax.random.normal(jax.random.PRNGKey(1), (64, 128, 3))
M = TMatrix(data)

U, S, Vt = t_svd(M)
print(U.shape, S.shape, Vt.shape)  # (64, 64, 3)  (64, 3)  (64, 128, 3)

# Reconstruct
M_recon = reconstruct_t_svd(U, S, Vt)   # JAX array, same shape as M.data
rel_err = jnp.linalg.norm(M.data - M_recon) / jnp.linalg.norm(M.data)
print(f"Relative error: {rel_err:.2e}")
```

### Truncated t-SVD

```python
from t_arp.tubal.utils import t_tsvd, reconstruct_t_svd

rank = 16
U, S, Vt = t_tsvd(M, n_slices=rank)
print(U.shape, S.shape, Vt.shape)  # (64, 16, 3)  (16, 3)  (16, 128, 3)

M_recon = reconstruct_t_svd(U, S, Vt)
```

### Randomised t-SVD (t-RSVD)

```python
from t_arp.tubal.utils import t_rsvd, reconstruct_t_svd

key = jax.random.PRNGKey(42)
U, S, Vt = t_rsvd(key, M, n_slices=16, n_oversamples=8, n_subspace_iters=2)
print(U.shape)   # (64, 16, 3)

M_recon = reconstruct_t_svd(U, S, Vt)
```
