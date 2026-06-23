import jax
import jax.numpy as jnp
import numpy as np
from pandas._libs.lib import dtypes_all_equal

from t_arp.tubal import TMatrix


def test_tproduct_properties():
    """
    Test the t‑product implementation for correctness.
    Checks associativity, distributivity, identity, invertibility,
    and consistency between time and Fourier domain.
    """
    key = jax.random.PRNGKey(42)
    shapes = [(3, 4, 5), (2, 3, 4), (5, 5, 3)]  # (rows, cols, tubes)
    tol = 1e-5

    for m, n, p in shapes:
        print(f"\nTesting shape ({m}, {n}, {p})")

        # Generate random tensors A (m x n x p) and B (n x k x p)
        # We'll test with k = m to check square-ish cases, but keep general.
        # k = n  # for simplicity, use square product
        key, subk = jax.random.split(key)
        A_data = jax.random.normal(subk, (m, n, p))
        key, subk = jax.random.split(key)
        B_data = jax.random.normal(subk, (n, n, p))
        key, subk = jax.random.split(key)
        C_data = jax.random.normal(subk, (n, n, p))  # for associativity

        A = TMatrix(A_data)
        B = TMatrix(B_data)
        C = TMatrix(C_data)

        # ---------- Associativity: (A * B) * C = A * (B * C) ----------
        AB = A @ B
        ABC1 = AB @ C
        BC = B @ C
        ABC2 = A @ BC
        diff_assoc = (ABC1 - ABC2).data
        rel_err = jnp.linalg.norm(diff_assoc) / jnp.linalg.norm(ABC1.data)
        print(f"  Associativity error: {rel_err:.2e}")
        assert rel_err < tol, "Associativity failed"

        # ---------- Distributivity: A * (B + C) = A*B + A*C ----------
        B_plus_C = B + C
        left = A @ B_plus_C
        right = A @ B + A @ C
        diff_dist = (left - right).data
        rel_err = jnp.linalg.norm(diff_dist) / jnp.linalg.norm(left.data)
        print(f"  Distributivity error: {rel_err:.2e}")
        assert rel_err < tol, "Distributivity failed"

        # ---------- Identity: A * I = I * A = A ----------
        _I = TMatrix.eye(
            shape=(n, n, p), dtype=A.dtype
        )  # assumes TMatrix has classmethod eye
        I_ = TMatrix.eye(
            shape=(m, m, p), dtype=A.dtype
        )  # assumes TMatrix has classmethod eye
        left_id = A @ _I
        right_id = I_ @ A
        diff_left = (left_id - A).data
        diff_right = (right_id - A).data
        rel_left = jnp.linalg.norm(diff_left) / jnp.linalg.norm(A.data)
        rel_right = jnp.linalg.norm(diff_right) / jnp.linalg.norm(A.data)
        print(f"  Identity (left) error: {rel_left:.2e}")
        print(f"  Identity (right) error: {rel_right:.2e}")
        assert rel_left < tol and rel_right < tol, "Identity failed"

        # ---------- Invertibility (for square frontal slices) ----------
        if m == n:
            # Construct a random invertible tensor (ensure non‑singular Fourier slices)
            # Use a small random matrix on each tube, but to guarantee invertibility,
            # we generate a tensor whose Fourier frontal slices are randomly perturbed
            # from identity so they are invertible with high probability.
            key, subk = jax.random.split(key)
            A_data = jax.random.normal(subk, (m, m, p))
            # Add a large diagonal to help invertibility
            for i in range(m):
                A_data = A_data.at[i, i, :].add(10.0)
            A = TMatrix(A_data)

            # Compute inverse via Fourier domain
            A_inv = A.facewise_operation(
                jnp.linalg.inv
            )  # assumes shape (m,m,p) -> (m,m,p)
            I_calc = A @ A_inv
            eye_tensor = TMatrix.eye(shape=(m, m, p), dtype=A.dtype)
            diff_inv = (I_calc - eye_tensor).data
            rel_err_inv = jnp.linalg.norm(diff_inv) / jnp.linalg.norm(eye_tensor.data)
            print(f"  Inverse (A * A^{-1} - I) error: {rel_err_inv:.2e}")
            assert rel_err_inv < tol, "Inverse failed"

        # ---------- Consistency with Fourier domain multiplication ----------
        # The t‑product should equal inverse FFT of pointwise multiplication
        # of the Fourier transformed frontal slices.
        A_hat = A.t_data  # shape (m, n, p) in Fourier domain
        B_hat = B.t_data  # shape (n, k, p)
        # Pointwise multiplication of each frontal slice
        C_hat_fft = jnp.zeros((m, n, p), dtype=jnp.complex64)
        for l in range(p):
            C_hat_fft = C_hat_fft.at[:, :, l].set(A_hat[:, :, l] @ B_hat[:, :, l])
        # Inverse FFT to get expected result
        C_expected_data = jnp.fft.ifft(C_hat_fft, axis=-1).real
        C_actual = (A @ B).data
        diff_fft = C_actual - C_expected_data
        rel_err_fft = jnp.linalg.norm(diff_fft) / jnp.linalg.norm(C_expected_data)
        print(f"  Fourier domain consistency error: {rel_err_fft:.2e}")
        assert rel_err_fft < tol, "Fourier domain mismatch"

    print("\nAll t‑product tests passed.")


if __name__ == "__main__":
    test_tproduct_properties()
