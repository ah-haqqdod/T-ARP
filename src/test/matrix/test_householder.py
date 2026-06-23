import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from t_arp.matrix.utils import HouseholderReflection  # your class

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SIZES = (10, 20, 40, 60, 80, 100)  # matrix sizes (n x n)
N_TRIALS = 5  # number of random matrices per size
SEED = 0
OUTPUT_PLOT = "src/test/figures/householder_qr_error.png"


# ----------------------------------------------------------------------
# Householder QR implementation (as provided)
# ----------------------------------------------------------------------
def householder_qr(A):
    """Compute reduced QR decomposition using Householder reflections."""
    m, n = A.shape
    R = A.astype(jnp.float64)
    Q = jnp.eye(m, dtype=R.dtype)

    # h = HouseholderReflection(is_rowwise=False)

    for k in range(min(m, n)):
        v = R[:, k]
        v = v.at[:k].set(0.0)  # caller's responsibility
        Q, R = HouseholderReflection.QR_step(Q, R, v, k)

    Q = Q[:, :n]
    R = R[:n, :]
    R = jnp.triu(R)
    return Q, R


# ----------------------------------------------------------------------
# Error metrics
# ----------------------------------------------------------------------
def reconstruction_error(A, Q, R):
    """Relative Frobenius error ||A - QR||_F / ||A||_F"""
    return jnp.linalg.norm(A - Q @ R) / jnp.linalg.norm(A)


def orthogonality_error(Q):
    """For reduced Q (m×n), compute ||Q^T Q - I||_F"""
    n = Q.shape[1]
    I = jnp.eye(n, dtype=Q.dtype)
    return jnp.linalg.norm(Q.conj().T @ Q - I)


# ----------------------------------------------------------------------
# Single test for one matrix
# ----------------------------------------------------------------------
def test_single(matrix):
    """Return (rec_err_qr, orth_err_qr, rec_err_hh, orth_err_hh)"""
    # JAX built-in QR
    Q_jax, R_jax = jnp.linalg.qr(matrix, mode="reduced")
    rec_jax = reconstruction_error(matrix, Q_jax, R_jax)
    orth_jax = orthogonality_error(Q_jax)

    # Householder QR
    Q_hh, R_hh = householder_qr(matrix)
    rec_hh = reconstruction_error(matrix, Q_hh, R_hh)
    orth_hh = orthogonality_error(Q_hh)

    return (rec_jax, orth_jax, rec_hh, orth_hh)


# ----------------------------------------------------------------------
# Main experiment: loop over matrix sizes and trials
# ----------------------------------------------------------------------
def run_experiment(sizes, n_trials, seed=0):
    qr_rec_means, qr_rec_stds = [], []
    qr_orth_means, qr_orth_stds = [], []
    hh_rec_means, hh_rec_stds = [], []
    hh_orth_means, hh_orth_stds = [], []

    base_key = jax.random.PRNGKey(seed)

    for n in sizes:
        rec_qr, orth_qr = [], []
        rec_hh, orth_hh = [], []

        for _ in range(n_trials):
            base_key, subkey = jax.random.split(base_key)
            A = jax.random.normal(subkey, (n, n))

            rec_j, orth_j, rec_h, orth_h = test_single(A)
            rec_qr.append(rec_j)
            orth_qr.append(orth_j)
            rec_hh.append(rec_h)
            orth_hh.append(orth_h)

        # Convert to numpy and aggregate
        rec_qr = np.array(rec_qr)
        orth_qr = np.array(orth_qr)
        rec_hh = np.array(rec_hh)
        orth_hh = np.array(orth_hh)

        qr_rec_means.append(rec_qr.mean())
        qr_rec_stds.append(rec_qr.std())
        qr_orth_means.append(orth_qr.mean())
        qr_orth_stds.append(orth_qr.std())
        hh_rec_means.append(rec_hh.mean())
        hh_rec_stds.append(rec_hh.std())
        hh_orth_means.append(orth_hh.mean())
        hh_orth_stds.append(orth_hh.std())

        print(
            f"Size {n:3d}: QR rec err = {qr_rec_means[-1]:.2e} ± {qr_rec_stds[-1]:.2e}  |  "
            f"HH rec err = {hh_rec_means[-1]:.2e} ± {hh_rec_stds[-1]:.2e}"
        )
        print(
            f"        QR orth err = {qr_orth_means[-1]:.2e} ± {qr_orth_stds[-1]:.2e}  |  "
            f"HH orth err = {hh_orth_means[-1]:.2e} ± {hh_orth_stds[-1]:.2e}"
        )

    return (
        qr_rec_means,
        qr_rec_stds,
        qr_orth_means,
        qr_orth_stds,
        hh_rec_means,
        hh_rec_stds,
        hh_orth_means,
        hh_orth_stds,
    )


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_errors(sizes, qr_rec, qr_orth, hh_rec, hh_orth, output_file):
    qr_rec_means, qr_rec_stds = qr_rec
    qr_orth_means, qr_orth_stds = qr_orth
    hh_rec_means, hh_rec_stds = hh_rec
    hh_orth_means, hh_orth_stds = hh_orth

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reconstruction error
    ax1.errorbar(
        sizes,
        qr_rec_means,
        yerr=qr_rec_stds,
        fmt="o-",
        capsize=5,
        label="jnp.linalg.qr",
        color="blue",
    )
    ax1.errorbar(
        sizes,
        hh_rec_means,
        yerr=hh_rec_stds,
        fmt="s-",
        capsize=5,
        label="Householder QR",
        color="red",
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Matrix size (n)")
    ax1.set_ylabel("Relative reconstruction error")
    ax1.set_title("||A - QR|| / ||A||")
    ax1.grid(True, which="both", linestyle="--", alpha=0.7)
    ax1.legend()

    # Orthogonality error
    ax2.errorbar(
        sizes,
        qr_orth_means,
        yerr=qr_orth_stds,
        fmt="o-",
        capsize=5,
        label="jnp.linalg.qr",
        color="blue",
    )
    ax2.errorbar(
        sizes,
        hh_orth_means,
        yerr=hh_orth_stds,
        fmt="s-",
        capsize=5,
        label="Householder QR",
        color="red",
    )
    ax2.set_yscale("log")
    ax2.set_xlabel("Matrix size (n)")
    ax2.set_ylabel("Orthogonality error")
    ax2.set_title("||QᵀQ - I||")
    ax2.grid(True, which="both", linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()
    print(f"Plot saved to {output_file}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Running experiment: sizes={SIZES}, trials={N_TRIALS}")
    (
        qr_rec_m,
        qr_rec_s,
        qr_orth_m,
        qr_orth_s,
        hh_rec_m,
        hh_rec_s,
        hh_orth_m,
        hh_orth_s,
    ) = run_experiment(SIZES, N_TRIALS, SEED)

    plot_errors(
        SIZES,
        (qr_rec_m, qr_rec_s),
        (qr_orth_m, qr_orth_s),
        (hh_rec_m, hh_rec_s),
        (hh_orth_m, hh_orth_s),
        OUTPUT_PLOT,
    )

# ---

# import jax
# import jax.numpy as jnp
# import matplotlib.pyplot as plt
# import numpy as np

# from t_arp.matrix.utils import HouseholderReflection

# # ----------------------------------------------------------------------
# # Configuration
# # ----------------------------------------------------------------------
# SHAPE = (100, 100)
# RANKS = (5, 10, 25, 50, 75, 100)
# N_TRIALS = 5  # number of random matrices per rank
# SEED = 0
# RSVD_OVERSAMPLING = 10  # p = rank + oversampling (if your RSVD supports it)
# RSVD_POWER_ITER = 2  # q = number of power iterations
# OUTPUT_PLOT = "src/test/figures/svd_vs_rsvd_error.png"


# # ----------------------------------------------------------------------
# # Helper: single test for a given matrix and rank
# # ----------------------------------------------------------------------
# def householder_qr(A, compute_q=True):
#     """Compute reduced QR decomposition using Householder reflections."""
#     m, n = A.shape
#     R = A.astype(jnp.float64)
#     Q = jnp.eye(m, dtype=R.dtype) if compute_q else None

#     # Configure for column‑wise QR (apply reflections on the left)
#     h = HouseholderReflection(is_rowwise=False, use_Q=compute_q)

#     for k in range(min(m, n)):
#         # 1. Take the full k‑th column
#         v = R[:, k]

#         # 2. ZERO out entries above the diagonal (this is the caller's responsibility)
#         v = v.at[:k].set(0.0)

#         # 3. Apply the Householder reflection
#         if compute_q:
#             Q, R = h.reflect_with_Q(Q, R, v, k)
#         else:
#             R = h.reflect_without_Q(R, v, k)

#     # Reduce to standard 'reduced' QR format
#     Q = None if Q is None else Q[:, :n]
#     R = R[:n, :]
#     R = jnp.triu(R)  # enforce exact upper triangular
#     return Q, R


# def test_single(matrix):
#     Q, R = jnp.linalg.qr(matrix)
#     Q_, R_ = householder_qr(matrix)
#     return 1, 2


# # ----------------------------------------------------------------------
# # Main experiment: average over trials for each rank
# # ----------------------------------------------------------------------
# def run_experiment(shape, ranks, n_trials, seed=0):
#     """
#     Returns:
#         qr_errors : list of (mean, std) for each rank
#         hh_errors: list of (mean, std) for each rank
#     """
#     qr_means, qr_stds = [], []
#     hh_means, hh_stds = [], []

#     base_key = jax.random.PRNGKey(seed)

#     for rank in ranks:
#         trial_err_qr = []
#         trial_err_hh = []
#         for trial in range(n_trials):
#             # Generate fresh random matrix for each trial
#             base_key, subkey = jax.random.split(base_key)
#             matrix = jax.random.normal(subkey, shape)

#             err_qr, err_hh = test_single(matrix)
#             trial_err_qr.append(err_qr)
#             trial_err_hh.append(err_hh)

#         # Convert to numpy for easy statistics
#         trial_err_qr = np.array(trial_err_qr)
#         trial_err_hh = np.array(trial_err_hh)
#         qr_means.append(trial_err_qr.mean())
#         qr_stds.append(trial_err_qr.std())
#         hh_means.append(trial_err_hh.mean())
#         hh_stds.append(trial_err_hh.std())

#         # Optional: print progress
#         print(
#             f"Rank {rank:3d}: QR error = {qr_means[-1]:.3e} ± {qr_stds[-1]:.3e}, "
#             f"Householder error = {hh_means[-1]:.3e} ± {hh_stds[-1]:.3e}"
#         )

#     return (qr_means, qr_stds), (hh_means, hh_stds)


# # ----------------------------------------------------------------------
# # Plotting
# # ----------------------------------------------------------------------
# def plot_errors(ranks, svd_stats, rsvd_stats, output_file): ...


# # ----------------------------------------------------------------------
# # Main
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     print(f"Running experiment: shape={SHAPE}, ranks={RANKS}, trials={N_TRIALS}")
#     svd_stats, rsvd_stats = run_experiment(
#         SHAPE,
#         RANKS,
#         N_TRIALS,
#         SEED,
#         oversampling=RSVD_OVERSAMPLING,
#         power_iter=RSVD_POWER_ITER,
#     )
#     plot_errors(RANKS, svd_stats, rsvd_stats, OUTPUT_PLOT)
