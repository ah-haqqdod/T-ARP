import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from t_arp.matrix.utils import RSVD  # your custom RSVD implementation

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SHAPE = (100, 100)
RANKS = (5, 10, 25, 50, 75, 100)
N_TRIALS = 5  # number of random matrices per rank
SEED = 0
RSVD_OVERSAMPLING = 10  # p = rank + oversampling (if your RSVD supports it)
RSVD_POWER_ITER = 2  # q = number of power iterations
OUTPUT_PLOT = "src/test/figures/svd_vs_rsvd_error.png"


# ----------------------------------------------------------------------
# Helper: single test for a given matrix and rank
# ----------------------------------------------------------------------
def test_single(key, matrix, rank, oversampling=10, power_iter=2):
    """
    Compute relative Frobenius errors for SVD and RSVD at given rank.
    Returns (err_svd, err_rsvd).
    """
    # SVD reconstruction error (optimal)
    U, S, Vt = jnp.linalg.svd(matrix, full_matrices=False)
    rec_svd = U[:, :rank] @ jnp.diag(S[:rank]) @ Vt[:rank, :]
    err_svd = jnp.linalg.norm(matrix - rec_svd, ord="fro")

    # RSVD approximation
    rsvd = RSVD(
        rank=rank,
        n_oversamples=oversampling,
        n_subspace_iters=power_iter,
        return_range=False,
    )
    key, subkey = jax.random.split(key)
    U_r, S_r, Vt_r = rsvd(subkey, matrix)
    rec_rsvd = U_r @ jnp.diag(S_r) @ Vt_r
    err_rsvd = jnp.linalg.norm(matrix - rec_rsvd, ord="fro")

    # Relative errors (divide by matrix norm)
    norm_mat = jnp.linalg.norm(matrix, ord="fro")
    return err_svd / norm_mat, err_rsvd / norm_mat


# ----------------------------------------------------------------------
# Main experiment: average over trials for each rank
# ----------------------------------------------------------------------
def run_experiment(shape, ranks, n_trials, seed=0, oversampling=10, power_iter=2):
    """
    Returns:
        svd_errors : list of (mean, std) for each rank
        rsvd_errors: list of (mean, std) for each rank
    """
    svd_means, svd_stds = [], []
    rsvd_means, rsvd_stds = [], []

    base_key = jax.random.PRNGKey(seed)

    for rank in ranks:
        trial_err_svd = []
        trial_err_rsvd = []
        for trial in range(n_trials):
            # Generate fresh random matrix for each trial
            base_key, subkey = jax.random.split(base_key)
            matrix = jax.random.normal(subkey, shape)

            base_key, rsvd_key = jax.random.split(base_key)
            err_svd, err_rsvd = test_single(
                rsvd_key, matrix, rank, oversampling, power_iter
            )
            trial_err_svd.append(err_svd)
            trial_err_rsvd.append(err_rsvd)

        # Convert to numpy for easy statistics
        trial_err_svd = np.array(trial_err_svd)
        trial_err_rsvd = np.array(trial_err_rsvd)
        svd_means.append(trial_err_svd.mean())
        svd_stds.append(trial_err_svd.std())
        rsvd_means.append(trial_err_rsvd.mean())
        rsvd_stds.append(trial_err_rsvd.std())

        # Optional: print progress
        print(
            f"Rank {rank:3d}: SVD error = {svd_means[-1]:.3e} ± {svd_stds[-1]:.3e}, "
            f"RSVD error = {rsvd_means[-1]:.3e} ± {rsvd_stds[-1]:.3e}"
        )

    return (svd_means, svd_stds), (rsvd_means, rsvd_stds)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_errors(ranks, svd_stats, rsvd_stats, output_file):
    svd_means, svd_stds = svd_stats
    rsvd_means, rsvd_stds = rsvd_stats

    plt.figure(figsize=(8, 6))

    # SVD error (deterministic in theory, but we still show std for consistency)
    plt.errorbar(
        ranks,
        svd_means,
        yerr=svd_stds,
        fmt="o-",
        capsize=5,
        label="SVD (optimal)",
        color="blue",
        markersize=8,
    )

    # RSVD error with error bars
    plt.errorbar(
        ranks,
        rsvd_means,
        yerr=rsvd_stds,
        fmt="s-",
        capsize=5,
        label=f"RSVD (oversampling={RSVD_OVERSAMPLING}, power_iter={RSVD_POWER_ITER})",
        color="red",
        markersize=8,
    )

    plt.xlabel("Rank", fontsize=12)
    plt.ylabel("Relative Frobenius Error", fontsize=12)
    plt.title("SVD vs RSVD Reconstruction Error", fontsize=14)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()
    print(f"Plot saved to {output_file}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Running experiment: shape={SHAPE}, ranks={RANKS}, trials={N_TRIALS}")
    svd_stats, rsvd_stats = run_experiment(
        SHAPE,
        RANKS,
        N_TRIALS,
        SEED,
        oversampling=RSVD_OVERSAMPLING,
        power_iter=RSVD_POWER_ITER,
    )
    plot_errors(RANKS, svd_stats, rsvd_stats, OUTPUT_PLOT)

# ---

# import jax
# import jax.numpy as jnp

# from t_arp.matrix.utils import RSVD

# SEED = 0
# SHAPE = (100, 100)
# RANKS = (5, 10, 25, 50, 75, 100)


# def test_rvsd(shape, rsvd_rank):
#     key = jax.random.PRNGKey(0)
#     key, subkey = jax.random.split(key)

#     random_matrix = jax.random.normal(subkey, shape)

#     # SVD
#     U, S, Vt = jnp.linalg.svd(random_matrix, full_matrices=False)
#     rec1 = U[:, :rsvd_rank] @ jnp.diag(S[:rsvd_rank]) @ Vt[:rsvd_rank, :]
#     err_svd = jnp.linalg.norm(random_matrix - rec1, ord="fro")

#     # RSVD
#     rsvd = RSVD(rank=rsvd_rank)
#     # since we are returning range then expect  U, S, Vt
#     key, subkey = jax.random.split(key)
#     U, S, Vt = rsvd(key, random_matrix)
#     rec2 = U @ jnp.diag(S) @ Vt
#     err_rsvd = jnp.linalg.norm(random_matrix - rec2, ord="fro")

#     return err_svd, err_rsvd


# if __name__ == "__main__":
#     ranks = RANKS
#     for r in ranks:
#         err_svd, err_rsvd = test_rvsd(SHAPE, r)
