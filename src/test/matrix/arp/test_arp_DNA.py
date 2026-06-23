import time
from typing import Any, Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from t_arp.matrix import ARP, RSVD

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_PATH = "src/test/matrix/arp/DNA_matrix.txt"
FIGURE_PATH = "src/test/figures/arp_analysis_DNA.png"
RANKS = [1, 5, 10, 25, 50, 75, 100]  # must be <= min(M,N)
N_TRIALS = 10  # number of independent repetitions
SEED = 42


# ----------------------------------------------------------------------
# Helper functions (module level)
# ----------------------------------------------------------------------
def compute_cssp_error(A: jnp.ndarray, indices: jnp.ndarray) -> float:
    """Relative Frobenius error after projecting A onto selected columns."""
    C = A[:, indices]
    Q, _ = jnp.linalg.qr(C, mode="reduced")
    A_proj = Q @ (Q.T @ A)
    return jnp.linalg.norm(A - A_proj) / jnp.linalg.norm(A)


def best_low_rank_error(A: jnp.ndarray, rank: int) -> float:
    """Relative error of optimal rank‑k SVD approximation."""
    U, S, Vt = jnp.linalg.svd(A, full_matrices=False)
    A_rank = U[:, :rank] @ jnp.diag(S[:rank]) @ Vt[:rank, :]
    return jnp.linalg.norm(A - A_rank) / jnp.linalg.norm(A)


def measure(
    key,
    jitted_func: Callable,
    args: Tuple[Any, ...],
    error_func: Callable[[Any], float],
) -> Tuple[float, float, Any]:
    """
    Measure execution time and compute error for a single run.

    Returns:
        elapsed_time (s), error value, function output
    """
    key, subkey = jax.random.split(key)
    start = time.perf_counter()
    output = jitted_func(subkey, *args)
    jax.block_until_ready(output)
    end = time.perf_counter()
    elapsed = end - start
    error = error_func(output)
    # counterintuitively, blocking error is required for proper time counting.
    jax.block_until_ready(error)
    return elapsed, error, output


# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep="\t", comment="!", header=0, index_col=0)
A = jnp.array(df.values, dtype=jnp.float64)
M, N = A.shape
print(f"Matrix shape: {M} x {N}")
print(f"Maximum possible rank: {min(M, N)}\n")

# Pre‑compute optimal SVD errors for all ranks (deterministic)
svd_errors = {rank: best_low_rank_error(A, rank) for rank in RANKS}

# ----------------------------------------------------------------------
# Data structures for results
# ----------------------------------------------------------------------
results = {
    "rank": RANKS,
    "svd_error": [svd_errors[r] for r in RANKS],
    "rsvd_error_mean": [],
    "rsvd_error_std": [],
    "rsvd_time_mean": [],
    "rsvd_time_std": [],
    "arp_prototype_error_mean": [],
    "arp_prototype_error_std": [],
    "arp_prototype_time_mean": [],
    "arp_prototype_time_std": [],
    "arp_house_error_mean": [],
    "arp_house_error_std": [],
    "arp_house_time_mean": [],
    "arp_house_time_std": [],
}


# ----------------------------------------------------------------------
# Experiment for a single rank (modular)
# ----------------------------------------------------------------------
def run_experiment_for_rank(base_key, rank: int, A: jnp.ndarray) -> dict:
    """
    Run all trials for a given rank.
    Returns dictionary with lists of errors and times for each method.
    """
    # Instantiate and JIT once per rank
    rsvd = RSVD(rank=rank)
    rsvd_jit = eqx.filter_jit(rsvd)

    arp_prototype = ARP(rank=rank, use_householder=False)
    arp_prototype_jit = eqx.filter_jit(arp_prototype)

    arp_house = ARP(rank=rank, use_householder=True)
    arp_house_jit = eqx.filter_jit(arp_house)

    # Warm‑up (compilation) – not timed.
    _, _, Vt_dummy = rsvd_jit(base_key, A)
    # jax.block_until_ready(Vt_dummy)
    _ = arp_prototype_jit(base_key, A, Vt_dummy.T)
    _ = arp_house_jit(base_key, A, Vt_dummy.T)

    # Storage for this rank
    rsvd_times, rsvd_errors = [], []
    prototype_times, prototype_errors = [], []
    house_times, house_errors = [], []

    # Run trials
    for trial in range(N_TRIALS):
        trial_key = jax.random.fold_in(base_key, trial)

        rsvd_key, arp_key = jax.random.split(trial_key)

        # ---- RSVD: measure time and error, also obtain Vt for ARP ----
        def rsvd_error(output):
            U, S, Vt = output
            A_approx = U @ jnp.diag(S) @ Vt
            return jnp.linalg.norm(A - A_approx) / jnp.linalg.norm(A)

        rsvd_time, rsvd_err, (U, S, Vt) = measure(rsvd_key, rsvd_jit, (A,), rsvd_error)
        rsvd_times.append(rsvd_time)
        rsvd_errors.append(rsvd_err)

        # ---- ARP Householder ----
        def house_error(output):
            _, J = output
            return compute_cssp_error(A, J)

        house_time, house_err, _ = measure(
            arp_key, arp_house_jit, (A, Vt.T), house_error
        )

        # ---- ARP prototype ----
        def prototype_error(output):
            _, J = output
            return compute_cssp_error(A, J)

        prototype_time, prototype_err, _ = measure(
            arp_key, arp_prototype_jit, (A, Vt.T), prototype_error
        )

        prototype_times.append(prototype_time)
        prototype_errors.append(prototype_err)

        house_times.append(house_time)
        house_errors.append(house_err)

    return {
        "rsvd_times": rsvd_times,
        "rsvd_errors": rsvd_errors,
        "prototype_times": prototype_times,
        "prototype_errors": prototype_errors,
        "house_times": house_times,
        "house_errors": house_errors,
    }


# ----------------------------------------------------------------------
# Main loop over ranks
# ----------------------------------------------------------------------
base_key = jax.random.PRNGKey(SEED)

for rank in RANKS:
    print(f"\n=== Rank = {rank} ===")

    rank_data = run_experiment_for_rank(base_key, rank, A)

    # Aggregate statistics
    results["rsvd_error_mean"].append(np.mean(rank_data["rsvd_errors"]))
    results["rsvd_error_std"].append(np.std(rank_data["rsvd_errors"]))
    results["rsvd_time_mean"].append(np.mean(rank_data["rsvd_times"]))
    results["rsvd_time_std"].append(np.std(rank_data["rsvd_times"]))

    results["arp_prototype_error_mean"].append(np.mean(rank_data["prototype_errors"]))
    results["arp_prototype_error_std"].append(np.std(rank_data["prototype_errors"]))
    results["arp_prototype_time_mean"].append(np.mean(rank_data["prototype_times"]))
    results["arp_prototype_time_std"].append(np.std(rank_data["prototype_times"]))

    results["arp_house_error_mean"].append(np.mean(rank_data["house_errors"]))
    results["arp_house_error_std"].append(np.std(rank_data["house_errors"]))
    results["arp_house_time_mean"].append(np.mean(rank_data["house_times"]))
    results["arp_house_time_std"].append(np.std(rank_data["house_times"]))

    # Print progress
    print(
        f"RSVD: error = {results['rsvd_error_mean'][-1]:.3e} ± {results['rsvd_error_std'][-1]:.3e}  time = {results['rsvd_time_mean'][-1]:.4f} ± {results['rsvd_time_std'][-1]:.4f} s"
    )
    print(
        f"ARP prototype: error = {results['arp_prototype_error_mean'][-1]:.3e} ± {results['arp_prototype_error_std'][-1]:.3e}  time = {results['arp_prototype_time_mean'][-1]:.4f} ± {results['arp_prototype_time_std'][-1]:.4f} s"
    )
    print(
        f"ARP house: error = {results['arp_house_error_mean'][-1]:.3e} ± {results['arp_house_error_std'][-1]:.3e}  time = {results['arp_house_time_mean'][-1]:.4f} ± {results['arp_house_time_std'][-1]:.4f} s"
    )

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy plot
ax1.plot(RANKS, results["svd_error"], "k--", label="SVD (optimal)", linewidth=2)
ax1.errorbar(
    RANKS,
    results["rsvd_error_mean"],
    yerr=results["rsvd_error_std"],
    fmt="o-",
    capsize=5,
    label="RSVD",
    color="blue",
)
ax1.errorbar(
    RANKS,
    results["arp_prototype_error_mean"],
    yerr=results["arp_prototype_error_std"],
    fmt="s-",
    capsize=5,
    label="ARP (prototype)",
    color="green",
)
ax1.errorbar(
    RANKS,
    results["arp_house_error_mean"],
    yerr=results["arp_house_error_std"],
    fmt="^-",
    capsize=5,
    label="ARP (Householder)",
    color="red",
)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Rank (k)")
ax1.set_ylabel("Relative Frobenius error")
ax1.set_title("Reconstruction error (lower is better)")
ax1.grid(True, which="both", linestyle="--", alpha=0.7)
ax1.legend()

# Timing plot
ax2.errorbar(
    RANKS,
    results["rsvd_time_mean"],
    yerr=results["rsvd_time_std"],
    fmt="o-",
    capsize=5,
    label="RSVD",
    color="blue",
)
ax2.errorbar(
    RANKS,
    results["arp_prototype_time_mean"],
    yerr=results["arp_prototype_time_std"],
    fmt="s-",
    capsize=5,
    label="ARP (prototype)",
    color="green",
)
ax2.errorbar(
    RANKS,
    results["arp_house_time_mean"],
    yerr=results["arp_house_time_std"],
    fmt="^-",
    capsize=5,
    label="ARP (Householder)",
    color="red",
)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Rank (k)")
ax2.set_ylabel("Execution time (seconds)")
ax2.set_title("Performance comparison")
ax2.grid(True, which="both", linestyle="--", alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig(FIGURE_PATH, dpi=150)
plt.show()
print(f"\nPlot saved to {FIGURE_PATH}")

# ---

# import equinox as eqx
# import jax
# import pandas as pd
# from jax import numpy as jnp

# from t_arp.matrix import ARP, RSVD

# key = jax.random.PRNGKey(0)

# M, N = 100, 10000
# RANK = 5


# # key, subkey = jax.random.split(key)
# # A = get_structured_data(subkey, M, N, RANK)
# df = pd.read_csv(
#     "src/test/matrix/arp/DNA_matrix.txt", sep="\t", comment="!", header=0, index_col=0
# )
# # print(df.head())

# # Convert to a JAX array for your CSSP testing
# A = jnp.array(df.values)
# print("A shape", A.shape)
# print("Selected Rank", RANK)
# A_norm = jnp.linalg.norm(A, ord="fro")

# # U, S, Vt = jax.jit(jnp.linalg.svd)(A)

# # # import matplotlib.pyplot as plt

# # # plt.figure(figsize=(8, 5))
# # # plt.semilogy(S, 'o-', markersize=4)  # Just pass S, not (U, S, Vt)
# # # plt.title('Singular Values of A (log scale)')
# # # plt.xlabel('Index')
# # # plt.ylabel('Singular Value')
# # # plt.grid(True, alpha=0.3)
# # # plt.tight_layout()
# # # plt.show()

# # # print("S", S)
# # U, S, Vt = U.at[:, :RANK].get(), S.at[:RANK].get(), Vt.at[:RANK, :].get()
# # print("SVD error ", jnp.linalg.matrix_norm(A - U @ jnp.diag(S) @ Vt) / A_norm)


# rsvd = RSVD(rank=RANK)
# key, subkey = jax.random.split(key)
# U, S, Vt = eqx.filter_jit(rsvd)(subkey, A)

# print("rSVD error ", jnp.linalg.matrix_norm(A - U @ jnp.diag(S) @ Vt) / A_norm)

# arp = ARP(rank=RANK, use_householder=False)
# # key, subkey = jax.random.split(key)
# C, J = eqx.filter_jit(arp)(subkey, A, Vt.T)


# def compute_cssp_error(A, indices):
#     """
#     Computes the Frobenius norm reconstruction error for selected columns.
#     Matches the MATLAB 'compute_error' logic.
#     """
#     C = A[:, indices]
#     # Use reduced QR to get orthonormal basis of the selected columns
#     Q, _ = jnp.linalg.qr(C, mode="reduced")

#     # Project A onto the subspace of Q
#     A_proj = Q @ (Q.T @ A)

#     # Return Frobenius norm of the residual
#     return jnp.linalg.norm(A - A_proj, ord="fro")


# print("J shape", J.shape, "J unique count", jnp.unique(J).shape)

# # To get the relative error as seen in the plot:
# cssp_error = compute_cssp_error(A, J)
# relative_error = cssp_error / A_norm
# print("CSSP error", relative_error)
# print(J)

# arp_householder = ARP(rank=RANK, use_householder=True)
# # key, subkey = jax.random.split(key)
# C, J = eqx.filter_jit(arp)(subkey, A, Vt.T)

# cssp_error = compute_cssp_error(A, J)
# relative_error = cssp_error / A_norm
# print("CSSP householder error", relative_error)
# print(J)
