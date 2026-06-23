import time
from typing import Any, Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from t_arp.matrix import ARP, RSVD

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Larger synthetic matrix (adjust for laptop: ~1000x5000)
M = 1000  # rows
N = 5000  # columns
TRUE_RANK = 200  # intrinsic rank
COND_NUM = 1e12  # highly ill‑conditioned
NOISE_STD = 1e-8  # tiny noise

RANKS = [10, 25, 50, 75, 100, 150, 200]  # target ranks
N_TRIALS = 5  # independent repetitions
SEED = 42

FIGURE_PATH_ILL = "src/test/figures/arp_analysis_synthethic_ill.png"
FIGURE_PATH_WELL = "src/test/figures/arp_analysis_synthethic_well.png"


# ----------------------------------------------------------------------
# Helper functions (same as before, reused)
# ----------------------------------------------------------------------
def generate_ill_conditioned_matrix(key, m, n, rank, cond, noise_std=0.0):
    key1, key2, key3 = jax.random.split(key, 3)
    # Random orthonormal bases
    U, _ = jnp.linalg.qr(jax.random.normal(key1, (m, rank)), mode="reduced")
    V, _ = jnp.linalg.qr(jax.random.normal(key2, (n, rank)), mode="reduced")
    # Singular values
    s = jnp.logspace(0, -jnp.log10(cond), rank)
    S = jnp.diag(s)
    A = U @ S @ V.T
    if noise_std > 0:
        A = A + jax.random.normal(key3, (m, n)) * noise_std
    return A


def compute_cssp_error(A, indices):
    C = A[:, indices]
    Q, _ = jnp.linalg.qr(C, mode="reduced")
    A_proj = Q @ (Q.T @ A)
    return jnp.linalg.norm(A - A_proj) / jnp.linalg.norm(A)


def measure(key, jitted_func, args, error_func):
    key, subkey = jax.random.split(key)
    start = time.perf_counter()
    output = jitted_func(subkey, *args)
    jax.block_until_ready(output)
    end = time.perf_counter()
    elapsed = end - start
    error = error_func(output)
    jax.block_until_ready(error)
    return elapsed, error, output


# ----------------------------------------------------------------------
# Single rank experiment (same logic, just using the larger dimensions)
# ----------------------------------------------------------------------
def run_experiment_for_rank(base_key, rank, m, n, true_rank, cond, noise_std, n_trials):
    rsvd = RSVD(rank=rank)
    rsvd_jit = eqx.filter_jit(rsvd)
    arp_proto = ARP(rank=rank, use_householder=False)
    arp_proto_jit = eqx.filter_jit(arp_proto)
    arp_house = ARP(rank=rank, use_householder=True)
    arp_house_jit = eqx.filter_jit(arp_house)

    # Warm-up
    dummy_A = jnp.ones((m, n), dtype=jnp.float64)
    _, _, Vt_dummy = rsvd_jit(base_key, dummy_A)
    _ = arp_proto_jit(base_key, dummy_A, Vt_dummy.T)
    _ = arp_house_jit(base_key, dummy_A, Vt_dummy.T)

    rsvd_times, rsvd_errors = [], []
    proto_times, proto_errors = [], []
    house_times, house_errors = [], []

    for trial in range(n_trials):
        trial_key = jax.random.fold_in(base_key, trial)
        mat_key, rsvd_key, arp_key = jax.random.split(trial_key, 3)
        A = generate_ill_conditioned_matrix(mat_key, m, n, true_rank, cond, noise_std)

        def rsvd_err_fun(out):
            U, S, Vt = out
            A_approx = U @ jnp.diag(S) @ Vt
            return jnp.linalg.norm(A - A_approx) / jnp.linalg.norm(A)

        rsvd_time, rsvd_err, (U, S, Vt) = measure(
            rsvd_key, rsvd_jit, (A,), rsvd_err_fun
        )
        rsvd_times.append(rsvd_time)
        rsvd_errors.append(rsvd_err)

        def proto_err_fun(out):
            _, J = out
            return compute_cssp_error(A, J)

        proto_time, proto_err, _ = measure(
            arp_key, arp_proto_jit, (A, Vt.T), proto_err_fun
        )
        proto_times.append(proto_time)
        proto_errors.append(proto_err)

        def house_err_fun(out):
            _, J = out
            return compute_cssp_error(A, J)

        house_time, house_err, _ = measure(
            arp_key, arp_house_jit, (A, Vt.T), house_err_fun
        )
        house_times.append(house_time)
        house_errors.append(house_err)

    return {
        "rsvd_times": rsvd_times,
        "rsvd_errors": rsvd_errors,
        "prototype_times": proto_times,
        "prototype_errors": proto_errors,
        "house_times": house_times,
        "house_errors": house_errors,
    }


# ----------------------------------------------------------------------
# Run and plot for given condition number
# ----------------------------------------------------------------------
def run_and_plot(cond, noise_std, output_path):
    print(f"\n=== Large‑scale test: cond = {cond:.0e}, matrix {M}×{N} ===")
    results = {
        "rank": RANKS,
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

    base_key = jax.random.PRNGKey(SEED)

    for rank in RANKS:
        print(f"  Rank {rank}...", end=" ", flush=True)
        data = run_experiment_for_rank(
            base_key, rank, M, N, TRUE_RANK, cond, noise_std, N_TRIALS
        )
        results["rsvd_error_mean"].append(np.mean(data["rsvd_errors"]))
        results["rsvd_error_std"].append(np.std(data["rsvd_errors"]))
        results["rsvd_time_mean"].append(np.mean(data["rsvd_times"]))
        results["rsvd_time_std"].append(np.std(data["rsvd_times"]))
        results["arp_prototype_error_mean"].append(np.mean(data["prototype_errors"]))
        results["arp_prototype_error_std"].append(np.std(data["prototype_errors"]))
        results["arp_prototype_time_mean"].append(np.mean(data["prototype_times"]))
        results["arp_prototype_time_std"].append(np.std(data["prototype_times"]))
        results["arp_house_error_mean"].append(np.mean(data["house_errors"]))
        results["arp_house_error_std"].append(np.std(data["house_errors"]))
        results["arp_house_time_mean"].append(np.mean(data["house_times"]))
        results["arp_house_time_std"].append(np.std(data["house_times"]))
        print("done")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Accuracy
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
    ax1.set_xlabel("Target rank")
    ax1.set_ylabel("Relative CSSP error")
    ax1.set_title(f"Column selection error (cond = {cond:.0e}, {M}×{N})")
    ax1.grid(True, which="both", linestyle="--", alpha=0.7)
    ax1.legend()

    # Timing
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
    ax2.set_xlabel("Target rank")
    ax2.set_ylabel("Execution time (seconds)")
    ax2.set_title("Performance comparison")
    ax2.grid(True, which="both", linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Plot saved to {output_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Ill‑conditioned test (main focus)
    run_and_plot(COND_NUM, NOISE_STD, FIGURE_PATH_ILL)
    # Optional well‑conditioned baseline (fast)
    run_and_plot(1e2, NOISE_STD, FIGURE_PATH_WELL)

# ---
#
# import jax
# from jax import numpy as jnp
# import pandas as pd

# from t_arp.matrix import arp, rsvd

# key = jax.random.PRNGKey(0)

# M, N = 100, 10000
# RANK = 100
# identity_matrice_fns = [
#     lambda i=i: jnp.eye(RANK).at[i - 1].get() for i in range(RANK, 0, -1)
# ]


# # print(len(identity_matrice_fns))
# # for f in identity_matrice_fns:
# #     print("!", f())
# def get_structured_data(key, m=100, n=100, rank=10):
#     k1, k2 = jax.random.split(key)

#     # 1. Base latent features
#     latent = jax.random.normal(k1, (m, rank))

#     # 2. Make columns represent certain latent features (Correlation)
#     # We assign columns to clusters so they are highly redundant
#     assignments = jnp.arange(n) % rank
#     A = latent[:, assignments]

#     # 3. Add low-magnitude noise to keep it full rank but with clear leverage
#     A += 0.05 * jax.random.normal(k2, (m, n))
#     return A


# # key, subkey = jax.random.split(key)
# # A = get_structured_data(subkey, M, N, RANK)
# df = pd.read_csv(
#     "src/test/DNA_matrix.txt", sep="\t", comment="!", header=0, index_col=0
# )
# # print(df.head())

# # Convert to a JAX array for your CSSP testing
# A = jnp.array(df.values)
# print("A shape", A.shape)
# print("Selected Rank", RANK)
# A_norm = jnp.linalg.norm(A, ord="fro")

# U, S, Vt = jax.jit(jnp.linalg.svd)(A)
# # print("S", S)
# U, S, Vt = U.at[:, :RANK].get(), S.at[:RANK].get(), Vt.at[:RANK, :].get()


# print("SVD error ", jnp.linalg.matrix_norm(A - U @ jnp.diag(S) @ Vt) / A_norm)

# key, subkey = jax.random.split(key)
# U, S, Vt = jax.jit(rsvd, static_argnames=("rank",))(subkey, A, RANK)

# print("rSVD error ", jnp.linalg.matrix_norm(A - U @ jnp.diag(S) @ Vt) / A_norm)


# key, subkey = jax.random.split(key)
# C, V, J = jax.jit(arp, static_argnames=("rank",))(subkey, A, RANK)


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


# # To get the relative error as seen in the plot:
# cssp_error = compute_cssp_error(A, J)
# relative_error = cssp_error / A_norm
# print("CSSP error", relative_error)

# # # print(
# # #     "CSSP error", jnp.linalg.matrix_norm(A - C @ jnp.linalg.inv(V.at[J].get().T) @ V.T)
# # # )
# # print(
# #     "CSSP error", jnp.linalg.matrix_norm(A - C @ jnp.linalg.pinv(C) @ A)
# # )
