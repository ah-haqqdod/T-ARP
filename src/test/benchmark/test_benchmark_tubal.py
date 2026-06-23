from functools import partial
from typing import Callable

import equinox as eqx
import jax
from jax import numpy as jnp

from t_arp.benchmark import KodakBenchmark
from t_arp.tubal import TARP, TMatrix, TMatrixTOnly


def reconstruct_tarp(*tarp_args) -> jnp.ndarray:
    C, U, R = tarp_args
    recon = C @ U @ R

    return jnp.clip(recon.data, 0.0, 1.0)


def reconstruct_tsvd(*tsvs_args) -> jnp.ndarray:
    U, S, Vt = tsvs_args
    S_tdiag = S.facewise_operation(lambda s: jnp.diag(s))
    return jnp.clip(U @ S_tdiag @ Vt, 0.0, 1.0)


def test_kodak_benchmark():
    # kernel_shape = (100, 100, 3)
    # hosvd = HOSVD(kernel_shape=kernel_shape, method="hosvd")
    # benchmark = KodakBenchmark(
    #     method_name="HOSVD",
    #     hyperparameters_log={"kernel_shape": kernel_shape, "method": "hosvd"},
    #     decomposition_blackbox_fn=hosvd,
    #     reconstruction_blackbox_fn=reconstruct_hosvd,
    #     reconstructions_path="src/test/benchmark/",
    #     results_path="src/test/benchmark/",
    # )

    kernel_shape = (100, 100, 3)
    rsvd_ranks = (100, 100, 3)
    key = jax.random.PRNGKey(0)
    hoarp = HOARP(
        kernel_shape=kernel_shape,
        rsvd_ranks=rsvd_ranks,
        use_householder=False,
        method="arp-st-hosvd",
        # method="s-hoarp",
        # method="arp-hosvd",
        # method="hoarp",
    )
    benchmark = KodakBenchmark(
        n_trials=1,
        method_name="HOARP",
        hyperparameters_log={
            "ranks": kernel_shape,
            "rsvd_ranks": rsvd_ranks,
            "use_householder": False,
            "method": "arp-st-hosvd",
        },
        decomposition_blackbox_fn=jax.jit(partial(hoarp, key)),
        # decomposition_blackbox_fn=partial(hoarp, key),
        reconstruction_blackbox_fn=jax.jit(reconstruct_hosvd),
        # reconstruction_blackbox_fn=reconstruct_hosvd,
        reconstructions_path="src/test/benchmark/",
        results_path="src/test/benchmark/",
    )
    result = benchmark()

    print(result)


if __name__ == "__main__":
    test_kodak_benchmark()
