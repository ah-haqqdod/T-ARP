import argparse
import os
from functools import partial
from typing import Callable, Literal, Optional, Tuple, Union

import chex
import equinox as eqx
import jax
import numpy as np
import tqdm
import yaml
from jax import numpy as jnp

from t_arp.benchmark import compute_relative_error, save_metrics
from t_arp.matrix import ARP_params, CSS_module_factory
from t_arp.tubal import (
    TARP,
    TCSSBaselines,
    TMatrix,
    TMatrixTOnly,
    reconstruct_t_cross,
    reconstruct_t_svd,
    t_cross,
    t_cur,
    t_rsvd,
    t_tsvd,
)
# enable float64 and complex128 in jax
jax.config.update("jax_enable_x64", True)
EXPERIMENTS_DTYPE = jnp.float64
# jax.config.update("jax_enable_complex128", True)

def _compute_relative_gaussian_noise(
    key: chex.PRNGKey,
    x: chex.Array,
    relative_noise: float,
) -> chex.Array:
    """Add Gaussian noise with prescribed relative Frobenius norm.

    The perturbation E is scaled so that
        ||E||_F / ||X||_F = relative_noise.
    """
    noise = jax.random.normal(key, shape=x.shape)
    noise_norm = jnp.linalg.norm(noise)
    x_norm = jnp.linalg.norm(x)

    return relative_noise * (x_norm / noise_norm) * noise


@jax.jit(static_argnums=(1, 2, 3))
def construct_random_tmatrix(
    key: chex.PRNGKey,
    t_rank: int,
    t_shape: Tuple[int, ...],
    relative_noise: Optional[float] = None,
) -> TMatrix:
    assert len(t_shape) > 2

    key_u, key_v, key_eps = jax.random.split(key, 3)

    u_shape = (t_shape[0], t_rank, *t_shape[2:])
    U = jax.random.uniform(key_u, u_shape, dtype=EXPERIMENTS_DTYPE)

    v_shape = (t_rank, *t_shape[1:])
    V = jax.random.uniform(key_v, v_shape, dtype=EXPERIMENTS_DTYPE)

    M = TMatrix(U) @ TMatrix(V)

    if relative_noise is not None and relative_noise > 0:
        noise = _compute_relative_gaussian_noise(key=key_eps, x=M.data, relative_noise=relative_noise)
        M = M + noise

    return M


@jax.jit(static_argnums=(1, 2, 3))
def construct_function_tmatrix(
    key: chex.PRNGKey,
    t_shape: Tuple[int, ...],
    p: float,
    relative_noise: Optional[float] = None,
) -> TMatrix:
    assert len(t_shape) > 2

    grids = jnp.ogrid[tuple(slice(1, d + 1) for d in t_shape)]
    denom = sum(grids)
    x = 1.0 / jnp.power(denom, 1.0 / p)
    x = x.astype(EXPERIMENTS_DTYPE)
    M = TMatrix(x)
    if relative_noise is not None and relative_noise > 0:
        noise = _compute_relative_gaussian_noise(key=key, x=M.data, relative_noise=relative_noise)
        M = M + noise
    return M



    # denom = jnp.zeros(t_shape)
    # for i in range(len(t_shape)):
    #     vec = jnp.arange(t_shape[i]) + 1
    #     shape = [t_shape[j] if j == i else 1 for j in range(len(t_shape))]
    #     vec = vec.reshape(tuple(shape))
    #     denom = denom + vec

    # denom = denom
    # denom = denom ** (1 / p)

    # M = TMatrix(1 / denom)

    # return M


@jax.jit(static_argnames=["method", "n_slices"])
def t_cross_helper(
    key: chex.PRNGKey,
    M: TMatrix,
    V: TMatrix,
    method: Literal[
        "uniform",
        "leverage_scores",
        "householder",
        "orth_proj_pinv",
        "orth_proj_normalized",
    ],
    n_slices: int,
) -> Tuple[
    Union[TMatrix, TMatrixTOnly],
    Union[TMatrix, TMatrixTOnly],
    Union[TMatrix, TMatrixTOnly],
]:
    if method == "uniform":
        tcss_module_constructor = partial(TCSSBaselines, method="uniform")
    elif method == "leverage_scores":
        tcss_module_constructor = partial(TCSSBaselines, method="leverage_scores")
    else:
        tcss_module_constructor = partial(TARP, method=method, use_derandomized=False)

    key, subkey = jax.random.split(key)

    C, W, R = t_cross(
        key=subkey,
        A=M,
        V=V,
        partial_tcss_module_constructor=tcss_module_constructor,
        n_vert_slices=n_slices,
        use_intersection=False,
    )

    return C, W, R

# @jax.jit(static_argnums=(1, 2, 3, 4, 5))
def benchmark_synth(
    key: chex.PRNGKey,
    decomposition_blackbox_fn: Callable,
    reconstruction_blackbox_fn: Callable,
    n_samples: int,
    benchmark_type: str,
    is_deterministic: bool = False,
) -> chex.Array:
    key_1, key_2 = jax.random.split(key)
    keys_gen = jax.random.split(key_1, n_samples)

    @jax.jit
    def benchmark_fn(key):
        if benchmark_type == "function_tensor":
            return construct_function_tmatrix(
                key=key,
                t_shape=config["t_shape"],
                p=config["function_tensor_p"],
                relative_noise=config["relative_noise"],
            )
        else:
            return construct_random_tmatrix(
                key=key,
                t_rank=config["t_rank"],
                t_shape=config["t_shape"],
                relative_noise=config["relative_noise"],
            )

    if is_deterministic:
        return jax.vmap(
            lambda key: (
                M := benchmark_fn(key),
                decomp := decomposition_blackbox_fn(M),
                recon := reconstruction_blackbox_fn(*decomp),
                compute_relative_error(predictions=recon, targets=M.data),
            )[-1]
        )(keys_gen)
    else:
        keys_decomp = jax.random.split(key_2, n_samples)
        return jax.vmap(
            lambda key_gen, key_decomp: (
                M := benchmark_fn(key_gen),
                decomp := decomposition_blackbox_fn(key_decomp, M),
                recon := reconstruction_blackbox_fn(*decomp),
                compute_relative_error(predictions=recon, targets=M.data),
            )[-1]
        )(keys_gen, keys_decomp)


def t_svd_pass(
    key: chex.PRNGKey,
    config,
) -> np.ndarray:
    relative_errors = []

    pbar = tqdm.tqdm(config["n_slices"], desc="", leave=False)
    for n_slices in pbar:
        relative_error_batch = benchmark_synth(
            key=key,
            decomposition_blackbox_fn=eqx.filter_jit(lambda M: t_tsvd(M, n_slices)),
            reconstruction_blackbox_fn=reconstruct_t_svd,
            is_deterministic=True,
            n_samples=config["n_trials"],
            benchmark_type=config["benchmark_type"],
        )

        relative_error = np.mean(relative_error_batch)
        relative_errors.append(relative_error)
        pbar.set_postfix({"relative_error": relative_error, "n_slices": n_slices})

    relative_errors = np.stack(relative_errors)

    return relative_errors


def t_cur_pass(
    key: chex.PRNGKey,
    method,
    config,
) -> np.ndarray:
    relative_errors = []

    arp_constr = CSS_module_factory(method=method)

    pbar = tqdm.tqdm(config["n_slices"], desc="", leave=False)
    for n_slices in pbar:
        relative_error_batch = benchmark_synth(
            key=key,
            decomposition_blackbox_fn=eqx.filter_jit(
                lambda key, M: (
                    keys := jax.random.split(key),
                    arp_params := ARP_params(rsvd_r=n_slices),
                    arp := arp_constr(r=n_slices, css_params=arp_params),
                    CWR_tuple := t_cur(M, css_method=arp, key=keys[0]),
                )[-1]
            ),
            reconstruction_blackbox_fn=reconstruct_t_cross,
            is_deterministic=False,
            n_samples=config["n_trials"],
            benchmark_type=config["benchmark_type"],
        )

        relative_error = np.mean(relative_error_batch)
        relative_errors.append(relative_error)
        pbar.set_postfix({"relative_error": relative_error, "n_slices": n_slices})

    relative_errors = np.stack(relative_errors)

    return relative_errors


def t_css_pass(
    key,
    method,
    config,
    is_deterministic: bool = False,
) -> np.ndarray:
    relative_errors = []
    # execution_times = []

    pbar = tqdm.tqdm(config["n_slices"], desc="", leave=False)
    for n_slices in pbar:
        relative_error_batch = benchmark_synth(
            key,
            decomposition_blackbox_fn=eqx.filter_jit(
                lambda key, M: (
                    keys := jax.random.split(key),
                    Vt := t_rsvd(
                        key=keys[0],
                        M=M,
                        n_slices=n_slices,
                        n_oversamples=config["n_oversamples"],
                        n_subspace_iters=config["n_subspace_iters"],
                    )[-1],
                    # Vt := t_tsvd(
                    #     M=M,
                    #     n_slices=n_slices,
                    # )[-1],
                    V := Vt.T.create_t_matrix(),
                    t_cross_helper(
                        key=keys[1], M=M, V=V, method=method, n_slices=n_slices
                    ),
                )[-1]
            ),
            reconstruction_blackbox_fn=reconstruct_t_cross,
            is_deterministic=is_deterministic,
            n_samples=config["n_trials"],
            benchmark_type=config["benchmark_type"],
        )
        relative_error = np.mean(relative_error_batch)

        relative_errors.append(relative_error)
        # execution_times.append(result.execution_time)

        pbar.set_postfix({"relative_error": relative_error, "n_slices": n_slices})

    relative_errors = np.stack(relative_errors)
    # execution_times = np.stack(execution_times)

    return relative_errors


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Convert lists to tuples
    cfg["t_shape"] = tuple(cfg["t_shape"])
    cfg["n_slices"] = tuple(cfg["n_slices"])
    # Build the map of methods to run
    if "active_methods" in cfg:
        cfg["methods_map"] = {
            k: cfg["methods"][k] for k in cfg["active_methods"] if k in cfg["methods"]
        }
    else:
        cfg["methods_map"] = cfg["methods"]
    return cfg


def get_benchmarking_task(key, method, config):
    """
    Dispatcher function to return the correct task execution.
    """
    if method == "t_svd":
        return t_svd_pass(key=key, config=config)

    if method == "arp_t_cur":
        return t_cur_pass(key=key, method="arp", config=config)

    t_arp_method_map = {
        "arp_householder": "householder",
        "arp_normalized": "orth_proj_normalized",
        "arp": "orth_proj_pinv",
    }

    if method in t_arp_method_map:
        method = t_arp_method_map[method]

    return t_css_pass(key=key, method=method, config=config)


def main(config):
    seed_key = jax.random.PRNGKey(config["seed"])

    relative_errors_map = {}

    pbar = tqdm.tqdm(config["methods_map"].items(), desc="Benchmarking")
    for method_ex, method in pbar:
        pbar.set_postfix(method=method_ex)
        result = get_benchmarking_task(key=seed_key, method=method, config=config)
        relative_errors_map[method_ex] = result

    save_metrics(
        save_path=os.path.join(config["save_path"], config["benchmark_type"]),
        x_axis=np.asarray(config["n_slices"]),
        x_label="# Slices",
        relative_error_map=relative_errors_map,
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, "config.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=default_config_path, help="Path to YAML config file"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
