import argparse
import os
from functools import partial
from typing import Literal, Tuple, Union

import chex
import equinox as eqx
import jax
import numpy as np
import tqdm
import yaml
from jax import numpy as jnp
from PIL import Image

from t_arp.benchmark import KodakBenchmark, save_metrics
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
# NOTE: benchmark loop
# 1. iterate over methods
# 2. iterate over n_slices_list
# 3. iterate over seeds
# The seed iteration is on third layer to minimize the number of jit compilations


def save_reconstruction(reconstruction, dir_path, idx):
    # Save as PNG
    img_pil = Image.fromarray(np.asarray(reconstruction * 255).astype(np.uint8))
    save_path = os.path.join(dir_path, f"{idx:04d}.png")
    img_pil.save(save_path)


def save_reconstructions(method_name, n_slices, reconstructions, dir_path):
    # prefix / method / sub-method / rank / reconstructions
    dir_path = os.path.join(dir_path, *("reconstructions", str(n_slices)), method_name)
    os.makedirs(dir_path, exist_ok=True)
    for idx, reconstruction in enumerate(reconstructions):
        save_reconstruction(reconstruction, dir_path, idx)


@jax.jit(static_argnames=["method", "n_slices"])
def t_cross_helper(
    key: chex.PRNGKey,
    M: TMatrix,
    V: TMatrix,
    method: Literal[
        "uniform",
        "length_squared",
        "leverage_scores",
        "householder",
        "orth_proj_pinv",
        "orth_proj_normalized",
        # "fast_arp",
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
    elif method == "length_squared":
        tcss_module_constructor = partial(TCSSBaselines, method="length_squared")
    # elif method == "fast_arp":
    #     tcss_module_constructor = partial(FastTARP, method="householder")
    else:
        tcss_module_constructor = partial(TARP, method=method, use_derandomized=False)

    key, subkey = jax.random.split(key)

    C, W, R = t_cross(
        key=subkey,
        A=M,
        V=V,
        partial_tcss_module_constructor=tcss_module_constructor,
        n_vert_slices=n_slices,
    )

    return C, W, R


def t_svd_pass(
    method_name,
    config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    relative_errors = []
    ssims = []
    psnrs = []
    execution_times = []

    # for n_slices in config["n_slices"]:
    pbar = tqdm.tqdm(config["n_slices"], desc="", leave=False)
    for n_slices in pbar:
        pbar.set_postfix({"n_slices": n_slices})
        # define benchmark
        benchmark = KodakBenchmark(
            decomposition_blackbox_fn=eqx.filter_jit(
                lambda image: (M := TMatrix(image), t_tsvd(M, n_slices))[-1]
            ),
            reconstruction_blackbox_fn=lambda U, S, Vt: jnp.clip(
                reconstruct_t_svd(U, S, Vt), 0.0, 1.0
            ).astype(EXPERIMENTS_DTYPE),
            data_dir=config["data_dir"],
        )

        result, reconstructions = benchmark(dtype=EXPERIMENTS_DTYPE)
        # log reconstructions
        if config["save_reconstruction"]:
            save_reconstructions(
                method_name=method_name,
                n_slices=n_slices,
                reconstructions=reconstructions,
                dir_path=config["save_path"],
            )
        relative_errors.append(result.relative_error)
        ssims.append(result.ssim)
        psnrs.append(result.psnr)
        execution_times.append(result.execution_time)

    ssims = np.stack(ssims)
    psnrs = np.stack(psnrs)
    relative_errors = np.stack(relative_errors)
    execution_times = np.stack(execution_times)

    return ssims, psnrs, relative_errors, execution_times


def t_cur_pass(
    key: chex.PRNGKey,
    method,
    method_name,
    config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    relative_errors = []
    ssims = []
    psnrs = []
    execution_times = []

    arp_constr = CSS_module_factory(method=method)

    pbar = tqdm.tqdm(config["n_slices"], desc="", leave=False)
    for n_slices in pbar:
        pbar.set_postfix({"n_slices": n_slices})
        benchmark = KodakBenchmark(
            n_trials=config["n_trials"],
            decomposition_blackbox_fn=eqx.filter_jit(
                lambda key, image: (
                    M := TMatrix(image.astype(EXPERIMENTS_DTYPE)),
                    arp_params := ARP_params(rsvd_r=n_slices),
                    arp := arp_constr(r=n_slices, css_params=arp_params),
                    CWR_tuple := t_cur(M, css_method=arp, key=key),
                )[-1]
            ),
            # reconstruction_blackbox_fn=reconstruct_t_cross,
            reconstruction_blackbox_fn=lambda C, W, R: jnp.clip(
                reconstruct_t_cross(C, W, R), 0.0, 1.0
            ).astype(EXPERIMENTS_DTYPE),
            data_dir=config["data_dir"],
        )

        result, reconstructions = benchmark(key, dtype=EXPERIMENTS_DTYPE)
        # log reconstructions
        if config["save_reconstruction"]:
            save_reconstructions(
                method_name=method_name,
                n_slices=n_slices,
                reconstructions=reconstructions,
                dir_path=config["save_path"],
            )
        relative_errors.append(result.relative_error)
        ssims.append(result.ssim)
        psnrs.append(result.psnr)
        execution_times.append(result.execution_time)

    ssims = np.stack(ssims)
    psnrs = np.stack(psnrs)
    relative_errors = np.stack(relative_errors)
    execution_times = np.stack(execution_times)

    return ssims, psnrs, relative_errors, execution_times


def t_css_pass(
    key,
    method,
    method_name,
    config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    relative_errors = []
    ssims = []
    psnrs = []
    execution_times = []

    # for n_slices in config["n_slices"]:
    pbar = tqdm.tqdm(config["n_slices"], desc="", leave=False)
    for n_slices in pbar:
        pbar.set_postfix({"n_slices": n_slices})
        # define benchmark
        benchmark = KodakBenchmark(
            n_trials=config["n_trials"],
            decomposition_blackbox_fn=eqx.filter_jit(
                lambda key, image: (
                    M := TMatrix(image.astype(EXPERIMENTS_DTYPE)),
                    Vt := t_rsvd(
                        key=key,
                        M=M,
                        n_slices=n_slices,
                        n_oversamples=config["n_oversamples"],
                        n_subspace_iters=config["n_subspace_iters"],
                    )[-1],
                    V := Vt.T.create_t_matrix(),
                    keys := jax.random.split(key),
                    t_cross_helper(
                        key=keys[1], M=M, V=V, method=method, n_slices=n_slices
                    ),
                )[-1]
            ),
            reconstruction_blackbox_fn=lambda C, W, R: jnp.clip(
                reconstruct_t_cross(C, W, R), 0.0, 1.0
            ).astype(EXPERIMENTS_DTYPE),
            data_dir=config["data_dir"],
        )

        result, reconstructions = benchmark(key, dtype=EXPERIMENTS_DTYPE)
        # log reconstructions
        if save_reconstruction:
            save_reconstructions(
                method_name=method_name,
                n_slices=n_slices,
                reconstructions=reconstructions,
                dir_path=config["save_path"],
            )
        relative_errors.append(result.relative_error)
        ssims.append(result.ssim)
        psnrs.append(result.psnr)
        execution_times.append(result.execution_time)

    ssims = np.stack(ssims)
    psnrs = np.stack(psnrs)
    relative_errors = np.stack(relative_errors)
    execution_times = np.stack(execution_times)

    return ssims, psnrs, relative_errors, execution_times


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Validate / set defaults if needed
    config.setdefault("n_trials", 1)
    config.setdefault("save_reconstruction", False)
    # Convert list to tuple if needed downstream (e.g., n_slices)
    if isinstance(config.get("n_slices"), list):
        config["n_slices"] = tuple(config["n_slices"])
    # Build METHODS_MAP or filter active methods
    if "active_methods" in config:
        config["methods_map"] = {
            k: config["methods"][k]
            for k in config["active_methods"]
            if k in config["methods"]
        }
    else:
        config["methods_map"] = config["methods"]
    return config


def get_benchmarking_task(key, method, method_name, config):
    """
    Dispatcher function to return the correct task execution.
    """
    if method == "t_svd":
        return t_svd_pass(method_name=method_name, config=config)

    if method == "arp_t_cur":
        return t_cur_pass(key=key, method="arp", method_name=method_name, config=config)

    t_arp_method_map = {
        "arp_householder": "householder",
        "arp_normalized": "orth_proj_normalized",
        "arp": "orth_proj_pinv",
    }

    if method in t_arp_method_map:
        method = t_arp_method_map[method]

    return t_css_pass(key=key, method=method, method_name=method_name, config=config)


def main(config):
    seed_key = jax.random.PRNGKey(config["seed"])

    ssims_map = {}
    psnrs_map = {}
    relative_errors_map = {}
    execution_times_map = {}

    pbar = tqdm.tqdm(config["methods_map"].items(), desc="Benchmarking")
    for method_ex, method in pbar:
        pbar.set_postfix(method=method_ex)
        ssims, psnrs, relative_errors, execution_times = get_benchmarking_task(
            key=seed_key,
            method=method,
            method_name=method_ex,
            config=config,
        )

        ssims_map[method_ex] = ssims
        psnrs_map[method_ex] = psnrs
        relative_errors_map[method_ex] = relative_errors
        execution_times_map[method_ex] = execution_times

    save_metrics(
        save_path=config["save_path"],
        x_axis=np.asarray(config["n_slices"]),
        x_label="# Slices",
        ssim_map=ssims_map,
        psnr_map=psnrs_map,
        relative_error_map=relative_errors_map,
        execution_time_map=execution_times_map,
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
