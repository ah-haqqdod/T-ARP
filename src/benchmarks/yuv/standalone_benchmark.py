import argparse
import os
from functools import partial
from typing import Optional, Sequence, Tuple

import chex
import jax
import numpy as np
import yaml
from jax import numpy as jnp
from tqdm import tqdm

from t_arp.benchmark import (
    compute_psnr,
    compute_relative_error,
    compute_ssim,
    read_yuv_tensor,
    save_metrics,
    write_tensor_to_yuv,
    yuv_to_mp4_ffmpeg,
)
from t_arp.tubal import (
    TARP,
    TCSSBaselines,
    TMatrix,
    t_cross,
    t_rsvd,
    t_tsvd,
)

# NOTE: this benchmark script was designed only for common-index tubal cross-approximation methods (T-ARP, T leverage scores, T uniform)

# NOTE: benchmark loop
# 1. iterate over methods
# 2. iterate over n_slices_list
# 3. iterate over seeds
# The seed iteration is on third layer to minimize the number of jit compilations


def load_yuv_tensor(
    config,  # video_path, width, height, save_path, video_name, framerate
) -> chex.Array:
    # GET TENSOR
    print("Reading YUV video...")
    try:
        tensor = read_yuv_tensor(
            config["video_path"], config["width"], config["height"]
        )
        print(f"Success! Tensor shape: {tensor.shape}")  # e.g., (5, 144, 176, 3)
        print(f"Data type: {tensor.dtype}")
        print(f"Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: File not found at {config['video_path']}. Please check the path."
        )
    except Exception as e:
        raise e
    # SAVE ORIGINAL VIDEO
    yuv_to_mp4_ffmpeg(
        config["video_path"],
        config["save_path"] + f"{config['video_name']}_org.mp4",
        config["width"],
        config["height"],
        config["framerate"],
        pix_fmt="yuv420p",
    )

    print("view of 0-th frame and 0-th channel", tensor[0, :, :, 0])

    return tensor


def save_reconstruction(
    M_recon,
    method,
    n_slices,
    config,
):
    rec_tensor = jnp.moveaxis(M_recon, -1, 0)
    rec_tensor = rec_tensor * 255.0
    rec_tensor = rec_tensor.astype(config["data_dtype"])

    # create dir for method
    method_path = os.path.join(config["save_path"], method)
    os.makedirs(method_path, exist_ok=True)
    # define file name
    file_name = f"rec_{n_slices}.yuv"
    file_path = os.path.join(method_path, file_name)

    write_tensor_to_yuv(
        rec_tensor, file_path, width=config["width"], height=config["height"]
    )

    yuv_to_mp4_ffmpeg(
        file_path,
        os.path.join(method_path, f"{config['video_name']}_{n_slices}.mp4"),
        config["width"],
        config["height"],
        config["framerate"],
        pix_fmt="yuv420p",
    )


@jax.jit
def compute_metrics(
    M_recon: chex.Array, M: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    # NOTE: metrax SSIM and PSNR support batch computation. Move frame axis to position 0 to make frame axis the batch axis.
    M_recon = jnp.moveaxis(M_recon, -1, 0).astype(jnp.float32)
    M = jnp.moveaxis(M, -1, 0).astype(jnp.float32)

    ssim = compute_ssim(predictions=M_recon, targets=M, max_val=1.0)
    psnr = compute_psnr(predictions=M_recon, targets=M, max_val=1.0)
    relative_error = compute_relative_error(predictions=M_recon, targets=M)

    return ssim, psnr, relative_error


@jax.jit(static_argnames=["method", "n_slices"])
def t_cross_helper(
    key: chex.PRNGKey, M: TMatrix, V: TMatrix, method: str, n_slices: int
) -> chex.Array:
    if method == "uniform":
        tcss_module_constructor = partial(TCSSBaselines, method="uniform")
    elif method == "leverage_scores":
        tcss_module_constructor = partial(TCSSBaselines, method="leverage_scores")
    else:
        tcss_module_constructor = partial(TARP, method=method, use_derandomized=True)

    key, subkey = jax.random.split(key)

    A_J, W, A_I = t_cross(
        key=subkey,
        A=M,
        V=V,
        partial_tcss_module_constructor=tcss_module_constructor,
        n_vert_slices=n_slices,
        # use_intersection=True,
    )

    M_recon = (A_J @ W @ A_I).create_t_matrix()

    M_recon = jnp.clip(M_recon.data, 0.0, 1.0)

    return M_recon


def single_method_benchmark(
    key: chex.PRNGKey,
    M: TMatrix,
    Vs: Sequence[TMatrix],
    method: str,
    method_ex,
    config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ssim_list = []
    psnr_list = []
    relative_error_list = []

    # second loop: iterate over list of numbers of slices
    pbar_n_slices = tqdm(
        config["n_slices"],
        desc=f"Looping over n_slices for {method}...",
        position=1,
        leave=True,
    )
    for i, n_slices in enumerate(pbar_n_slices):
        pbar_n_slices.set_postfix(n_slices=n_slices)
        subkey = jax.random.fold_in(key, i)
        t_cross_keys = (
            (subkey,)
            if config["n_trials"] == 1
            else jax.random.split(subkey, config["n_trials"])
        )

        ssim_trials = []
        psnr_trials = []
        relative_error_trials = []

        # third loop: iterate for N_TRIALS to collect statistics
        pbar_trials = tqdm(
            range(config["n_trials"]),
            desc=f"Trials for {method}, n_slices={n_slices}",
            position=2,
            leave=False,
        )
        for trial in pbar_trials:
            pbar_trials.set_postfix(trial=trial, doing="decomposition")

            # Tubal cross approximation
            M_recon = t_cross_helper(
                key=t_cross_keys[trial], M=M, V=Vs[i], method=method, n_slices=n_slices
            )
            jax.block_until_ready(M_recon)

            if config["save_reconstructions"] and trial == 0:
                # save only the first trial
                if not method_ex:
                    method_ex = method
                save_reconstruction(
                    M_recon, method=method_ex, n_slices=n_slices, config=config
                )

            # Compute metrics
            pbar_trials.set_postfix(trial=trial, doing="metrics")
            ssim, psnr, rel_err = compute_metrics(M_recon=M_recon, M=M.data)
            jax.block_until_ready(ssim)
            jax.block_until_ready(psnr)
            jax.block_until_ready(rel_err)

            ssim_trials.append(ssim)
            psnr_trials.append(psnr)
            relative_error_trials.append(rel_err)

        ssim_list.append(np.stack(ssim_trials))
        psnr_list.append(np.stack(psnr_trials))
        relative_error_list.append(np.stack(relative_error_trials))

    return np.stack(ssim_list), np.stack(psnr_list), np.stack(relative_error_list)


def precompute_Vs(
    M: TMatrix,
    config,
    key: Optional[chex.PRNGKey] = None,
) -> Sequence[TMatrix]:
    Vs = []
    pbar = tqdm(config["n_slices"], desc="Computing V for all n_slices...")
    for n in pbar:
        if config["use_rsvd"]:
            if key is None:
                raise ValueError("key must be provided when use_rsvd is True")
            _, _, t_Vt = t_rsvd(
                key,
                M,
                n_slices=n,
                n_oversamples=config["n_oversamples"],
                n_subspace_iters=config["n_subspace_iters"],
            )
        else:
            _, _, t_Vt = t_tsvd(M)
        V = t_Vt.T.create_t_matrix(config["dtype"])
        jax.block_until_ready(V)
        Vs.append(V)
        pbar.set_postfix(n_slices=n)

    return Vs


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Dynamically construct the video path
    if "video_dir" in cfg and "video_name" in cfg:
        cfg["video_path"] = os.path.join(cfg["video_dir"], f"{cfg['video_name']}.yuv")

    # TODO: verify that the video path is valid

    # 2. Map string representations to JAX data types
    dtype_map = {
        "float32": jnp.float32,
        "uint8": jnp.uint8,
        "float64": jnp.float64,
        "int32": jnp.int32,
    }
    if "dtype" in cfg:
        cfg["dtype"] = dtype_map.get(cfg["dtype"], cfg["dtype"])
    if "data_dtype" in cfg:
        cfg["data_dtype"] = dtype_map.get(cfg["data_dtype"], cfg["data_dtype"])

    # 3. Convert lists to tuples where required
    if "n_slices" in cfg:
        cfg["n_slices"] = tuple(cfg["n_slices"])

    # 4. Build the map of methods to run based on active_methods selection
    if "active_methods" in cfg and "methods" in cfg:
        cfg["methods_map"] = {
            k: cfg["methods"][k] for k in cfg["active_methods"] if k in cfg["methods"]
        }
    else:
        cfg["methods_map"] = cfg.get("methods", {})

    # 5. determine the video shape from the video file suffix
    if "width" not in cfg or "height" not in cfg:
        if "video_path" in cfg:
            # suffix = os.path.splitext(cfg["video_path"])[1]
            if cfg["video_name"].endswith("_qcif"):
                cfg["width"] = 176
                cfg["height"] = 144
            elif cfg["video_name"].endswith("_cif"):
                cfg["width"] = 352
                cfg["height"] = 288
            else:
                raise ValueError(f"Unknown video suffix: {cfg['video_name']}")
        else:
            raise ValueError("video_path was not resolved")

    return cfg


def main(config):
    tensor = load_yuv_tensor(config)
    tensor_HWCF = jnp.moveaxis(tensor, source=0, destination=-1)
    tensor_HWCF = tensor_HWCF.astype(config["dtype"]) / 255.0

    M = TMatrix(tensor_HWCF)
    print("Shape of TMatrix", M.shape)

    seed_key = jax.random.PRNGKey(config["seed"])
    key, subkey = jax.random.split(seed_key)

    # PRECOMPUTE Vs for all n_slices
    Vs = precompute_Vs(
        M,
        key=subkey,
        config=config,
    )
    print("V dtype", Vs[0].dtype)

    # BENCHMARK METHODS

    ssim_map = {}
    psnr_map = {}
    relative_error_map = {}
    pbar = tqdm(
        config["methods_map"].items(),
        desc="Benchmarking methods...",
        position=0,
        leave=True,
    )

    # first loop: benchmark each method
    for method_ex, method in pbar:
        pbar.set_postfix(method=method_ex)
        ssim, psnr, relative_error = single_method_benchmark(
            key=key,
            M=M,
            Vs=Vs,
            method=method,
            method_ex=method_ex,
            config=config,
        )
        ssim_map[method_ex] = ssim
        psnr_map[method_ex] = psnr
        relative_error_map[method_ex] = relative_error

    # SAVE METRICS
    print("Saving metrics...")
    save_metrics(
        save_path=config["save_path"],
        x_axis=np.asarray(config["n_slices"]),
        x_label="# Slices",
        ssim_map=ssim_map,
        psnr_map=psnr_map,
        relative_error_map=relative_error_map,
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
