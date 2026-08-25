"""YUV video benchmark for tubal cross-approximation methods.

Benchmark loop:
  1. Iterate over methods
  2. Iterate over n_slices
  3. Iterate over trials (to collect statistics for randomised methods)

The trial loop is innermost to minimise the number of JAX JIT recompilations.
"""

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
from t_arp.matrix.css import CSS_module_factory
from t_arp.matrix.css_modules.arp import ARP_params
from t_arp.tubal import (
    TARP,
    TCSSBaselines,
    TMatrix,
    t_cross,
    t_cur,
    t_rsvd,
    t_tsvd,
)
from t_arp.tubal.utils import reconstruct_t_cross

# enable float64 and complex128 in jax
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# JIT-compiled helpers
# ---------------------------------------------------------------------------

@jax.jit
def compute_metrics(
    M_recon: chex.Array,
    M: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Compute SSIM, PSNR and relative error between a reconstruction and the original.

    Frames are expected in HWC-F layout (height, width, channels, frames).
    The frame axis is moved to position 0 to use frame as the batch dimension.
    """
    M_recon = jnp.moveaxis(M_recon, -1, 0).astype(jnp.float32)
    M = jnp.moveaxis(M, -1, 0).astype(jnp.float32)
    ssim = compute_ssim(predictions=M_recon, targets=M, max_val=1.0)
    psnr = compute_psnr(predictions=M_recon, targets=M, max_val=1.0)
    relative_error = compute_relative_error(predictions=M_recon, targets=M)
    return ssim, psnr, relative_error


@jax.jit(static_argnames=["n_slices"])
def t_cur_helper(key: chex.PRNGKey, M: TMatrix, n_slices: int) -> chex.Array:
    """Run T-CUR decomposition and return the clipped reconstruction."""
    arp_constr = CSS_module_factory(method="arp")
    arp_params = ARP_params(rsvd_r=n_slices)
    arp = arp_constr(r=n_slices, css_params=arp_params)
    C, U, R = t_cur(M, css_method=arp, key=key)
    return jnp.clip(reconstruct_t_cross(C, U, R), 0.0, 1.0)


@jax.jit(static_argnames=["method", "n_slices"])
def t_cross_helper(
    key: chex.PRNGKey,
    M: TMatrix,
    V: TMatrix,
    method: str,
    n_slices: int,
) -> chex.Array:
    """Run tubal cross-approximation with the given CSS method and return the clipped reconstruction."""
    if method == "uniform":
        tcss_module_constructor = partial(TCSSBaselines, method="uniform")
    elif method == "leverage_scores":
        tcss_module_constructor = partial(TCSSBaselines, method="leverage_scores")
    else:
        tcss_module_constructor = partial(TARP, method=method, use_derandomized=False)

    key, subkey = jax.random.split(key)
    A_J, W, A_I = t_cross(
        key=subkey,
        A=M,
        V=V,
        partial_tcss_module_constructor=tcss_module_constructor,
        n_vert_slices=n_slices,
    )
    return jnp.clip(reconstruct_t_cross(A_J, W, A_I), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Data loading and saving
# ---------------------------------------------------------------------------

def load_yuv_tensor(config: dict) -> chex.Array:
    """Read the YUV video, save the original as MP4 and return the raw tensor."""
    print("Reading YUV video...")
    tensor = read_yuv_tensor(config["video_path"], config["width"], config["height"])
    print(f"Tensor shape : {tensor.shape}  |  dtype: {tensor.dtype}  |  "
          f"range: [{tensor.min():.0f}, {tensor.max():.0f}]")

    os.makedirs(config["save_path"], exist_ok=True)
    yuv_to_mp4_ffmpeg(
        config["video_path"],
        os.path.join(config["save_path"], f"{config['video_name']}_org.mp4"),
        config["width"],
        config["height"],
        config["framerate"],
        pix_fmt="yuv420p",
    )
    return tensor


def save_reconstruction(
    M_recon: chex.Array,
    method_ex: str,
    n_slices: int,
    config: dict,
) -> None:
    """Write a reconstructed tensor to a YUV file and convert it to MP4."""
    rec_tensor = jnp.moveaxis(M_recon, -1, 0) * 255.0
    rec_tensor = rec_tensor.astype(config["data_dtype"])

    method_path = os.path.join(config["save_path"], method_ex)
    os.makedirs(method_path, exist_ok=True)

    yuv_path = os.path.join(method_path, f"rec_{n_slices}.yuv")
    write_tensor_to_yuv(rec_tensor, yuv_path, width=config["width"], height=config["height"])

    yuv_to_mp4_ffmpeg(
        yuv_path,
        os.path.join(method_path, f"{config['video_name']}_{n_slices}.mp4"),
        config["width"],
        config["height"],
        config["framerate"],
        pix_fmt="yuv420p",
    )


# ---------------------------------------------------------------------------
# Per-method benchmark passes
# ---------------------------------------------------------------------------

def _collect_trial_metrics(
    M_recon: chex.Array,
    M_data: chex.Array,
    method_ex: str,
    n_slices: int,
    trial: int,
    config: dict,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Optionally save the reconstruction (first trial only) and return metrics."""
    jax.block_until_ready(M_recon)
    if config["save_reconstructions"] and trial == 0:
        save_reconstruction(M_recon, method_ex=method_ex, n_slices=n_slices, config=config)
    ssim, psnr, rel_err = compute_metrics(M_recon=M_recon, M=M_data)
    jax.block_until_ready(ssim)
    jax.block_until_ready(psnr)
    jax.block_until_ready(rel_err)
    return ssim, psnr, rel_err


def t_cur_pass(
    key: chex.PRNGKey,
    M: TMatrix,
    method_ex: str,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Benchmark T-CUR decomposition across all n_slices and trials."""
    ssim_list, psnr_list, rel_err_list = [], [], []

    pbar_slices = tqdm(config["n_slices"], desc=f"{method_ex}", position=1, leave=True)
    for i, n_slices in enumerate(pbar_slices):
        pbar_slices.set_postfix(n_slices=n_slices)
        subkey = jax.random.fold_in(key, i)
        trial_keys = (
            (subkey,) if config["n_trials"] == 1
            else jax.random.split(subkey, config["n_trials"])
        )

        ssim_trials, psnr_trials, rel_err_trials = [], [], []
        pbar_trials = tqdm(
            range(config["n_trials"]),
            desc=f"n_slices={n_slices}",
            position=2,
            leave=False,
        )
        for trial in pbar_trials:
            pbar_trials.set_postfix(trial=trial)
            M_recon = t_cur_helper(key=trial_keys[trial], M=M, n_slices=n_slices)
            ssim, psnr, rel_err = _collect_trial_metrics(
                M_recon, M.data, method_ex, n_slices, trial, config
            )
            ssim_trials.append(ssim)
            psnr_trials.append(psnr)
            rel_err_trials.append(rel_err)

        ssim_list.append(np.stack(ssim_trials))
        psnr_list.append(np.stack(psnr_trials))
        rel_err_list.append(np.stack(rel_err_trials))

    return np.stack(ssim_list), np.stack(psnr_list), np.stack(rel_err_list)


def t_css_pass(
    key: chex.PRNGKey,
    M: TMatrix,
    Vs: Sequence[TMatrix],
    method: str,
    method_ex: str,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Benchmark a tubal CSS method across all n_slices and trials."""
    ssim_list, psnr_list, rel_err_list = [], [], []

    pbar_slices = tqdm(
        enumerate(config["n_slices"]),
        total=len(config["n_slices"]),
        desc=f"{method_ex}",
        position=1,
        leave=True,
    )
    for i, n_slices in pbar_slices:
        pbar_slices.set_postfix(n_slices=n_slices)
        subkey = jax.random.fold_in(key, i)
        trial_keys = (
            (subkey,) if config["n_trials"] == 1
            else jax.random.split(subkey, config["n_trials"])
        )

        ssim_trials, psnr_trials, rel_err_trials = [], [], []
        pbar_trials = tqdm(
            range(config["n_trials"]),
            desc=f"n_slices={n_slices}",
            position=2,
            leave=False,
        )
        for trial in pbar_trials:
            pbar_trials.set_postfix(trial=trial)
            M_recon = t_cross_helper(
                key=trial_keys[trial], M=M, V=Vs[i], method=method, n_slices=n_slices
            )
            ssim, psnr, rel_err = _collect_trial_metrics(
                M_recon, M.data, method_ex, n_slices, trial, config
            )
            ssim_trials.append(ssim)
            psnr_trials.append(psnr)
            rel_err_trials.append(rel_err)

        ssim_list.append(np.stack(ssim_trials))
        psnr_list.append(np.stack(psnr_trials))
        rel_err_list.append(np.stack(rel_err_trials))

    return np.stack(ssim_list), np.stack(psnr_list), np.stack(rel_err_list)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def precompute_Vs(
    M: TMatrix,
    config: dict,
    key: Optional[chex.PRNGKey] = None,
) -> Sequence[TMatrix]:
    """Precompute the right singular vector basis V for each value of n_slices."""
    Vs = []
    pbar = tqdm(config["n_slices"], desc="Precomputing V for all n_slices")
    for n in pbar:
        pbar.set_postfix(n_slices=n)
        if config["use_rsvd"]:
            if key is None:
                raise ValueError("key must be provided when use_rsvd is True")
            _, _, t_Vt = t_rsvd(
                key, M,
                n_slices=n,
                n_oversamples=config["n_oversamples"],
                n_subspace_iters=config["n_subspace_iters"],
            )
        else:
            _, _, t_Vt = t_tsvd(M)
        V = t_Vt.T.create_t_matrix(config["dtype"])
        jax.block_until_ready(V)
        Vs.append(V)
    return Vs


# ---------------------------------------------------------------------------
# Dispatcher and main
# ---------------------------------------------------------------------------

def get_benchmarking_task(
    key: chex.PRNGKey,
    M: TMatrix,
    Vs: Sequence[TMatrix],
    method: str,
    method_ex: str,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Route a method identifier to the appropriate benchmark pass."""
    if method == "arp_t_cur":
        return t_cur_pass(key=key, M=M, method_ex=method_ex, config=config)

    return t_css_pass(key=key, M=M, Vs=Vs, method=method, method_ex=method_ex, config=config)


def main(config: dict) -> None:
    # --- Data preparation ---
    tensor = load_yuv_tensor(config)
    tensor_HWCF = jnp.moveaxis(tensor, source=0, destination=-1)
    tensor_HWCF = tensor_HWCF.astype(config["dtype"]) / 255.0
    M = TMatrix(tensor_HWCF)
    print(f"TMatrix shape: {M.shape}")

    seed_key = jax.random.PRNGKey(config["seed"])
    key, subkey = jax.random.split(seed_key)

    # --- Precompute bases ---
    Vs = precompute_Vs(M, key=subkey, config=config)
    print(f"V dtype: {Vs[0].dtype}")

    # --- Benchmark ---
    ssim_map, psnr_map, rel_err_map = {}, {}, {}
    pbar = tqdm(config["methods_map"].items(), desc="Benchmarking", position=0, leave=True)
    for method_ex, method in pbar:
        pbar.set_postfix(method=method_ex)
        ssim, psnr, rel_err = get_benchmarking_task(
            key=key, M=M, Vs=Vs, method=method, method_ex=method_ex, config=config
        )
        ssim_map[method_ex] = ssim
        psnr_map[method_ex] = psnr
        rel_err_map[method_ex] = rel_err

    # --- Save results ---
    print("Saving metrics...")
    save_metrics(
        save_path=config["save_path"],
        x_axis=np.asarray(config["n_slices"]),
        x_label="# Slices",
        ssim_map=ssim_map,
        psnr_map=psnr_map,
        relative_error_map=rel_err_map,
    )


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve video path
    if "video_dir" in cfg and "video_name" in cfg:
        cfg["video_path"] = os.path.join(cfg["video_dir"], f"{cfg['video_name']}.yuv")

    # Map dtype strings to JAX dtypes
    dtype_map = {
        "float32": jnp.float32,
        "float64": jnp.float64,
        "uint8": jnp.uint8,
        "int32": jnp.int32,
    }
    cfg["dtype"] = dtype_map.get(cfg.get("dtype"), jnp.float32)
    cfg["data_dtype"] = dtype_map.get(cfg.get("data_dtype"), jnp.uint8)

    # Convert n_slices list to tuple (required for JAX static args)
    if "n_slices" in cfg:
        cfg["n_slices"] = tuple(cfg["n_slices"])

    # Resolve video resolution from filename suffix if not explicit
    if "width" not in cfg or "height" not in cfg:
        name = cfg.get("video_name", "")
        if name.endswith("_qcif"):
            cfg["width"], cfg["height"] = 176, 144
        elif name.endswith("_cif"):
            cfg["width"], cfg["height"] = 352, 288
        else:
            raise ValueError(
                f"Cannot infer resolution from video_name '{name}'. "
                "Specify 'width' and 'height' explicitly in the config."
            )

    # Build active methods map
    if "active_methods" in cfg and "methods" in cfg:
        cfg["methods_map"] = {
            k: cfg["methods"][k] for k in cfg["active_methods"] if k in cfg["methods"]
        }
    else:
        cfg["methods_map"] = cfg.get("methods", {})

    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, "config.yaml")

    parser = argparse.ArgumentParser(description="YUV video benchmark for tubal methods.")
    parser.add_argument("--config", default=default_config_path, help="Path to YAML config file")
    args = parser.parse_args()

    main(load_config(args.config))
