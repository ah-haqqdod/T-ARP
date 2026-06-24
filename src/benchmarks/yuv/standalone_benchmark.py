import os
from functools import partial
from typing import List, Optional, Sequence, Tuple

import chex
import jax
import numpy as np
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
from t_arp.matrix import RSVD
from t_arp.tubal import TARP, TCSSBaselines, TMatrix, TMatrixTOnly, t_cross

# NOTE: benchmark loop
# 1. iterate over methods
# 2. iterate over n_slices_list
# 3. iterate over seeds
# The seed iteration is on third layer to minimize the number of jit compilations

METHODS_MAP = {
    "Uniform": "uniform",
    "Leverage Scores": "leverage_scores",
    # "T-ARP": "orth_proj_pinv",
    "T-ARP (Householder)": "householder",
}

N_SLICES_LIST = [50]
SEED = 42
N_TRIALS = 1
DTYPE = jnp.float32
DATA_DTYPE = jnp.uint8
SAVE_RECONSTRUCTIONS = True
USE_RSVD = True

VIDEO_DIR = "datasets/yuv_video/Video/"
VIDEO_NAME = "waterfall_cif"
VIDEO_PATH = os.path.join(VIDEO_DIR, f"{VIDEO_NAME}.yuv")

WIDTH = 176 if VIDEO_NAME.endswith("qcif") else 352  # QCIF width
HEIGHT = 144 if VIDEO_NAME.endswith("qcif") else 288  # QCIF height
FRAMERATE = 25
SAVE_PATH = "src/benchmarks/yuv/results/"

N_OVERSAMPLES = 5
N_SUBSPACE_ITERS = 1


def load_yuv_tensor() -> chex.Array:
    # GET TENSOR
    print("Reading YUV video...")
    try:
        tensor = read_yuv_tensor(VIDEO_PATH, WIDTH, HEIGHT)
        print(f"Success! Tensor shape: {tensor.shape}")  # e.g., (5, 144, 176, 3)
        print(f"Data type: {tensor.dtype}")
        print(f"Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: File not found at {VIDEO_PATH}. Please check the path."
        )
    except Exception as e:
        raise e
    # SAVE ORIGINAL VIDEO
    yuv_to_mp4_ffmpeg(
        VIDEO_PATH,
        SAVE_PATH + f"{VIDEO_NAME}_org.mp4",
        WIDTH,
        HEIGHT,
        FRAMERATE,
        pix_fmt="yuv420p",
    )

    print("view of 0-th frame and 0-th channel", tensor[0, :, :, 0])

    return tensor


def save_reconstruction(M_recon, method, n_slices):
    rec_tensor = jnp.moveaxis(M_recon, -1, 0)
    rec_tensor = rec_tensor * 255.0
    rec_tensor = rec_tensor.astype(DATA_DTYPE)

    # create dir for method
    method_path = os.path.join(SAVE_PATH, method)
    os.makedirs(method_path, exist_ok=True)
    # define file name
    file_name = f"rec_{n_slices}.yuv"
    file_path = os.path.join(method_path, file_name)

    write_tensor_to_yuv(rec_tensor, file_path, width=WIDTH, height=HEIGHT)

    yuv_to_mp4_ffmpeg(
        file_path,
        os.path.join(method_path, f"{VIDEO_NAME}_{n_slices}.mp4"),
        WIDTH,
        HEIGHT,
        FRAMERATE,
        pix_fmt="yuv420p",
    )


@jax.jit
def t_svd(M: TMatrix) -> TMatrixTOnly:
    raise NotImplementedError
    svd_fn = partial(jnp.linalg.svd, full_matrices=False)
    t_U, t_S, t_V = M.facewise_operation(svd_fn)

    return t_V


@jax.jit(static_argnames=["n_slices"])
def t_rsvd(key: chex.PRNGKey, M: TMatrix, n_slices: int) -> TMatrixTOnly:
    # rsvd = RSVD(
    #     rank=min(M.shape[0], M.shape[1]) // 4, n_oversamples=5, n_subspace_iters=1
    # )
    rsvd = RSVD(
        rank=min(n_slices * 2, min(M.shape[0], M.shape[1])),
        # rank=min(n_slices, M.shape[0], M.shape[1]),
        n_oversamples=N_OVERSAMPLES,
        n_subspace_iters=N_SUBSPACE_ITERS,
    )

    def rsvd_fn(key: chex.PRNGKey, A: chex.Array):
        return rsvd(key=key, A=A)

    # svd_fn = partial(jnp.linalg.svd, full_matrices=False)
    _, _, t_Vt = M.facewise_operation(rsvd_fn, key=key)

    return t_Vt


# NOTE: it takes approximately 16s to compute metrics
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

    # print(f"\nCompiling t_cross for method={method}, n_slices={n_slices}, key={key}\n")
    key, subkey = jax.random.split(key)

    # print(f"V shape: {V.shape}")

    A_J, W, A_I = t_cross(
        key=subkey,
        A=M,
        V=V,
        partial_tcss_module_constructor=tcss_module_constructor,
        n_vert_slices=n_slices,
        # use_intersection=True,
    )

    # print(A_J.shape, W.shape, A_I.shape)
    M_recon = (A_J @ W @ A_I).create_t_matrix()
    # M_recon = M_recon.data * 255.0

    # M_recon = jnp.clip(M_recon.data, jnp.min(M.data), jnp.max(M.data))
    M_recon = jnp.clip(M_recon.data, 0.0, 1.0)

    return M_recon


def single_method_benchmark(
    key: chex.PRNGKey,
    M: TMatrix,
    Vs: Sequence[TMatrix],
    method: str,
    n_slices_list: List[int],
    n_trials: int = 1,
    save_reconstructions: bool = False,
    method_ex: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # print(f"Benchmarking {method}...")
    ssim_list = []
    psnr_list = []
    relative_error_list = []

    # second loop: iterate over list of numbers of slices
    pbar_n_slices = tqdm(
        n_slices_list,
        desc=f"Looping over n_slices for {method}...",
        position=1,
        leave=True,
    )
    for i, n_slices in enumerate(pbar_n_slices):
        pbar_n_slices.set_postfix(n_slices=n_slices)
        subkey = jax.random.fold_in(key, i)
        t_cross_keys = (
            (subkey,) if n_trials == 1 else jax.random.split(subkey, n_trials)
        )

        ssim_trials = []
        psnr_trials = []
        relative_error_trials = []

        # third loop: iterate for N_TRIALS to collect statistics
        pbar_trials = tqdm(
            range(n_trials),
            desc=f"Trials for {method}, n_slices={n_slices}",
            position=2,
            leave=False,  # don't leave the trial bar after it finishes
        )
        for trial in pbar_trials:
            pbar_trials.set_postfix(trial=trial, doing="decomposition")

            # Tubal cross approximation
            M_recon = t_cross_helper(
                key=t_cross_keys[trial], M=M, V=Vs[i], method=method, n_slices=n_slices
            )
            jax.block_until_ready(M_recon)

            if save_reconstructions and trial == 0:
                # save only the first trial
                if not method_ex:
                    method_ex = method
                save_reconstruction(M_recon, n_slices=n_slices, method=method_ex)

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
    use_rsvd: bool = True,
    key: Optional[chex.PRNGKey] = None,
) -> Sequence[TMatrix]:
    Vs = []
    pbar = tqdm(N_SLICES_LIST, desc="Computing V for all n_slices...")
    for n in pbar:
        if use_rsvd:
            if key is None:
                raise ValueError("key must be provided when use_rsvd is True")
            t_Vt = t_rsvd(key, M, n_slices=n)
        else:
            t_Vt = t_svd(M)
        V = t_Vt.T.create_t_matrix(DTYPE)
        jax.block_until_ready(V)
        Vs.append(V)
        pbar.set_postfix(n_slices=n)

    return Vs


def main():
    # tensor shape is (height, width, channels, frames)
    tensor = load_yuv_tensor()
    tensor_HWCF = jnp.moveaxis(tensor, source=0, destination=-1)
    tensor_HWCF = tensor_HWCF.astype(jnp.float32) / 255.0
    # print("tensor_HWCF dtype", tensor_HWCF.dtype)

    M = TMatrix(tensor_HWCF)
    print("Shape of TMatrix", M.shape)

    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key)

    # PRECOMPUTE Vs for all n_slices
    Vs = precompute_Vs(M, use_rsvd=USE_RSVD, key=subkey)
    print("V dtype", Vs[0].dtype)

    # BENCHMARK METHODS

    ssim_map = {}
    psnr_map = {}
    relative_error_map = {}
    pbar = tqdm(
        METHODS_MAP.items(),
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
            n_slices_list=N_SLICES_LIST,
            save_reconstructions=SAVE_RECONSTRUCTIONS,
            method_ex=method_ex,
            n_trials=N_TRIALS,
        )
        ssim_map[method_ex] = ssim
        psnr_map[method_ex] = psnr
        relative_error_map[method_ex] = relative_error

    # SAVE METRICS
    print("Saving metrics...")
    save_metrics(
        save_path=SAVE_PATH,
        x_axis=np.asarray(N_SLICES_LIST),
        x_label="# Slices",
        ssim_map=ssim_map,
        psnr_map=psnr_map,
        relative_error_map=relative_error_map,
    )


if __name__ == "__main__":
    main()
