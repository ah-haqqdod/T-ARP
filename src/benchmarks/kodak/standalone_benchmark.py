import os
from functools import partial
from typing import Literal, Sequence, Tuple, Union

import chex
import equinox as eqx
import jax
import numpy as np
import tqdm
from jax import numpy as jnp
from PIL import Image

from t_arp.benchmark import KodakBenchmark, save_metrics
from t_arp.matrix import RSVD, ARP_params, CSS_module_factory
from t_arp.tubal import (
    TARP,
    TCSSBaselines,
    TMatrix,
    TMatrixTOnly,
    t_cross,
    t_cur,
)

# NOTE: benchmark loop
# 1. iterate over methods
# 2. iterate over n_slices_list
# 3. iterate over seeds
# The seed iteration is on third layer to minimize the number of jit compilations

SAVE_PATH = "src/benchmarks/kodak/results/"
N_TRIALS = 1  # number of trials of randomized methods to collect statistics
SAVE_RECONSTRUCTION = True

METHODS_MAP = {
    "ARP T-CUR": "arp_t_cur",
    "T-SVD": "t_svd",
    "T Uniform Sampling": "uniform",
    "T Lengths Squared Sampling": "length_squared",
    "T Leverage Scores Sampling": "leverage_scores",
    "T-ARP": "arp",
    # "ARP (normalized)": "arp_normalized",
    # "Fast ARP": "fast_arp",
    "T-ARP (Householder)": "arp_householder",
}

# N_SLICES = (10, 25, 50, 100, 150)
# N_SLICES = (10, 20, 40, 80)
N_SLICES = (10, 20, 40, 60, 80, 100)
# N_SLICES = (10, 100)
SEED = 42
N_OVERSAMPLES = 5
N_SUBSPACE_ITERS = 1

# Note: log of hyperparams was removed.


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


@jax.jit(static_argnames=["n_slices"])
def t_svd(
    M: TMatrix, n_slices: int
) -> Tuple[
    Union[TMatrix, TMatrixTOnly],
    Union[TMatrix, TMatrixTOnly],
    Union[TMatrix, TMatrixTOnly],
]:
    U, S, Vt = M.facewise_operation(lambda A: jnp.linalg.svd(A, full_matrices=False))

    # Truncate to n_slices
    indices = jnp.arange(n_slices)
    U = TMatrix(jnp.take(U.data, indices=indices, axis=1))
    S = TMatrix(jnp.take(S.data, indices=indices, axis=0))
    Vt = TMatrix(jnp.take(Vt.data, indices=indices, axis=0))

    return U, S, Vt


@jax.jit(static_argnames=["n_slices"])
def t_rsvd(key: chex.PRNGKey, M: TMatrix, n_slices: int) -> TMatrix:
    rsvd = RSVD(
        rank=min(n_slices, M.shape[0], M.shape[1]),
        n_oversamples=N_OVERSAMPLES,
        n_subspace_iters=N_SUBSPACE_ITERS,
    )

    # rsvd_fn = lambda A: rsvd(key=key, A=A)
    def rsvd_fn(key: chex.PRNGKey, A: chex.Array):
        return rsvd(key=key, A=A)

    # _, _, t_Vt = M.facewise_operation(rsvd_fn)
    _, _, t_Vt = M.facewise_operation(rsvd_fn, key=key)

    V = t_Vt.T.create_t_matrix()
    return V


@jax.jit
def reconstruct_t_svd(U: TMatrix, S: TMatrix, Vt: TMatrix) -> jnp.ndarray:
    # print("S shape before:", S.shape)
    S = S.facewise_operation(lambda x: jnp.diag(x.flatten()))
    # print("S shape after:", S.shape)
    recon = (U @ S @ Vt).create_t_matrix()
    recon = jnp.clip(recon.data, 0.0, 1.0)
    return recon


@jax.jit
def reconstruct_t_cross(C: TMatrix, W: TMatrix, R: TMatrix) -> jnp.ndarray:
    recon = (C @ W @ R).create_t_matrix()
    recon = jnp.clip(recon.data, 0.0, 1.0)
    return recon.astype(jnp.float32)


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
        tcss_module_constructor = partial(TARP, method=method, use_derandomized=True)

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
    n_slices_list: Sequence[int] = N_SLICES,
    method_name: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    relative_errors = []
    ssims = []
    psnrs = []
    execution_times = []

    for n_slices in n_slices_list:
        # define benchmark
        benchmark = KodakBenchmark(
            decomposition_blackbox_fn=eqx.filter_jit(
                lambda image: (M := TMatrix(image), t_svd(M, n_slices))[-1]
            ),
            reconstruction_blackbox_fn=reconstruct_t_svd,
        )

        result, reconstructions = benchmark()
        # log reconstructions
        if SAVE_RECONSTRUCTION:
            save_reconstructions(
                method_name=method_name,
                n_slices=n_slices,
                reconstructions=reconstructions,
                dir_path=SAVE_PATH,
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
    method="arp",
    n_slices_list: Sequence[int] = N_SLICES,
    n_trials: int = N_TRIALS,
    method_name: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    relative_errors = []
    ssims = []
    psnrs = []
    execution_times = []

    arp_constr = CSS_module_factory(method=method)

    pbar = tqdm.tqdm(n_slices_list, desc="", leave=False)
    for n_slices in pbar:
        benchmark = KodakBenchmark(
            n_trials=n_trials,
            decomposition_blackbox_fn=eqx.filter_jit(
                lambda key, image: (
                    M := TMatrix(image.astype(jnp.float32)),
                    arp_params := ARP_params(rsvd_r=n_slices),
                    arp := arp_constr(r=n_slices, css_params=arp_params),
                    CWR_tuple := t_cur(M, css_method=arp, key=key),
                )[-1]
            ),
            reconstruction_blackbox_fn=reconstruct_t_cross,
        )

        result, reconstructions = benchmark(key)
        # log reconstructions
        if SAVE_RECONSTRUCTION:
            save_reconstructions(
                method_name=method_name,
                n_slices=n_slices,
                reconstructions=reconstructions,
                dir_path=SAVE_PATH,
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
    n_slices_list: Sequence[int] = N_SLICES,
    n_trials: int = N_TRIALS,
    method_name: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    relative_errors = []
    ssims = []
    psnrs = []
    execution_times = []

    for n_slices in n_slices_list:
        # define benchmark
        benchmark = KodakBenchmark(
            n_trials=n_trials,
            decomposition_blackbox_fn=eqx.filter_jit(
                lambda key, image: (
                    M := TMatrix(image.astype(jnp.float32)),
                    V := t_rsvd(key=key, M=M, n_slices=n_slices),
                    keys := jax.random.split(key),
                    t_cross_helper(
                        key=keys[1], M=M, V=V, method=method, n_slices=n_slices
                    ),
                )[-1]
            ),
            reconstruction_blackbox_fn=reconstruct_t_cross,
        )

        result, reconstructions = benchmark(key)
        # log reconstructions
        if SAVE_RECONSTRUCTION:
            save_reconstructions(
                method_name=method_name,
                n_slices=n_slices,
                reconstructions=reconstructions,
                dir_path=SAVE_PATH,
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


def main():
    master_key = jax.random.PRNGKey(SEED)

    ssims_map = {}
    psnrs_map = {}
    relative_errors_map = {}
    execution_times_map = {}

    pbar = tqdm.tqdm(METHODS_MAP.items(), desc="Benchmarking")
    for method_ex, method in pbar:
        pbar.set_postfix(method=method_ex)

        if method == "t_svd":
            ssims, psnrs, relative_errors, execution_times = t_svd_pass(
                method_name=method_ex,
            )
            ssims_map[method_ex] = ssims
            psnrs_map[method_ex] = psnrs
            relative_errors_map[method_ex] = relative_errors
            execution_times_map[method_ex] = execution_times
            # pbar.update(1)
            continue

        if method == "arp_t_cur":
            ssims, psnrs, relative_errors, execution_times = t_cur_pass(
                key=master_key,
                method="arp",
                method_name=method_ex,
            )
            ssims_map[method_ex] = ssims
            psnrs_map[method_ex] = psnrs
            relative_errors_map[method_ex] = relative_errors
            execution_times_map[method_ex] = execution_times
            # pbar.update(1)
            continue

        if method == "arp_householder":
            method = "householder"
        elif method == "arp_normalized":
            method = "orth_proj_normalized"
        elif method == "arp":
            method = "orth_proj_pinv"

        ssims, psnrs, relative_errors, execution_times = t_css_pass(
            key=master_key,
            method=method,
            method_name=method_ex,
        )
        ssims_map[method_ex] = ssims
        psnrs_map[method_ex] = psnrs
        relative_errors_map[method_ex] = relative_errors
        execution_times_map[method_ex] = execution_times
        # pbar.update(1)

    save_metrics(
        save_path=SAVE_PATH,
        x_axis=np.asarray(N_SLICES),
        x_label="# Slices",
        ssim_map=ssims_map,
        psnr_map=psnrs_map,
        relative_error_map=relative_errors_map,
        execution_time_map=execution_times_map,
    )


if __name__ == "__main__":
    main()
