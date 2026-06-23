from functools import partial
from typing import Callable, Literal, Optional, Sequence, Tuple, Union

import chex
import equinox as eqx
import jax
import numpy as np
import tqdm
from jax import numpy as jnp

from t_arp.benchmark import compute_relative_error, save_metrics
from t_arp.matrix import RSVD, ARP_params, CSS_module, CSS_module_factory
from t_arp.tubal import TARP, TCSSBaselines, TMatrix, TMatrixTOnly, t_cross, t_cur

BENCHMARK_TYPE: Literal["random_tensor", "function_tensor"] = "random_tensor"

SAVE_PATH = "src/benchmarks/synth_tubal/results/"
N_TRIALS = 1  # number of trials of randomized methods to collect statistics
SAVE_RECONSTRUCTION = True


METHODS_MAP = {
    "ARP T-CUR": "arp_t_cur",
    "T-SVD": "t_svd",
    "T Uniform Sampling": "uniform",
    "T Leverage Scores Sampling": "leverage_scores",
    "T-ARP": "arp",
    "T-ARP (Householder)": "arp_householder",
}

N_SLICES = tuple([i for i in range(2, 23)])
SEED = 42
N_OVERSAMPLES = 5
N_SUBSPACE_ITERS = 1

T_SHAPE = (60, 60, 60)

FUNCTION_TENSOR_P = 2

T_RANK = 7
EPS = 1e-6


@jax.jit(static_argnums=(1, 2, 3))
def construct_random_tmatrix(
    key: chex.PRNGKey,
    t_rank: int = T_RANK,
    t_shape: Tuple[int, ...] = T_SHAPE,
    eps: Optional[float] = EPS,
) -> TMatrix:
    assert len(t_shape) > 2

    key_u, key_v, key_eps = jax.random.split(key, 3)

    u_shape = (t_shape[0], t_rank, *t_shape[2:])
    U = jax.random.uniform(key_u, u_shape)

    v_shape = (t_rank, *t_shape[1:])
    V = jax.random.uniform(key_v, v_shape)

    if eps:
        noise = eps * jax.random.normal(key_eps, t_shape)
        M = TMatrix(U) @ TMatrix(V) + noise
    else:
        M = TMatrix(U) @ TMatrix(V)

    return M


def construct_function_tmatrix(
    dummy_key: Optional[chex.PRNGKey] = None,
    t_shape: Tuple[int, ...] = T_SHAPE,
    p: float = FUNCTION_TENSOR_P,
) -> TMatrix:
    assert len(t_shape) > 2

    denom = jnp.zeros(t_shape)
    for i in range(len(t_shape)):
        vec = jnp.arange(t_shape[i]) + 1
        shape = [t_shape[j] if j == i else 1 for j in range(len(t_shape))]
        vec = vec.reshape(tuple(shape))
        denom = denom + vec

    denom = denom
    denom = denom ** (1 / p)

    M = TMatrix(1 / denom)

    return M


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
    # recon = jnp.clip(recon.data, 0.0, 1.0)
    return recon.data


@jax.jit
def reconstruct_t_cross(C: TMatrix, W: TMatrix, R: TMatrix) -> jnp.ndarray:
    recon = (C @ W @ R).create_t_matrix()
    # recon = jnp.clip(recon.data, 0.0, 1.0)
    return recon.data


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
        tcss_module_constructor = partial(TARP, method=method, use_derandomized=True)

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


def benchmark_synth(
    key: chex.PRNGKey,
    decomposition_blackbox_fn: Callable,
    reconstruction_blackbox_fn: Callable,
    is_deterministic: bool = False,
    n_samples: int = N_TRIALS,
) -> chex.Array:
    keys_gen = jax.random.split(key, n_samples)

    benchmark_fn = (
        construct_function_tmatrix
        if BENCHMARK_TYPE == "function_tensor"
        else construct_random_tmatrix
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
        subkey = jax.random.fold_in(key, 1234)
        keys_decomp = jax.random.split(subkey, n_samples)
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
    n_slices_list: Sequence[int] = N_SLICES,
) -> np.ndarray:
    relative_errors = []

    pbar = tqdm.tqdm(n_slices_list, desc="", leave=False)
    for n_slices in pbar:
        relative_error_batch = benchmark_synth(
            key=key,
            decomposition_blackbox_fn=eqx.filter_jit(lambda M: t_svd(M, n_slices)),
            reconstruction_blackbox_fn=reconstruct_t_svd,
            is_deterministic=True,
        )

        relative_error = np.mean(relative_error_batch)
        relative_errors.append(relative_error)
        pbar.set_postfix({"relative_error": relative_error, "n_slices": n_slices})

    relative_errors = np.stack(relative_errors)

    return relative_errors


def t_cur_pass(
    key: chex.PRNGKey, n_slices_list: Sequence[int] = N_SLICES, method="arp"
) -> np.ndarray:
    relative_errors = []

    arp_constr = CSS_module_factory(method=method)

    pbar = tqdm.tqdm(n_slices_list, desc="", leave=False)
    for n_slices in pbar:
        relative_error_batch = benchmark_synth(
            key=key,
            decomposition_blackbox_fn=eqx.filter_jit(
                lambda M: (
                    arp_params := ARP_params(rsvd_r=n_slices),
                    arp := arp_constr(r=n_slices, css_params=arp_params),
                    CWR_tuple := t_cur(M, css_method=arp, key=key),
                )[-1]
            ),
            reconstruction_blackbox_fn=reconstruct_t_cross,
            is_deterministic=True,
        )

        relative_error = np.mean(relative_error_batch)
        relative_errors.append(relative_error)
        pbar.set_postfix({"relative_error": relative_error, "n_slices": n_slices})

    relative_errors = np.stack(relative_errors)

    return relative_errors


def t_css_pass(
    key,
    method,
    n_slices_list: Sequence[int] = N_SLICES,
    is_deterministic: bool = False,
) -> np.ndarray:
    relative_errors = []
    # execution_times = []

    pbar = tqdm.tqdm(n_slices_list, desc="", leave=False)
    for n_slices in pbar:
        relative_error_batch = benchmark_synth(
            key,
            decomposition_blackbox_fn=eqx.filter_jit(
                lambda key, M: (
                    keys := jax.random.split(key),
                    V := t_rsvd(key=keys[0], M=M, n_slices=n_slices),
                    t_cross_helper(
                        key=keys[1], M=M, V=V, method=method, n_slices=n_slices
                    ),
                )[-1]
            ),
            reconstruction_blackbox_fn=reconstruct_t_cross,
            is_deterministic=is_deterministic,
        )
        relative_error = np.mean(relative_error_batch)

        relative_errors.append(relative_error)
        # execution_times.append(result.execution_time)

        pbar.set_postfix({"relative_error": relative_error, "n_slices": n_slices})

    relative_errors = np.stack(relative_errors)
    # execution_times = np.stack(execution_times)

    return relative_errors


def main():
    master_key = jax.random.PRNGKey(jnp.array(SEED))

    relative_errors_map = {}

    pbar = tqdm.tqdm(METHODS_MAP.items(), desc="Benchmarking")
    for method_ex, method in pbar:
        pbar.set_postfix(method=method_ex)

        if method == "t_svd":
            relative_errors = t_svd_pass(
                key=master_key,
            )
            relative_errors_map[method_ex] = relative_errors
            continue

        if method == "arp_t_cur":
            relative_errors = t_cur_pass(key=master_key, method="arp")
            relative_errors_map[method_ex] = relative_errors
            continue

        if method == "arp_householder":
            method = "householder"
        elif method == "arp_normalized":
            method = "orth_proj_normalized"
        elif method == "arp":
            method = "orth_proj_pinv"

        relative_errors = t_css_pass(
            key=master_key,
            method=method,
        )
        relative_errors_map[method_ex] = relative_errors

    save_metrics(
        save_path=SAVE_PATH,
        x_axis=np.asarray(N_SLICES),
        x_label="# Slices",
        relative_error_map=relative_errors_map,
    )


if __name__ == "__main__":
    main()
