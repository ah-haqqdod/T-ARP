import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import chex
import equinox as eqx
import jax
from jax import numpy as jnp
from tqdm import tqdm

from t_arp.benchmark.metrics import (
    compute_psnr,
    compute_relative_error,
    compute_ssim,
)
from t_arp.benchmark.utils.kodak import KodakDataLoader


@dataclass
class KodakResult(eqx.Module):
    relative_error: chex.Array
    ssim: chex.Array
    psnr: chex.Array
    execution_time: chex.Array


@dataclass
class KodakBenchmark(eqx.Module):
    """
    Benchmarks a reconstruction blackbox function on the Kodak dataset.
    """

    # blackbox functions
    decomposition_blackbox_fn: Callable = eqx.field(static=True)
    reconstruction_blackbox_fn: Callable = eqx.field(static=True)
    data_dir: str = eqx.field(static=True)
    artifacts_blackbox_fn: Optional[Callable] = eqx.field(default=None, static=True)

    n_trials: int = eqx.field(default=1, static=True)

    def __call__(self, key: Optional[chex.PRNGKey] = None) -> Tuple[KodakResult, list]:
        dataloader = KodakDataLoader(self.data_dir)
        # pbar = tqdm(dataloader, desc=f"Benchmarking {self.method_name}")
        pbar = tqdm(dataloader, desc=f"Kodak dataset", leave=False)
        reconstructions = []
        results = []
        # loop over all images in the dataset
        for i, image in enumerate(pbar, 1):
            # NOTE: we intentionally do not split the key to let all images share the same key
            reconstruction, result = self.step(image, key=key)

            reconstructions.append(reconstruction)
            # aggregate over trials
            result = jax.tree_util.tree_map(lambda x: jnp.mean(x), result)
            results.append(result)
            pbar.set_postfix(
                {
                    "error": f"{float(result.relative_error):.4f}",
                    "last_exec_time": f"{float(result.execution_time):.4f}s",
                }
            )

        # stack results over number of data samples.
        stacked: KodakResult = jax.tree_util.tree_map(
            lambda *vals: jnp.stack(vals), *results
        )

        return stacked, reconstructions

    def _loop_over_n_trials(
        self, key: chex.PRNGKey, image: jnp.ndarray
    ) -> Tuple[jnp.ndarray]:
        """Returns an array of decompositions which are are tuples of arrays.

        First dimension is the trial index which will be used as a batch dimension."""

        # Parallel version
        keys = jax.random.split(key, self.n_trials)
        decompositions = jax.vmap(self.decomposition_blackbox_fn, in_axes=(0, None))(
            keys, image
        )
        jax.block_until_ready(decompositions)
        return decompositions

    def _step_randomized_method(
        self, key: chex.PRNGKey, image: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray], jnp.ndarray, float]:
        start_time = time.time()
        if self.n_trials > 1:
            # key, subkey = jax.random.split(key)
            decompositions = self._loop_over_n_trials(key, image)
        else:
            image_batched = jnp.expand_dims(image, 0)
            decompositions = jax.vmap(
                self.decomposition_blackbox_fn, in_axes=(None, 0), out_axes=0
            )(key, image_batched)
        jax.block_until_ready(decompositions)
        end_time = time.time()
        execution_time = (end_time - start_time) / self.n_trials

        num_args = len(decompositions)
        reconstructions = jax.vmap(
            self.reconstruction_blackbox_fn, in_axes=(0,) * num_args
        )(*decompositions)

        return decompositions, reconstructions, execution_time

    def _step_deterministic_method(
        self, image: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray], jnp.ndarray, float]:
        # simulate vectorized reconstructions to keep unified downstream logic for randomized and deterministic methods
        image_batched = jnp.expand_dims(image, 0)

        start_time = time.time()
        decompositions = jax.vmap(
            self.decomposition_blackbox_fn, in_axes=0, out_axes=0
        )(image_batched)
        jax.block_until_ready(decompositions)
        end_time = time.time()
        execution_time = end_time - start_time

        num_args = len(decompositions)
        in_axes = (0,) * num_args
        reconstructions = jax.vmap(self.reconstruction_blackbox_fn, in_axes=in_axes)(
            *decompositions
        )

        return decompositions, reconstructions, execution_time

    def step(
        self, image: jnp.ndarray, key: Optional[chex.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, KodakResult]:
        """Runs a single step of the benchmark, returning a reconstruction and Kodak results with leading batch dimension corresponding to the number of trials."""
        # NOTE: artifacts serialization logic can be added. for example, saving the decompositions to disk.
        if key is not None:
            _, reconstructions, execution_time = self._step_randomized_method(
                key, image
            )
        else:
            _, reconstructions, execution_time = self._step_deterministic_method(image)

        kodak_results = eqx.filter_vmap(
            KodakBenchmark.compute_result, in_axes=(None, 0, None), out_axes=0
        )(image, reconstructions, execution_time)

        return reconstructions[0], kodak_results

    @eqx.filter_jit
    @staticmethod
    def compute_result(
        image: jnp.ndarray, reconstruction: jnp.ndarray, execution_time: float
    ) -> KodakResult:
        """Compute the result of the benchmark, including SSIM, PSNR, tensor accuracy and execution time.

        Args:
            image: The original image.
            reconstruction: The reconstructed image.
            execution_time: The time taken to execute the benchmark.

        Returns:
            A `KodakResult` object containing the computed metrics.
        """

        ssim = compute_ssim(targets=image, predictions=reconstruction, max_val=1.0)
        psnr = compute_psnr(targets=image, predictions=reconstruction, max_val=1.0)
        relative_error = compute_relative_error(
            targets=image, predictions=reconstruction
        )

        return KodakResult(
            ssim=ssim,
            psnr=psnr,
            relative_error=relative_error,
            execution_time=jnp.asarray(execution_time),
        )
