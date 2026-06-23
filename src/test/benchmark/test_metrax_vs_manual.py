"""
decompose an image using HOSVD.
Compute the metrics using metrax and manual implementation.
Compare the outputs of both implementations.
"""

from typing import Tuple

import chex
import equinox as eqx
import jax
from jax import numpy as jnp
from metrax import PSNR, SSIM
from tqdm import tqdm

from t_arp.benchmark import (
    KodakDataLoader,
    compute_psnr,
    compute_relative_error,
    compute_ssim,
)
from t_arp.tensor import HOSVD, full_tucker


@jax.jit
def reconstruct_tucker(*hosvd_args) -> jnp.ndarray:
    return jnp.clip(full_tucker(*hosvd_args[:2]), 0.0, 1.0)


@jax.jit
def compute_metrics_metrax(images, recons) -> Tuple[chex.Array, chex.Array]:
    max_val = 1.0

    ssim = SSIM.from_model_output(
        targets=images,
        predictions=recons,
        max_val=max_val,
    ).compute()
    psnr = PSNR.from_model_output(
        targets=images,
        predictions=recons,
        max_val=max_val,
    ).compute()

    return psnr, ssim


@jax.jit
def compute_metrics_manual(images, recons) -> Tuple[chex.Array, chex.Array]:
    max_val = 1.0
    psnr = compute_psnr(images, recons, max_val)
    ssim = compute_ssim(images, recons, max_val)
    return psnr.mean(), ssim.mean()


dataloader = KodakDataLoader()
hosvd = HOSVD(kernel_shape=(50, 50, 3))
hosvd_fn = eqx.filter_jit(hosvd)

images = []
reconstructions = []
pbar = tqdm(dataloader, desc="Reconstructing")
for i, img in enumerate(pbar):
    G, factors, _ = hosvd_fn(img)
    recon = reconstruct_tucker(G, factors)

    jax.block_until_ready(recon)

    images.append(img)
    reconstructions.append(recon)

unique_shape = set([img.shape for img in images])
pairs_by_data_shape = [
    (
        jnp.stack([img for img in images if img.shape == shape]),
        jnp.stack([recon for recon in reconstructions if recon.shape == shape]),
    )
    for shape in unique_shape
]

psnrs_metrax = []
ssims_metrax = []
psnrs_manual = []
ssims_manual = []

for images, recons in pairs_by_data_shape:
    print(f"For shape {images.shape}")
    print("Computing metrics (metrax)...")
    psnr_metrax, ssim_metrax = compute_metrics_metrax(images, recons)
    jax.block_until_ready(psnr_metrax)
    jax.block_until_ready(ssim_metrax)
    psnrs_metrax.append(psnr_metrax)
    ssims_metrax.append(ssim_metrax)

    print("Computing metrics (manual)...")
    psnr_manual, ssim_manual = compute_metrics_manual(images, recons)
    jax.block_until_ready(psnr_manual)
    jax.block_until_ready(ssim_manual)
    psnrs_manual.append(psnr_manual)
    ssims_manual.append(ssim_manual)

ssims_metrax = jnp.stack(ssims_metrax)
psnrs_metrax = jnp.stack(psnrs_metrax)
ssims_manual = jnp.stack(ssims_manual)
psnrs_manual = jnp.stack(psnrs_manual)

ssim_metrax = ssims_metrax.mean()
psnr_metrax = psnrs_metrax.mean()
ssim_manual = ssims_manual.mean()
psnr_manual = psnrs_manual.mean()

print("---")

print(f"PSNR (manual): {psnr_manual:.2f}, SSIM (manual): {ssim_manual:.2f}")
print(f"PSNR (metrax): {psnr_metrax:.2f}, SSIM (metrax): {ssim_metrax:.2f}")

print()

print(
    f"PSNR (manual) - PSNR (metrax): {psnr_manual - psnr_metrax:.2f}, SSIM (manual) - SSIM (metrax): {ssim_manual - ssim_metrax:.2f}"
)
