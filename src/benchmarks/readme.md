# Experiments

All scripts are run from the **repository root**. Paths in `config.yaml` files are relative to that root. To use a custom config file, pass `--config <path>` to the `standalone_benchmark` script.

View each corresponding `config.yaml` file for more accurate description of parameters that need to be defined.

---

## Kodak dataset

Benchmarks tubal decomposition methods on the 24 Kodak PNG images.

**Run:**
```bash
python src/benchmarks/kodak/standalone_benchmark.py
```

**Config:** `src/benchmarks/kodak/config.yaml`

| Key | Default | Description |
|---|---|---|
| `data_dir` | `datasets/kodak` | Path to Kodak PNG images |
| `save_path` | `src/benchmarks/kodak/results/` | Output directory |
| `n_slices` | `[10, 20, 40, 60, 80, 100]` | Ranks to sweep |
| `n_trials` | `1` | Trials per rank (randomised methods) |
| `seed` | `42` | Global random seed |
| `n_oversamples` | `5` | RSVD oversampling |
| `n_subspace_iters` | `1` | RSVD subspace iterations |
| `save_reconstruction` | `true` | Save reconstructed PNGs |
| `active_methods` | all | List of method names to run; remove key to run all |

**Outputs** written to `save_path`:
- `metrics/` — Parquet files for SSIM, PSNR, relative error, execution time
- `reconstructions/<n_slices>/<method>/` — PNG reconstructions (if enabled)

---

## Synthetic data

Benchmarks on procedurally generated tensors; no dataset download required.

**Run:**
```bash
python src/benchmarks/synth/standalone_benchmark.py
```

**Config:** `src/benchmarks/synth/config.yaml`

| Key | Default | Description |
|---|---|---|
| `save_path` | `src/benchmarks/synth/results/` | Output directory |
| `benchmark_type` | `function_tensor` | `"function_tensor"` or `"random_tensor"` |
| `t_shape` | `[60, 60, 60]` | Tensor dimensions |
| `function_tensor_p` | `2` | Decay exponent (function tensor only) |
| `t_rank` | `7` | Rank (random tensor only) |
| `eps` | `1e-6` | Noise level (random tensor only; `null` for none) |
| `n_slices` | `[2 … 22]` | Ranks to sweep |
| `n_trials` | `1` | Samples averaged per rank |
| `seed` | `42` | Global random seed |

**Outputs** written to `save_path/<benchmark_type>/metrics/` — Parquet file for relative error.

---

## YUV video dataset

Benchmarks common-index tubal cross-approximation methods on a raw YUV video. Requires `ffmpeg` on `PATH` to convert `.yuv` files to `.mp4`.

**Run:**
```bash
python src/benchmarks/yuv/standalone_benchmark.py
```

**Config:** `src/benchmarks/yuv/config.yaml`

| Key | Default | Description |
|---|---|---|
| `video_dir` | `datasets/yuv_video/Video/` | Directory containing `.yuv` files |
| `video_name` | `tempete_cif` | Filename stem (without `.yuv`); suffix `_qcif` → 176×144, `_cif` → 352×288 |
| `save_path` | `src/benchmarks/yuv/results/` | Output directory |
| `n_slices` | `[50]` | Ranks to sweep |
| `n_trials` | `1` | Trials per rank |
| `seed` | `42` | Global random seed |
| `use_rsvd` | `true` | Use randomised SVD for basis computation |
| `save_reconstructions` | `true` | Save `.yuv` and `.mp4` reconstructions |

**Outputs** written to `save_path`:
- `metrics/` — Parquet files for SSIM, PSNR, relative error
- `<method>/rec_<n_slices>.yuv` and `.mp4` reconstructions (if enabled)

---

## Post-processing

### Plot figures

```bash
python src/benchmarks/plot_benchmark.py --save_dir <results_dir>
```

Reads `<results_dir>/metrics/` and writes plots to `<results_dir>/figures/`.

### Generate tables (CSV + LaTeX)

```bash
python src/benchmarks/aggregate_tables.py --load-dir <results_dir>/metrics/
```

Produces `*_summary.csv` and `*_table.tex` files in the same directory. Override output locations with `--save-dir-csv` and `--save-dir-latex`.
