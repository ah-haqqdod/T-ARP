# Adaptive Randomized Pivoting for Tensor Singular Value Decomposition Model

## installation

Step 1: Download python

Step 2: Create a virtual environment

Step 3: Download the repository using git:

```bash
git clone --single-branch --branch main https://github.com/ah-haqqdod/T-ARP.git
```

Step 4: Install the library using pip:

```bash
pip install -e src/
```

The installation will also install the required dependencies; python >3.11 is recommended.

## T-Matrix structure

The T-Matrix module (`src/t_arp/tubal/t_matrix.py`) is used to define the interface for working with tensor structures subject to `t-product`. The name `T-Matrix` stands for `Tubal Matrix`, which has a matrix-like structure, but each element is an array, e.g., a vector representing RGB values of a pixel.

...

## Numeric evaluations

The numeric evaluations and empirical results are presented in the `experiments` branch, under `src/benchmarks`.
