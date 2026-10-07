# Whittaker-type Sturm–Liouville operator and half-wave transform — reproducibility code

Code accompanying the manuscript

> R. Singh, N. Wagh, *Spectral theory of a Whittaker-type Sturm–Liouville operator and an associated half-wave transform* (submitted to *Integral Transforms and Special Functions*).

It reproduces every numerical table and figure in the paper.

## Requirements
Python ≥ 3.9 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Files

| Script | Reproduces |
|---|---|
| `wlcore.py` | Core library: Gauss–Laguerre (Golub–Welsch) matrices with robust sign convention, half-wave matrix, real orthogonal (reality-preserving) matrix |
| `verify2.py` | Spectral theory checks (Table 1): characteristic function, eigenfunction norms, Weyl–Titchmarsh residues, Green's function vs spectral series, Hilbert–Schmidt norm, Parseval, operational rule, Hankel eigen-relation |
| `paper_checks.py` | β-sensitivity norm formula, θ-norm = 2, non-closure asymptotics, discretisation convergence |
| `check2.py` | Discretisation: orthogonality for N ≤ 512, geometric convergence, polynomial exactness, group law of the real orthogonal variant |
| `expt.py` | Image-mixing diagnostic (Tables 2–3, Figure 2 → `fig_encryption.pdf`) |
| `bounds.py` | Measured parameter sensitivity vs the theoretical bounds (Table 4) |
| `make_kernel_figure.py` | Figure 1 → `wl_kernels.pdf` |

## Usage
```bash
python verify2.py        # a few minutes (high-precision quadrature)
python paper_checks.py
python check2.py
python expt.py
python bounds.py
python make_kernel_figure.py
```

Default parameters: α = 0.3, β = 0.25 (λ₀ = 1.8), θ = 2.7. Test images are the public scikit-image samples (`camera`, `astronaut`, `coffee`).

## Note
The image experiment is a diagnostic of a linear mixing layer only; no cryptographic security is claimed.

## License
MIT — see `LICENSE`.
