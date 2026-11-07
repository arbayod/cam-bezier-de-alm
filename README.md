# Multi-parameter Bézier optimization of cam mechanisms with negative radius follower

Python code for the paper

> A. Revuelta-Bayod, *Multi-parameter Bézier optimization of cam mechanisms with negative radius follower using Augmented Lagrangian Differential Evolution*, 2026.
> SSRN: <https://ssrn.com/abstract=7275640> · DOI: [10.2139/ssrn.7275640](http://dx.doi.org/10.2139/ssrn.7275640)

The paper extends the single-parameter Bézier design of Hidalgo-Martínez et al. (2014) by freeing two Bézier ordinates. It then maximizes the follower lift with Differential Evolution (DE), an Augmented Lagrangian Method (ALM) for the constraints and a short local search. The constraints are no undercutting (ρ/|R<sub>f</sub>| ≤ 1) and a pressure angle φ ≤ 41°.

## Contents

| File | Purpose |
|---|---|
| `camBezierOptimization_DE_ALM.py` | Main optimizer. Sweeps α = R<sub>a</sub>/\|R<sub>f</sub>\| and returns the maximum lift *h* and the optimal ordinates (*b*<sub>lo</sub>, *b*<sub>hi</sub>). Reproduces Tables 2 and 3 of the paper. |
| `cam_profile.py` | Draws the cam profile of one design (*n*, β, α, *h*, *b*<sub>lo</sub>, *b*<sub>hi</sub>), checks its constraints and writes the profile coordinates to a CSV file. |

Neither script takes command-line options. All inputs sit in a block marked `USER INPUTS` at the top of each file.

## Requirements

- Python ≥ 3.9
- NumPy, SciPy, matplotlib

```bash
pip install numpy scipy matplotlib
```

Tested on:

| OS | NumPy | SciPy |
|---|---|---|
| Windows 11 | 1.26.4 | 1.16.0 |
| Linux | 2.5.3 | 1.18.1 |

## 1. Optimizer: `camBezierOptimization_DE_ALM.py`

### Run

```bash
python camBezierOptimization_DE_ALM.py
```

With the default inputs the script reproduces Table 3 (*n* = 8, two free ordinates, α = 0.2 … 0.9). The full sweep takes a few seconds per value of α on a desktop PC. DE uses all CPU cores.

### Main inputs (`USER INPUTS` block)

| Variable | Default | Meaning |
|---|---|---|
| `N_ORDER` | 8 | Bézier order *n* (6, 7, 8 or 9) |
| `N_VARS` | 3 | 3 = two free ordinates (this work); 2 = one free ordinate (HSB benchmark, *n* = 6 or 8 only) |
| `BETA_DEG` | 180 | Rise angle β [deg]; the return is symmetric |
| `PHI_MAX_DEG` | 41 | Pressure-angle limit [deg] |
| `ALPHAS` | 0.2 … 0.9 | Values of α = R<sub>a</sub>/\|R<sub>f</sub>\| to optimize |
| `B_MIN`, `B_MAX` | 0.50, 1.00 | Bounds of the free ordinates |
| `NP`, `N_OUTER`, `G_MAX` | 30, 15, 88 | DE population, ALM outer iterations, DE generations per outer iteration |
| `BASE_SEED` | 42 | Random seed (see *Reproducibility*) |

The algorithm parameters of Table 1 of the paper (DE strategy, ALM penalties and damping, local search) sit in a separate block below the user inputs. You do not need to change them to reproduce the paper.

Useful settings:

- **Table 2** (*n* = 6): `N_ORDER = 6`
- **HSB single-parameter benchmark**: `N_VARS = 2`

### Output (`./output/`)

| File | Content |
|---|---|
| `cam_results_n{n}_v{N_VARS}_{timestamp}.csv` | One row per α: `alpha, b_lo, b_hi, h, phi_max_deg, rho_max, feasible` |
| `cam_log_n{n}_v{N_VARS}_{timestamp}.dat` | Full log of every ALM outer iteration |
| `cam_plots_n{n}_v{N_VARS}_{timestamp}.png` | Motion law, pressure angle and curvature of the last α, plus *h*(α) |

Expected result with the default inputs (*n* = 8, β = 180°):

| α | *b*<sub>lo</sub> | *b*<sub>hi</sub> | *h* |
|---|---|---|---|
| 0.2 | 0.9886 | 0.5965 | 0.160970 |
| 0.3 | 0.9725 | 0.6058 | 0.240774 |
| 0.4 | 0.9365 | 0.6294 | 0.319533 |
| 0.5 | 0.9716 | 0.6281 | 0.398711 |
| 0.6 | 0.9925 | 0.6241 | 0.476970 |
| 0.7 | 0.9994 | 0.6361 | 0.554188 |
| 0.8 | 0.9961 | 0.6465 | 0.630847 |
| 0.9 | 0.9898 | 0.6572 | 0.707229 |

All designs are feasible, with both constraints active at the optimum (φ<sub>max</sub> ≈ 40.9°).

### Reproducibility

- **Fixed seeds.** Every run is seeded, so the same machine and the same library versions give identical results.
- **Seed depends on list position.** The *k*-th value in `ALPHAS` (counting from 0) uses seed `BASE_SEED + k`. To reproduce one entry of the table on its own, adjust the seed. For example, α = 0.9 alone needs `ALPHAS = [0.9]` and `BASE_SEED = 49`.
- **Library versions.** Different NumPy/SciPy versions can change the last digits of *h*, typically by about 10<sup>−4</sup>, and the ordinates by more, because the optimum is flat in *b*<sub>lo</sub>. The gains reported in the paper are not affected.
- **Windows, macOS and Linux behave the same.** The ALM multipliers are passed explicitly to every parallel DE worker, so the result does not depend on how the operating system starts processes.

## 2. Cam profile: `cam_profile.py`

### Run

```bash
python cam_profile.py
```

Copy *h*, *b*<sub>lo</sub> and *b*<sub>hi</sub> from the optimizer CSV, or from Tables 2–3 of the paper, into the `USER INPUTS` block. The default is the Table 3 design for *n* = 8, α = 0.6. For the single-parameter HSB design set `B_LO = 0.0`.

| Variable | Meaning |
|---|---|
| `N_ORDER`, `BETA_DEG` | Bézier order and rise angle |
| `ALPHA`, `H` | α = R<sub>a</sub>/\|R<sub>f</sub>\| and lift *h* = *h*\*/\|R<sub>f</sub>\| |
| `B_LO`, `B_HI` | Free Bézier ordinates |
| `N_POINTS` | Points along one revolution (1441 → 0.25° step) |
| `LENGTH_SCALE` | Factor applied to the coordinates. Set it to \|R<sub>f</sub>\| in mm to get mm. |

Before drawing, the script checks the design against the same constraints as the optimizer and prints the maximum pressure angle and curvature ratio. If the design is infeasible it prints a warning and still writes the profile.

### Output (`./output/`)

- `cam_profile_n{n}_a{alpha}.png` / `.pdf`: the cam profile, the follower-centre curve and the prime circle.
- `cam_profile_n{n}_a{alpha}.csv`: the coordinates of the profile.

The CSV starts with six comment lines (`#`) listing the inputs and the design check. Then come the columns:

```
theta_deg,x,y,x_C,y_C
```

- `theta_deg` is the cam rotation angle in degrees.
- `x, y` is the cam profile.
- `x_C, y_C` is the follower-centre (pitch) curve.

Both curves are given in the cam frame, with the follower centre *C* on the +*x* axis at θ = 0. The last row repeats the first, so the curve is closed. Lengths are divided by |R<sub>f</sub>| unless `LENGTH_SCALE` is set.

Reading the file:

```python
import pandas as pd
df = pd.read_csv("output/cam_profile_n8_a0.60.csv", comment="#")

import numpy as np
data = np.loadtxt("output/cam_profile_n8_a0.60.csv", delimiter=",", skiprows=7)
```

## Notation and conventions

- All lengths are dimensionless, divided by the follower radius |R<sub>f</sub>|. In particular α = R<sub>a</sub>/|R<sub>f</sub>| and *h* = *h*\*/|R<sub>f</sub>|.
- The Bézier ordinate vectors are:

  | *n* | ordinates |
  |---|---|
  | 6 | [0, 0, *b*<sub>lo</sub>, *b*<sub>hi</sub>, 1, 1, 1] |
  | 7 | [0, 0, 0, *b*<sub>lo</sub>, *b*<sub>hi</sub>, 1, 1, 1] |
  | 8 | [0, 0, 0, *b*<sub>lo</sub>, *b*<sub>hi</sub>, 1, 1, 1, 1] |
  | 9 | [0, 0, 0, 0, *b*<sub>lo</sub>, *b*<sub>hi</sub>, 1, 1, 1, 1] |

  With *b*<sub>lo</sub> = 0, *n* = 6 and *n* = 8 recover the single-parameter vectors of Hidalgo-Martínez et al., whose *b*<sub>3</sub> and *b*<sub>4</sub> correspond to *b*<sub>hi</sub>.
- The paper writes the design vector as *x* = (*h*, *b*<sub>lo</sub>, *b*<sub>hi</sub>). Internally the optimizer stores it as `[h, b_hi, b_lo]`. The order is kept because changing it would change the random sequence of DE. All output files list (*b*<sub>lo</sub>, *b*<sub>hi</sub>).

## Citation

If you use this code, please cite:

```bibtex
@misc{RevueltaBayod2026cam,
  author       = {Revuelta-Bayod, Antonio},
  title        = {Multi-parameter {B}{\'e}zier optimization of cam mechanisms with
                  negative radius follower using {A}ugmented {L}agrangian
                  {D}ifferential {E}volution},
  year         = {2026},
  howpublished = {SSRN},
  doi          = {10.2139/ssrn.7275640},
  url          = {https://ssrn.com/abstract=7275640}
}
```

The benchmark single-parameter method is:

> M. Hidalgo-Martínez, E. Sanmiguel-Rojas, M.A. Burgos, *Design of cams with negative radius follower using Bézier curves*, Mechanism and Machine Theory 82 (2014) 87–96. DOI: [10.1016/j.mechmachtheory.2014.08.001](https://doi.org/10.1016/j.mechmachtheory.2014.08.001)

## License

See [`LICENSE`](LICENSE).
