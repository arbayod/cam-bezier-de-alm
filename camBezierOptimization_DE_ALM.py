# -*- coding: utf-8 -*-
"""
camBezierOptimization_DE_ALM.py
===============================================================================
Optimal Bezier motion laws for cam mechanisms with a negative-radius follower,
solved with Differential Evolution (DE) + Augmented Lagrangian Method (ALM)
+ periodic Local Search (LS).

This script reproduces the multi-parameter results of the manuscript
(Table 3: n = 8, beta = 180 deg; Table 2 with N_ORDER = 6):

    A. Revuelta-Bayod, "Multi-parameter Bezier optimization of cam mechanisms
    with negative radius follower using Augmented Lagrangian Differential
    Evolution", 2026. SSRN: https://ssrn.com/abstract=7275640
    doi:10.2139/ssrn.7275640

Benchmark (single-parameter design) that this work extends:

    J.A. Hidalgo-Martinez et al. (2014), "Design of cams with negative
    radius follower using Bezier curves", Mechanism and Machine Theory,
    82, 87-96.  (Referred to below as "HSB".)

-------------------------------------------------------------------------------
QUICK START
-------------------------------------------------------------------------------
    1. Requirements: Python >= 3.9, NumPy, SciPy (matplotlib optional, plots).
           pip install numpy scipy matplotlib
    2. Edit the block "USER INPUTS" below (there are no command-line options).
    3. Run:
           python camBezierOptimization_DE_ALM.py
    4. Results are written to ./output/ (next to this script):
           cam_results_n{n}_v{N_VARS}_{timestamp}.csv   -> results table
           cam_log_n{n}_v{N_VARS}_{timestamp}.dat       -> full run log
           cam_plots_n{n}_v{N_VARS}_{timestamp}.png     -> diagnostic plots

    Default inputs reproduce Table 3 of the manuscript (n = 8, two free
    ordinates, alpha = 0.2 ... 0.9). Set N_ORDER = 6 for Table 2, or
    N_VARS = 2 to reproduce the single-parameter HSB benchmark.

-------------------------------------------------------------------------------
PROBLEM (manuscript Section 2, Eqs. 12-15), all lengths divided by |Rf|
-------------------------------------------------------------------------------
    maximize   h                                  (lift, h = h*/|Rf|)
    subject to g1 = max_theta rho(theta)/|Rf| - 1 <= 0   (no undercutting)
               g2 = max_theta phi(theta) - phi_max <= 0  (pressure angle)
               box bounds on h and on the free Bezier ordinates.

    alpha = Ra/|Rf| is the dimensionless prime-circle radius (a fixed input
    of each run, swept over ALPHAS).

-------------------------------------------------------------------------------
DESIGN VECTOR -- note on ordering
-------------------------------------------------------------------------------
    The manuscript writes x = (h, b_lo, b_hi) (Eq. 16). The code stores
        N_VARS = 2 :  x = [h, b_hi]           (b_lo fixed at 0 -> HSB)
        N_VARS = 3 :  x = [h, b_hi, b_lo]
    i.e. b_hi before b_lo. The order is kept on purpose: DE draws its random
    numbers per vector component, so swapping the order would change the
    random sequence and therefore the last digits of the published results.
    Output tables and CSV files always list (b_lo, b_hi), as in the paper.

-------------------------------------------------------------------------------
REPRODUCIBILITY
-------------------------------------------------------------------------------
    * Every run is seeded (BASE_SEED); repeated runs on the same machine
      and library versions give identical results.
    * The ALM multipliers/penalties are passed explicitly to the objective
      function, so the parallel DE workers always see the current values,
      whatever the multiprocessing start method of the platform
      (fork on Linux, spawn on Windows/macOS).
    * Different NumPy/SciPy versions may change the last digits of h
      (typically O(1e-4)); the reported trends and gains are unaffected.

Author : Antonio Revuelta-Bayod
Version: 1.0 (2026-11)   -- public release of the research code v06
License: see LICENSE file in the repository.
===============================================================================
"""

import os

# DE parallelism is at process level (one evaluation per worker). Limiting
# each process to one BLAS/OpenMP thread avoids CPU over-subscription.
# Must be set BEFORE NumPy is imported to have any effect.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import datetime
import multiprocessing
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import comb


# =============================================================================
# USER INPUTS  --  edit this block to set up a run
# =============================================================================

# --- Cam design problem ------------------------------------------------------
N_ORDER     = 8                  # Bezier order n: 6, 7, 8 or 9
N_VARS      = 3                  # 2 = single free ordinate (HSB benchmark,
                                 #     only n = 6 or 8)
                                 # 3 = two free ordinates (present work)
BETA_DEG    = 180.0              # rise angle beta [deg] (return is symmetric)
PHI_MAX_DEG = 41.0               # pressure-angle limit phi_max [deg]
ALPHAS      = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # alpha = Ra/|Rf|

# --- Search bounds -----------------------------------------------------------
H_BOUNDS    = (0.01, 1.0)        # bounds on the lift h = h*/|Rf|
B_MIN       = 0.50               # lower bound of the free ordinates
B_MAX       = 1.00               # upper bound of the free ordinates

# --- Evaluation grid ---------------------------------------------------------
N_THETA     = 360                # samples per revolution (1 deg resolution)

# --- Optimizer budget (manuscript Table 1) -----------------------------------
NP          = 30                 # DE population size (total, all variables)
N_OUTER     = 15                 # ALM outer iterations (N_outer)
G_MAX       = 88                 # DE generations per outer iteration (G_max)
BASE_SEED   = 42                 # run k of the alpha sweep uses BASE_SEED + k
                                 # (k = position in ALPHAS, starting at 0).
                                 # NOTE: the seed depends on the POSITION of
                                 # alpha in the list, not on its value. To
                                 # reproduce one entry of Table 2/3 alone, keep
                                 # the full list or set BASE_SEED = 42 + k
                                 # (e.g. ALPHAS = [0.9], BASE_SEED = 49).

# --- Output ------------------------------------------------------------------
MAKE_PLOTS  = True               # diagnostic plots of the last alpha
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# =============================================================================
# ALGORITHM PARAMETERS  (manuscript Table 1 -- change only if you know why)
# =============================================================================

# Differential Evolution
DE_STRATEGY      = "randtobest1bin"   # rand-to-best/1/bin, Eq. (17)
DE_MUTATION      = (0.3, 1.0)         # F ~ U(0.3, 1.0), dithered per generation
DE_RECOMBINATION = 0.9                # crossover rate CR, Eq. (18)

# Augmented Lagrangian
ALM_RHO_0        = 10.0               # initial penalty parameters rho_k
ALM_RHO_MAX      = 1e6                # cap on rho_k
ALM_DAMPING      = 0.8                # alpha_d in the multiplier update, Eq. (22)
ALM_ETA_0        = 0.25               # initial target reduction ratio eta
ALM_ETA_RANGE    = (0.1, 0.5)         # limits for eta (adaptive roll-up)
ALM_BETA_0       = 10.0               # initial penalty growth factor beta_ALM
ALM_BETA_RANGE   = (5.0, 50.0)        # limits for beta_ALM
MAX_CONSECUTIVE_BAD = 3               # diverging iterations before roll-back
ROLLBACK_FACTOR  = 2.0                # "diverging" = worst violation x2

# Local search (manuscript Section 3.4, Eq. 25)
LS_FREQUENCY     = 1                  # run LS every LS_FREQUENCY outer iterations
LS_N_TRIALS      = 30                 # Gaussian trials per LS call (N_ls)
LS_SIGMA_RATIO   = 0.066              # sigma_i = sigma_r * (x_i^U - x_i^L)

# Convergence
H_TOL            = 1e-4               # stop when best feasible h changes < H_TOL


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CamDesignParams:
    """
    Fixed data of one cam design problem (everything except alpha and x).

    Attributes
    ----------
    n       : Bezier order (6, 7, 8 or 9).
    beta    : rise angle [rad].
    phi_max : maximum allowable pressure angle [deg].
    n_theta : number of angular intervals per revolution.
    b_min, b_max : bounds of the free Bezier ordinates.
    """
    n: int = 8
    beta: float = np.pi
    phi_max: float = 41.0
    n_theta: int = 360
    b_min: float = 0.50
    b_max: float = 1.0


@dataclass
class AugmentedLagrangianState:
    """
    Multipliers, penalties and adaptive parameters of the ALM outer loop.

    The constrained problem is replaced by a sequence of box-constrained
    subproblems on the augmented Lagrangian (manuscript Eq. 20):

        L(x; mu, rho) = -h + sum_k psi_k(g_k(x); mu_k, rho_k),   k = 1, 2

    with g1 = curvature (no-undercutting) and g2 = pressure angle.

    An instance of this class is passed to every objective evaluation (also
    in the parallel DE workers), so it must stay picklable.
    """
    # Lagrange multipliers mu_k (start at 0, Table 1)
    mu_rho: float = 0.0          # multiplier of g1 (curvature)
    mu_phi: float = 0.0          # multiplier of g2 (pressure angle)
    # Penalty parameters rho_k
    rho_rho: float = ALM_RHO_0   # penalty of g1
    rho_phi: float = ALM_RHO_0   # penalty of g2
    # Adaptive roll-up parameters
    eta: float = ALM_ETA_0       # target ratio for violation reduction
    beta_alm: float = ALM_BETA_0 # penalty growth factor
    # History of (g1, g2) at each accepted outer iteration
    constraint_violations: List[Tuple[float, float]] = field(default_factory=list)
    outer_iterations: int = 0    # number of accepted multiplier updates

    def copy_parameters(self) -> "AugmentedLagrangianState":
        """Snapshot of mu, rho, eta, beta (used for roll-back)."""
        return AugmentedLagrangianState(
            mu_rho=self.mu_rho, mu_phi=self.mu_phi,
            rho_rho=self.rho_rho, rho_phi=self.rho_phi,
            eta=self.eta, beta_alm=self.beta_alm,
        )

    def update_multipliers(self, g1: float, g2: float) -> None:
        """
        Outer-loop update after each DE + LS run (manuscript Eqs. 22-24).

        Order of operations:
          1. adapt eta and beta_ALM from the recent violation history (Eq. 24);
          2. damped multiplier update (Eq. 22);
          3. penalty growth for constraints that did not improve enough (Eq. 23).
        """
        self.constraint_violations.append((g1, g2))

        # --- 1. Adaptive roll-up of eta and beta_ALM (Eq. 24) ---------------
        # theta_t = worst positive violation; r_red compares it with the value
        # stored three entries back in the history.
        if self.outer_iterations >= 3:
            theta_curr = max(max(0.0, g1), max(0.0, g2))
            theta_prev = max(max(0.0, self.constraint_violations[-3][0]),
                             max(0.0, self.constraint_violations[-3][1]))
            if theta_prev > 1e-10:
                r_red = (theta_curr - theta_prev) / (3.0 * theta_prev)
                if r_red > -1e-3:            # slow progress -> push harder
                    self.eta = min(ALM_ETA_RANGE[1], 1.2 * self.eta)
                    self.beta_alm = min(ALM_BETA_RANGE[1], 1.5 * self.beta_alm)
                elif r_red < -1e-1:          # fast progress -> relax
                    self.eta = max(ALM_ETA_RANGE[0], 0.8 * self.eta)
                    self.beta_alm = max(ALM_BETA_RANGE[0], 0.8 * self.beta_alm)

        # --- 2. Damped multiplier update (Eq. 22) ---------------------------
        # Active constraint:   mu <- a*mu + (1-a)*(mu + rho*g) = mu + (1-a)*rho*g
        # Inactive constraint: mu <- a*mu   (multiplier decays towards zero)
        a = ALM_DAMPING
        if g1 > 0:
            self.mu_rho = max(0.0, a * self.mu_rho
                              + (1 - a) * (self.mu_rho + self.rho_rho * g1))
        else:
            self.mu_rho = max(0.0, a * self.mu_rho)

        if g2 > 0:
            self.mu_phi = max(0.0, a * self.mu_phi
                              + (1 - a) * (self.mu_phi + self.rho_phi * g2))
        else:
            self.mu_phi = max(0.0, a * self.mu_phi)

        # --- 3. Penalty growth (Eq. 23) -------------------------------------
        # rho_k grows by beta_ALM only if g_k is still violated and has not
        # dropped below eta times its previous value.
        if self.outer_iterations > 0:
            prev_g1, prev_g2 = self.constraint_violations[-2]
            if g1 > 0 and g1 > self.eta * max(0.0, prev_g1):
                self.rho_rho = min(ALM_RHO_MAX, self.beta_alm * self.rho_rho)
            if g2 > 0 and g2 > self.eta * max(0.0, prev_g2):
                self.rho_phi = min(ALM_RHO_MAX, self.beta_alm * self.rho_phi)

        self.outer_iterations += 1

    def penalty(self, g1: float, g2: float) -> float:
        """
        Shifted penalty for inequality constraints (manuscript Eq. 21):

            psi_k = mu_k*g_k + (rho_k/2)*g_k^2   if mu_k + rho_k*g_k > 0
            psi_k = -mu_k^2 / (2*rho_k)          otherwise (constraint inactive)
        """
        return (self._psi(g1, self.mu_rho, self.rho_rho)
                + self._psi(g2, self.mu_phi, self.rho_phi))

    @staticmethod
    def _psi(g: float, mu: float, rho: float) -> float:
        if mu + rho * g > 0:
            return mu * g + 0.5 * rho * g * g
        return -0.5 * mu * mu / rho if rho > 0 else 0.0


# =============================================================================
# LOGGING
# =============================================================================

_output_buffer: List[str] = []


def log_message(message: str) -> None:
    """Print to console and keep a copy for the .dat log file."""
    print(message)
    _output_buffer.append(message)


def write_log_file(filename: str) -> None:
    """Write all logged messages to OUTPUT_DIR/filename."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        full_path = os.path.join(OUTPUT_DIR, filename)
        with open(full_path, "w", encoding="utf-8") as fh:
            fh.write("Cam Optimization Results (DE + ALM + LS) -- "
                     f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
            fh.write("-" * 80 + "\n\n")
            fh.write("\n".join(_output_buffer) + "\n")
        log_message(f"Log written to {full_path}")
    except OSError as exc:
        print(f"Error writing log file: {exc}")


# =============================================================================
# BEZIER CURVE EVALUATION  (manuscript Eqs. 7-9)
# =============================================================================

def bernstein_poly(i: int, n: int, u: np.ndarray) -> np.ndarray:
    """Bernstein basis polynomial B_i^n(u) = C(n,i) u^i (1-u)^(n-i), Eq. (8)."""
    return comb(n, i, exact=True) * (u ** i) * ((1 - u) ** (n - i))


def bezier_curve(u: np.ndarray, b: np.ndarray, n: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a 1-D Bezier curve P(u) and its first two derivatives.

    The derivatives are Bezier curves of order n-1 and n-2 whose ordinates
    are the forward differences of b (Eq. 9):
        b^(1)_i = n (b_{i+1} - b_i),   b^(2)_i = (n-1) (b^(1)_{i+1} - b^(1)_i)

    Parameters
    ----------
    u : parameter values in [0, 1]
    b : the n+1 Bezier ordinates
    n : curve order

    Returns
    -------
    P, dP/du, d2P/du2 : arrays with the shape of u
    """
    u = np.atleast_1d(u)

    P = np.zeros_like(u)
    for i in range(n + 1):
        P += b[i] * bernstein_poly(i, n, u)

    b1 = n * np.diff(b)
    P1 = np.zeros_like(u)
    for i in range(n):
        P1 += b1[i] * bernstein_poly(i, n - 1, u)

    b2 = (n - 1) * np.diff(b1)
    P2 = np.zeros_like(u)
    for i in range(n - 1):
        P2 += b2[i] * bernstein_poly(i, n - 2, u)

    return P, P1, P2


# =============================================================================
# BEZIER ORDINATE VECTOR  (manuscript Section 2.2)
# =============================================================================
#
#  Valid (n, N_VARS) combinations and resulting ordinate vectors.
#  Continuity is quoted as  C^k at u = 0  /  C^k at u = 1.
#
#  +-----+--------+----------------------------------+-----------------------+
#  |  n  | N_VARS |  ordinates b                     |  continuity           |
#  +-----+--------+----------------------------------+-----------------------+
#  |  6  |   2    |  [0,0,0, b_hi, 1,1,1]            |  C2 / C2   (HSB)      |
#  |  6  |   3    |  [0,0, b_lo, b_hi, 1,1,1]        |  C1 / C2              |
#  |  7  |   3    |  [0,0,0, b_lo, b_hi, 1,1,1]      |  C2 / C2              |
#  |  8  |   2    |  [0,0,0,0, b_hi, 1,1,1,1]        |  C3 / C3   (HSB)      |
#  |  8  |   3    |  [0,0,0, b_lo, b_hi, 1,1,1,1]    |  C2 / C3              |
#  |  9  |   3    |  [0,0,0,0, b_lo, b_hi, 1,1,1,1]  |  C3 / C3              |
#  +-----+--------+----------------------------------+-----------------------+
#
#  b_hi : free ordinate next to the fixed "1" block (HSB's b3 for n = 6,
#         b4 for n = 8).
#  b_lo : free ordinate next to the fixed "0" block. With N_VARS = 2 it is
#         fixed to 0, which recovers the HSB vectors for n = 6 and n = 8.
#  For n = 6 and n = 8 the extra freedom costs one order of continuity at
#  u = 0, because the free b_lo replaces a fixed zero (manuscript Section 4).

VALID_CONFIGS = {
    (6, 2): "HSB single-parameter: C2/C2, 1 free ordinate",
    (6, 3): "Multi-parameter: C1/C2, 2 free ordinates (loses C2 at u=0)",
    (7, 3): "Multi-parameter: C2/C2, 2 free ordinates",
    (8, 2): "HSB single-parameter: C3/C3, 1 free ordinate",
    (8, 3): "Multi-parameter: C2/C3, 2 free ordinates (loses C3 at u=0)",
    (9, 3): "Multi-parameter: C3/C3, 2 free ordinates",
}


def validate_config(n: int, n_vars: int) -> None:
    """Check that (n, n_vars) is one of VALID_CONFIGS and log its properties."""
    key = (n, n_vars)
    if key not in VALID_CONFIGS:
        valid = ", ".join(f"(n={k[0]}, N_VARS={k[1]})" for k in VALID_CONFIGS)
        raise ValueError(f"Invalid configuration (n={n}, N_VARS={n_vars}). "
                         f"Valid configurations: {valid}")
    log_message(f"  Config: {VALID_CONFIGS[key]}")


def build_bezier_ordinates(b_hi: float, b_lo: float, n: int) -> np.ndarray:
    """
    Full ordinate vector b (length n+1) from the free ordinates.

    Parameters
    ----------
    b_hi : free ordinate next to the "1" block.
    b_lo : free ordinate next to the "0" block (0.0 -> HSB vector, n = 6, 8).
    n    : Bezier order (6, 7, 8 or 9).
    """
    if n == 6:
        return np.array([0.0, 0.0, b_lo, b_hi, 1.0, 1.0, 1.0])
    if n == 7:
        return np.array([0.0, 0.0, 0.0, b_lo, b_hi, 1.0, 1.0, 1.0])
    if n == 8:
        return np.array([0.0, 0.0, 0.0, b_lo, b_hi, 1.0, 1.0, 1.0, 1.0])
    if n == 9:
        return np.array([0.0, 0.0, 0.0, 0.0, b_lo, b_hi, 1.0, 1.0, 1.0, 1.0])
    raise ValueError(f"Unsupported Bezier order n={n}. Use 6, 7, 8 or 9.")


# =============================================================================
# CAM KINEMATICS
# =============================================================================

def compute_motion_law(theta: np.ndarray, beta: float, b: np.ndarray, n: int
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalized motion law f(theta) and its theta-derivatives over one turn.

    Piecewise definition (manuscript Eq. 10, with f normalized by the lift):
        rise   0          <= theta <  beta      : f = P(theta/beta)
        dwell  beta       <= theta <= 2pi-beta  : f = 1
        return 2pi - beta <  theta <= 2pi       : f = P((2pi - theta)/beta)
    The return is the mirror image of the rise (symmetric motion law).
    Chain rule: df/dtheta = P'/beta (sign flips on the return),
                d2f/dtheta2 = P''/beta^2.

    Returns
    -------
    f, f1, f2 : f, df/dtheta, d2f/dtheta2
    """
    f = np.zeros(len(theta))
    f1 = np.zeros(len(theta))
    f2 = np.zeros(len(theta))

    mask_rise = theta < beta
    mask_return = theta > (2 * np.pi - beta)
    mask_dwell = ~(mask_rise | mask_return)

    if np.any(mask_rise):
        P, P1, P2 = bezier_curve(theta[mask_rise] / beta, b, n)
        f[mask_rise] = P
        f1[mask_rise] = P1 / beta
        f2[mask_rise] = P2 / beta**2

    f[mask_dwell] = 1.0               # f1 = f2 = 0 on the dwell

    if np.any(mask_return):
        P, P1, P2 = bezier_curve((2 * np.pi - theta[mask_return]) / beta, b, n)
        f[mask_return] = P
        f1[mask_return] = -P1 / beta
        f2[mask_return] = P2 / beta**2

    return f, f1, f2


def compute_cam_geometry(alpha: float, h: float, f: np.ndarray,
                         f1: np.ndarray, f2: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pressure angle and dimensionless curvature ratio (HSB Eqs. 3-4).

        q       = alpha - h f
        phi     = arccos( |q| / sqrt(h^2 f'^2 + q^2) )
        rho/|Rf| = (h^2 f'^2 + q^2)^(3/2) / D,
        D       = 2 h^2 f'^2 + q (q + h f'')

    Parameters
    ----------
    alpha : Ra/|Rf|
    h     : h*/|Rf|
    f, f1, f2 : motion law and derivatives from compute_motion_law()

    Returns
    -------
    phi : pressure angle [deg]
    rho : curvature ratio rho/|Rf| (inf where D = 0)
    D   : curvature denominator; a sign change of D means the pitch curve
          has a cusp, i.e. undercutting (manuscript Section 3.4)
    """
    q = alpha - h * f
    hf1_2 = h * h * f1 * f1
    denom_sq = hf1_2 + q * q

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt(denom_sq)
        cos_phi = np.where(denom > 1e-12, np.abs(q) / denom, 1.0)
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.degrees(np.arccos(cos_phi))

    D = 2 * hf1_2 + q * (q + h * f2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(np.abs(D) > 1e-12, denom_sq ** 1.5 / D, np.inf)

    return phi, rho, D


# =============================================================================
# DESIGN EVALUATION  (constraints g1, g2 -- manuscript Eqs. 13-14)
# =============================================================================

def unpack_design(x: np.ndarray) -> Tuple[float, float, float]:
    """x = [h, b_hi] or [h, b_hi, b_lo]  ->  (h, b_hi, b_lo)."""
    return x[0], x[1], (x[2] if len(x) > 2 else 0.0)


def evaluate_cam_design(x: np.ndarray, design_params: CamDesignParams,
                        alpha: float) -> Dict[str, Any]:
    """
    Kinematics, constraint values and feasibility of one design.

    Parameters
    ----------
    x : design vector [h, b_hi] (N_VARS = 2) or [h, b_hi, b_lo] (N_VARS = 3)
    design_params : fixed problem data
    alpha : Ra/|Rf|

    Returns
    -------
    dict with h, b_hi, b_lo, max_rho, max_phi, g1_violation, g2_violation,
    is_feasible and the sampled curves theta, phi, rho, f, f1, f2.
    """
    n, beta = design_params.n, design_params.beta
    h, b_hi, b_lo = unpack_design(x)

    b = build_bezier_ordinates(b_hi, b_lo, n)
    theta = np.linspace(0, 2 * np.pi, design_params.n_theta + 1)
    f, f1, f2 = compute_motion_law(theta, beta, b, n)
    phi, rho, D = compute_cam_geometry(alpha, h, f, f1, f2)

    max_rho = np.max(rho[np.isfinite(rho)])
    max_phi = np.max(phi[np.isfinite(phi)])

    # Robust undercutting test: if D(theta) changes sign, rho has passed
    # through +-inf between two samples (cusp) and the sampled max(rho) can
    # miss it. Such designs are marked infeasible with max_rho >= 2, i.e. at
    # least twice the limit, whatever the grid resolution.
    sign_changes = np.where(np.diff(np.sign(D)))[0]
    if len(sign_changes) > 0:
        rho_cross = np.abs(rho[sign_changes])
        rho_cross = rho_cross[np.isfinite(rho_cross)]
        max_rho = max(max_rho, np.max(rho_cross) if len(rho_cross) else 100.0)
        max_rho = max(max_rho, 2.0)

    g1 = max_rho - 1.0                       # g1 <= 0 : no undercutting
    g2 = max_phi - design_params.phi_max     # g2 <= 0 : pressure angle OK

    return {
        "h": h, "b_hi": b_hi, "b_lo": b_lo,
        "max_rho": max_rho, "max_phi": max_phi,
        "g1_violation": g1, "g2_violation": g2,
        "is_feasible": (g1 <= 0) and (g2 <= 0),
        "theta": theta, "phi": phi, "rho": rho,
        "f": f, "f1": f1, "f2": f2,
    }


# =============================================================================
# OBJECTIVE  (augmented Lagrangian, manuscript Eq. 20)
# =============================================================================

def objective_function(x: np.ndarray, design_params: CamDesignParams,
                       alpha: float, alm: AugmentedLagrangianState) -> float:
    """
    L(x) = -h + psi_1(g1) + psi_2(g2), minimized by DE and by the local search.

    The ALM state is an explicit argument (not a global) so that the parallel
    DE workers always use the current multipliers and penalties.
    Returns 1e10 for designs that cannot be evaluated.
    """
    h, b_hi, b_lo = unpack_design(x)
    if h <= 0 or not (0.0 <= b_hi <= 1.0) or not (0.0 <= b_lo <= 1.0):
        return 1e10
    try:
        res = evaluate_cam_design(x, design_params, alpha)
        return -res["h"] + alm.penalty(res["g1_violation"], res["g2_violation"])
    except Exception:
        return 1e10


# =============================================================================
# LOCAL SEARCH  (manuscript Section 3.4, Eq. 25)
# =============================================================================

def local_search_refinement(x_best: np.ndarray, design_params: CamDesignParams,
                            alpha: float, alm: AugmentedLagrangianState,
                            bounds: List[Tuple[float, float]],
                            rng: np.random.Generator,
                            n_trials: int = LS_N_TRIALS,
                            sigma_ratio: float = LS_SIGMA_RATIO
                            ) -> Tuple[np.ndarray, float, int]:
    """
    Gaussian random search around the best DE point.

    Each trial is x' = x_best + xi, xi_i ~ N(0, sigma_i^2) with
    sigma_i = sigma_ratio * (upper_i - lower_i), clipped to the bounds.
    A trial is accepted only if it lowers L; accepted trials become the new
    centre. The DE population is not modified.

    Returns
    -------
    x_refined  : best point found (x_best itself if nothing improved)
    val_refined: its augmented-Lagrangian value
    n_improved : number of accepted trials
    """
    best_x = x_best.copy()
    best_val = objective_function(best_x, design_params, alpha, alm)
    n_improved = 0

    lower = np.array([lo for lo, _ in bounds])
    upper = np.array([hi for _, hi in bounds])
    sigmas = sigma_ratio * (upper - lower)

    for _ in range(n_trials):
        x_new = np.clip(best_x + rng.normal(0, sigmas), lower, upper)
        val_new = objective_function(x_new, design_params, alpha, alm)
        if val_new < best_val:
            best_x, best_val = x_new, val_new
            n_improved += 1

    return best_x, best_val, n_improved


# =============================================================================
# ROLL-BACK TEST
# =============================================================================

def should_rollback(current: Tuple[float, float], previous: Tuple[float, float],
                    threshold: float = ROLLBACK_FACTOR) -> bool:
    """True if the worst positive violation grew by more than `threshold`."""
    curr = max(max(0.0, current[0]), max(0.0, current[1]))
    prev = max(max(0.0, previous[0]), max(0.0, previous[1]))
    return prev > 0 and curr > threshold * prev


# =============================================================================
# MAIN OPTIMIZER: DE + ALM + LS  (manuscript Algorithm 1)
# =============================================================================

def optimize_cam(design_params: CamDesignParams, alpha: float,
                 n_vars: int = 3, population_size: int = 30,
                 outer_iterations: int = 15, de_maxiter: int = 88,
                 seed: int = 42) -> Optional[Dict[str, Any]]:
    """
    Maximize the lift h for one value of alpha = Ra/|Rf|.

    Double loop (Algorithm 1):

        for t = 1 .. N_outer                        (ALM outer loop)
            DE for G_max generations on L(x; mu, rho)   (inner loop)
            local search around the best DE point
            evaluate g1, g2; keep the best feasible design
            roll back mu, rho if the violation keeps diverging, else
            update mu, rho, eta, beta_ALM
            stop if feasible, delta h_best < H_TOL and t >= N_outer/2

    Parameters
    ----------
    design_params    : fixed problem data
    alpha            : Ra/|Rf|
    n_vars           : 2 (HSB, x = [h, b_hi]) or 3 (x = [h, b_hi, b_lo])
    population_size  : total DE population NP
    outer_iterations : N_outer
    de_maxiter       : G_max
    seed             : seed of this run (DE uses seed + t at iteration t)

    Returns
    -------
    dict with the final design and its sampled curves, or None if no design
    could be evaluated. The best FEASIBLE design is returned if one was
    found; otherwise the one with the lowest augmented Lagrangian.
    """
    validate_config(design_params.n, n_vars)
    alm = AugmentedLagrangianState()

    # Search box. Same bounds for both free ordinates.
    b_box = (design_params.b_min, design_params.b_max)
    bounds = [H_BOUNDS, b_box] + ([b_box] if n_vars == 3 else [])

    log_message(f"\n{'=' * 60}")
    log_message(f"Optimizing for α = {alpha:.2f}")
    log_message(f"{'=' * 60}")
    log_message(f"  Bézier order: n = {design_params.n}")
    log_message(f"  Rise angle: β = {np.degrees(design_params.beta):.1f}°")
    log_message(f"  Max pressure angle: φ_max = {design_params.phi_max}°")
    log_message(f"  Design variables: {n_vars}  (x = [h, b_hi"
                f"{', b_lo' if n_vars == 3 else ''}])")
    log_message(f"  Bounds: {bounds}")

    best_solution, best_objective = None, np.inf          # lowest L overall
    best_feasible_solution, best_feasible_h = None, 0.0   # highest feasible h
    consecutive_bad = 0
    rng = np.random.default_rng(seed)                     # local-search RNG

    for t in range(outer_iterations):
        log_message(f"\n--- ALM outer iteration {t + 1}/{outer_iterations} ---")
        log_message(f"  μ_ρ={alm.mu_rho:.4f}, μ_φ={alm.mu_phi:.4f}, "
                    f"ρ_ρ={alm.rho_rho:.2f}, ρ_φ={alm.rho_phi:.2f}")

        saved_state = alm.copy_parameters()     # for a possible roll-back

        try:
            # ---------------- Inner loop: Differential Evolution ------------
            # SciPy's `popsize` is a multiplier: total population =
            # popsize * len(bounds), so NP = 30 with 3 variables -> popsize 10.
            result = differential_evolution(
                func=objective_function,
                bounds=bounds,
                args=(design_params, alpha, alm),
                strategy=DE_STRATEGY,
                mutation=DE_MUTATION,
                recombination=DE_RECOMBINATION,
                popsize=max(5, population_size // len(bounds)),
                maxiter=de_maxiter,
                seed=seed + t,
                polish=False,             # no gradient polishing (constraints)
                workers=-1,               # all CPU cores
                updating="deferred",      # required for workers != 1
            )
            if result.fun >= 1e9:
                log_message("  DE failed to find an evaluable solution")
                consecutive_bad += 1
                continue
            x_de = result.x.copy()

            # ---------------- Local search ----------------------------------
            x_opt = x_de
            if t % LS_FREQUENCY == 0:
                x_ls, val_ls, n_improved = local_search_refinement(
                    x_de, design_params, alpha, alm, bounds, rng)
                if val_ls < result.fun:
                    log_message(f"  LS improved solution "
                                f"({n_improved} accepted perturbations)")
                    x_opt = x_ls

            # ---------------- Constraint evaluation -------------------------
            res = evaluate_cam_design(x_opt, design_params, alpha)
            g1, g2 = res["g1_violation"], res["g2_violation"]
            log_message(f"  h = {res['h']:.6f}")
            log_message(f"  max(ρ/|Rf|) = {res['max_rho']:.6f}  (g1 = {g1:+.6f})")
            log_message(f"  max(φ) = {res['max_phi']:.4f}°  (g2 = {g2:+.4f}°)")
            log_message(f"  Feasible: {res['is_feasible']}")

            obj_val = objective_function(x_opt, design_params, alpha, alm)
            if obj_val < best_objective:
                best_solution, best_objective = x_opt.copy(), obj_val

            previous_best_h = best_feasible_h
            if res["is_feasible"] and res["h"] > best_feasible_h:
                best_feasible_solution = x_opt.copy()
                best_feasible_h = res["h"]
                log_message(f"  *** New best feasible: h = {best_feasible_h:.6f} ***")

            # ---------------- Roll-back -------------------------------------
            # If the worst violation more than doubles MAX_CONSECUTIVE_BAD
            # times in a row, restore mu (and slightly relaxed rho) from the
            # start of this iteration and skip the multiplier update.
            if (alm.constraint_violations
                    and should_rollback((g1, g2), alm.constraint_violations[-1])):
                consecutive_bad += 1
                log_message(f"  Warning: constraint violation increased "
                            f"({consecutive_bad}/{MAX_CONSECUTIVE_BAD})")
            else:
                consecutive_bad = 0

            if consecutive_bad >= MAX_CONSECUTIVE_BAD:
                log_message("  ROLLBACK triggered")
                alm.mu_rho = saved_state.mu_rho
                alm.mu_phi = saved_state.mu_phi
                alm.rho_rho = max(1.0, 0.8 * saved_state.rho_rho)
                alm.rho_phi = max(1.0, 0.8 * saved_state.rho_phi)
                consecutive_bad = 0
                continue

            # ---------------- ALM update (Eqs. 22-24) -----------------------
            alm.update_multipliers(g1, g2)

            # ---------------- Convergence -----------------------------------
            if res["is_feasible"] and t >= 3:
                delta_h = best_feasible_h - previous_best_h
                if delta_h < H_TOL and t >= outer_iterations // 2:
                    log_message(f"\n  Converged: Δh = {delta_h:.6f} < {H_TOL}")
                    break

        except Exception as exc:
            log_message(f"  Error in outer iteration {t + 1}: {exc}")
            consecutive_bad += 1
            continue

    # ---------------- Final selection ---------------------------------------
    if best_feasible_solution is not None:
        final, tag = best_feasible_solution, "FEASIBLE"
    elif best_solution is not None:
        final, tag = best_solution, "INFEASIBLE"
    else:
        log_message(f"\nNo valid solution found for α = {alpha}")
        return None

    final_res = evaluate_cam_design(final, design_params, alpha)
    log_message(f"\nFinal ({tag}): h = {final_res['h']:.6f}")

    return {
        "x": final, "alpha": alpha,
        "h": final_res["h"], "b_hi": final_res["b_hi"], "b_lo": final_res["b_lo"],
        "max_rho": final_res["max_rho"], "max_phi": final_res["max_phi"],
        "is_feasible": final_res["is_feasible"],
        "theta": final_res["theta"], "phi": final_res["phi"],
        "rho": final_res["rho"],
    }


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def save_results_csv(results: List[Dict[str, Any]], path: str) -> None:
    """One row per alpha: alpha, b_lo, b_hi, h, max_phi, max_rho, feasible."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("alpha,b_lo,b_hi,h,phi_max_deg,rho_max,feasible\n")
        for r in results:
            fh.write(f"{r['alpha']:.6f},{r['b_lo']:.6f},{r['b_hi']:.6f},"
                     f"{r['h']:.6f},{r['max_phi']:.6f},{r['max_rho']:.6f},"
                     f"{int(r['is_feasible'])}\n")


def make_plots(results: List[Dict[str, Any]], design_params: CamDesignParams,
               path: str) -> None:
    """Motion law, pressure angle and curvature of the last alpha, and h(alpha)."""
    import matplotlib
    matplotlib.use("Agg")                     # file output only, no window
    import matplotlib.pyplot as plt

    last = results[-1]
    n, beta = design_params.n, design_params.beta
    b = build_bezier_ordinates(last["b_hi"], last["b_lo"], n)
    theta = np.linspace(0, 2 * np.pi, design_params.n_theta + 1)
    f, f1, f2 = compute_motion_law(theta, beta, b, n)
    phi, rho, _ = compute_cam_geometry(last["alpha"], last["h"], f, f1, f2)

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    ax[0, 0].plot(theta / np.pi, f, "b-", lw=2)
    ax[0, 0].set(xlabel=r"$\theta/\pi$", ylabel=r"$f(\theta)$",
                 title=rf"Motion law ($\alpha$ = {last['alpha']:.2f})")

    ax[0, 1].plot(theta / np.pi, phi, "r-", lw=2)
    ax[0, 1].axhline(design_params.phi_max, color="k", ls="--",
                     label=rf"$\varphi_{{max}}$ = {design_params.phi_max}$^\circ$")
    ax[0, 1].set(xlabel=r"$\theta/\pi$", ylabel=r"$\varphi$ [deg]",
                 title="Pressure angle")
    ax[0, 1].legend()

    ax[1, 0].plot(theta / np.pi, 1 - np.clip(rho, -5, 5), "g-", lw=2)
    ax[1, 0].axhline(0, color="k", ls="--", label=r"$\rho_c = 0$")
    ax[1, 0].set(xlabel=r"$\theta/\pi$", ylabel=r"$1 - \rho/|R_f|$",
                 title=r"Cam curvature ($\rho_c > 0$ required)")
    ax[1, 0].legend()

    ax[1, 1].plot([r["alpha"] for r in results], [r["h"] for r in results],
                  "ko-", lw=2, ms=8)
    ax[1, 1].set(xlabel=r"$\alpha = R_a/|R_f|$", ylabel=r"$h^*/|R_f|$",
                 title="Optimal lift vs prime-circle ratio")

    for a in ax.flat:
        a.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log_message(f"Plots saved to: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Sweep ALPHAS with the USER INPUTS and write CSV, log and plots."""
    # Greek symbols are printed to the console. On Windows, output redirected
    # to a file may use cp1252 (no Greek); replace instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass

    np.seterr(all="ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    start_time = time.time()

    design_params = CamDesignParams(
        n=N_ORDER, beta=np.radians(BETA_DEG), phi_max=PHI_MAX_DEG,
        n_theta=N_THETA, b_min=B_MIN, b_max=B_MAX,
    )

    log_message(f"Detected {multiprocessing.cpu_count()} CPU cores")
    log_message("\n" + "=" * 60)
    log_message("CAM MECHANISM OPTIMIZATION")
    log_message("Bézier curves + DE + ALM + Local Search")
    log_message("=" * 60)
    log_message("\nParameters:")
    log_message(f"  Bézier order n = {N_ORDER}")
    log_message(f"  Rise angle β = {BETA_DEG:.1f}°")
    log_message(f"  Max pressure angle φ_max = {PHI_MAX_DEG}°")
    log_message(f"  Design variables: {N_VARS}")
    log_message(f"  Ordinate bounds: [{B_MIN}, {B_MAX}]")
    log_message(f"  Population NP = {NP}")
    log_message(f"  ALM outer iterations = {N_OUTER}")
    log_message(f"  DE generations/outer = {G_MAX}")
    log_message(f"  α sweep: {ALPHAS}")

    # ---------------- Alpha sweep ---------------------------------------------
    results: List[Dict[str, Any]] = []
    for idx, alpha in enumerate(ALPHAS):
        log_message(f"\n{'#' * 60}")
        log_message(f"# α = {alpha:.2f}  ({idx + 1}/{len(ALPHAS)})")
        log_message(f"{'#' * 60}")
        res = optimize_cam(design_params, alpha, n_vars=N_VARS,
                           population_size=NP, outer_iterations=N_OUTER,
                           de_maxiter=G_MAX, seed=BASE_SEED + idx)
        if res is not None:
            results.append(res)
        else:
            log_message(f"No solution found for α = {alpha}")

    # ---------------- Summary table -------------------------------------------
    log_message("\n" + "=" * 60)
    log_message("OPTIMIZATION COMPLETED")
    log_message("=" * 60)
    log_message(f"Total time: {time.time() - start_time:.2f} s")

    hdr = (f"{'α':>6} | {'b_lo':>8} | {'b_hi':>8} | {'h':>10} | "
           f"{'φ_max':>8} | {'ρ_max':>8} | feasible")
    log_message("\n" + "-" * len(hdr))
    log_message("RESULTS TABLE")
    log_message("-" * len(hdr))
    log_message(hdr)
    log_message("-" * len(hdr))
    for r in results:
        log_message(f"{r['alpha']:>6.2f} | {r['b_lo']:>8.4f} | {r['b_hi']:>8.4f} | "
                    f"{r['h']:>10.6f} | {r['max_phi']:>8.2f} | "
                    f"{r['max_rho']:>8.4f} | {r['is_feasible']}")

    # ---------------- Files ---------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = f"n{N_ORDER}_v{N_VARS}_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    if results:
        csv_path = os.path.join(OUTPUT_DIR, f"cam_results_{stem}.csv")
        save_results_csv(results, csv_path)
        log_message(f"\nCSV saved to: {csv_path}")

        if MAKE_PLOTS:
            try:
                make_plots(results, design_params,
                           os.path.join(OUTPUT_DIR, f"cam_plots_{stem}.png"))
            except ImportError:
                log_message("matplotlib not available -- plots skipped")

    write_log_file(f"cam_log_{stem}.dat")
    log_message("\nDone.")


if __name__ == "__main__":
    # The guard is required: DE runs in parallel worker processes, which
    # re-import this file on Windows/macOS.
    main()
