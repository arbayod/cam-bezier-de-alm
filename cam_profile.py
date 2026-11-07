# -*- coding: utf-8 -*-
"""
cam_profile.py
===============================================================================
Cam profile for a negative-radius follower driven by a Bezier motion law.

Given the Bezier order n, the rise angle beta, the prime-circle ratio alpha,
the lift h and the free Bezier ordinates (b_lo, b_hi) -- for instance an
optimum produced by camBezierOptimization_DE_ALM.py -- this script:

    1. checks the design (max pressure angle, max curvature ratio,
       undercutting), with the same kernel as the optimizer;
    2. builds the cam profile (Hidalgo-Martinez et al. 2014, Eqs. 1-4);
    3. saves a figure (.png and .pdf) and the profile coordinates (.csv).

Companion code of:
    A. Revuelta-Bayod, "Multi-parameter Bezier optimization of cam mechanisms
    with negative radius follower using Augmented Lagrangian Differential
    Evolution", 2026. SSRN: https://ssrn.com/abstract=7275640
    doi:10.2139/ssrn.7275640

-------------------------------------------------------------------------------
QUICK START
-------------------------------------------------------------------------------
    1. Requirements: Python >= 3.9, NumPy, SciPy, matplotlib.
    2. Edit the block "USER INPUTS" below (there are no command-line options).
       Copy (b_lo, b_hi, h) from the results CSV of the optimizer or from
       Tables 2-3 of the manuscript. For the single-parameter HSB design set
       B_LO = 0.0 (n = 6 or 8 only).
    3. Run:
           python cam_profile.py
    4. Results are written to ./output/ (next to this script):
           cam_profile_n{n}_a{alpha}.png / .pdf   -> figure
           cam_profile_n{n}_a{alpha}.csv          -> coordinates

-------------------------------------------------------------------------------
CSV FILE
-------------------------------------------------------------------------------
    Header lines start with '#' and list the inputs and the design checks.
    Columns (comma separated):
        theta_deg : cam rotation angle [deg], 0 ... 360
        x, y      : cam profile (contact point), cam reference frame
        x_C, y_C  : follower-centre curve (pitch curve), same frame
    Lengths are dimensionless (divided by |Rf|); multiply by |Rf| to get
    physical units, or set LENGTH_SCALE = |Rf| below.
    The last row repeats the first one, so the curve is closed.
    Reading it back:
        pandas:  df = pd.read_csv(path, comment="#")
        numpy :  data = np.loadtxt(path, delimiter=",", skiprows=7)
                 (6 '#' lines + 1 line of column names; columns in the
                 order theta_deg, x, y, x_C, y_C)
        Excel / CAD: open as CSV; ignore the lines starting with '#'.

-------------------------------------------------------------------------------
GEOMETRY (Hidalgo-Martinez et al. 2014, Eqs. 1-4), lengths / |Rf|
-------------------------------------------------------------------------------
    Fixed frame Oxy, O = cam rotation centre, follower moving along y:
        r_C = (0, alpha - h f)          follower (arc) centre
        r_I = (-h f', 0)                instantaneous centre of velocity
        r_P = r_C - (r_C - r_I)/|r_C - r_I|   contact point (|Rf| = 1)
    The cam rotates by theta; a point is taken to the cam frame with
        [A] = [[cos theta,  sin theta],
               [-sin theta, cos theta]]
    Finally the picture is rotated by -90 deg (X = y, Y = -x) so that the
    start position of C lies on the +x axis, as in Fig. 2 of the manuscript.

Author : Antonio Revuelta-Bayod
Version: 1.0 (2026-11)
License: see LICENSE file in the repository.
===============================================================================
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                    # write files only, no window needed
import matplotlib.pyplot as plt
from scipy.special import comb


# =============================================================================
# USER INPUTS  --  edit this block
# =============================================================================

# --- Motion law --------------------------------------------------------------
N_ORDER   = 8          # Bezier order n: 6, 7, 8 or 9
BETA_DEG  = 180.0      # rise angle beta [deg] (return is symmetric)

# --- Design (default: Table 3 of the manuscript, n = 8, alpha = 0.6) ---------
ALPHA     = 0.6        # prime-circle ratio alpha = Ra/|Rf|
H         = 0.476376   # dimensionless lift h = h*/|Rf|
B_LO      = 0.9781     # free ordinate next to the "0" block (0.0 -> HSB)
B_HI      = 0.6319     # free ordinate next to the "1" block

# --- Design limits used by the check (same as the optimizer) -----------------
PHI_MAX_DEG = 41.0     # pressure-angle limit [deg]

# --- Output ------------------------------------------------------------------
N_POINTS     = 1441    # points along one revolution (1441 -> 0.25 deg step)
LENGTH_SCALE = 1.0     # multiply coordinates by this (e.g. |Rf| in mm)
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Angular grid of the design check. Keep 360 (1 deg): it is the grid used by
# the optimizer, so the check reproduces the constraint values of the paper.
N_THETA_CHECK = 360


# =============================================================================
# BEZIER MOTION LAW  (identical to camBezierOptimization_DE_ALM.py)
# =============================================================================

def bernstein_poly(i, n, u):
    """Bernstein basis polynomial B_i^n(u) = C(n,i) u^i (1-u)^(n-i)."""
    return comb(n, i, exact=True) * (u ** i) * ((1 - u) ** (n - i))


def bezier_curve(u, b, n):
    """Bezier curve P(u) and its derivatives P'(u), P''(u) (forward differences)."""
    u = np.atleast_1d(u)
    P = sum(b[i] * bernstein_poly(i, n, u) for i in range(n + 1))
    b1 = n * np.diff(b)
    P1 = sum(b1[i] * bernstein_poly(i, n - 1, u) for i in range(n))
    b2 = (n - 1) * np.diff(b1)
    P2 = sum(b2[i] * bernstein_poly(i, n - 2, u) for i in range(n - 1))
    return P, P1, P2


def build_bezier_ordinates(b_hi, b_lo, n):
    """
    Full ordinate vector b from the two free ordinates (manuscript Sec. 2.2).

        n = 6 : [0,0, b_lo, b_hi, 1,1,1]
        n = 7 : [0,0,0, b_lo, b_hi, 1,1,1]
        n = 8 : [0,0,0, b_lo, b_hi, 1,1,1,1]
        n = 9 : [0,0,0,0, b_lo, b_hi, 1,1,1,1]

    With b_lo = 0, n = 6 and n = 8 give the single-parameter HSB vectors.
    """
    vectors = {
        6: [0.0, 0.0, b_lo, b_hi, 1.0, 1.0, 1.0],
        7: [0.0, 0.0, 0.0, b_lo, b_hi, 1.0, 1.0, 1.0],
        8: [0.0, 0.0, 0.0, b_lo, b_hi, 1.0, 1.0, 1.0, 1.0],
        9: [0.0, 0.0, 0.0, 0.0, b_lo, b_hi, 1.0, 1.0, 1.0, 1.0],
    }
    if n not in vectors:
        raise ValueError(f"Unsupported Bezier order n={n}. Use 6, 7, 8 or 9.")
    return np.array(vectors[n])


def compute_motion_law(theta, beta, b, n):
    """
    Normalized follower motion f(theta) and its derivatives over one turn.

        rise   0          <= theta <  beta     : f = P(theta/beta)
        dwell  beta       <= theta <= 2pi-beta : f = 1
        return 2pi - beta <  theta <= 2pi      : f = P((2pi - theta)/beta)

    Returns f, df/dtheta, d2f/dtheta2.
    """
    f, f1, f2 = (np.zeros(len(theta)) for _ in range(3))
    rise = theta < beta
    ret = theta > (2 * np.pi - beta)

    P, P1, P2 = bezier_curve(theta[rise] / beta, b, n)
    f[rise], f1[rise], f2[rise] = P, P1 / beta, P2 / beta**2

    f[~(rise | ret)] = 1.0                      # dwell: f' = f'' = 0

    P, P1, P2 = bezier_curve((2 * np.pi - theta[ret]) / beta, b, n)
    f[ret], f1[ret], f2[ret] = P, -P1 / beta, P2 / beta**2
    return f, f1, f2


# =============================================================================
# DESIGN CHECK  (same constraints as the optimizer, manuscript Eqs. 13-14)
# =============================================================================

def check_design(alpha, h, b, n, beta, phi_max, n_theta=N_THETA_CHECK):
    """
    Maximum pressure angle and curvature ratio of a design.

    Returns a dict with max_phi [deg], max_rho (= max rho/|Rf|), a flag
    `cusp` (sign change of the curvature denominator D -> undercutting) and
    `feasible` (max_rho <= 1, no cusp, max_phi <= phi_max).
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    f, f1, f2 = compute_motion_law(theta, beta, b, n)

    q = alpha - h * f                           # y-coordinate of C
    hf1_2 = (h * f1) ** 2
    denom_sq = hf1_2 + q * q
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_phi = np.clip(np.abs(q) / np.sqrt(denom_sq), -1.0, 1.0)
        phi = np.degrees(np.arccos(cos_phi))    # pressure angle
        D = 2 * hf1_2 + q * (q + h * f2)        # curvature denominator
        rho = np.where(np.abs(D) > 1e-12, denom_sq ** 1.5 / D, np.inf)

    max_phi = np.max(phi[np.isfinite(phi)])
    max_rho = np.max(rho[np.isfinite(rho)])
    cusp = bool(np.any(np.diff(np.sign(D)) != 0))
    feasible = (max_rho <= 1.0) and (not cusp) and (max_phi <= phi_max)
    return {"max_phi": max_phi, "max_rho": max_rho,
            "cusp": cusp, "feasible": feasible}


# =============================================================================
# CAM PROFILE  (Hidalgo-Martinez et al. 2014, Eqs. 1-4)
# =============================================================================

def cam_profile(alpha, h, b, n, beta, n_points=N_POINTS):
    """
    Cam profile and follower-centre curve in the cam frame.

    Returns
    -------
    theta  : rotation angles [rad], 0 ... 2pi (closed curve)
    xP, yP : contact-point curve = cam profile
    xC, yC : follower-centre (pitch) curve
    All in display orientation (start position of C on the +x axis).
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    f, f1, _ = compute_motion_law(theta, beta, b, n)

    # Fixed frame: r_C = (0, q), r_C - r_I = (h f', q)
    q = alpha - h * f
    hf1 = h * f1
    dist = np.sqrt(hf1**2 + q**2)               # |r_C - r_I|

    # Contact point: one follower radius (= 1) from C towards I
    rPx = -hf1 / dist
    rPy = q - q / dist

    c, s = np.cos(theta), np.sin(theta)

    def to_display(rx, ry):
        """Fixed frame -> cam frame ([A] r) -> rotate -90 deg (X=y1, Y=-x1)."""
        x1 = c * rx + s * ry
        y1 = -s * rx + c * ry
        return y1, -x1

    xP, yP = to_display(rPx, rPy)
    xC, yC = to_display(np.zeros_like(q), q)
    return theta, xP, yP, xC, yC


# =============================================================================
# OUTPUT
# =============================================================================

def save_csv(path, theta, xP, yP, xC, yC, header_lines):
    """Coordinates to CSV; header lines are written as '#' comments."""
    data = np.column_stack([np.degrees(theta), xP, yP, xC, yC])
    header = "\n".join(header_lines) + "\ntheta_deg,x,y,x_C,y_C"
    np.savetxt(path, data, delimiter=",", fmt="%.8f",
               header=header, comments="# ")
    # np.savetxt prefixes every header line with '# ', also the column names;
    # rewrite that last line without '# ' so pandas/genfromtxt can read it.
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    lines[len(header_lines)] = "theta_deg,x,y,x_C,y_C\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def plot_profile(path_stem, xP, yP, xC, yC, alpha, title):
    """Cam profile, follower-centre curve and prime circle; .png and .pdf."""
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(alpha * np.cos(t), alpha * np.sin(t), "k--", lw=0.8, alpha=0.6,
            label="prime circle")
    ax.plot(xC, yC, color="0.5", lw=1.0, label="follower-centre curve")
    ax.plot(xP, yP, color="#c81e1e", lw=2.2, label="cam profile")

    ax.plot(0, 0, "k+", ms=10, mew=1.4)                 # rotation centre O
    ax.annotate(r"$O$", (0, 0), xytext=(4, 4), textcoords="offset points")
    ax.plot(alpha, 0, "ko", ms=3)                       # start position of C
    ax.annotate(r"$C$", (alpha, 0), xytext=(4, 4), textcoords="offset points")

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/|R_f|$")
    ax.set_ylabel(r"$y/|R_f|$")
    ax.set_title(title, fontsize=11)
    ax.grid(True, lw=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              fontsize=9, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Greek symbols are printed; never crash on a console that cannot show them.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass

    beta = np.radians(BETA_DEG)
    b = build_bezier_ordinates(B_HI, B_LO, N_ORDER)

    # ---------------- 1. Design check ----------------------------------------
    chk = check_design(ALPHA, H, b, N_ORDER, beta, PHI_MAX_DEG)
    print(f"Design: n = {N_ORDER}, β = {BETA_DEG:.1f}°, α = {ALPHA}, h = {H}")
    print(f"        b = {np.array2string(b, precision=4)}")
    print(f"  max(φ)      = {chk['max_phi']:.3f}°   (limit {PHI_MAX_DEG}°)")
    print(f"  max(ρ/|Rf|) = {chk['max_rho']:.4f}    (limit 1)")
    print(f"  cusp        = {chk['cusp']}")
    print(f"  feasible    = {chk['feasible']}")
    if not chk["feasible"]:
        print("  WARNING: design violates the constraints; the profile is "
              "still written but may be undercut or exceed φ_max.")

    # ---------------- 2. Profile ---------------------------------------------
    theta, xP, yP, xC, yC = cam_profile(ALPHA, H, b, N_ORDER, beta)
    xP, yP, xC, yC = (LENGTH_SCALE * v for v in (xP, yP, xC, yC))

    # ---------------- 3. Files -----------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.join(OUTPUT_DIR, f"cam_profile_n{N_ORDER}_a{ALPHA:.2f}")

    header = [
        "Cam profile, negative-radius follower, Bezier motion law",
        f"n = {N_ORDER}, beta = {BETA_DEG} deg, alpha = Ra/|Rf| = {ALPHA}, "
        f"h = h*/|Rf| = {H}",
        f"b_lo = {B_LO}, b_hi = {B_HI}, ordinates = "
        + " ".join(f"{v:g}" for v in b),
        f"check: max_phi = {chk['max_phi']:.4f} deg (limit {PHI_MAX_DEG}), "
        f"max_rho/|Rf| = {chk['max_rho']:.5f} (limit 1), "
        f"cusp = {chk['cusp']}, feasible = {chk['feasible']}",
        f"lengths multiplied by LENGTH_SCALE = {LENGTH_SCALE} "
        "(1.0 -> dimensionless, divided by |Rf|)",
        "x, y: cam profile; x_C, y_C: follower-centre curve; cam frame, "
        "C on +x at theta = 0",
    ]
    save_csv(f"{stem}.csv", theta, xP, yP, xC, yC, header)

    title = (rf"$n={N_ORDER}$, $\beta={BETA_DEG:.0f}^\circ$, $\alpha={ALPHA}$, "
             rf"$h={H:.4f}$" "\n"
             rf"$(b_{{lo}}, b_{{hi}}) = ({B_LO:.4f},\ {B_HI:.4f})$")
    plot_profile(stem, xP / LENGTH_SCALE, yP / LENGTH_SCALE,
                 xC / LENGTH_SCALE, yC / LENGTH_SCALE, ALPHA, title)

    print(f"\nSaved: {stem}.csv")
    print(f"       {stem}.png / .pdf")


if __name__ == "__main__":
    main()
