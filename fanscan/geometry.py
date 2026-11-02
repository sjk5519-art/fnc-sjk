"""Blade camber-line model shared by synth.py, analyze.py and cad.py.

Model: the blade angle beta varies *linearly with radius* between beta1 at r1
and beta2 at r2.  With beta measured from the circumferential direction,

    tan(beta) = dr / (r dtheta)      =>      dtheta/dr = 1 / (r tan beta(r))

Integrating from r1 gives theta(r); the blade sweeps toward +theta as r grows.
Everything here is unit-agnostic: pass r in mm and you get mm, pass r in px
and you get px.  theta is always radians.
"""
from __future__ import annotations

import math

import numpy as np


def beta_of_r(r: np.ndarray | float, r1: float, r2: float,
              beta1_deg: float, beta2_deg: float) -> np.ndarray | float:
    """Blade angle [rad] at radius r (linear in r)."""
    t = (np.asarray(r, dtype=float) - r1) / (r2 - r1)
    return np.radians(beta1_deg + (beta2_deg - beta1_deg) * t)


def camber_line(r1: float, r2: float, beta1_deg: float, beta2_deg: float,
                n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return (r, theta) arrays of the camber line, theta(r1) = 0.

    Uses trapezoidal integration of dtheta/dr on a fine grid.
    """
    r = np.linspace(r1, r2, n)
    dtheta_dr = 1.0 / (r * np.tan(beta_of_r(r, r1, r2, beta1_deg, beta2_deg)))
    theta = np.concatenate([[0.0], np.cumsum(0.5 * (dtheta_dr[1:] + dtheta_dr[:-1]) * np.diff(r))])
    return r, theta


def wrap_angle(theta_rad: float) -> float:
    """Total angular sweep [rad] of the blade from r1 to r2."""
    return float(theta_rad)


def polar_to_xy(r: np.ndarray, theta: np.ndarray,
                theta0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Convert (r, theta) to Cartesian, rotating by theta0 [rad]."""
    return r * np.cos(theta + theta0), r * np.sin(theta + theta0)


def offset_camber(r: np.ndarray, theta: np.ndarray, thickness: float,
                  theta0: float = 0.0) -> np.ndarray:
    """Closed 2-D polygon (N,2) of a blade of uniform *thickness* around the
    camber line, built by offsetting along the local normal.  Ends are flat.
    """
    x, y = polar_to_xy(r, theta, theta0)
    dx, dy = np.gradient(x), np.gradient(y)
    norm = np.hypot(dx, dy)
    nx, ny = -dy / norm, dx / norm
    h = thickness / 2.0
    left = np.stack([x + nx * h, y + ny * h], axis=1)
    right = np.stack([x - nx * h, y - ny * h], axis=1)[::-1]
    return np.vstack([left, right])


def beta_from_trace(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Inverse model: local blade angle [deg] from a traced (r, theta) curve.

    beta = atan( 1 / (r * dtheta/dr) ).  Robust to sign: returns values in
    (0, 180) with the sweep direction normalised to +theta.
    """
    r = np.asarray(r, float)
    theta = np.asarray(theta, float)
    dth = np.gradient(theta, r)
    # Normalise sweep direction so the inverse model matches camber_line().
    if np.nanmean(dth) < 0:
        dth = -dth
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.degrees(np.arctan2(1.0, r * dth))
    return beta


def fit_linear_beta(r: np.ndarray, theta: np.ndarray, r1: float, r2: float
                    ) -> tuple[float, float]:
    """Least-squares fit of the linear-beta model to a traced curve.

    Returns (beta1_deg, beta2_deg) evaluated at r1 and r2 (extrapolated if the
    trace does not reach them).  Points near the ends of the trace are
    down-weighted because np.gradient is one-sided there.
    """
    r = np.asarray(r, float)
    beta = beta_from_trace(r, theta)
    ok = np.isfinite(beta)
    r, beta = r[ok], beta[ok]
    if len(r) < 4:
        raise ValueError("trace too short to fit beta")
    w = np.ones_like(r)
    w[0] = w[-1] = 0.25
    A = np.stack([np.ones_like(r), (r - r1) / (r2 - r1)], axis=1)
    coef, *_ = np.linalg.lstsq(A * w[:, None], beta * w, rcond=None)
    b1 = float(coef[0])
    b2 = float(coef[0] + coef[1])
    return b1, b2


def total_sweep_deg(r1: float, r2: float, beta1_deg: float, beta2_deg: float) -> float:
    _, th = camber_line(r1, r2, beta1_deg, beta2_deg, n=256)
    return math.degrees(th[-1])


# ---------------------------------------------------------------------------
# Wrap-angle relations (used by the shrouded-mode beta2 solver)
# ---------------------------------------------------------------------------

def wrap_from_betas(r1: float, r2: float, beta1_deg: float, beta2_deg: float) -> float:
    """Total blade wrap angle [deg] from r1 to r2 under the linear-beta model."""
    return total_sweep_deg(r1, r2, beta1_deg, beta2_deg)


def beta2_from_wrap(r1: float, r2: float, beta1_deg: float, wrap_deg: float,
                    lo: float = 5.0, hi: float = 170.0) -> float:
    """Solve wrap_from_betas(r1, r2, beta1, beta2) == wrap_deg for beta2 [deg].

    The wrap angle is monotonically decreasing in beta2 on (0, 90) and the
    model is only meaningful for 0 < beta < 180, so a bracketed root find is
    safe.  Raises ValueError if wrap_deg is outside the attainable range.
    """
    from scipy.optimize import brentq

    f = lambda b2: wrap_from_betas(r1, r2, beta1_deg, b2) - wrap_deg  # noqa: E731
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"wrap {wrap_deg:.1f} deg not attainable with beta1={beta1_deg} "
            f"(range {f(hi)+wrap_deg:.1f}..{f(lo)+wrap_deg:.1f} deg)")
    return float(brentq(f, lo, hi, xtol=1e-6))


def wrap_mod_pitch_candidates(wrap_mod_deg: float, blade_count: int,
                              est_wrap_deg: float, k_max: int = 6) -> list[float]:
    """Resolve a wrap angle known only modulo the blade pitch.

    Returns candidate wrap angles wrap_mod + k*pitch (k >= 0) sorted by
    closeness to *est_wrap_deg* (e.g. the extrapolated estimate).
    """
    pitch = 360.0 / blade_count
    base = wrap_mod_deg % pitch
    cands = [base + k * pitch for k in range(k_max + 1)]
    return sorted(cands, key=lambda w: abs(w - est_wrap_deg))
