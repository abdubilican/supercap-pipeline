"""
peis_kk.py
----------
Kramers-Kronig test by the Lin-KK method (lab guide PEIS 3.3).

impedance.py 1.7.1 ships impedance.validation.linKK, but it raises a
NameError ('np' is not defined) -- see linkk_impedance_py_works(). The test is
therefore implemented here, exactly as specified in the guide:

  Z_KK(w) = R0 + sum_k R_k / (1 + j w tau_k) + a_C / (j w) + a_L j w
  tau_k log-spaced from 1/(2 pi f_max) to 1/(2 pi f_min), k = 1..M
  linear least squares on real and imaginary parts together, every row
  weighted by 1/|Z|; M increased from KK_M_START until
  mu = 1 - sum|R_k < 0| / sum|R_k >= 0| < KK_MU  (Schönleber criterion)
  residuals dZ'/|Z| and dZ''/|Z| in % per point.

The KK test only checks and flags; it never removes points.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from . import peis_config as cfg

log = logging.getLogger("supercap_pipeline.peis").getChild("kk")


def _design(omega: np.ndarray, tau: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted design matrix [real rows; imaginary rows] for (R0, R_1..R_M, a_C, a_L)."""
    wt = omega[:, None] * tau[None, :]
    re_k = 1.0 / (1.0 + wt ** 2)
    im_k = -wt / (1.0 + wt ** 2)
    n = len(omega)
    one, zero = np.ones((n, 1)), np.zeros((n, 1))
    re = np.hstack([one, re_k, zero, zero])
    im = np.hstack([zero, im_k, (-1.0 / omega)[:, None], omega[:, None]])
    return np.vstack([re * weight[:, None], im * weight[:, None]])


def _fit(f: np.ndarray, Z: np.ndarray, M: int) -> tuple[np.ndarray, np.ndarray, float]:
    omega = 2.0 * np.pi * f
    tau = np.logspace(np.log10(1.0 / (2 * np.pi * f.max())), np.log10(1.0 / (2 * np.pi * f.min())), M)
    w = 1.0 / np.abs(Z)
    A = _design(omega, tau, w)
    b = np.r_[Z.real * w, Z.imag * w]
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    Rk = x[1:M + 1]
    pos, neg = np.abs(Rk[Rk >= 0]).sum(), np.abs(Rk[Rk < 0]).sum()
    mu = 1.0 - neg / pos if pos > 0 else -np.inf
    return x, tau, mu


def kk_model(f: np.ndarray, x: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Z_KK at the frequencies f for the coefficients x = (R0, R_1..R_M, a_C, a_L)."""
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    M = len(tau)
    Z = x[0] + (x[1:M + 1][None, :] / (1.0 + 1j * omega[:, None] * tau[None, :])).sum(axis=1)
    return Z + x[M + 1] / (1j * omega) + x[M + 2] * 1j * omega


def lin_kk(f, Z, mu_crit: float = cfg.KK_MU, m_start: int = cfg.KK_M_START,
           m_max: int | None = None) -> dict:
    """
    Lin-KK test of one spectrum (see module docstring). Returns
    dict(M, mu, converged, coef, tau, Z_fit, res_re, res_im [%], max_res [%],
    f_max_res) where max_res is the largest |residual| of either part.
    """
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    m_max = m_max or len(f)
    M, mu, converged = m_start, np.inf, False
    x = tau = None
    for M in range(m_start, max(m_start, m_max) + 1):
        x, tau, mu = _fit(f, Z, M)
        if mu < mu_crit:
            converged = True
            break
    if not converged:
        log.warning("Lin-KK: mu did not fall below %.2f up to M = %d (mu = %.3f)", mu_crit, M, mu)
    Zk = kk_model(f, x, tau)
    res_re = 100.0 * (Z.real - Zk.real) / np.abs(Z)
    res_im = 100.0 * (Z.imag - Zk.imag) / np.abs(Z)
    worst = np.maximum(np.abs(res_re), np.abs(res_im))
    k = int(np.argmax(worst))
    return dict(M=M, mu=float(mu), converged=converged, coef=x, tau=tau, Z_fit=Zk,
                res_re=res_re, res_im=res_im, max_res=float(worst[k]), f_max_res=float(f[k]))


def linkk_impedance_py_works() -> tuple[bool, str]:
    """Whether impedance.validation.linKK runs (it raises a NameError in 1.7.1)."""
    try:
        from impedance.validation import linKK
        f = np.logspace(4, -1, 21)
        Z = 0.2 + 1.0 / (1j * 2 * np.pi * f * 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            linKK(f, Z, c=cfg.KK_MU, max_M=20)
        return True, "impedance.validation.linKK works"
    except Exception as exc:  # noqa: BLE001  (any failure means: use the own implementation)
        return False, f"impedance.validation.linKK fails ({type(exc).__name__}: {exc})"
