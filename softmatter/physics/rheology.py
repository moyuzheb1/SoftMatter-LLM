"""流变学 master curve：WLF 时温等效平移 + Maxwell 模谱。"""

import numpy as np
from typing import Dict, List


def wlf_log10_shift(temperature: float, t_ref: float = 25.0,
                    c1: float = 17.44, c2: float = 51.6) -> float:
    """log10(a_T) = -C1 (T - Tref) / (C2 + T - Tref)。"""
    delta = temperature - t_ref
    if c2 + delta <= 1e-6:
        return 0.0
    return float(-c1 * delta / (c2 + delta))


def maxwell_response(omega, g0: float, tau: float):
    x = np.asarray(omega, dtype=float) * tau
    return g0 * x ** 2 / (1.0 + x ** 2), g0 * x / (1.0 + x ** 2)


def simulate_rheology(xi: float, temperature: float, rng,
                      n_points: int = 48) -> Dict[str, object]:
    """用关联长度驱动松弛时间，生成单温度频率扫描（含测量噪声）。"""
    omega = np.logspace(-2.0, 3.0, n_points)
    tau = 0.05 * (max(xi, 1.0) / 20.0) ** 2 * np.exp(-0.03 * (temperature - 25.0))
    g_prime, g_double_prime = maxwell_response(omega, 1.0e3, tau)
    noise = 1.0 + 0.03 * rng.standard_normal(omega.size)
    return {
        "omega": omega.tolist(),
        "g_prime": (g_prime * noise).tolist(),
        "g_double_prime": (g_double_prime * noise).tolist(),
        "temperature": float(temperature),
        "a_T": float(10.0 ** wlf_log10_shift(temperature)),
        "tau": float(tau),
    }


def build_master_curve(curves: List[Dict[str, object]], t_ref: float = 25.0,
                       n_points: int = 120) -> Dict[str, object]:
    """按 a_T 平移各温度频率扫描并叠加成参考温度下的 master curve。"""
    omega_ref = np.logspace(-3.0, 4.0, n_points)
    log_ref = np.log(omega_ref)
    gp_stack, gpp_stack, shifts = [], [], []
    for curve in curves:
        omega = np.asarray(curve["omega"], dtype=float)
        a_t = float(curve.get("a_T", 1.0))
        reduced = omega * a_t / max(float(10.0 ** wlf_log10_shift(t_ref)), 1e-12)
        order = np.argsort(reduced)
        gp_stack.append(np.interp(log_ref, np.log(reduced[order]),
                                  np.asarray(curve["g_prime"], dtype=float)[order],
                                  left=np.nan, right=np.nan))
        gpp_stack.append(np.interp(log_ref, np.log(reduced[order]),
                                   np.asarray(curve["g_double_prime"], dtype=float)[order],
                                   left=np.nan, right=np.nan))
        shifts.append(a_t)
    with np.errstate(all="ignore"):
        gp = np.nanmean(np.asarray(gp_stack, dtype=float), axis=0)
        gpp = np.nanmean(np.asarray(gpp_stack, dtype=float), axis=0)
    return {
        "omega": omega_ref.tolist(),
        "g_prime": np.nan_to_num(gp).tolist(),
        "g_double_prime": np.nan_to_num(gpp).tolist(),
        "shift_factors": shifts,
        "t_ref": float(t_ref),
    }


def crossing_omega(master: Dict[str, object]) -> float:
    """G'=G'' 交叉频率，用于与散射得到的关联长度交叉验证。"""
    gp = np.asarray(master["g_prime"], dtype=float)
    gpp = np.asarray(master["g_double_prime"], dtype=float)
    omega = np.asarray(master["omega"], dtype=float)
    return float(omega[int(np.argmin(np.abs(gp - gpp)))])
