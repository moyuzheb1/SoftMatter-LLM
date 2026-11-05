"""一维 SAXS/SANS 散射强度的解析模型与反演估计器。

约定：q 单位为 nm^-1，I 为任意单位 (a.u.)。高 q 区满足 Porod / 幂律衰减
I(q) ~ q^-n；低 q 区用 Ornstein-Zernike (OZ) 形式描述临界涨落。
"""

import numpy as np
from typing import Dict, Sequence


def ornstein_zernike(q, xi: float, amplitude: float = 1.0):
    """临界涨落 / 关联长度模型 I(q) = A / (1 + q^2 xi^2)。"""
    q = np.asarray(q, dtype=float)
    return amplitude / (1.0 + (q * max(xi, 1e-6)) ** 2)


def guinier(q, rg: float, amplitude: float = 1.0):
    """Guinier 区近似 I(q) = A exp(-q^2 Rg^2 / 3)。"""
    q = np.asarray(q, dtype=float)
    return amplitude * np.exp(-(q * rg) ** 2 / 3.0)


def power_law(q, n: float, amplitude: float = 1.0, q_ref: float = 1.0):
    q = np.asarray(q, dtype=float)
    return amplitude * np.power(q / q_ref, -n)


def gaussian(q, center: float, width: float):
    q = np.asarray(q, dtype=float)
    return np.exp(-0.5 * ((q - center) / max(width, 1e-9)) ** 2)


def lorentzian(q, center: float, width: float):
    q = np.asarray(q, dtype=float)
    return 1.0 / (1.0 + ((q - center) / max(width, 1e-9)) ** 2)


def peak_train(q, q_star: float, ratios: Sequence[float], width: float,
               amplitude: float, shape: str = "gaussian"):
    """有序相 Bragg 峰序列；高阶峰强度按 k^-1.3 衰减。"""
    q = np.asarray(q, dtype=float)
    out = np.zeros_like(q)
    fn = lorentzian if shape == "lorentzian" else gaussian
    for k, ratio in enumerate(ratios):
        out += amplitude / (1.0 + k) ** 1.3 * fn(q, q_star * ratio,
                                                 width * max(ratio, 1.0))
    return out


def scattering_profile(q, params: Dict[str, float]):
    """合成散射谱：OZ 低 q 背景 + 有序峰 + 平滑接入的幂律尾 + 背景。

    尾项写成 ``A (q/q_c)^-n * (q/q_c)^k / (1 + (q/q_c)^k)``，
    保证低 q 不发散、高 q 严格趋于 q^-n（便于 Porod 指数反演）。
    """
    q = np.asarray(q, dtype=float)
    xi = float(params.get("xi", 20.0))
    n = float(params.get("n", 4.0))
    q_star = float(params.get("q_star", 0.2))
    q_c = float(params.get("q_c", max(3.0 * q_star, 0.4)))
    ratios = tuple(params.get("ratios", (1.0,)))
    amplitude = float(params.get("amplitude", 1.0))
    background = float(params.get("background", 1e-4))
    peak_amp = float(params.get("peak_amp", 1.5))
    peak_width = float(params.get("peak_width", 0.015))
    tail_amp = float(params.get("tail_amp", 0.20))
    shape = str(params.get("peak_shape", "gaussian"))

    # 幂律尾以 q_c 为参考幅度，并在低 q 用 12 次幂平滑关闭；
    # OZ 背景加高斯截断，避免 q^-2 在高 q 掩盖 Porod 区。
    x = np.power(q / q_c, 12.0)
    tail = tail_amp * np.power(q / q_c, -n) * x / (1.0 + x)
    broad = ornstein_zernike(q, xi, amplitude) * np.exp(-np.power(q / 0.6, 2.0))
    peaks = peak_train(q, q_star, ratios, peak_width, peak_amp, shape)
    return broad + peaks + tail + background


def estimate_porod_exponent(q, intensity, frac: float = 0.18) -> float:
    """高 q 窗口内 log-log 线性拟合，返回幂律指数 n = -slope。"""
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    mask = np.isfinite(q) & np.isfinite(intensity) & (q > 0) & (intensity > 0)
    q, intensity = q[mask], intensity[mask]
    if q.size < 8:
        return float("nan")
    window = max(8, int(q.size * frac))
    qs, ins = q[-window:], intensity[-window:]
    log_i = np.log(ins)
    if log_i.size >= 5:
        kernel = np.ones(5) / 5.0
        log_i = np.convolve(log_i, kernel, mode="same")
        log_i[:2], log_i[-2:] = log_i[2], log_i[-3]
    # 用局部斜率的中位数抑制高阶 Bragg 峰与噪声对 Porod 拟合的污染
    slopes = np.diff(log_i) / np.diff(np.log(qs))
    slopes = slopes[np.isfinite(slopes) & (np.abs(slopes) < 10.0)]
    if slopes.size == 0:
        return float("nan")
    return float(-np.median(slopes))


def estimate_xi(q, intensity, frac: float = 0.18, q_cut: float = 0.6) -> float:
    """低 q 区用 OZ x 仪器截断模型做一维网格最小二乘反演 xi。

    模型 I(q) = A / (1 + q^2 xi^2) * exp(-(q/q_cut)^2)。
    相比经典的 1/I ~ q^2 线性化，它对 q 窗口边缘的弯曲更稳健。
    """
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    mask = np.isfinite(q) & np.isfinite(intensity) & (q > 0) & (intensity > 0)
    q, intensity = q[mask], intensity[mask]
    if q.size < 8:
        return float("nan")
    window = max(8, int(q.size * frac))
    qs, ins = q[:window], intensity[:window]
    cutoff = np.exp(-np.power(qs / q_cut, 2.0))
    candidates = np.exp(np.linspace(np.log(1.0), np.log(300.0), 140))
    best_xi, best_loss = 10.0, float("inf")
    denominator = max(float(np.mean(ins ** 2)), 1e-12)
    for xi in candidates:
        shape = cutoff / (1.0 + np.power(qs * xi, 2.0))
        amplitude = float(np.dot(shape, ins) / max(np.dot(shape, shape), 1e-12))
        loss = float(np.mean((ins - amplitude * shape) ** 2) / denominator)
        if loss < best_loss:
            best_loss, best_xi = loss, float(xi)
    return float(np.clip(best_xi, 0.5, 300.0))
