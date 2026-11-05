"""有序相峰比判别与（简化的）嵌段共聚物相图。

峰位比参考：层状相 1:2:3:4、六方柱状相 1:sqrt(3):2:sqrt(7):3、
双连续 Ia3d gyroid 相 sqrt(6):sqrt(8):sqrt(14):sqrt(16):sqrt(20)。
"""

import math
import numpy as np
from typing import Dict, Sequence, Tuple

PHASES = ("lamellar", "cylindrical", "gyroid", "disordered", "critical")

PHASE_RATIOS: Dict[str, Tuple[float, ...]] = {
    "lamellar": (1.0, 2.0, 3.0, 4.0),
    "cylindrical": (1.0, math.sqrt(3), 2.0, math.sqrt(7), 3.0),
    "gyroid": tuple(math.sqrt(r / 6.0) for r in (6.0, 8.0, 14.0, 16.0, 20.0)),
    "disordered": (1.0,),
    "critical": (1.0,),
}


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, window | 1)
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, np.ones(window) / float(window), mode="valid")


def find_peaks(q, intensity, smooth: int = 3, min_rel: float = 0.12,
               max_peaks: int = 6) -> np.ndarray:
    """平滑 + 宽带基线提升度（prominence）筛选 Bragg 峰，返回按 q 升序的峰位。"""
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    series = _moving_average(intensity, smooth)
    baseline = _moving_average(intensity, 25)
    lo, hi = max(2, smooth), series.size - max(2, smooth)
    threshold = min_rel * float(series.max())
    candidates = [i for i in range(lo, hi)
                  if series[i] > series[i - 1] and series[i] >= series[i + 1]
                  and series[i] > threshold
                  and series[i] > 1.12 * baseline[i]]
    dq = float(np.median(np.diff(q))) if q.size > 1 else 1.0
    min_separation = 2.0 * dq
    kept = []
    for index in sorted(candidates, key=lambda j: -series[j]):
        if all(abs(q[index] - q[j]) > min_separation for j in kept):
            kept.append(index)
        if len(kept) >= max_peaks:
            break
    return np.array(sorted(q[i] for i in kept), dtype=float)


def ratio_distance(peaks: Sequence[float], ratios: Sequence[float]) -> float:
    """观测峰比与模板峰比的双向匹配偏差（对噪声导致的多余峰稳健）。"""
    if len(peaks) == 0:
        return 1.0
    raw = np.sort(np.asarray(peaks, dtype=float))
    target = np.asarray(ratios, dtype=float)

    def distance_for(observed: np.ndarray) -> float:
        per_observed = np.asarray([min(abs(o - t) / max(t, 1e-9) for t in target)
                                   for o in observed[:8]], dtype=float)
        per_target = np.asarray([min(abs(o - t) / max(t, 1e-9) for o in observed)
                                 for t in target], dtype=float)
        extra = int((per_observed > 0.20).sum())
        missing = int((per_target > 0.15).sum())
        matched = np.sort(per_observed)[:max(1, len(target))]
        return float(0.5 * np.mean(matched) + 0.5 * np.mean(per_target)
                     + 0.12 * extra + 0.25 * missing / len(target))

    # 前几个峰都可能是真实主峰（伪峰仍可能存在），取最优归一化
    return min(distance_for(raw / q0) for q0 in raw[:3])


def classify_by_ratios(peaks: Sequence[float]) -> Tuple[str, Dict[str, float]]:
    distances = {phase: ratio_distance(peaks, ratios)
                 for phase, ratios in PHASE_RATIOS.items()}
    return min(distances, key=distances.get), distances


def phase_from_composition(chi_n: float, f: float) -> str:
    """用于基准评测的解析相图（chi*N 为分离强度，f 为嵌段体积分数）。"""
    if chi_n < 9.7:
        return "disordered"
    if chi_n <= 11.3:
        return "critical"
    if f < 0.30 or f > 0.70:
        return "cylindrical"
    if 0.40 <= f <= 0.60:
        return "lamellar"
    return "gyroid"


def phase_diagram_grid(chi_values, f_values) -> np.ndarray:
    chi_values = np.asarray(chi_values, dtype=float)
    f_values = np.asarray(f_values, dtype=float)
    grid = np.empty((chi_values.size, f_values.size), dtype=object)
    for i, chi_n in enumerate(chi_values):
        for j, f in enumerate(f_values):
            grid[i, j] = phase_from_composition(float(chi_n), float(f))
    return grid


def _boundary_mask(grid: np.ndarray) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] \
                        and grid[ni, nj] != grid[i, j]:
                    mask[i, j] = True
                    break
    return mask


def boundary_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    """相边界位置 F1（同时衡量自动绘制相图的边界准确性）。"""
    a, b = _boundary_mask(np.asarray(truth, dtype=object)), \
        _boundary_mask(np.asarray(pred, dtype=object))
    tp = float(np.logical_and(a, b).sum())
    fp = float(np.logical_and(~a, b).sum())
    fn = float(np.logical_and(a, ~b).sum())
    denominator = 2.0 * tp + fp + fn
    return 1.0 if denominator == 0 else 2.0 * tp / denominator
