"""可直接运行的参考分析器（CPU baseline / 策略载体）。

真实系统中该位置是 SFT -> DAPO 之后的多模态大模型；本模块提供一个
基于物理特征 + 可训练 softmax/线性头的等价替身，使整条流水线在没有
GPU、没有模型权重、没有真实实验数据时也能被完整复现与测试。
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .data.schema import Prediction
from .physics.models import estimate_porod_exponent, estimate_xi
from .physics.phase import _moving_average, classify_by_ratios, find_peaks

PHASE_ORDER = ("lamellar", "cylindrical", "gyroid", "disordered", "critical")

FEATURE_NAMES = (
    "bias", "log_q_star", "porod_n", "log_xi", "peak_count", "log_snr",
    "lowq_slope", "peak_prominence", "dist_lamellar", "dist_cylindrical",
    "dist_gyroid", "dist_disorder", "log_d_spacing", "highq_curvature",
    "critical_indicator", "cot_consistency", "chi_n_normalized", "f_asymmetry",
)
N_FEATURES = len(FEATURE_NAMES)


def _safe(value: float, default: float) -> float:
    return float(value) if np.isfinite(value) else float(default)


def _refine_peak(q: np.ndarray, intensity: np.ndarray, q0: float) -> float:
    """扣除缓变背景后对主峰做抛物线插值，修正宽峰被低 q 背景拉偏的峰位。"""
    baseline = _moving_average(intensity, 25)
    excess = np.clip(intensity - baseline, 0.0, None)
    mask = (q > 0.5 * q0) & (q < 1.7 * q0)
    if int(mask.sum()) < 5:
        return q0
    qs, ys = q[mask], excess[mask]
    center = int(np.argmax(ys))
    lo, hi = max(0, center - 3), min(qs.size, center + 4)
    if hi - lo < 4:
        return float(qs[center])
    coefficients = np.polyfit(qs[lo:hi], ys[lo:hi], 2)
    if coefficients[0] >= 0:
        return float(qs[center])
    vertex = -coefficients[1] / (2.0 * coefficients[0])
    return float(vertex) if 0.5 * q0 < vertex < 1.7 * q0 else float(qs[center])


def curve_features(sample) -> Dict[str, Any]:
    """从谱图上抽取 16 维可解释特征所需的中间量。"""
    q = np.asarray(sample.q, dtype=float)
    intensity = np.asarray(sample.intensity, dtype=float)
    peaks = find_peaks(q, intensity)
    q_star = float(peaks[0]) if peaks.size else float(q[int(np.argmax(intensity))])
    q_star = _refine_peak(q, intensity, q_star)
    n_est = _safe(estimate_porod_exponent(q, intensity), 2.0)
    xi_est = _safe(estimate_xi(q, intensity), 10.0)
    heuristic, distances = classify_by_ratios(peaks)

    tail = max(5, int(0.10 * intensity.size))
    high_q, high_i = q[-tail:], intensity[-tail:]
    good = (high_q > 0) & (high_i > 0)
    curvature = 0.0
    if good.sum() >= 6:
        curvature = float(np.polyfit(np.log(high_q[good]), np.log(high_i[good]), 2)[0])
    low_n = max(6, int(0.20 * intensity.size))
    low_good = (q[:low_n] > 0) & (intensity[:low_n] > 0)
    low_slope = 0.0
    if low_good.sum() >= 4:
        low_slope = float(np.polyfit(np.log(q[:low_n][low_good]),
                                    np.log(intensity[:low_n][low_good]), 1)[0])
    noise = float(np.std(np.diff(intensity[-tail:]))) / math.sqrt(2.0) + 1e-9
    snr = float(np.clip(intensity.max() / noise, 1.0, 1e6))
    peak_index = int(np.argmin(np.abs(q - q_star)))
    prominence = float(intensity[peak_index] / (np.median(intensity) + 1e-9))
    critical_indicator = float(np.exp(-((n_est - 2.0) / 0.8) ** 2))
    consistency = float(np.clip(1.0 - abs(distances[heuristic] - distances["disordered"]), 0.0, 1.0))
    return {
        "peaks": peaks,
        "q_star": q_star,
        "n_est": n_est,
        "xi_est": xi_est,
        "heuristic": heuristic,
        "distances": distances,
        "snr": snr,
        "lowq_slope": low_slope,
        "curvature": curvature,
        "prominence": prominence,
        "critical_indicator": critical_indicator,
        "consistency": consistency,
    }


def stage_mask(level: int = 3) -> np.ndarray:
    """课程学习特征门控：1=物理概念，2=曲线拟合，3=CoT 复杂推理。"""
    mask = np.zeros(N_FEATURES, dtype=float)
    mask[:4 if level == 1 else (8 if level == 2 else N_FEATURES)] = 1.0
    return mask


def extract_features(sample, level: int = 3,
                     observations: Optional[Dict[str, Any]] = None) -> np.ndarray:
    obs = observations if observations is not None else curve_features(sample)
    distances = obs["distances"]
    q_star = max(obs["q_star"], 1e-6)
    features = np.zeros(N_FEATURES, dtype=float)
    features[0] = 1.0
    features[1] = math.log(q_star)
    features[2] = np.clip(obs["n_est"], 0.0, 6.0) / 4.0
    features[3] = math.log(max(obs["xi_est"], 0.5))
    features[4] = min(len(obs["peaks"]), 6) / 6.0
    features[5] = math.log10(max(obs["snr"], 1.0)) / 4.0
    features[6] = np.clip(obs["lowq_slope"], -5.0, 5.0) / 4.0
    features[7] = np.clip(math.log(max(obs["prominence"], 1e-3)), -3.0, 5.0) / 4.0
    features[8] = min(distances["lamellar"], 3.0) / 2.0
    features[9] = min(distances["cylindrical"], 3.0) / 2.0
    features[10] = min(distances["gyroid"], 3.0) / 2.0
    features[11] = min(distances["disordered"], 3.0) / 2.0
    features[12] = math.log(2.0 * math.pi / q_star)
    features[13] = np.clip(obs["curvature"], -3.0, 3.0) / 2.0
    features[14] = obs["critical_indicator"]
    features[15] = obs["consistency"]
    features[16] = float(sample.params.get("chi_n", 20.0)) / 30.0
    features[17] = abs(float(sample.params.get("f", 0.5)) - 0.5) * 2.0
    return np.nan_to_num(features * stage_mask(level), nan=0.0, posinf=0.0, neginf=0.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp = np.exp(np.clip(shifted, -60.0, 60.0))
    return exp / max(float(exp.sum()), 1e-12)


class ReferenceAnalyst:
    """特征驱动的 phase softmax + xi/n 回归头；支持 GRPO 动作注入。"""

    def __init__(self, stage: int = 1, seed: int = 0):
        self.W = np.zeros((len(PHASE_ORDER), N_FEATURES), dtype=float)
        self.theta_xi = np.zeros(N_FEATURES, dtype=float)
        self.theta_n = np.zeros(N_FEATURES, dtype=float)
        self.stage = int(stage)
        self.seed = int(seed)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def curve_features(self, sample) -> Dict[str, Any]:
        if sample.id not in self._cache:
            self._cache[sample.id] = curve_features(sample)
        return self._cache[sample.id]

    def features(self, sample, level: Optional[int] = None) -> np.ndarray:
        level = self.stage if level is None else int(level)
        return extract_features(sample, level, self.curve_features(sample))

    def base_logits(self, sample, observations: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """物理先验（峰比模板 + 峰数 + 临界指数），SFT 学习的是其残差修正。"""
        obs = observations if observations is not None else self.curve_features(sample)
        distances = obs["distances"]
        logits = np.array([-4.0 * distances[p] for p in PHASE_ORDER], dtype=float)
        critical = obs["critical_indicator"]
        peak_count = min(len(obs["peaks"]), 6)
        logits[3] += 0.8 - 0.5 * peak_count + 0.5 * (1.0 - critical)  # 单峰 -> 无序相
        logits[4] += 2.0 * critical - 0.8 - 0.5 * peak_count          # n≈2 且单峰 -> 临界涨落
        return logits

    def phase_distribution(self, sample, level: Optional[int] = None) -> np.ndarray:
        level = self.stage if level is None else int(level)
        observations = self.curve_features(sample)
        x = extract_features(sample, level, observations)
        return _softmax(self.base_logits(sample, observations) + self.W @ x)

    def predict(self, sample, action=None, level: Optional[int] = None) -> Prediction:
        level = self.stage if level is None else int(level)
        probabilities = self.phase_distribution(sample, level)
        phase = PHASE_ORDER[int(np.argmax(probabilities))]
        observations = self.curve_features(sample)
        x = extract_features(sample, level, observations)
        delta_xi, delta_n = 0.0, 0.0
        if action is not None:
            delta_xi, delta_n = float(action[0]), float(action[1])
        xi = observations["xi_est"] * math.exp(
            float(np.dot(self.theta_xi, x)) + 0.25 * delta_xi)
        n = observations["n_est"] + float(np.dot(self.theta_n, x)) + 0.35 * delta_n
        q_star = observations["q_star"]
        return Prediction(
            phase=phase,
            xi=float(np.clip(xi, 0.5, 300.0)),
            n=float(np.clip(n, 0.5, 6.0)),
            q_star=float(q_star),
            d_spacing=float(2.0 * math.pi / max(q_star, 1e-9)),
            confidence=float(np.max(probabilities)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "seed": self.seed,
                "W": self.W.tolist(), "theta_xi": self.theta_xi.tolist(),
                "theta_n": self.theta_n.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceAnalyst":
        analyst = cls(stage=int(data.get("stage", 3)), seed=int(data.get("seed", 0)))
        analyst.W = np.asarray(data.get("W", analyst.W.tolist()), dtype=float)
        analyst.theta_xi = np.asarray(data.get("theta_xi", analyst.theta_xi.tolist()), dtype=float)
        analyst.theta_n = np.asarray(data.get("theta_n", analyst.theta_n.tolist()), dtype=float)
        return analyst

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ReferenceAnalyst":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
