"""DAPO / GRPO 的复合奖励，核心是「数值一致性奖励」。

数值一致性奖励对 xi、幂律指数 n、主峰 q* 的相对误差做指数软衰减，
并额外要求 d = 2π/q* 等物理量之间自洽，从而抑制「相判对但参数乱编」。
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Union

import numpy as np

from ..data.schema import PHASES, Prediction
from ..physics.models import estimate_porod_exponent

REQUIRED_KEYS = ("phase", "xi", "n", "q_star", "d_spacing")
ORDERED_PHASES = ("lamellar", "cylindrical", "gyroid")


@dataclass
class RewardConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "format": 0.05, "phase": 0.20, "parameter": 0.55, "consistency": 0.20})
    tolerances: Dict[str, float] = field(default_factory=lambda: {
        "xi": 0.15, "n": 0.25, "q_star": 0.15})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardConfig":
        return cls(weights=dict(data.get("weights", cls().weights)),
                   tolerances=dict(data.get("tolerances", cls().tolerances)))


@dataclass
class RewardBreakdown:
    total: float
    format: float
    phase: float
    parameter: float
    consistency: float

    def to_dict(self) -> Dict[str, float]:
        return {"total": self.total, "format": self.format, "phase": self.phase,
                "parameter": self.parameter, "consistency": self.consistency}


def _as_dict(prediction: Union[Prediction, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(prediction, Prediction) or hasattr(prediction, "to_dict"):
        return prediction.to_dict()
    return dict(prediction)


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def relative_error(predicted: float, truth: float) -> float:
    return abs(predicted - truth) / max(abs(truth), 1e-9)


def numerical_consistency_reward(pred: Dict[str, Any], truth: Dict[str, Any],
                                 cfg: RewardConfig) -> float:
    """xi / n / q* 的加权相对误差软奖励（对应「拟合参数提取」）。"""
    scores = []
    for key, tolerance in cfg.tolerances.items():
        value, target = _number(pred.get(key)), _number(truth.get(key))
        if not (np.isfinite(value) and np.isfinite(target)):
            scores.append(0.0)
            continue
        error = (abs(value - target) if key == "n" else relative_error(value, target))
        scores.append(math.exp(-error / max(tolerance, 1e-6)))
    return float(np.mean(scores)) if scores else 0.0


def physics_consistency_reward(pred: Dict[str, Any], sample, cfg: RewardConfig) -> float:
    """物理量之间的自洽性：d=2π/q*、Porod 指数、参数取值域。"""
    checks = []
    q_star, d_spacing = _number(pred.get("q_star")), _number(pred.get("d_spacing"))
    if np.isfinite(q_star) and np.isfinite(d_spacing) and q_star > 0:
        error = relative_error(d_spacing, 2.0 * math.pi / q_star)
        checks.append(math.exp(-error / 0.20))
    else:
        checks.append(0.0)
    n = _number(pred.get("n"))
    checks.append(1.0 if np.isfinite(n) and 0.8 <= n <= 4.6 else 0.0)
    xi = _number(pred.get("xi"))
    checks.append(1.0 if np.isfinite(xi) and 0 < xi <= 300.0 else 0.0)
    measured = estimate_porod_exponent(sample.q, sample.intensity)
    if np.isfinite(n) and np.isfinite(measured):
        checks.append(math.exp(-abs(n - measured) / 0.8))
    else:
        checks.append(0.0)
    return float(np.mean(checks))


def phase_reward(predicted: str, truth: str) -> float:
    if predicted == truth:
        return 1.0
    if predicted in ORDERED_PHASES and truth in ORDERED_PHASES:
        return 0.30  # 同为有序相但晶型判错，给部分Credit
    return 0.0


def format_reward(pred: Dict[str, Any]) -> float:
    """JSON 可解析、字段齐全、数值有限、相标签合法。"""
    if pred.get("phase") not in PHASES:
        return 0.0
    return float(np.mean([1.0 if np.isfinite(_number(pred.get(key))) else 0.0
                          for key in REQUIRED_KEYS[1:]]))


def reward(prediction: Union[Prediction, Dict[str, Any]], sample,
           cfg: RewardConfig = None) -> RewardBreakdown:
    cfg = cfg or RewardConfig()
    pred = _as_dict(prediction)
    truth = sample.params if hasattr(sample, "params") else dict(sample.get("params", {}))
    truth_phase = sample.phase if hasattr(sample, "phase") else str(sample.get("phase", ""))
    components = {
        "format": format_reward(pred),
        "phase": phase_reward(str(pred.get("phase", "")), truth_phase),
        "parameter": numerical_consistency_reward(pred, truth, cfg),
        "consistency": physics_consistency_reward(pred, sample, cfg),
    }
    total = sum(cfg.weights.get(key, 0.0) * value for key, value in components.items())
    return RewardBreakdown(total=float(np.clip(total, 0.0, 1.0)), **components)
