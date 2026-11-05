"""Hardcase 挖掘：相判错、低置信度、参数误差大的样本优先进入 DAPO。"""

from dataclasses import dataclass
from typing import Any, Dict, List

from ..data.schema import Prediction


@dataclass
class HardcaseConfig:
    confidence_threshold: float = 0.55
    xi_rtol: float = 0.25
    n_atol: float = 0.40
    max_samples: int = 600

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardcaseConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in allowed})


def prediction_gap(sample, pred: Prediction) -> Dict[str, float]:
    truth = sample.params
    def rel(key: str) -> float:
        value, target = float(pred.to_dict().get(key, float("nan"))), float(truth.get(key, 1.0))
        return abs(value - target) / max(abs(target), 1e-9)

    return {
        "phase_wrong": 0.0 if pred.phase == sample.phase else 1.0,
        "low_confidence": 1.0 - float(pred.confidence),
        "xi_rel_error": rel("xi"),
        "n_abs_error": abs(float(pred.n) - float(truth.get("n", 0.0))),
        "q_star_rel_error": rel("q_star"),
    }


def is_hard(sample, pred: Prediction, cfg: HardcaseConfig = None) -> bool:
    cfg = cfg or HardcaseConfig()
    gap = prediction_gap(sample, pred)
    return bool(
        gap["phase_wrong"] > 0
        or gap["low_confidence"] > 1.0 - cfg.confidence_threshold
        or gap["xi_rel_error"] > cfg.xi_rtol
        or gap["n_abs_error"] > cfg.n_atol
        or gap["q_star_rel_error"] > cfg.xi_rtol
    )


def mine_hardcases(samples: List[Any], analyst,
                   cfg: HardcaseConfig = None) -> List[Any]:
    cfg = cfg or HardcaseConfig()
    scored = []
    for sample in samples:
        pred = analyst.predict(sample)
        gap = prediction_gap(sample, pred)
        score = (2.0 * gap["phase_wrong"] + gap["low_confidence"]
                 + gap["xi_rel_error"] + 0.5 * gap["n_abs_error"])
        if is_hard(sample, pred, cfg) or sample.hardcase:
            scored.append((score, sample))
    scored.sort(key=lambda item: -item[0])
    return [sample for _, sample in scored[:cfg.max_samples]]
