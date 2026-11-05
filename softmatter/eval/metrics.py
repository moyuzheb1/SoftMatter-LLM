"""评测指标：结构判定准确率、参数提取准确率、相图自动绘制评分。"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from ..physics.phase import boundary_f1, phase_diagram_grid


@dataclass
class EvalConfig:
    xi_rtol: float = 0.15
    n_atol: float = 0.25
    q_rtol: float = 0.15
    chi_bins: int = 8
    f_bins: int = 7

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in allowed})


def _get(record: Any, key: str) -> Any:
    return record[key] if isinstance(record, dict) else getattr(record, key)


def phase_accuracy(records: List[Any]) -> float:
    if not records:
        return 0.0
    return float(np.mean([_get(r, "pred").phase == _get(r, "sample").phase
                          for r in records]))


def parameter_hits(records: List[Any], cfg: EvalConfig = None) -> np.ndarray:
    cfg = cfg or EvalConfig()
    hits = []
    for record in records:
        sample, pred = _get(record, "sample"), _get(record, "pred")
        truth = sample.params
        try:
            xi_ok = abs(pred.xi - truth["xi"]) / max(abs(truth["xi"]), 1e-9) <= cfg.xi_rtol
            n_ok = abs(pred.n - truth["n"]) <= cfg.n_atol
            q_ok = abs(pred.q_star - truth["q_star"]) / max(abs(truth["q_star"]), 1e-9) <= cfg.q_rtol
        except (KeyError, TypeError, ValueError):
            xi_ok = n_ok = q_ok = False
        hits.append(bool(np.isfinite(pred.xi) and np.isfinite(pred.n)
                         and np.isfinite(pred.q_star) and xi_ok and n_ok and q_ok))
    return np.asarray(hits, dtype=bool)


def parameter_accuracy(records: List[Any], cfg: EvalConfig = None) -> float:
    hits = parameter_hits(records, cfg)
    return float(hits.mean()) if hits.size else 0.0


def per_regime_breakdown(records: List[Any], cfg: EvalConfig = None) -> Dict[str, Dict[str, float]]:
    cfg = cfg or EvalConfig()
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for record in records:
        grouped[_get(record, "sample").regime].append(record)
    return {regime: {"n": len(group), "phase_accuracy": phase_accuracy(group),
                     "parameter_accuracy": parameter_accuracy(group, cfg)}
            for regime, group in sorted(grouped.items())}


def bootstrap_ci(values: List[float], n_boot: int = 300,
                 seed: int = 0) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = [float(np.mean(values[rng.integers(0, values.size, values.size)]))
             for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def phase_diagram_from_records(records: List[Any], cfg: EvalConfig = None
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 (chi*N, f) 网格汇总模型预测，构成待评分的自动绘制相图。"""
    cfg = cfg or EvalConfig()
    chi_edges = np.linspace(7.0, 50.0, cfg.chi_bins + 1)
    f_edges = np.linspace(0.20, 0.80, cfg.f_bins + 1)
    chi_centers = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    f_centers = 0.5 * (f_edges[:-1] + f_edges[1:])
    truth = phase_diagram_grid(chi_centers, f_centers)
    pred = np.full(truth.shape, "", dtype=object)
    coverage = np.zeros(truth.shape, dtype=bool)
    buckets: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for record in records:
        params = _get(record, "sample").params
        if "chi_n" not in params or "f" not in params:
            continue
        i = int(np.clip(np.digitize(params["chi_n"], chi_edges) - 1, 0, cfg.chi_bins - 1))
        j = int(np.clip(np.digitize(params["f"], f_edges) - 1, 0, cfg.f_bins - 1))
        buckets[(i, j)].append(_get(record, "pred").phase)
    for (i, j), votes in buckets.items():
        pred[i, j] = Counter(votes).most_common(1)[0][0]
        coverage[i, j] = True
    return truth, pred, coverage


def phase_diagram_score(truth: np.ndarray, pred: np.ndarray,
                        coverage: np.ndarray) -> Dict[str, float]:
    cell_accuracy = float(np.mean(truth[coverage] == pred[coverage])) if coverage.any() else 0.0
    f1 = boundary_f1(truth, pred)
    return {"cell_accuracy": cell_accuracy, "boundary_f1": f1,
            "score": 0.6 * cell_accuracy + 0.4 * f1,
            "coverage": float(coverage.mean())}


def summarize(records: List[Any], cfg: EvalConfig = None) -> Dict[str, Any]:
    cfg = cfg or EvalConfig()
    truth, pred, coverage = phase_diagram_from_records(records, cfg)
    correct = [1.0 if _get(r, "pred").phase == _get(r, "sample").phase else 0.0
               for r in records]
    return {
        "n_samples": len(records),
        "phase_accuracy": phase_accuracy(records),
        "phase_accuracy_ci95": bootstrap_ci(correct),
        "parameter_accuracy": parameter_accuracy(records, cfg),
        "per_regime": per_regime_breakdown(records, cfg),
        "phase_diagram": phase_diagram_score(truth, pred, coverage),
        "hardcase_phase_accuracy": phase_accuracy(
            [r for r in records if _get(r, "sample").hardcase]),
    }
