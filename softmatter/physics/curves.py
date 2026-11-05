"""物理约束驱动的数据合成器（可复现实验，用于替代不可公开的真实数据）。

生成 1D SAXS/SANS 谱与配套的流变频率扫描；真值包含 xi、n、q*、d、
chi*N、f。hardcase 通过弱化高阶峰、加宽峰形、提升噪声和基线漂移构造。
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from ..data.schema import Sample, ensure_parent
from .models import scattering_profile
from .phase import PHASE_RATIOS
from .rheology import simulate_rheology


@dataclass
class SynthConfig:
    n_samples: int = 2400
    seed: int = 7
    q_min: float = 0.03
    q_max: float = 3.0
    n_points: int = 180
    noise: float = 0.035
    hardcase_fraction: float = 0.22
    regime_weights: Dict[str, float] = field(default_factory=lambda: {
        "ordered": 0.50, "disordered": 0.32, "critical": 0.18})
    with_rheology: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SynthConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in allowed})


def sample_parameters(rng: np.random.Generator, cfg: SynthConfig,
                      index: int) -> Dict[str, object]:
    regimes = list(cfg.regime_weights.keys())
    weights = np.asarray([cfg.regime_weights[r] for r in regimes], dtype=float)
    weights = weights / weights.sum()
    regime = str(rng.choice(regimes, p=weights))

    if regime == "ordered":
        phase = str(rng.choice(["lamellar", "cylindrical", "gyroid"]))
        n = float(np.clip(rng.normal(4.0, 0.12), 3.6, 4.35))  # Porod 区
        chi_n = float(rng.uniform(11.5, 48.0))
        ratios = tuple(PHASE_RATIOS[phase])
        peak_shape, peak_amp, peak_width = "gaussian", 1.5, 0.015
    elif regime == "disordered":
        phase, n = "disordered", float(rng.uniform(2.6, 3.4))
        chi_n = float(rng.uniform(7.0, 9.6))
        ratios, peak_shape, peak_amp, peak_width = (1.0,), "lorentzian", 0.8, 0.05
    else:
        phase, n = "critical", float(rng.uniform(1.7, 2.2))
        chi_n = float(rng.uniform(9.8, 11.2))
        ratios, peak_shape, peak_amp, peak_width = (1.0,), "lorentzian", 0.9, 0.065

    # chi*N 与 f 和解析相图保持一致，使「相图自动绘制」评测有明确真值
    if phase == "lamellar":
        f = float(rng.uniform(0.41, 0.59))
    elif phase == "cylindrical":
        f = float(rng.uniform(0.21, 0.29) if rng.random() < 0.5
                  else rng.uniform(0.71, 0.79))
    elif phase == "gyroid":
        f = float(rng.uniform(0.31, 0.39) if rng.random() < 0.5
                  else rng.uniform(0.61, 0.69))
    else:
        f = float(rng.uniform(0.20, 0.80))

    xi = float(np.exp(rng.uniform(math.log(4.0), math.log(70.0))))
    if regime == "critical":
        xi *= 1.5
    d_spacing = float(np.exp(rng.uniform(math.log(12.0), math.log(60.0))))
    q_star = 2.0 * math.pi / d_spacing

    hardcase = bool(rng.random() < cfg.hardcase_fraction)
    noise = float(cfg.noise)
    drift = 0.0
    if hardcase:
        ratios = tuple(ratios[:max(1, min(2, len(ratios)))])
        peak_amp *= 0.45
        peak_width *= 1.8
        noise *= 2.6
        drift = float(rng.uniform(0.002, 0.02))

    return {
        "index": index,
        "id": "sm-%06d" % index,
        "phase": phase,
        "regime": regime,
        "hardcase": hardcase,
        "xi": xi,
        "n": n,
        "q_star": q_star,
        "d_spacing": d_spacing,
        "chi_n": chi_n,
        "f": f,
        "ratios": ratios,
        "peak_shape": peak_shape,
        "peak_amp": peak_amp,
        "peak_width": peak_width,
        "q_c": max(1.2 * q_star, 0.18),
        "amplitude": 1.0,
        "background": 1e-9,
        "tail_amp": 0.20,
        "noise": noise,
        "drift": drift,
        "temperature": float(rng.uniform(15.0, 65.0)),
    }


def synthesize(params: Dict[str, object], cfg: SynthConfig,
               rng: np.random.Generator) -> Sample:
    q = np.logspace(math.log10(cfg.q_min), math.log10(cfg.q_max), cfg.n_points)
    intensity = scattering_profile(q, params)
    noise = float(params["noise"])
    intensity = intensity * (1.0 + noise * rng.standard_normal(q.size))
    drift = float(params["drift"])
    if drift > 0:
        intensity = intensity + drift * np.exp(-q / 0.3) * float(intensity.max())
    if params["hardcase"]:
        spikes = rng.random(q.size) < 0.015
        intensity[spikes] *= 1.0 + rng.uniform(0.5, 3.0, size=int(spikes.sum()))
    intensity = np.clip(intensity, 1e-8, None)

    truth = {key: float(value) for key, value in params.items()
             if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)}
    rheology = None
    if cfg.with_rheology:
        rheology = simulate_rheology(truth["xi"], truth["temperature"], rng)
    return Sample(
        id=str(params["id"]), q=q, intensity=intensity,
        phase=str(params["phase"]), regime=str(params["regime"]),
        params=truth, temperature=float(params["temperature"]),
        rheology=rheology, hardcase=bool(params["hardcase"]), source="synthetic",
    )


def generate_dataset(cfg: SynthConfig, n_samples: Optional[int] = None,
                     seed: Optional[int] = None) -> List[Sample]:
    total = int(n_samples if n_samples is not None else cfg.n_samples)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    return [synthesize(sample_parameters(rng, cfg, i), cfg, rng)
            for i in range(total)]


def write_raw_files(samples: List[Sample], out_dir: Union[str, Path],
                    max_files: Optional[int] = None) -> int:
    """把样本写回带 '# metadata' 头的 .dat 文件，供流式读取器解析。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for sample in samples[:max_files]:
        metadata = {
            "id": sample.id,
            "phase": sample.phase,
            "regime": sample.regime,
            "params": sample.params,
            "temperature": sample.temperature,
            "hardcase": sample.hardcase,
        }
        path = ensure_parent(out_dir / (sample.id + ".dat"))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# softmatter 1D scattering profile (SAXS/SANS)\n")
            handle.write("# q_unit: nm^-1\n")
            handle.write("# metadata: " + json.dumps(metadata, ensure_ascii=False) + "\n")
            handle.write("# q intensity\n")
            for q_value, i_value in zip(sample.q, sample.intensity):
                handle.write("%.8g %.8g\n" % (q_value, i_value))
        count += 1
    return count
