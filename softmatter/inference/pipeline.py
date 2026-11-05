"""端到端推理：prompt 构造 -> 模型输出 -> JSON 解析 -> 奖励打分。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..analyst import ReferenceAnalyst
from .parser import format_answer, parse_prediction
from .prompt import build_messages, render_prompt
from .vision import render_curve_png


@dataclass
class AnalysisResult:
    sample_id: str
    prompt: str
    prediction: Any
    reward: Any
    cot: List[str] = field(default_factory=list)
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"sample_id": self.sample_id, "prediction": self.prediction.to_dict(),
                "reward": self.reward.to_dict(), "cot": self.cot,
                "image_path": self.image_path}


def analyze(sample, analyst=None, config: Dict[str, Any] = None,
            image_path: str = None) -> AnalysisResult:
    from ..training.reward import reward  # 延迟导入避免包循环

    config = config or {}
    analyst = analyst or ReferenceAnalyst(stage=3)
    rendered = image_path
    if config.get("render_image") and rendered:
        rendered = render_curve_png(sample, rendered)
    messages = build_messages(sample, config.get("prompt", {}), rendered)
    text = render_prompt(messages)
    raw_answer = format_answer(analyst.predict(sample))
    prediction = parse_prediction(raw_answer)
    return AnalysisResult(
        sample_id=sample.id,
        prompt=text,
        prediction=prediction,
        reward=reward(prediction, sample),
        cot=list(sample.cot),
        image_path=rendered,
    )


def batch_analyze(samples: List[Any], analyst=None,
                  config: Dict[str, Any] = None) -> List[AnalysisResult]:
    analyst = analyst or ReferenceAnalyst(stage=3)
    return [analyze(sample, analyst, config) for sample in samples]
