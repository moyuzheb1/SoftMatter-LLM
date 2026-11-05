"""LLM 输出解析：兼容 ```json 围栏、<answer> 标签与 CoT 末尾 JSON。"""

import json
import math
import re
from typing import Any, Dict

import numpy as np

from ..data.schema import PHASES, Prediction


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = text or ""
    if "<answer>" in cleaned and "</answer>" in cleaned:
        cleaned = cleaned.split("<answer>", 1)[1].split("</answer>", 1)[0]
    cleaned = re.sub(r"```(?:json)?", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(cleaned[start:end + 1])


def parse_prediction(text: str) -> Prediction:
    try:
        data = extract_json(text)
    except (ValueError, json.JSONDecodeError):
        return Prediction(raw=text or "")
    pred = Prediction.from_dict(data)
    pred.raw = text or ""
    if pred.phase not in PHASES:
        pred.phase = "disordered"
    if not np.isfinite(pred.d_spacing) and np.isfinite(pred.q_star) and pred.q_star > 0:
        pred.d_spacing = 2.0 * math.pi / pred.q_star
    return pred


def format_answer(pred: Prediction) -> str:
    return json.dumps(pred.to_dict(), ensure_ascii=False)
