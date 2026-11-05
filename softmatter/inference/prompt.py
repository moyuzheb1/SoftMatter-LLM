"""多模态 prompt：谱图 PNG (<image>) + 下采样数值序列 + 实验条件。"""

import json
from typing import Any, Dict, List, Optional

import numpy as np

SYSTEM_PROMPT = (
    "你是软物质散射/流变数据分析专家。给定 1D SAXS/SANS 谱图与实验条件，"
    "先用 CoT 逐步分析：（1）单位与背景预处理；（2）高 q Porod/幂律区拟合指数 n；"
    "（3）低 q Ornstein-Zernike 区反演相关长度 xi；（4）主峰 q* 与高阶峰比；"
    "（5）匹配层状/六方柱状/双连续 gyroid 峰比模板；（6）流变 master curve 交叉验证。"
    "最后只输出一个 JSON，字段为 phase, xi, n, q_star, d_spacing，"
    "其中 q 单位 nm^-1，xi/d_spacing 单位 nm，且必须满足 d_spacing = 2*pi/q_star。"
)


def downsample_curve(sample, n_points: int = 64) -> str:
    q = np.asarray(sample.q, dtype=float)
    intensity = np.asarray(sample.intensity, dtype=float)
    if q.size == 0:
        return ""
    indexes = np.unique(np.linspace(0, q.size - 1, min(n_points, q.size)).astype(int))
    return "; ".join("%.5g:%.5g" % (q[i], intensity[i]) for i in indexes)


def build_messages(sample, config: Dict[str, Any] = None,
                   image_path: Optional[str] = None) -> List[Dict[str, str]]:
    config = config or {}
    n_points = int(config.get("max_curve_points", 64))
    conditions = []
    if sample.temperature is not None:
        conditions.append("温度 %.1f C" % float(sample.temperature))
    if "chi_n" in sample.params:
        conditions.append("chi*N %.2f" % float(sample.params["chi_n"]))
    if "f" in sample.params:
        conditions.append("嵌段体积分数 f %.3f" % float(sample.params["f"]))
    user = (
        "<image>\n" + ("谱图渲染: %s\n" % image_path if image_path else "") +
        "1D 散射谱数值序列（q[nm^-1]: I[a.u.]，" + downsample_curve(sample, n_points) + "）\n"
        "实验条件：" + ("、".join(conditions) if conditions else "未提供") + "\n"
        "请给出 CoT 推理，并以 JSON 输出最终物理参数。"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def render_prompt(messages: List[Dict[str, str]]) -> str:
    return "\n\n".join("[%s] %s" % (m["role"], m["content"]) for m in messages)


def build_cot_text(record: Dict[str, Any]) -> str:
    """把 CoT 数据记录拼成一条 SFT 训练文本。"""
    cot = list(record.get("cot", []))
    answer = record.get("answer", {})
    body = cot[:-1] if cot and str(cot[-1]).startswith("最终答案") else cot
    return (
        "[system] " + SYSTEM_PROMPT + "\n\n"
        + "[user] " + str(record.get("curve", "")) + "\n\n"
        + "<think>\n" + "\n".join(str(step) for step in body) + "\n</think>\n"
        + "<answer>\n" + json.dumps(answer, ensure_ascii=False) + "\n</answer>"
    )
