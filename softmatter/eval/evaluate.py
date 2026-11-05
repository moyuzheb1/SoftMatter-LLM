"""评测入口与 Markdown 报告生成。"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .metrics import EvalConfig, summarize


def evaluate_samples(samples: List[Any], analyst,
                     cfg: Dict[str, Any] = None, action=None,
                     policy=None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    eval_config = EvalConfig.from_dict(cfg or {})
    records = []
    for sample in samples:
        effective_action = (policy.mu_for(policy.project(analyst.features(sample)))
                            if policy else action)
        records.append({"sample": sample,
                        "pred": analyst.predict(sample, action=effective_action)})
    return records, summarize(records, eval_config)


def compare(baseline: Dict[str, Any], improved: Dict[str, Any]) -> Dict[str, float]:
    keys = ("phase_accuracy", "parameter_accuracy", "hardcase_phase_accuracy")
    result = {key + "_delta": float(improved.get(key, 0.0) - baseline.get(key, 0.0))
              for key in keys}
    result["phase_diagram_score_delta"] = float(
        improved.get("phase_diagram", {}).get("score", 0.0)
        - baseline.get("phase_diagram", {}).get("score", 0.0))
    return result


def write_report(path: Any, summary: Dict[str, Any],
                 stages: Optional[List[Dict[str, Any]]] = None,
                 dapo_history: Optional[List[Dict[str, float]]] = None,
                 extra: Dict[str, Any] = None) -> str:
    lines = [
        "# SoftMatter-LLM 评测报告", "",
        "本报告由 `python -m softmatter.cli demo` 自动生成，",
        "所有数值均来自本次本地运行的合成数据，非预训练模型权重结果。", "",
        "## 核心指标", "",
        "| 指标 | 数值 |", "| --- | --- |",
        "| 评测样本数 | %d |" % summary.get("n_samples", 0),
        "| 结构判定准确率 | %.4f |" % summary.get("phase_accuracy", 0.0),
        "| 参数提取准确率 | %.4f |" % summary.get("parameter_accuracy", 0.0),
        "| 相图绘制综合分 | %.4f |" % summary.get("phase_diagram", {}).get("score", 0.0),
        "| hardcase 相判定准确率 | %.4f |" % summary.get("hardcase_phase_accuracy", 0.0),
        "",
    ]
    ci = summary.get("phase_accuracy_ci95", (0.0, 0.0))
    lines.append("结构判定 95%% bootstrap CI: [%.4f, %.4f]" % (ci[0], ci[1]))
    lines += ["", "## 分区域表现", "", "| 区域 | 样本数 | 相判定 | 参数提取 |",
              "| --- | --- | --- | --- |"]
    for regime, values in sorted(summary.get("per_regime", {}).items()):
        lines.append("| %s | %d | %.4f | %.4f |" %
                     (regime, values["n"], values["phase_accuracy"],
                      values["parameter_accuracy"]))
    if stages:
        lines += ["", "## 三阶段 SFT", "", "| 阶段 | 语料规模 | 本次采样 | 相判定 | 参数提取 |",
                  "| --- | --- | --- | --- | --- |"]
        for stage in stages:
            lines.append("| %s | %d | %d | %.4f | %.4f |" %
                         (stage["stage"], stage["num_samples"], stage["num_samples_used"],
                          stage["phase_accuracy"], stage["parameter_accuracy"]))
    if dapo_history:
        first, last = dapo_history[0], dapo_history[-1]
        lines += ["", "## DAPO / GRPO", "",
                  "| 指标 | 起始 | 结束 |", "| --- | --- | --- |",
                  "| 训练奖励 | %.4f | %.4f |" % (first["train_reward"], last["train_reward"]),
                  "| holdout 奖励 | %.4f | %.4f |" %
                  (first["holdout_reward"], last["holdout_reward"]),
                  "| 参数提取准确率 | %.4f | %.4f |" %
                  (first["parameter_accuracy"], last["parameter_accuracy"]), ""]
    if extra:
        lines += ["## 数据与清洗", ""]
        for key, value in extra.items():
            lines.append("- %s: `%s`" % (key, value))
        lines.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    return content
