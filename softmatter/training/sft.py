"""三阶段课程式 SFT（10w 概念 -> 50w 拟合 -> 144w 复杂推理）。

CPU 可运行路径：在 CoT 特征上训练参考分析器的 softmax 相分类头与
xi/n 校准回归头；三阶段通过 feature gate 逐步放开特征维度。
GPU 路径：``train_with_hf`` 用 LoRA + TRL SFTTrainer 微调多模态基座模型。
"""

from typing import Any, Dict, List, Sequence

import numpy as np

from ..analyst import PHASE_ORDER, N_FEATURES, ReferenceAnalyst

DEFAULT_STAGES: Sequence[Dict[str, Any]] = (
    {"name": "physics_concepts", "num_samples": 100000, "feature_level": 1, "lr": 0.25},
    {"name": "curve_fitting", "num_samples": 500000, "feature_level": 2, "lr": 0.30},
    {"name": "complex_reasoning", "num_samples": 1440000, "feature_level": 3, "lr": 0.35},
)


def draw_subset(samples: List[Any], size: int,
                rng: np.random.Generator) -> List[Any]:
    """按语料规模做有放回子采样（生产环境等价于遍历完整语料）。"""
    if not samples:
        return []
    indexes = rng.integers(0, len(samples), size=int(size))
    return [samples[int(i)] for i in indexes]


def train_phase_classifier(analyst: ReferenceAnalyst, samples: List[Any],
                           level: int, cfg: Dict[str, Any], lr: float) -> Dict[str, float]:
    features = np.stack([analyst.features(sample, level) for sample in samples])
    labels = np.asarray([PHASE_ORDER.index(sample.phase) for sample in samples], dtype=int)
    priors = np.stack([analyst.base_logits(sample) for sample in samples])
    weight = analyst.W.copy()
    rng = np.random.default_rng(int(cfg.get("seed", 23)) + level)
    epochs = int(cfg.get("epochs", 14))
    batch_size = int(cfg.get("batch_size", 64))
    for _ in range(epochs):
        order = rng.permutation(len(labels))
        for start in range(0, len(labels), batch_size):
            index = order[start:start + batch_size]
            xb, yb = features[index], labels[index]
            logits = xb @ weight.T + priors[index]
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(np.clip(logits, -60.0, 60.0))
            probabilities = exp / exp.sum(axis=1, keepdims=True)
            probabilities[np.arange(len(yb)), yb] -= 1.0
            gradient = probabilities.T @ xb / len(yb) + 1e-4 * weight
            weight -= lr * gradient
    analyst.W = weight
    accuracy = float(np.mean(np.argmax(features @ weight.T + priors, axis=1) == labels))
    return {"train_phase_accuracy": accuracy, "n_used": len(labels)}


def train_calibration(analyst: ReferenceAnalyst, samples: List[Any],
                      level: int, shrink: float = 0.25) -> Dict[str, float]:
    """岭回归学习 xi 的乘性残差与 n 的加性残差。"""
    features = np.stack([analyst.features(sample, level) for sample in samples])
    target_xi, target_n = [], []
    for sample in samples:
        observations = analyst.curve_features(sample)
        truth_xi = float(sample.params.get("xi", observations["xi_est"]))
        truth_n = float(sample.params.get("n", observations["n_est"]))
        target_xi.append(np.clip(np.log(max(truth_xi, 1e-6) /
                                        max(observations["xi_est"], 1e-6)), -2.0, 2.0))
        target_n.append(np.clip(truth_n - observations["n_est"], -3.0, 3.0))
    # 只用少量稳健的物理特征做标定，避免高阶峰比等弱相关特征过拟合
    columns = (0, 2, 3, 14)
    design = features[:, columns]
    analyst.theta_xi = np.zeros(N_FEATURES, dtype=float)
    analyst.theta_n = np.zeros(N_FEATURES, dtype=float)

    def robust_ridge(target: np.ndarray, scale: float) -> np.ndarray:
        """IRLS / Huber：抑制极端失拟样本，让标定对准残差主体而非长尾。"""
        identity = 1.0 * np.eye(len(columns))
        weights = np.ones(len(target), dtype=float)
        coefficient = np.zeros(len(columns), dtype=float)
        for _ in range(6):
            weighted = design.T @ (weights[:, None] * design) + identity
            coefficient = np.linalg.solve(weighted, design.T @ (weights * target))
            residual = design @ coefficient - target
            weights = 1.0 / (1.0 + (residual / scale) ** 2)
        return coefficient

    # 收缩是保守正则：合成谱上 xi 残差有长尾，过度校正会伤害多数样本
    analyst.theta_xi[list(columns)] = shrink * robust_ridge(np.asarray(target_xi), 0.25)
    analyst.theta_n[list(columns)] = shrink * robust_ridge(np.asarray(target_n), 0.25)
    return {"xi_rmse": float(np.sqrt(np.mean((features @ analyst.theta_xi -
                                              np.asarray(target_xi)) ** 2))),
            "n_rmse": float(np.sqrt(np.mean((features @ analyst.theta_n -
                                             np.asarray(target_n)) ** 2)))}


def run_sft_pipeline(samples: List[Any], analyst: ReferenceAnalyst,
                     cfg: Dict[str, Any], eval_samples: List[Any],
                     eval_cfg: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """逐阶段训练并在固定评测集上记录准确率曲线。"""
    from ..eval.evaluate import evaluate_samples  # 延迟导入，避免包循环

    eval_cfg = eval_cfg or {}
    rng = np.random.default_rng(int(cfg.get("seed", 23)))
    results: List[Dict[str, Any]] = []
    for stage in cfg.get("stages", DEFAULT_STAGES):
        logical = int(stage["num_samples"])
        used = max(200, int(logical * float(cfg.get("demo_scale", 0.001))))
        batch = draw_subset(samples, used, rng)
        level = int(stage["feature_level"])
        analyst.stage = level
        classifier = train_phase_classifier(analyst, batch, level, cfg,
                                            float(stage.get("lr", cfg.get("lr", 0.3))))
        calibration = train_calibration(analyst, batch, level,
                                        float(cfg.get("calibration_shrink", 0.25)))
        _, summary = evaluate_samples(eval_samples, analyst, eval_cfg)
        results.append({
            "stage": stage["name"],
            "feature_level": level,
            "num_samples": logical,
            "num_samples_used": used,
            "phase_accuracy": summary["phase_accuracy"],
            "parameter_accuracy": summary["parameter_accuracy"],
            "train_phase_accuracy": classifier["train_phase_accuracy"],
            "xi_rmse": calibration["xi_rmse"],
            "n_rmse": calibration["n_rmse"],
        })
    return results


def train_with_hf(cot_records: List[Dict[str, Any]], llm_cfg: Dict[str, Any],
                  output_dir: str, stage_name: str = "sft") -> str:
    """真实 LLM 训练入口：需要 ``pip install -e '.[train]'`` 与 GPU。

    cot_records 为 ``softmatter.data.cot_builder.build_cot_dataset`` 的输出，
    每条包含 CoT 步骤与最终 JSON 答案；这里把 (谱图数值序列 + CoT + 答案)
    拼成单条文本，用 LoRA 做监督微调。
    """
    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - 依赖可选
        raise RuntimeError(
            "缺少训练依赖，请先执行: pip install -e '.[train]'（torch/transformers/trl/peft/datasets）"
        ) from exc

    from ..inference.prompt import build_cot_text

    rows = [{"text": build_cot_text(record)} for record in cot_records]
    dataset = Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(llm_cfg["base_model"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(llm_cfg["base_model"], trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        r=int(llm_cfg.get("lora_r", 16)),
        lora_alpha=int(llm_cfg.get("lora_alpha", 32)),
        lora_dropout=float(llm_cfg.get("lora_dropout", 0.05)),
        task_type="CAUSAL_LM",
    ))
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(output_dir=output_dir,
                       max_length=int(llm_cfg.get("max_length", 2048)),
                       num_train_epochs=1, report_to="none",
                       run_name=stage_name),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
