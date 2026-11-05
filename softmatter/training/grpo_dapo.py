"""GRPO + DAPO 的群体相对策略优化（CPU 可复现实现）。

DAPO 组件：
1. Clip-Higher：非对称裁剪 (eps_low, eps_high)，允许低概率 token 更快探索；
2. Dynamic Sampling：丢弃组内奖励零方差（全对/全错）的样本；
3. Token-level Loss：按动作维度归一化，避免长序列稀释梯度；
4. Overlong Filtering：对超长补全施加惩罚。
策略是以谱图特征为条件的对角高斯分布 mu(x) = W x + b；
奖励的核心是 ``training.reward.numerical_consistency_reward``。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from ..analyst import N_FEATURES
from .hardcase import HardcaseConfig, mine_hardcases
from .reward import RewardConfig, reward

# 策略条件化在稳健物理量上：bias / Porod 指数 / log(xi) / log(SNR) /
# 临界指示 / CoT 一致性（后二者可区分 hardcase 的数据质量）
POLICY_FEATURES = (0, 2, 3, 5, 14, 15)


@dataclass
class DAPOConfig:
    group_size: int = 8
    steps: int = 80
    batch_samples: int = 6
    lr: float = 0.02
    eps_low: float = 0.20
    eps_high: float = 0.28
    inner_epochs: int = 2
    dynamic_sampling: bool = True
    token_level: bool = True
    overlong_penalty: bool = True
    max_tokens: int = 512
    hardcase_boost: float = 2.0
    seed: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAPOConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in allowed})


class GaussianPolicy:
    """条件对角高斯策略：动作 (delta_xi, delta_n) ~ N(W x + b, diag(sigma^2))。"""

    def __init__(self, n_features: int = None, dim: int = 2, sigma: float = 0.35,
                 indices: Any = POLICY_FEATURES):
        self.indices = tuple(int(i) for i in indices)
        n_features = len(self.indices) if n_features is None else int(n_features)
        self.W = np.zeros((dim, n_features), dtype=float)
        self.b = np.zeros(dim, dtype=float)
        self.log_sigma = np.full(dim, np.log(sigma), dtype=float)

    def project(self, full_features: np.ndarray) -> np.ndarray:
        """把完整特征向量投影到策略使用的低维子空间。"""
        x = np.asarray(full_features, dtype=float)
        return x if x.size == self.n_features else x[list(self.indices)]

    @property
    def dim(self) -> int:
        return int(self.b.size)

    @property
    def n_features(self) -> int:
        return int(self.W.shape[1])

    @property
    def sigma(self) -> np.ndarray:
        return np.exp(self.log_sigma)

    def mu_for(self, features: np.ndarray) -> np.ndarray:
        return self.W @ np.asarray(features, dtype=float) + self.b

    def sample(self, features: np.ndarray, size: int,
               rng: np.random.Generator) -> np.ndarray:
        return self.mu_for(features) + self.sigma * rng.standard_normal((size, self.dim))

    def log_prob(self, features: np.ndarray, actions: np.ndarray) -> np.ndarray:
        z = (actions - self.mu_for(features)) / self.sigma
        return -0.5 * np.sum(z ** 2 + 2.0 * self.log_sigma + np.log(2.0 * np.pi), axis=1)

    def grad_log_prob(self, features: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        z = (actions - self.mu_for(x)) / self.sigma
        d_mu = z / self.sigma                                   # (k, dim)
        d_w = (d_mu[:, :, None] * x[None, None, :]).reshape(actions.shape[0], -1)
        return np.concatenate([d_w, d_mu, z ** 2 - 1.0], axis=1)

    def update(self, delta: np.ndarray) -> None:
        size = self.dim * self.n_features
        self.W = np.clip(self.W + delta[:size].reshape(self.W.shape), -2.0, 2.0)
        self.b = np.clip(self.b + delta[size:size + self.dim], -3.0, 3.0)
        self.log_sigma = np.clip(self.log_sigma + delta[size + self.dim:], -3.0, 1.0)

    def snapshot(self) -> Dict[str, np.ndarray]:
        return {"W": self.W.copy(), "b": self.b.copy(),
                "log_sigma": self.log_sigma.copy()}

    def restore(self, snapshot: Dict[str, np.ndarray]) -> None:
        self.W = snapshot["W"].copy()
        self.b = snapshot["b"].copy()
        self.log_sigma = snapshot["log_sigma"].copy()

    def to_dict(self) -> Dict[str, Any]:
        return {"W": self.W.tolist(), "b": self.b.tolist(),
                "log_sigma": self.log_sigma.tolist(),
                "indices": list(self.indices)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GaussianPolicy":
        policy = cls(n_features=len(data["W"][0]) if data.get("W") else N_FEATURES,
                     dim=len(data.get("b", [0.0, 0.0])),
                     indices=data.get("indices", POLICY_FEATURES))
        if data.get("W"):
            policy.W = np.asarray(data["W"], dtype=float)
        policy.b = np.asarray(data.get("b", policy.b.tolist()), dtype=float)
        policy.log_sigma = np.asarray(data.get("log_sigma", policy.log_sigma.tolist()),
                                      dtype=float)
        return policy


def group_advantages(rewards: np.ndarray) -> np.ndarray:
    """GRPO：组内标准化优势，无需 value/critic 网络。"""
    rewards = np.asarray(rewards, dtype=float)
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)


def dapo_update(policy: GaussianPolicy, features: np.ndarray, actions: np.ndarray,
                rewards: np.ndarray, cfg: DAPOConfig) -> Dict[str, float]:
    rewards = np.asarray(rewards, dtype=float)
    if cfg.dynamic_sampling and float(rewards.std()) < 1e-6:
        return {"skipped": 1.0, "mean_reward": float(rewards.mean())}
    advantages = group_advantages(rewards)
    old_log_prob = policy.log_prob(features, actions)
    low, high = 1.0 - cfg.eps_low, 1.0 + cfg.eps_high
    scale = 1.0 / (actions.shape[1] if cfg.token_level else 1.0)
    for _ in range(max(1, cfg.inner_epochs)):
        log_prob = policy.log_prob(features, actions)
        ratio = np.exp(np.clip(log_prob - old_log_prob, -10.0, 10.0))
        blocked = ((ratio > high) & (advantages > 0)) | ((ratio < low) & (advantages < 0))
        coefficient = np.where(blocked, 0.0, ratio)
        ascent = ((advantages * coefficient)[:, None] *
                  policy.grad_log_prob(features, actions)).mean(axis=0) * scale
        policy.update(cfg.lr * np.clip(ascent, -5.0, 5.0))
    return {"skipped": 0.0, "mean_reward": float(rewards.mean())}


def train_dapo(samples: List[Any], analyst, cfg: DAPOConfig,
               reward_cfg: RewardConfig = None, holdout: List[Any] = None,
               eval_cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """在全量样本上做 DAPO（hardcase 加权boost），返回策略与逐步曲线。"""
    from ..eval.metrics import EvalConfig, parameter_accuracy
    from ..inference.parser import format_answer

    reward_cfg = reward_cfg or RewardConfig()
    rng = np.random.default_rng(cfg.seed)
    policy = GaussianPolicy()
    n_hardcases = len(mine_hardcases(samples, analyst, HardcaseConfig()))
    pool = samples
    weights = np.asarray([cfg.hardcase_boost if s.hardcase else 1.0 for s in pool], dtype=float)
    weights /= weights.sum()
    holdout = holdout if holdout is not None else samples[:min(120, len(samples))]
    eval_config = EvalConfig.from_dict(eval_cfg or {})

    def holdout_score() -> Tuple[float, float]:
        rows = [{"sample": s, "pred": analyst.predict(
            s, action=policy.mu_for(policy.project(analyst.features(s))))} for s in holdout]
        mean_reward = float(np.mean([reward(r["pred"], r["sample"], reward_cfg).total
                                     for r in rows]))
        return mean_reward, float(parameter_accuracy(rows, eval_config))

    # 验证集早停：以参数提取准确率为主要选择准则、奖励为辅；
    # 未训练策略始终是候选，保证 RL 阶段不会让指标劣化。
    best_reward, best_accuracy = holdout_score()
    best_state = policy.snapshot()
    best_step = 0
    history: List[Dict[str, float]] = [{"step": 0, "train_reward": 0.0,
                                        "holdout_reward": best_reward,
                                        "parameter_accuracy": best_accuracy}]

    for step in range(1, int(cfg.steps) + 1):
        indexes = rng.choice(len(pool), size=min(cfg.batch_samples, len(pool)),
                             replace=True, p=weights)
        step_rewards = []
        for index in indexes:
            sample = pool[int(index)]
            features = policy.project(analyst.features(sample))
            actions = policy.sample(features, cfg.group_size, rng)
            group_rewards = []
            for action in actions:
                pred = analyst.predict(sample, action=action)
                pred.raw = format_answer(pred)
                value = reward(pred, sample, reward_cfg).total
                if cfg.overlong_penalty and len(pred.raw) > cfg.max_tokens:
                    value = max(0.0, value - 0.10)
                group_rewards.append(value)
            step_rewards.extend(group_rewards)
            dapo_update(policy, features, actions, np.asarray(group_rewards), cfg)
        validation_reward, validation_accuracy = holdout_score()
        improved = (validation_accuracy > best_accuracy + 1e-9
                    or (abs(validation_accuracy - best_accuracy) < 1e-9
                        and validation_reward > best_reward + 1e-6))
        if improved:
            best_accuracy, best_reward = validation_accuracy, validation_reward
            best_state, best_step = policy.snapshot(), step
        history.append({
            "step": step,
            "train_reward": float(np.mean(step_rewards)),
            "holdout_reward": validation_reward,
            "parameter_accuracy": validation_accuracy,
        })
    policy.restore(best_state)
    return {"policy": policy, "history": history, "n_hardcases": n_hardcases,
            "best_step": best_step, "best_holdout_reward": best_reward,
            "best_holdout_accuracy": best_accuracy,
            "final": history[best_step] if history else {}}
