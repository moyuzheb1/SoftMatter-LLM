"""训练层：三阶段 SFT、CoT 数据、DAPO/GRPO 与复合奖励。"""

from .reward import (  # noqa: F401
    RewardBreakdown, RewardConfig, numerical_consistency_reward,
    physics_consistency_reward, reward,
)
from .sft import run_sft_pipeline, train_with_hf  # noqa: F401
from .grpo_dapo import (  # noqa: F401
    DAPOConfig, GaussianPolicy, dapo_update, group_advantages, train_dapo,
)
from .hardcase import HardcaseConfig, is_hard, mine_hardcases  # noqa: F401
