"""SoftMatter-LLM: 软物质散射/流变数据智能解析与相行为预测。

可在不依赖 GPU / 基座模型权重的情况下运行完整的数据 -> 训练 -> 评估流水线；
接入 HuggingFace 多模态模型时，可使用 ``softmatter.training.sft.train_with_hf``。
"""

__version__ = "0.1.0"

from .data.schema import Prediction, Sample  # noqa: F401,E402
from .analyst import ReferenceAnalyst  # noqa: F401,E402

__all__ = ["Sample", "Prediction", "ReferenceAnalyst", "__version__"]
