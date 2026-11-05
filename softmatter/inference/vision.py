"""可选的谱图 PNG 渲染（多模态输入的图像通道）。"""

from pathlib import Path
from typing import Optional, Union

import numpy as np


def render_curve_png(sample, path: Union[str, Path]) -> Optional[str]:
    """log-log 绘制 I(q)；未安装 matplotlib 时返回 None（走纯数值通道）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q = np.asarray(sample.q, dtype=float)
    intensity = np.asarray(sample.intensity, dtype=float)
    figure, axis = plt.subplots(figsize=(4.2, 3.2), dpi=110)
    axis.loglog(q, np.clip(intensity, 1e-12, None), lw=1.2, color="#1f5faa")
    axis.set_xlabel("q / nm$^{-1}$")
    axis.set_ylabel("I (a.u.)")
    axis.grid(alpha=0.25, which="both")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return str(path)
