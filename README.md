# SoftMatter-LLM

**基于多模态大模型的软物质散射 / 流变数据智能解析与相行为预测系统**

针对嵌段共聚物、胶体溶液、高分子凝胶等软物质体系中 SAXS/SANS 一维谱图解析复杂、
流变 master curve 构建依赖人工经验的问题，本仓库给出一条完整的
**数据构建 → CoT/SFT → GRPO/DAPO 强化优化 → 相图自动绘制** 技术流水线。

技术栈关键词：`GRPO` / `DAPO`、`SFT`、`CoT`、数值一致性奖励、多模态（谱图 PNG + 数值序列）、
Porod 定律、Ornstein-Zernike 关联长度、WLF 时温等效。

---

## 1. 系统设计

### 1.1 数据构建

| 环节 | 实现 | 说明 |
| --- | --- | --- |
| 语料读取 | `softmatter/data/readers.py` | `os.walk` 生成器流式扫描 10 万级 `.dat/.txt/.csv/.asc`，不一次性载入内存；支持 `# metadata` 头与 `A^-1 → nm^-1` 单位换算 |
| 清洗 | `softmatter/data/clean.py` | 有限值过滤、q 排序去重、尖峰剔除、保守背景扣除、SNR 门槛、**高 q 区 Porod 幂律校验**（稳健局部斜率中位数） |
| 多维配比 | `softmatter/data/mixture.py` | 按 `(regime, phase)` 分层抽样，平衡有序相 / 无序相 / 临界涨落区；输出卡方偏差、Porod 合规率、`log ξ` 分布等物理统计校验 |
| CoT 构造 | `softmatter/data/cot_builder.py` | 七步推理链：预处理 → 高 q 幂律 → 低 q OZ → 峰位/峰比 → 结构判定 → 流变交叉验证 → hardcase 不确定性 → JSON 答案；每个中间量都从曲线实测量得到 |

物理内核在 `softmatter/physics/`：解析散射模型（OZ 低 q 背景 + Bragg 峰列 + 平滑接入的
`q^-n` 尾）、峰比模板（层状 `1:2:3:4`、六方柱状 `1:√3:2`、gyroid `√6:√8:√14`）、
WLF master curve、简化的 `χN–f` 相图。

### 1.2 三阶段 SFT + CoT

`softmatter/training/sft.py` 按课程逐步放开特征维度：

| 阶段 | 语料规模（配置值） | 特征级别 | 训练目标 |
| --- | --- | --- | --- |
| `physics_concepts` | 100,000 | 1（4 维） | Guinier / Porod / OZ 区间识别 |
| `curve_fitting` | 500,000 | 2（8 维） | 峰位、幂律指数 n、相关长度 ξ 的数值回归 |
| `complex_reasoning` | 1,440,000 | 3（18 维） | CoT 峰比匹配、临界涨落、χN/f 相边界与 hardcase 判别 |

分类头在**物理先验 logits 上做残差学习**，回归头用 Huber/IRLS 鲁棒岭回归学习 ξ 与 n 的残差并做保守收缩。
真实 LLM 训练入口为 `train_with_hf()`（LoRA + TRL `SFTTrainer`，需 `pip install -e ".[train]"` 与 GPU）。

### 1.3 GRPO / DAPO 与数值一致性奖励

`softmatter/training/grpo_dapo.py` 实现以谱图特征为条件的高斯策略 `μ(x) = Wx + b`，包含：

- **GRPO**：组内标准化优势，无需 critic；
- **DAPO Clip-Higher**：非对称裁剪 `eps_low=0.20 / eps_high=0.28`；
- **Dynamic Sampling**：丢弃组内奖励零方差样本；
- **Token-level Loss / Overlong Filtering**：按动作维度归一化并惩罚超长补全；
- **Hardcase 挖掘与加权**：相判错、低置信度、参数误差大的样本优先采样；
- **验证集早停**：以参数提取准确率为主、奖励为辅选择最佳策略快照，保证 RL 阶段不劣化。

`softmatter/training/reward.py` 的复合奖励为

\[
R = w_f R_{\text{format}} + w_p R_{\text{phase}} + w_n R_{\text{num}} + w_c R_{\text{consistency}}
\]

其中 **数值一致性奖励** `R_num` 对 ξ、`n`、主峰 `q*` 的（相对/绝对）误差做指数软衰减，
`R_consistency` 额外约束 `d = 2π/q*`、Porod 指数区间与参数取值域，抑制「相判对但参数乱编」。

### 1.4 推理与评测

- 多模态 prompt：`softmatter/inference/prompt.py`（`<image>` + 64 点数值序列 + 实验条件），
  `vision.py` 可选渲染 log-log 谱图 PNG；
- 输出解析：`parser.py` 兼容 ```json 围栏、`<answer>` 标签与 CoT 末尾 JSON；
- 指标：`phase_accuracy`、`parameter_accuracy`（ξ/n/q* 联合容差）、
  分区域拆解、bootstrap CI、相图 `cell_accuracy` + 边界 `F1` + 综合分。

---

## 2. 快速开始

需要 Python ≥ 3.9 与 `numpy`，无需 GPU、无需下载模型权重。

```bash
pip install -e .
python -m softmatter.cli demo --out .
```

`demo` 会依次生成 2400 条带物理真值的散射/流变样本 → 写回 `.dat` → 流式扫描 → 清洗 →
配比 → CoT → 三阶段 SFT → DAPO/GRPO → 评测，并把结果写到 `outputs/report.md`。

分步执行：

```bash
python -m softmatter.cli synth   --out data/raw --n 2400
python -m softmatter.cli ingest  --root data/raw --out data/manifest.jsonl
python -m softmatter.cli clean   --manifest data/manifest.jsonl --out data/clean.jsonl
python -m softmatter.cli mixture --input data/clean.jsonl --out data/train.jsonl
python -m softmatter.cli cot     --input data/train.jsonl --out data/cot.jsonl
python -m softmatter.cli train   --input data/train.jsonl --out .
python -m softmatter.cli eval    --input data/train.jsonl --out outputs/eval_report.md
python -m softmatter.cli infer   --input data/train.jsonl --id sm-000001
```

Windows 可直接 `powershell -File scripts/run_pipeline.ps1`，Linux/macOS 使用 `bash scripts/run_pipeline.sh`。
所有参数集中在 `configs/default.json`，可用 `--config my.json` 做深度覆盖。

---

## 3. 目录结构

```
softmatter/
  physics/     散射模型、峰比/相图、流变 master curve、数据合成
  data/        样本结构、流式读取、清洗、配比、CoT
  analyst.py   可训练的参考分析器（CPU baseline / 策略载体）
  training/    三阶段 SFT、复合奖励、GRPO/DAPO、hardcase 挖掘
  inference/   多模态 prompt、谱图渲染、输出解析、端到端分析
  eval/        指标、相图评分、Markdown 报告
  cli.py       全部命令行入口
configs/       默认配置
scripts/       一键流水线脚本
tests/         unittest 测试（无需 pytest）
```

---

## 4. 关于数据与指标的重要说明

> **请勿把本仓库的 demo 数字当作原始项目的结果，也不要反过来。**

- 原始项目描述中的「10 万级实验文件筛选清洗出 144w 样本」「相判定 72% → 82%」
  「参数提取准确率 85%、超越基线 7%、相图任务领先 14%」属于**在私有真实数据与 GPU 集群上
  训练多模态基座模型得到的结果**，本仓库**不附带**这些数据、模型权重与算力环境，因此**无法复现上述数值**。
- `configs/default.json` 里的 `100000 / 500000 / 1440000` 是生产语料规模参数；
  CPU demo 通过 `sft.demo_scale` 做等比例子采样（`num_samples_used` 会写进报告）。
- 本仓库用物理解析模型合成带真值的数据，并提供一个无需 GPU 的 `ReferenceAnalyst`
  作为「模型位置」的等价替身，目的是让**整条流水线的代码、算法与评测逻辑可被审查和复现**。
- `outputs/report.md` 中的数字是本机本次运行在**合成数据**上的实测结果。
  某次默认配置运行示例（seed=7，2400 条）：

  | 指标 | 基线 | 三阶段 SFT + DAPO |
  | --- | --- | --- |
  | 结构判定准确率 | 0.667 | 0.823 |
  | 参数提取准确率 | 0.490 | 0.503 |
  | 相图绘制综合分 | — | 0.875 |
  | 有序相 Porod 合规率 | — | 0.902 |

  阶段曲线：`0.583 → 0.707 → 0.823`。不同 numpy 版本/随机种子会有小幅波动。

接入真实数据时：把实验文件放到 `data/raw`，在 `# metadata` 头中给出 `phase/regime/params`
真值（或自行接标注管线），然后执行 `ingest → clean → mixture → cot`；
接入真实模型时先 `pip install -e ".[train]"`，再调用 `train_with_hf(cot_records, llm_cfg, out_dir)`。

---

## 5. 测试

```bash
python -m unittest discover -s tests -v
```

覆盖：Porod 指数/ξ 反演、峰比模板判别、WLF master curve、清洗与 Porod 校验、
配比分布、CoT/JSON 解析、复合奖励、三阶段 SFT、DAPO 早停、评测指标与相图评分。

---

## 6. 许可

MIT License.
