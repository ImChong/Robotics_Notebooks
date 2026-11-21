# ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models

> 来源归档（论文原文与官方开源仓库）

- **标题：** ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models
- **类型：** paper + repo
- **arXiv：** <https://arxiv.org/abs/2601.11404>（v2；HTML：<https://arxiv.org/html/2601.11404>）
- **代码：** <https://github.com/AgibotTech/ACoT-VLA>（快照 `cb9d195`，CVPR 2026 / AgiBot World Challenge @ ICRA 2026 基线）
- **作者：** Linqing Zhong, Yi Liu, Yifei Wei, Ziyu Xiong, Maoqing Yao, Si Liu, Guanghui Ren（AgiBot 等）
- **许可：** 代码 MIT；模型/内容声明见仓内徽章（CC BY 4.0 / MIT）
- **入库日期：** 2026-11-21
- **核对范围：** 通读 arXiv HTML 全文（<https://arxiv.org/html/2601.11404>，§1–§5，含 EAR/IAR/AGP 方法、LIBERO/LIBERO-Plus/VLABench 实验表、AgiBot G1 真机实验与消融）以及 `AgibotTech/ACoT-VLA@cb9d195` 源码（`README.md`、`scripts/train.sh`、`scripts/train.py`、`scripts/server.sh`、`scripts/serve_policy.py`、`src/openpi/models/acot_vla.py`、`src/openpi/policies/go2_policy.py`）；未运行 GPU 训练或推理服务。
- **通读：** 已完成
- **一句话说明：** 智元 GO-2 核心动作思维链（ACoT）技术报告与官方开源实现（基于 OpenPI）：将 VLA 中间推理从语言/图像空间移入动作空间，通过显式动作推理器（EAR）与隐式动作推理器（IAR）经双交叉注意力协同引导动作头生成，仓内提供 Go2 策略输入输出转换与 LIBERO/VLABench 训练配置，模型权重仍标 TODO。

## 核心论文与源码要点

1. **EAR（Explicit Action Reasoner，§3.2）：** 轻量 Transformer（flow matching）从 VLM 缓存条件生成粗粒度参考动作轨迹 $a^{\text{ref}}$，经 MLP 投影为显式先验 $Z^{\text{ex}}$（训练时采用参考轨迹 Teacher Forcing 稳定优化，推理时 `jax.lax.while_loop` 自条件去噪）。
2. **IAR（Implicit Action Reasoner，§3.3）：** 通过可学习查询矩阵下采样并交叉注意提取 VLM 每层 KV Cache 中的潜动作线索 $Z^{\text{im}}$（对应 `acot_vla.py` 中 `LearnableQueryExtractor` 等提取器）。
3. **AGP（Action-Guided Prediction，§3.4）：** 动作查询 $Q_{\text{action}}$ 分别与 $Z^{\text{ex}}$、$Z^{\text{im}}$ 做双交叉注意力并用自注意力融合后送入 `action_out_proj` 输出动作块。
4. **实验与开源边界（§4、`README.md`）：** LIBERO 均分 **98.5%**（Frozen/Unfrozen），LIBERO-Plus SFT **88.0%**，AgiBot G1 三项真机任务平均 **66.7%**（vs $\pi_{0.5}$ 61.0%）。仓库 `src/openpi/policies/go2_policy.py` 已定义 `Go2Inputs` / `Go2ACOTInputs`（`top_head`/`hand_left`/`hand_right` 三相机）与 `Go2Outputs`（22 维动作）；`README.md` 显示核心 EAR/IAR 训练与推理代码已开源，**预训练模型权重（checkpoints）仍为 TODO**。

## 对 wiki 的映射

- [GO-2（智元执行基座）](../../wiki/entities/go-2.md)
