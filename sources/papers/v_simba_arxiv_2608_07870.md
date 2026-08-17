# V-Simba: Unleashing the Architectural Potential of RL in Visual Continuous Control（arXiv:2608.07870）

> 来源归档（ingest）

- **标题：** V-Simba: Unleashing the Architectural Potential of RL in Visual Continuous Control
- **缩写 / 框架：** **V-Simba**
- **类型：** paper / visual-rl / sac / architecture
- **arXiv：** <https://arxiv.org/abs/2608.07870>
- **会议：** RLC 2026 / Reinforcement Learning Journal
- **代码：** <https://github.com/DAVIAN-Robotics/V-Simba>（归档见 [`sources/repos/v-simba.md`](../repos/v-simba.md)）
- **作者：** Donghu Kim、Youngdo Lee、Hojoon Lee、Johan Obando-Ceron、Byungkun Lee、Aaron Courville、Pablo Samuel Castro、Jaegul Choo、Clare Lyle
- **机构：** KAIST；其余作者含 Mila / Google DeepMind 一线（本库有 `deepmind`，无独立 `mila` alias）
- **入库日期：** 2026-08-17
- **一句话说明：** 把状态基 RL 的 Simba 架构原则迁到视觉连续控制：在带数据增强的 SAC 上加归一化层与 pointwise convolution，样本效率对齐或超过 SOTA，算力低于 DrQ-v2。

## 开源状态（步骤 2.5）

- **项目页：** 无独立站点。
- **代码仓核查（2026-08-17）：** [DAVIAN-Robotics/V-Simba](https://github.com/DAVIAN-Robotics/V-Simba)（Apache-2.0）。入口 `run_online.py`、`run_parallel.py`；复现脚本 `scripts/vsimba_dmc.sh` / `vsimba_adroit.sh` / `vsimba_metaworld.sh`。智能体实现 `scale_rl/agents/vsimba/`，对照 DrQ-v2 同仓。`uv sync` + MuJoCo 2.1.0。
- **结论：** **已开源、可运行训练**（DMC / Adroit / Meta-World）。

## 摘录 1：论点

视觉 RL 过去把样本效率问题交给动力学模型或探索算法；状态基 RL 已证明 **架构本身** 能抬样本效率。V-Simba 问：这些原则能否迁到像素输入？

## 摘录 2：做法

SAC + 数据增强；normalization 稳住高维视觉训练；pointwise conv 降计算。报告在 DMC、Adroit、Meta-World 匹配或超过当时方法，且比 DrQ-v2 更省算力。

**对 wiki 的映射：** [`wiki/entities/paper-v-simba.md`](../../wiki/entities/paper-v-simba.md)；交叉 [SAC](../../wiki/methods/sac.md)、[强化学习](../../wiki/methods/reinforcement-learning.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（Apache-2.0 可运行）
