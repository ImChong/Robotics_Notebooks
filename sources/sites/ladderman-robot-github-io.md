# ladderman-robot.github.io（LadderMan 项目页）

- **标题：** LadderMan — Learning Humanoid Perceptive Ladder Climbing
- **类型：** site / project-page
- **URL：** <https://ladderman-robot.github.io/>
- **入库日期：** 2026-06-09
- **配套论文：** [LadderMan（arXiv:2606.05873）](https://arxiv.org/abs/2606.05873) — 归档见 [`sources/papers/ladderman_arxiv_2606_05873.md`](../papers/ladderman_arxiv_2606_05873.md)

## 一句话摘要

Amazon FAR 等人提出的 **LadderMan** 官方站点：展示 **Unitree G1** 在 **多样梯子几何** 上的 **零样本 sim-to-real 爬梯**、**键盘双向攀爬控制**、**梯上遥操作**（调画、换灯泡、高处递箱），以及 **VFM 深度 / rung-focused masking** 消融与 **仿真成功率热力图**（相对盲 motion tracking 基线）。

## 公开信息要点（截至入库日）

- **机构：** Amazon FAR（Frontier AI & Robotics）、USC、UC Berkeley、Stanford University、CMU；† Amazon FAR Co-Lead。
- **演示板块：**
  - **On-ladder manipulation** — 梯顶调画、高处递箱、拧紧灯泡
  - **Keyboard-based climbing direction control** — 连续双向攀爬
  - **Zero-shot sim-to-real transfer** — Raw Depth vs VFM Depth 对比；rung-focused masking 示意
  - **Robust ladder climbing across geometries** — 仿真中变化踏棍间距 $z$ 与倾角 $\phi$ 的成功率对比（LadderMan vs blind motion tracking）
- **硬件（论文/页面一致）：** Unitree G1（1.3 m、29-DoF）；机载 **Intel RealSense D435i**；机载 **NVIDIA Jetson Orin** 推理。
- **代码：** 论文称训练与推理代码及可部署模型 **将开源**；项目页 **未提供** 公开仓库链接（截至入库日）。

## 为何值得保留

- **非 PDF 证据：** 梯上操作、双向攀爬与深度消融视频比摘要更直观呈现 **稀疏踏棍感知 + 全身多接触协调** 的难度与系统组件必要性。
- **与 arXiv 三角互证：** 页面成功率热力图、真机梯子 A/B/C 与论文 Fig. 4–5、Table 2 一致，便于维护者核对表述。
- **同系工作锚点：** 与 [RPL](https://rpl-humanoid.github.io/)、[PHP](https://perceptive-humanoid-parkour.github.io/) 同属 Amazon FAR 人形 **感知移动/操作** 线；LadderMan 聚焦 **梯子稀疏结构** 与 **梯上 manipulation**。

## 关联资料

- 论文归档：[`sources/papers/ladderman_arxiv_2606_05873.md`](../papers/ladderman_arxiv_2606_05873.md)
- 同系多向深度行走：[`sources/papers/rpl_arxiv_2602_03002.md`](../papers/rpl_arxiv_2602_03002.md)
- 同系感知跑酷：[`sources/papers/php_parkour_arxiv_2602_15827.md`](../papers/php_parkour_arxiv_2602_15827.md)
- 感知 LLC 对照：[`sources/papers/pilot_arxiv_2601_17440.md`](../papers/pilot_arxiv_2601_17440.md)
