# rpl-humanoid.github.io（RPL 项目页）

- **标题：** RPL — Learning Robust Humanoid Perceptive Locomotion on Challenging Terrains
- **类型：** site / project-page
- **URL：** <https://rpl-humanoid.github.io/>
- **PDF 镜像：** <https://rpl-humanoid.github.io/static/rpl.pdf>
- **入库日期：** 2026-06-07
- **配套论文：** [RPL（arXiv:2602.03002）](https://arxiv.org/abs/2602.03002) — 归档见 [`sources/papers/rpl_arxiv_2602_03002.md`](../papers/rpl_arxiv_2602_03002.md)

## 一句话摘要

Amazon FAR 等人提出的 **RPL** 官方站点：展示 **单一深度策略** 在坡面、双向楼梯与垫脚石上的长程行走、**2 kg 载荷 loco-manipulation**，以及 **DFSV / RSM 消融**、**多相机配置** 与 **多深度渲染 benchmark** 表格（相对 IsaacGym / IsaacSim 基线约 5× 加速）。

## 公开信息要点（截至入库日）

- **机构：** Amazon FAR、CMU、Stanford、UC Berkeley（* internship at Amazon FAR；† FAR co-lead）。
- **演示板块：**
  - **Bidirectional Locomotion** — 前后双向穿越复杂地形
  - **Back and Forth on Different Stairs** — 不同台阶尺寸往返
  - **Loco-Manipulation with 2kg Payload** — 负重搬运中的鲁棒行走
- **消融视频：** RPL vs w/o DFSV vs w/o RSM（OOD 窄地形与非对称多视角）
- **Benchmark 表：** IsaacGym PhysX、IsaacSim RTX、IsaacSim Warp vs RPL Warp 射线管线（VRAM / iter. time，$N_{\text{cam}}=1,2,4$）
- **多相机表：** 双向与全向 locomotion 下 $N_{\text{cam}}=1/2/4$ 的地形等级对比
- **代码：** 页面 **未提供** 公开仓库链接（截至入库日）

## 为何值得保留

- **非 PDF 证据：** 双向楼梯往返、载荷搬运与消融对比比摘要更直观呈现 **多向感知 + 蒸馏技巧** 的必要性。
- **与 arXiv 三角互证：** 项目页表格与论文 Table I/II 一致，便于维护者核对数值表述。
- **相关工作锚点：** [WholebodyVLA](https://github.com/OpenDriveLab/WholebodyVLA) README 引用本项目页作为 Demo 链接。

## 关联资料

- 论文归档：[`sources/papers/rpl_arxiv_2602_03002.md`](../papers/rpl_arxiv_2602_03002.md)
- 同系感知跑酷：[`sources/papers/php_parkour_arxiv_2602_15827.md`](../papers/php_parkour_arxiv_2602_15827.md)
- 感知 LLC 对照：[`sources/papers/pilot_arxiv_2601_17440.md`](../papers/pilot_arxiv_2601_17440.md)
