# warp-retargeting.github.io（WARP 项目页）

- **标题：** WARP — Whole-Body Retargeting for Learning from Offline Human Demonstrations
- **类型：** site / project-page
- **URL：** <https://warp-retargeting.github.io/>
- **arXiv：** <https://arxiv.org/abs/2606.29940>
- **入库日期：** 2026-08-20
- **配套论文：** [WARP（arXiv:2606.29940）](../papers/warp_arxiv_2606_29940.md)

## 一句话摘要

Georgia Tech（Danfei Xu 组）提出的 **WARP** 官方站点：强调 **离线人类演示** 经 **闭式 c-SEW** 转为可开环回放的全身机器人动作；展示相对 MINK-EF / MINK-TE / SEW-M 的重定向精度与一致性对比，以及 DexMimicGen 仿真策略与 RB-Y1 真机四任务评测视频。

## 公开信息要点（截至 2026-08-20 核查）

- **机构：** Georgia Institute of Technology（Zhenyang Chen、Chuizheng Kong、Chuye Zhang 等；Danfei Xu）。
- **核心叙事：** 人类数据是全身移动操作最廉价来源；难点在转化为可学习的机器人数据。离线设定无人在环纠错 → 重定向必须 **精确 + 一致**。
- **方法板块：**
  - **c-SEW** — palm 硬约束 + adaptive offset + 肘 nullspace（Stereo-sew / SP3）闭式解
  - **Lazy mobile base** — torso 吸收微调，base 仅 genuine relocation
  - **Hierarchical policy** — 单 flow-matching 头 + block-causal attention（base ≼ torso ≼ arm ≼ hand）
- **数据与平台：** Meta Quest 60 Hz 全身+手采集；部署 **RB-Y1**（holonomic base、6-DoF torso、双 7-DoF 臂、XHands）100 Hz 关节阻抗控制；物体 AprilTag + Vicon 定位（隔离感知/里程计因素）。
- **结果板块：**
  - **Retargeting quality** — BONES-SEED-SOMA 514 clips；交互式任务 qualitative 对比（cupboard、cooking、street 等）
  - **Policy learning in simulation** — DexMimicGen 三任务；WARP 策略平均 **71%** vs MINK **59%**
  - **Real-world** — 四任务真机；冰箱肘接触 replay **90%**
- **局限：** 当前无图像观测的策略训练；计划 visual-motor 扩展。
- **代码 / 数据（步骤 2.5）：** 页面 **未发现** GitHub、Hugging Face、Zenodo 或 Code 按钮；仅 BibTeX 与演示视频。按 **截至入库日未开源** 处理。

## 为何值得保留

- **非 PDF 证据：** c-SEW 动画、多方法并排重定向视频比表格更直观呈现「精确 vs 不一致」失败模式。
- **离线 vs 遥操作对照：** 页面强调 open-loop replay 对监督质量的苛刻要求，与 Teleop 闭环叙事形成对照。
- **评测任务列表：** 20+ qualitative 任务名（crouch cupboard、cutting bread、street avoid 等）便于与 BONES-SEED-SOMA 基准交叉引用。

## 关联资料

- 论文归档：[`sources/papers/warp_arxiv_2606_29940.md`](../papers/warp_arxiv_2606_29940.md)
- 概念域：[`wiki/concepts/motion-retargeting.md`](../../wiki/concepts/motion-retargeting.md)
- 任务域：[`wiki/tasks/loco-manipulation.md`](../../wiki/tasks/loco-manipulation.md)
