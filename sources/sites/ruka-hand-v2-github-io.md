# ruka-hand-v2.github.io（RUKA-v2 项目页）

- **标题：** RUKA-v2: Tendon Driven Open-Source Dexterous Hand with Wrist and Abduction for Robot Learning — 官方项目页
- **类型：** site / project-page
- **URL：** <https://ruka-hand-v2.github.io/>
- **入库日期：** 2026-06-13
- **配套论文：** [RUKA-v2（arXiv:2603.26660）](https://arxiv.org/abs/2603.26660) — 归档见 [`sources/papers/ruka_v2_arxiv_2603_26660.md`](../papers/ruka_v2_arxiv_2603_26660.md)
- **配套代码：** <https://github.com/ruka-hand-v2/RUKA-v2>

## 一句话摘要

NYU 团队 **RUKA-v2** 官方站点：**全硬件（3D 打印/CAD）、全软件（控制器/校准/重定向）、全文档（装配/视频）免费开放**；突出 **2-DoF 平行腕**、**指根外展/内收**、OpenTeach VR 遥操作与 BAKU 模仿学习演示。

## 公开信息要点（截至入库日）

- **作者与机构：** Xinqi (Lucas) Liu、Ruoxi Hu 等；**New York University** + **New York University Shanghai**（* equal contribution）；联系：`irmakguzey@nyu.edu`、`rh4073@nyu.edu`（arXiv 页）。
- ** venue：** ICRA Workshop 2026（项目页标题区）。
- **开源承诺（摘要原文）：** All 3D print files, assembly instructions, controller software, and videos are **open-sourced**。
- **硬件亮点（页面分区）：**
  - **Hardware Design**：16 主动指 DoF + **2-DoF 腕**；材料成本 **<$2,000**（页内表述；论文 Table 1 约 **$1,500**）。
  - **Wrist Kinematics**：球铰共 pivot + 腱过 rotation center。
  - **Finger Abduction/Adduction**：独立 knuckle；中指固定。
- **软件与演示：**
  - **Controller**：AnyTeleop vector retargeting + linear interpolation；人视频 → URDF 仿真 → 真机 transfer 视频链。
  - **Single Arm Teleoperation**：OpenTeach + **Oculus VR**；列举 10 项单臂任务（捡笔、书写、开烤箱等）。
  - **Bimanual Teleoperation**：3 项双臂任务（烤箱取面包、整理、擦桌）。
  - **Policy Learning**：**BAKU** 视觉 BC；3 任务 rollout 与泛化视频（如笔→塑料刀）。
  - **Payload Tests**：静态载荷表（DIP–PIP 1200 g/15 s 等）。
- **BibTeX：** 页脚提供 arXiv:2603.26660 条目。

## 为何值得保留

- **一手入口聚合**：论文、代码、CAD、装配与演示视频均从该页链出，适合作为 curator 核验 **「是否真·全开源」** 的锚点。
- **任务级证据**：13 遥操作 + 3 IL 任务的视频比摘要更能支撑 **research platform** 定位。
- **与 v1 对照**：前代站点 <https://ruka-hand.github.io/> / 仓库 `ruka-hand/RUKA` 仍独立；v2 为 **新组织 + 新 DoF 栈**，不宜混为同一 BOM。

## 关联资料

- 论文归档：[`sources/papers/ruka_v2_arxiv_2603_26660.md`](../papers/ruka_v2_arxiv_2603_26660.md)
- 代码归档：[`sources/repos/ruka-v2.md`](../repos/ruka-v2.md)
- 前代 RUKA v1：<https://github.com/ruka-hand/RUKA>

## 对 wiki 的映射

- [RUKA-v2 Hand](../../wiki/entities/ruka-v2-hand.md)
