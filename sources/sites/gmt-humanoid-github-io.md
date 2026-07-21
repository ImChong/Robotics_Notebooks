# gmt-humanoid.github.io（GMT 项目页）

- **标题：** GMT: General Motion Tracking for Humanoid Whole-Body Control
- **类型：** site / project-page
- **URL：** <https://gmt-humanoid.github.io/>
- **配套论文：** [GMT（arXiv:2506.14770）](https://arxiv.org/abs/2506.14770) — 归档见 [`sources/papers/gmt_arxiv_2506_14770.md`](../papers/gmt_arxiv_2506_14770.md)
- **代码：** <https://github.com/zixuan417/humanoid-general-motion-tracking> — 归档见 [`sources/repos/humanoid-general-motion-tracking.md`](../repos/humanoid-general-motion-tracking.md)
- **机构：** UC San Diego；Simon Fraser University
- **入库日期：** 2026-07-21

## 一句话摘要

UCSD × SFU 团队的 **GMT** 官方站点：展示 **单一统一策略** 在 Unitree G1 真机上跟踪 **长技能序列、功夫/高踢/射门等敏捷技能、舞蹈与风格化 locomotion**；方法核心为 **Adaptive Sampling + Motion MoE**，相对 HumanPlus / OmniH2O / ExBody2 / ASAP 强调 **单策略、全身、多样、大规模 filtered MoCap**。

## 公开信息要点（截至入库日）

- **作者：** Zixuan Chen\*、Mazeyu Ji\*、Xuxin Cheng、Xuanbin Peng、Xue Bin Peng†、Xiaolong Wang†（\* Equal Contribution；† Equal Advising）。
- **演示板块：**
  - **Long Skill Sequencing** — 未剪辑长技能序列跟踪
  - **Agile Skills** — Kungfu、Airkick、Ready Kick、Kick & Walk、High Kick、Soccer Shoot
  - **Dancing**
  - **Stylized Locomotion** — Crouch Walk、Punch & Stand、Loco-Manip(Walk & Squat)、Pass、Drunk Walk、Spinning、Squatting、Side Step、Throw、Stretch、Stylized Walk、Warm Up、Stomping 等
- **Abstract 要点：** 时间/运动学多样性、策略容量、上下身协调是通用全身跟踪难点；Adaptive Sampling 平衡难易样本；Motion MoE 在流形不同区域特化；仿真 + 真机报告 SOTA 级统一策略。
- **Q&A（页内）：** 相对 HumanPlus/OmniH2O（loco-manip 偏上、下肢有限）、ExBody2（小规模人工策展 + 分类微调）、ASAP（敏捷但每动作单独策略），GMT 目标是 **单策略高保真覆盖广谱动作**。
- **BibTeX：** `@article{chen2025gmt,... arXiv:2506.14770}`。

## 开源核查（步骤 2.5）

| 项 | 状态（2026-07-21） |
|----|-------------------|
| 项目页代码链 | 指向 [zixuan417/humanoid-general-motion-tracking](https://github.com/zixuan417/humanoid-general-motion-tracking) |
| 已发布 | MuJoCo **sim2sim**、`pretrained.pt`、示例 motion `.pkl`、G1 多 DoF 模型 |
| 待发布 / 未列 | README News：`Data processing and retargeter code will be released soon`；**训练管线（IsaacGym PPO / DAgger）不在公开仓** |
| 结论 | **部分开源**（可复现推理/可视化；完整训练与数据处理待后续） |

## 为何值得保留

- **非 PDF 证据：** 长序列与敏捷/风格化真机视频是「统一策略是否真能跟」的直观判据。
- **与 arXiv / GitHub 三角互证：** 方法叙事（Adaptive Sampling + MoE）与仓库入口一致。
- **下游锚点：** ResMimic / PhyGile / EGM / SONIC 等常以 GMT 为「通用全身跟踪底座」对照或命名族。

## 关联资料

- 论文归档：[`sources/papers/gmt_arxiv_2506_14770.md`](../papers/gmt_arxiv_2506_14770.md)
- 代码仓库：[`sources/repos/humanoid-general-motion-tracking.md`](../repos/humanoid-general-motion-tracking.md)
