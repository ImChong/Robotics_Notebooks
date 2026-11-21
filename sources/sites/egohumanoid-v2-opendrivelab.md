# EgoHumanoid-V2 官方项目页核查

> 来源归档（site / OpenDriveLab at The University of Hong Kong）

- **URL：** <https://opendrivelab.com/EgoHumanoid-V2/>
- **标题：** EgoHumanoid-V2: Human-to-Humanoid Transfer of Coordinated Whole-Body Skills for Loco-Manipulation
- **核查日期：** 2026-10-12（2026-11-21 复核）
- **核对范围：** 读取 `https://opendrivelab.com/EgoHumanoid-V2/` 官方项目页全文与按钮链接（Paper、Code -> `OpenDriveLab/EgoHumanoid-V2`），并复核 2026-11-04 仓库 commit `682199e` 后 checklist 全部勾齐的状态。
- **通读：** 已完成
- **代码：** <https://github.com/OpenDriveLab/EgoHumanoid-V2>（项目页 Code 按钮）
- **论文：** <https://arxiv.org/abs/2609.37181>
- **许可与开放状态：** 官方仓库标注 Apache-2.0；截至 2026-11-04 commit `682199e` 已开放全流程代码（采集、动作对齐、视觉对齐、训练与部署），论文全量训练数据与任务微调后权重仍需自备/自训
- **源码归档：** [OpenDriveLab/EgoHumanoid-V2](../repos/egohumanoid-v2.md)
- **论文归档：** [arXiv:2609.37181](../papers/egohumanoid_v2_arxiv_2609_37181.md)
- **覆盖 wiki：** [EgoHumanoid-V2](../../wiki/entities/paper-egohumanoid-v2.md)

## 项目页内容

主页把贡献概括为「第一视角人体示范到协调全身 loco-manipulation 的人形技能迁移」：粗到细动作对齐由运动学参考修正与动力学感知精炼组成，四项真机任务展示零目标任务机器人示范的技能迁移。项目页交互区域展示人体采集、未对齐动作回放、对齐后推理结果、动作对齐误差和视觉处理流程；正文数字以论文为准。

## 代码开放核查

项目页提供 Code 链接。复核其指向的官方 GitHub README（@ `682199e`，2026-11-04），仓库公开并使用 Apache-2.0，Workflow & Release Checklist 中**硬件配置、环境安装、人类数据采集、动作对齐、视觉对齐、策略训练流水线（OpenPI π₀.₅）、策略部署流水线（`eh2-serve` + SONIC `run_vla_inference.py`）**均已勾选发布。

仓库提供四项任务的动作对齐示例及 3 个任务的视觉对齐示例（`motion.npz` + 对比动画），但未列出论文全量 200 条/任务人类训练集或微调后任务 checkpoint 的直接下载包，复现仍需自备 `pi05_base`、SONIC、SAM 3.1 权重及机器人硬件。

## 项目关系

- 项目主页：<https://opendrivelab.com/EgoHumanoid-V2/>
- 代码：<https://github.com/OpenDriveLab/EgoHumanoid-V2>
- 论文：<https://arxiv.org/abs/2609.37181>
- Wiki：[EgoHumanoid-V2](../../wiki/entities/paper-egohumanoid-v2.md)
