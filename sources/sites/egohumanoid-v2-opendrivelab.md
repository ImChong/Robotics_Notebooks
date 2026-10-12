# EgoHumanoid-V2 官方项目页核查

> 来源归档（site / OpenDriveLab at The University of Hong Kong）

- **URL：** <https://opendrivelab.com/EgoHumanoid-V2/>
- **标题：** EgoHumanoid-V2: Human-to-Humanoid Transfer of Coordinated Whole-Body Skills for Loco-Manipulation
- **核查日期：** 2026-10-12
- **代码：** <https://github.com/OpenDriveLab/EgoHumanoid-V2>（项目页 Code 按钮）
- **论文：** <https://arxiv.org/abs/2609.37181>
- **许可与开放状态：** 官方仓库标注 Apache-2.0；截至核查日属于**部分开源**，不是论文全部训练/部署流水线的完整复现包
- **源码归档：** [OpenDriveLab/EgoHumanoid-V2](../repos/egohumanoid-v2.md)
- **论文归档：** [arXiv:2609.37181](../papers/egohumanoid_v2_arxiv_2609_37181.md)
- **覆盖 wiki：** [EgoHumanoid-V2](../../wiki/entities/paper-egohumanoid-v2.md)

## 项目页内容

主页把贡献概括为「第一视角人体示范到协调全身 loco-manipulation 的人形技能迁移」：粗到细动作对齐由运动学参考修正与动力学感知精炼组成，四项真机任务展示零目标任务机器人示范的技能迁移。项目页交互区域展示人体采集、未对齐动作回放、对齐后推理结果、动作对齐误差和视觉处理流程；正文数字以论文为准。

## 代码开放核查

项目页提供 Code 链接。核查其指向的官方 GitHub README，仓库公开并使用 Apache-2.0。README release checklist 将**硬件配置、环境安装、人类数据采集、动作对齐**列为已提供；其中动作对齐能用 `eh2-align` 和 `eh2-replay` 跑通样例轨迹。下列部分仍标记为 Coming soon / 未完成：机器人数据采集流程、视觉对齐代码、策略训练流水线、策略部署流水线。

仓库提供四项任务的动作对齐示例，但 README 未列出论文全部人类训练数据或策略 checkpoint 的完整下载入口。故此页记录为**训练/数据采集/动作对齐部分开源**：已经有可运行的对齐入口，但研究论文的端到端训练与部署链尚未全部公开。

## 项目关系

- 项目主页：<https://opendrivelab.com/EgoHumanoid-V2/>
- 代码：<https://github.com/OpenDriveLab/EgoHumanoid-V2>
- 论文：<https://arxiv.org/abs/2609.37181>
- Wiki：[EgoHumanoid-V2](../../wiki/entities/paper-egohumanoid-v2.md)
