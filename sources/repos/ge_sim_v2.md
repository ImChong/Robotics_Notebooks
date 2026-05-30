# GE-Sim 2.0（AgibotTech 官方仓库）

> 来源归档

- **标题：** Genie Envisioner World Simulator 2.0 (GE-Sim 2.0)
- **类型：** repo
- **组织：** AgiBot / AgibotTech
- **代码：** <https://github.com/AgibotTech/GE-Sim-V2>
- **论文：** <https://arxiv.org/abs/2605.27491>
- **项目页：** <https://ge-sim-v2.github.io/>
- **入库日期：** 2026-05-30
- **一句话说明：** GE-Sim 2.0 技术报告与项目入口；README 标明代码、模型权重与评测工具链 **待发布**（ingest 时仅技术报告与项目页可用）。
- **沉淀到 wiki：** [GE-Sim 2.0](../../wiki/entities/ge-sim-2.md)

---

## 仓库状态（README，ingest 快照）

| 项 | 状态 |
|----|------|
| 技术报告 | 已发布（arXiv:2605.27491，2026-05-28） |
| 代码 | TODO |
| 模型权重 | TODO |
| 评测工具链 | TODO |

## 许可与上游

- 改编自 [Diffusers](https://github.com/huggingface/diffusers)、[Cosmos](https://github.com/nvidia-cosmos) 等部分遵循 **Apache-2.0**。
- 本仓其余数据与代码：**CC BY-NC-SA 4.0**（以 README 为准）。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 2B 动作条件多视角视频世界模型 + 闭环模块 |
| [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) | 像素级操纵仿真与内置任务裁判 |
| [EWMBench](../../wiki/entities/ewmbench.md) | 同 Agibot 生态：EWMBench 偏 **开环生成质量** 三维量；GE-Sim 2.0 偏 **闭环 rollout + 奖励** |
| [robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) | ③ 机器人视频世界模型 · 训练闭环案例 |

## 对 wiki 的映射

- 主实体页：**`wiki/entities/ge-sim-2.md`**
- 论文摘录：**`sources/papers/ge_sim_2_arxiv_2605_27491.md`**
- 项目页：**`sources/sites/ge-sim-v2-project.md`**
