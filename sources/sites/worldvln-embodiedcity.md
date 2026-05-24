# WorldVLN 项目页（embodiedcity.github.io/WorldVLN）

> 来源归档

- **标题：** WorldVLN — Autoregressive World Action Model for Aerial Vision-Language Navigation
- **类型：** site（项目页 + 仿真/真机演示视频）
- **URL：** <https://embodiedcity.github.io/WorldVLN/>
- **论文：** <https://arxiv.org/abs/2605.15964>
- **代码：** <https://github.com/EmbodiedCity/WorldVLN.code>
- **入库日期：** 2026-05-24
- **一句话说明：** EmbodiedCity 官方页：World–Action 闭环推理示意、两阶段训练框架图、室内外 UAV 基准定量结果、WAM vs VLA / 自回归必要性 / Action-aware GRPO 消融叙事，以及室内外真机与仿真演示视频。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Highlights | 首个空中 VLN **自回归 WAM**；短视界潜转移 → waypoint；**Action-aware GRPO**；室内外 SOTA + 真机迁移 |
| World-Action Inference | 闭环：预测潜世界转移 → 解码动作 → 新观测回写上下文 |
| Two-Stage Training | SFT 接地 + GRPO 段级奖励与时间衰减 |
| Quantitative Results | 室外/室内 UAV 基准对比图（`figurenew.png`） |
| Training Analysis | WAM vs VLA、自回归必要性、GRPO 贡献（曲线 + demo 视频） |
| Demo Videos | Outdoor/Indoor 真机；UAV-Flow / IndoorUAV 仿真 |
| Citation | BibTeX |

## 对 wiki 的映射

- 主实体：[WorldVLN（论文实体）](../../wiki/entities/paper-worldvln-aerial-vln-wam.md)
- 论文摘录：[worldvln_arxiv_2605_15964.md](../papers/worldvln_arxiv_2605_15964.md)
