# [风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第四篇

> 来源归档（blog / 微信公众号）

- **标题：** [风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第四篇-顺祝大家双节快乐
- **类型：** blog
- **作者：** 多模空间（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/Eae0_0kuz-Hz8mmbpK3qBA
- **入库日期：** 2026-09-27
- **抓取方式：** WebFetch（wechat-article-for-ai 不可用）
- **原始抓取落盘：** [`sources/raw/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md`](../raw/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md)
- **一句话说明：** 14 篇 VLA 论文：三系统基础模型、过程奖励、推理期恢复、动态低延迟、力觉超声、端边协同、推测推理、比特翻转安全、flow RL 探索、阶段 LoRA、长程信用与记忆；**14/14 独立 canonical 详情节点**（本 ingest **新建 10**、**复用 4**）。

## 14 篇 → 本库 canonical 节点

| # | 论文 | 章节 | arXiv | wiki（canonical） |
|---|------|------|-------|-------------------|
| 01 | GigaBrain-0.7 | 架构模块 | [2608.15875](https://arxiv.org/abs/2608.15875) | [paper-gigabrain-0-7](../../wiki/entities/paper-gigabrain-0-7.md) |
| 02 | Robo-Dopamine 2.0 | 架构模块 | [2608.15680](https://arxiv.org/abs/2608.15680) | [paper-robo-dopamine-2](../../wiki/entities/paper-robo-dopamine-2.md) |
| 03 | CoRe | 架构模块 | [2608.14822](https://arxiv.org/abs/2608.14822) | [paper-core-vla-counterfactual-realignment](../../wiki/entities/paper-core-vla-counterfactual-realignment.md) |
| 04 | ReflexVLA | 架构模块 | [2608.14379](https://arxiv.org/abs/2608.14379) | [paper-reflexvla](../../wiki/entities/paper-reflexvla.md) |
| 05 | ForceU-VLA | 架构模块/医疗 | [2608.15009](https://arxiv.org/abs/2608.15009) | [paper-forceu-vla](../../wiki/entities/paper-forceu-vla.md) |
| 06 | ViTaR | 架构模块 | [2608.15816](https://arxiv.org/abs/2608.15816) | [paper-vitar](../../wiki/entities/paper-vitar.md) |
| 07 | EcoVLA | 性能提升 | [2608.15502](https://arxiv.org/abs/2608.15502) | [paper-ecovla](../../wiki/entities/paper-ecovla.md) |
| 08 | SpecVLA | 性能提升 | [2608.15636](https://arxiv.org/abs/2608.15636) | [paper-specvla](../../wiki/entities/paper-specvla.md) |
| 09 | VLA Bit-Flip Attacks | 安全防御 | [2608.15475](https://arxiv.org/abs/2608.15475) | [paper-vla-bit-flip-attacks-int8](../../wiki/entities/paper-vla-bit-flip-attacks-int8.md) |
| 10 | StructRL | 训练范式 | [2608.15139](https://arxiv.org/abs/2608.15139) | [paper-structrl](../../wiki/entities/paper-structrl.md) |
| 11 | PhaseLoRA | 训练范式 | [2608.15285](https://arxiv.org/abs/2608.15285) | [paper-phaselora](../../wiki/entities/paper-phaselora.md) |
| 12 | PACE（VLA 长程信用） | 长程记忆 | [2608.15026](https://arxiv.org/abs/2608.15026) | [paper-pace-phase-progress-vla](../../wiki/entities/paper-pace-phase-progress-vla.md) |
| 13 | Remember Smarter（RS） | 长程记忆 | [2608.15269](https://arxiv.org/abs/2608.15269) | [paper-remember-smarter-vla-memory](../../wiki/entities/paper-remember-smarter-vla-memory.md) |
| 14 | EvoScene-VLA | 长程记忆 | [2605.21862](https://arxiv.org/abs/2605.21862) | [paper-evoscene-vla](../../wiki/entities/paper-evoscene-vla.md) |

## 核心摘录（MVP）

### 1) 主题分布

- **架构与恢复：** GigaBrain-0.7 三系统扩展；Robo-Dopamine 2.0 过程奖励；CoRe 推理期反事实重对齐；Reflex 动态操纵；ForceU-VLA / ViTaR 模态扩展。
- **部署效率与安全：** EcoVLA 端–边协同；SpecVLA 推测–验证；INT8 比特翻转威胁模型。
- **训练与长程：** StructRL flow 动作空间探索；PhaseLoRA 阶段 LoRA；PACE / RS / EvoScene 长程信用与场景信念。

### 2) 节点去重结论（2026-09-27 核查）

- **14/14** 各有独立 canonical 页（10 新建 `paper-*` + 4 复用）；**0** 重复 arXiv ID。

## 对 wiki 的映射

- 阅读坐标：[一周 VLA 趋势技术地图（2026.08.10 第四篇）](../../wiki/overview/vla-weekly-trends-2026-08-10-part4-technology-map.md)
- 同系列：[第一篇](wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) · [第三篇](wechat_duomo_vla_weekly_trends_2026-08-10_part3.md)

## 当前提炼状态

- [x] 14 篇索引与独立详情节点
- [x] 技术地图
- [ ] 各篇深读（待原文 / 项目页 follow-up）
