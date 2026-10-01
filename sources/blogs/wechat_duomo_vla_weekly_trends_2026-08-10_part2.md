# [风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第二篇

> 来源归档（blog / 微信公众号）

- **标题：** [风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第二篇
- **类型：** blog
- **作者：** 多模空间（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/OWV8elPNXIds3MCR5CU5VQ
- **入库日期：** 2026-10-01
- **抓取方式：** WebFetch（wechat-article-for-ai 不可用）
- **原始抓取落盘：** [`sources/raw/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md`](../raw/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md)
- **一句话说明：** 15 篇 VLA 论文：RLinf 统一 RL 平台、G0.5/StellaVLA 架构、智驾 XCoT、SALT 动作–语言对齐、TMRL/MiDAS 训练、KV 门控、SJTU RoboHarness 记忆、VGA/3DGS 空间、TCAM/HandPrior 操控、DriveVLA-M0/DURA 安全；**15/15 独立 canonical 详情节点**（本 ingest **新建 12**、**复用 3**）。

## 15 篇 → 本库 canonical 节点

| # | 论文 | 章节 | arXiv | 节点 | wiki（canonical） |
|---|------|------|-------|------|-------------------|
| 01 | RLinf-VLA | 架构模块 | [2510.06710](https://arxiv.org/abs/2510.06710) | **新建** | [paper-rlinf-vla](../../wiki/entities/paper-rlinf-vla.md) |
| 02 | G0.5 | 架构模块 | [2608.11739](https://arxiv.org/abs/2608.11739) | **复用** | [paper-galaxea-g05](../../wiki/entities/paper-galaxea-g05.md) |
| 03 | StellaVLA | 架构模块 | [2608.11671](https://arxiv.org/abs/2608.11671) | **复用** | [paper-stellavla-structured-icl-vla](../../wiki/entities/paper-stellavla-structured-icl-vla.md) |
| 04 | XCoT-VLA | 架构模块/智驾 | [2608.10976](https://arxiv.org/abs/2608.10976) | **新建** | [paper-xcot-vla-driving](../../wiki/entities/paper-xcot-vla-driving.md) |
| 05 | SALT | 架构模块 | [2608.10484](https://arxiv.org/abs/2608.10484) | **新建** | [paper-salt-vla-action-language-alignment](../../wiki/entities/paper-salt-vla-action-language-alignment.md) |
| 06 | TMRL | 训练范式 | [2605.12236](https://arxiv.org/abs/2605.12236) | **新建** | [paper-tmrl-diffusion-timestep-pretraining](../../wiki/entities/paper-tmrl-diffusion-timestep-pretraining.md) |
| 07 | MiDAS | 训练范式 | [2608.11363](https://arxiv.org/abs/2608.11363) | **新建** | [paper-midas-minimal-data-vla-adaptation](../../wiki/entities/paper-midas-minimal-data-vla-adaptation.md) |
| 08 | Neural Introspection Gating | 性能提升 | [2608.10824](https://arxiv.org/abs/2608.10824) | **复用** | [paper-neural-introspection-gating](../../wiki/entities/paper-neural-introspection-gating.md) |
| 09 | RoboHarness（SJTU） | 长程记忆 | [2603.24060](https://arxiv.org/abs/2603.24060) | **新建** | [paper-robo-harness-memory-ic-adaptation](../../wiki/entities/paper-robo-harness-memory-ic-adaptation.md) |
| 10 | VGA | 空间感知 | [2604.12908](https://arxiv.org/abs/2604.12908) | **新建** | [paper-vga-vision-geometry-action](../../wiki/entities/paper-vga-vision-geometry-action.md) |
| 11 | Embodied MM Grounding | 空间感知/Agent | [2608.10756](https://arxiv.org/abs/2608.10756) | **新建** | [paper-embodied-multimodal-grounding-3dgs-mobile-manipulation](../../wiki/entities/paper-embodied-multimodal-grounding-3dgs-mobile-manipulation.md) |
| 12 | TCAM | 末端操控 | [2608.10718](https://arxiv.org/abs/2608.10718) | **新建** | [paper-tcam-deformable-manipulation-wbcd](../../wiki/entities/paper-tcam-deformable-manipulation-wbcd.md) |
| 13 | HandPriorScore | 末端操控 | [2608.11769](https://arxiv.org/abs/2608.11769) | **新建** | [paper-handprior-score-humanoid-dual-arm](../../wiki/entities/paper-handprior-score-humanoid-dual-arm.md) |
| 14 | DriveVLA-M0 | 异常处理/智驾 | [2608.10413](https://arxiv.org/abs/2608.10413) | **新建** | [paper-drivevla-m0-failure-aware-memory](../../wiki/entities/paper-drivevla-m0-failure-aware-memory.md) |
| 15 | DURA | 异常处理/安全 | [2608.10393](https://arxiv.org/abs/2608.10393) | **新建** | [paper-dura-diffusion-vla-visual-attack](../../wiki/entities/paper-dura-diffusion-vla-visual-attack.md) |

## 核心摘录（MVP）

### 1) 主题分布

- **架构与平台：** RLinf-VLA 统一 RL；G0.5 单流推理+动作；StellaVLA 结构化 ICL；XCoT-VLA / SALT 分别面向智驾 CoT 与动作语义对齐。
- **训练与效率：** TMRL 扩探索预训练；MiDAS 极少示范启动 RL；Neural Introspection Gating 训练无关 KV 复用。
- **空间与长程：** SJTU RoboHarness（**arXiv:2603.24060**，勿与 2607.18060 异构策略 Harness 混淆）；VGA 几何骨干；3DGS 移动操作 grounding。
- **操控与安全：** TCAM 柔性衣物 WBCD 冠军；HandPriorScore 双手先验诊断；DriveVLA-M0 失败记忆；DURA 视觉对抗。

### 2) 节点去重结论（2026-10-01 核查）

- **15/15** 各有独立 canonical 页（12 新建 + 3 复用）；**0** 重复 arXiv ID。

## 对 wiki 的映射

- 阅读坐标：[一周 VLA 趋势技术地图（2026.08.10 第二篇）](../../wiki/overview/vla-weekly-trends-2026-08-10-part2-technology-map.md)
- 同系列：[第一篇](wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) · [第三篇](wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) · [第四篇](wechat_duomo_vla_weekly_trends_2026-08-10_part4.md)

## 当前提炼状态

- [x] 15 篇索引与独立详情节点
- [x] 技术地图
- [ ] 各篇深读（待原文 / 项目页 follow-up）
