# [风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第三篇

> 来源归档（blog / 微信公众号）

- **标题：** [风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第三篇-顺祝大家双节快乐
- **类型：** blog
- **作者：** 多模空间（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/YangjxIBI8830OtlBBn61g
- **入库日期：** 2026-09-26
- **抓取方式：** WebFetch（wechat-article-for-ai 不可用）
- **原始抓取落盘：** [`sources/raw/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md`](../raw/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md)
- **一句话说明：** 12 篇 VLA/VLN/智驾论文：异步切换、安全对抗纹理、智驾加速与 FIRE 自进化、空域任务生成、工具化 Agent、空中 VLN 记忆与跨域评测；**12/12 独立 canonical 详情节点**（本 ingest **新建 10**、**复用 2**）。

## 12 篇 → 本库 canonical 节点

| # | 论文 | 章节 | arXiv | wiki（canonical） |
|---|------|------|-------|-------------------|
| 01 | AtomBridge | 架构/异步 | [2602.09430](https://arxiv.org/abs/2602.09430) | [paper-atombridge](../../wiki/entities/paper-atombridge.md) |
| 02 | BICPO-VLA | 架构/异步 | [2608.13924](https://arxiv.org/abs/2608.13924) | [paper-bicpo-vla](../../wiki/entities/paper-bicpo-vla.md) |
| 03 | SONIC | 架构/全身 | [2511.07820](https://arxiv.org/abs/2511.07820) | [sonic-motion-tracking](../../wiki/methods/sonic-motion-tracking.md)（复用） |
| 04 | Decoding Task Progress from VLA | 分析诊断 | [2608.13474](https://arxiv.org/abs/2608.13474) | [paper-vla-representation-task-progress](../../wiki/entities/paper-vla-representation-task-progress.md) |
| 05 | UniTexture | 安全防御 | [2608.13453](https://arxiv.org/abs/2608.13453) | [paper-unitexture-vla-adversarial](../../wiki/entities/paper-unitexture-vla-adversarial.md) |
| 06 | FlashDrive | 性能/智驾 | [2608.12932](https://arxiv.org/abs/2608.12932) | [paper-flashdrive-vla-autonomous-driving](../../wiki/entities/paper-flashdrive-vla-autonomous-driving.md) |
| 07 | Temporal GRPO | 训练范式 | [2608.13026](https://arxiv.org/abs/2608.13026) | [paper-temporal-grpo](../../wiki/entities/paper-temporal-grpo.md)（复用） |
| 08 | FIRE-VLA | 训练范式/智驾 | [2608.13395](https://arxiv.org/abs/2608.13395) | [paper-fire-vla](../../wiki/entities/paper-fire-vla.md) |
| 09 | ARIES-Mission2 | 类 Agent/导航 | [2608.12763](https://arxiv.org/abs/2608.12763) | [paper-aries-mission2-aerial](../../wiki/entities/paper-aries-mission2-aerial.md) |
| 10 | ART（VLA + Tool-use） | 类 Agent | [2608.14047](https://arxiv.org/abs/2608.14047) | [paper-art-vla-on-the-fly-tool-use](../../wiki/entities/paper-art-vla-on-the-fly-tool-use.md) |
| 11 | DreamFly | 长程记忆/导航 | [2608.12308](https://arxiv.org/abs/2608.12308) | [paper-dreamfly-aerial-vln](../../wiki/entities/paper-dreamfly-aerial-vln.md) |
| 12 | SSP | 评测/智驾 | [2608.14024](https://arxiv.org/abs/2608.14024) | [paper-ssp-syn2sim2phy-vla-eval](../../wiki/entities/paper-ssp-syn2sim2phy-vla-eval.md) |

## 核心摘录（MVP）

### 1) 主题分布

- **异步与切换：** AtomBridge、BICPO-VLA — 新指令到达时状态已变，分阶段动作合成 + 偏好/Flow 优化缩短接管。
- **诊断与安全：** VLA 内部任务进度探针；UniTexture 跨任务对抗纹理。
- **智驾效率与 RL：** FlashDrive 推理加速；Temporal GRPO / FIRE-VLA 改进 GRPO 信用分配与失败自进化。
- **Agent 与空域：** ARIES-Mission2 任务生成；ART 工具化 VLA（CVPR 2026）。
- **记忆与评测：** DreamFly 因果记忆 + 滚动 horizon；SSP 事件对齐 Syn2Sim2Phy 评测。

### 2) 节点去重结论（2026-09-26 核查）

- **12/12** 各有独立 canonical 页（10 新建 `paper-*` + 2 复用）；**0** 重复 arXiv ID。

## 对 wiki 的映射

- 阅读坐标：[一周 VLA 趋势技术地图（2026.08.10 第三篇）](../../wiki/overview/vla-weekly-trends-2026-08-10-part3-technology-map.md)
- 同系列：[第一篇](wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) · [第一篇技术地图](../../wiki/overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)

## 当前提炼状态

- [x] 12 篇索引与独立详情节点
- [x] 技术地图
- [ ] 各篇深读（待原文 / 项目页 follow-up）
