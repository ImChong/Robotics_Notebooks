# Generalist AI 官网 Blog 列表核查

- **类型：** 官方站点索引（博客列表）
- **链接：** https://generalistai.com/blog （栏目筛选：Research / Ideas / Stories / Company）
- **机构：** Generalist AI, Inc.；官网 About 写明团队位于 Bay Area（CA）与 Boston（MA），未写成立日期。成立年份 **2024** 据 TechCrunch（2026-08-25）：<https://techcrunch.com/2026/08/25/robotics-startup-generalist-reaches-3b-valuation-sources-say/>
- **核查日期：** 2026-10-09
- **方法：** curl 读取 `/blog` 列表页 HTML（服务端渲染）中的标题、日期与链接；列表页未见分页或「加载更多」。
- **用途：** 确认公司页 [Generalist AI](../../wiki/entities/generalist-ai-robotics.md) 覆盖全部官方博文。

## 官网列表（2026-10-09 共 10 篇）

| 官网日期 | 标题 | 官网入口 | 本库节点 |
| --- | --- | --- | --- |
| 2026-08-19 | GEN-1.5 / Embodied Foundation Models are One-Shot Learners | `/blog/gen-1.5` | [GEN-1.5 一次示范学习](../../wiki/entities/generalist-gen15-one-shot.md)；归档 [generalist_gen15_one_shot](../blogs/generalist_gen15_one_shot.md) |
| 2026-07-23 | Towards Machines with a Thousand Hands | `/blog/towards-machines-with-a-thousand-hands` | [GEN-1 千手](../../wiki/entities/generalist-gen1-thousand-hands.md)；归档 [generalist_thousand_hands](../blogs/generalist_thousand_hands.md) |
| 2026-06-04 | Accelerating the Next Phase of Physical AI（Company） | `/blog/accelerating-the-next-phase-of-physical-ai` | 公司页小节；归档 [generalist_accelerating_physical_ai](../blogs/generalist_accelerating_physical_ai.md) |
| 2026-04-07 | Going Beyond World Models & VLAs | `/blog/beyond-world-models` | [GEN-1](../../wiki/entities/generalist-gen1.md)（与 GEN-1 主文合并） |
| 2026-04-02 | GEN-1 / Scaling Embodied Foundation Models to Mastery | `/blog/gen-1` | [GEN-1](../../wiki/entities/generalist-gen1.md) |
| 2026-03-24 | The Real Breakthrough Behind Our GTC Demo | `/blog/the-real-breakthrough-behind-our-gtc-demo` | [GEN-0](../../wiki/entities/generalist-gen0.md)（与 GEN-0 主文合并） |
| 2026-01-29 | The Dark Matter of Robotics: Physical Commonsense | `/blog/physical-commonsense` | [Physical Commonsense](../../wiki/entities/physical-commonsense-generalist.md)；归档 [generalist_physical_commonsense_2026](../blogs/generalist_physical_commonsense_2026.md) |
| 2025-11-04 | GEN-0 / Embodied Foundation Models That Scale with Physical Interaction | `/blog/gen-0` | [GEN-0](../../wiki/entities/generalist-gen0.md) |
| 2025-09-24 | The Robots Build Now, Too | `/blog/the-robots-build-now-too` | 公司页小节；归档 [generalist_robots_build_now_too](../blogs/generalist_robots_build_now_too.md) |
| 2025-06-17 | Research Preview | `/blog/research-preview` | 公司页小节；归档 [generalist_research_preview](../blogs/generalist_research_preview.md) |

## 三篇未单独建页博文要点

- **Research Preview（2025-06-17）：** 首篇公开博文；端到端网络（像素等传感 → 100 Hz 动作）全自主完成 4 个灵巧任务（分拣紧固件、折盒装链锁并合盖、收回 M4 螺丝、拆 / 分拣 / 抛掷乐高）；自报跨具身模型可在 7-DoF Flexiv Rizon 4 与 6-DoF UR5 间迁移，紧固件任务未用 UR5 数据。无定量指标。
- **The Robots Build Now, Too（2025-09-24）：** 内部评测任务「one-shot assembly」——人搭小乐高结构，机器人看后端到端复制；仅测过 4 色、3 块 2×4 砖的结构（作者估算组合空间 99,840）；作者称据其所知是首个端到端视觉运动控制拼装乐高的机器人。无成功率。
- **Accelerating the Next Phase of Physical AI（2026-06-04）：** 融资公告——新融资 4 亿美元、累计超 5 亿美元；Radical Ventures 领投，8VC、USV、Hanabi Capital、Norwest 新进，NVIDIA、Boldstart、Spark、Bezos Expeditions、NFDG 跟投；天使 Bin Lin、Fei-Fei Li、Naval Ravikant。回顾 GEN-0 / GEN-1 并称形成数据飞轮。估值（$2B）与「Series B」称谓仅见媒体报道。

## 列表外的相关入口

- **About 页**（<https://generalistai.com/about>）：使命「general intelligence for the physical world」；团队来自 OpenAI、Boston Dynamics、Google DeepMind 等；从 **灵巧性** 切入做具身基础模型。不计入博文。
- **YouTube / X / LinkedIn：** 演示视频分发渠道，不计入博文。

## 日期口径

本表日期取官网列表页显示日期；公司页时间线与之一致。GTC 演示博文与「Beyond World Models」分别并入 GEN-0、GEN-1 详情页（由其他页面维护）。
