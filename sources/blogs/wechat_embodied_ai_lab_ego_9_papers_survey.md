# 机器人下一代数据入口，可能就是 Ego：9 篇论文讲透第一视角技术路线

> 来源归档（blog / 微信公众号）

- **标题：** 机器人下一代数据入口，可能就是 Ego：9 篇论文讲透第一视角技术路线
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA
- **发表日期：** 2026-06-01
- **入库日期：** 2026-06-01
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 + `wechat-article-for-ai`（Camoufox）；正文约 0.94 万字 / 10 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **关联姊妹篇：** [42 篇 humanoid RL 身体系统栈](wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)、[机器人世界模型训练闭环](wechat_embodied_ai_lab_robot_world_model_training_loop.md)、[BFM 41 篇专题](wechat_embodied_ai_lab_bfm_41_papers_survey.md)
- **一句话说明：** 按 **四个问题**（Ego 数据怎么采、人→机策略数据、世界模型里 world/ego 分工、Ego 是否足够）串读 9 篇论文；核心判断：**Ego 是具身智能规模化数据入口，但不会自动变成机器人数据**——中间需手部追踪、重定向、物理过滤与世界模型推演。

## 核心摘录（归纳，非全文）

### 问题重框

- **Ego ≠ 头戴摄像头**：同时记录视线、手、身体、任务过程、遮挡、接触与临场决策；对机器人而言，执行前注意力与失败前调整往往比「最终动作结果」更稀缺。
- **读法：** 不按时间堆摘要，而按 **采集 → 人→机 → 世界模型 → Ego+Exo** 四组组织。
- **收束判断：** Ego 重要因它记录 **人类真实任务过程** 且贴近机器人 **从自身传感器看世界**；但第一视角视频须经轨迹重建、对齐、重定向与策略学习管线，才是可用机器人数据。

### 四个问题（对应 9 篇分组）

| 组 | 篇数 | 核心问题 | 代表论文 |
|----|------|----------|----------|
| **01 数据采集** | 2 | Ego 数据如何 **大规模、低成本** 采？ | AoE、EgoLive |
| **02 人→机器人** | 3 | 人类第一视角如何变成 **可训练策略数据**？ | EgoMimic、EMMA、Gaze2Act |
| **03 世界模型** | 2 | 长时程里 **world vs ego** 如何分工？ | Ego-Vision World Model、WEM |
| **04 Ego+Exo** | 2 | 只看第一视角 **够不够**？ | EgoExoMem、E³C |

## 9 篇论文索引

### 01 — 数据采集（2）

| # | 标题 | 年份 | 链接 |
|---|------|------|------|
| 01 | AoE: Always-on Egocentric Human Video Collection for Embodied AI | 2026 | https://arxiv.org/abs/2602.23893 |
| 02 | EgoLive: A Large-Scale Egocentric Dataset from Real-World Human Tasks | 2026 | https://arxiv.org/abs/2604.23570 · https://robotdata-market.jdcloud.com/console/market |

### 02 — 人→机器人（3）

| # | 标题 | 年份 | 链接 |
|---|------|------|------|
| 03 | EgoMimic: Scaling Imitation Learning via Egocentric Video | 2024 | https://arxiv.org/abs/2410.24221 |
| 04 | EMMA: Scaling Mobile Manipulation via Egocentric Human Data | 2025 | https://arxiv.org/abs/2509.04443 |
| 05 | Gaze2Act: Gaze-Conditioned Vision-Language-Action Policies for Interactive Robot Manipulation | 2026 | https://arxiv.org/abs/2605.30282 |

### 03 — 世界模型（2）

| # | 标题 | 年份 | 链接 |
|---|------|------|------|
| 06 | Ego-Vision World Model for Humanoid Contact Planning | 2025 | https://ego-vcp.github.io/ |
| 07 | World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks | 2026 | https://arxiv.org/abs/2605.19957 |

### 04 — Ego+Exo（2）

| # | 标题 | 年份 | 链接 |
|---|------|------|------|
| 08 | EgoExoMem: Cross-View Memory Reasoning over Synchronized Egocentric and Exocentric Videos | 2026 | https://arxiv.org/abs/2605.18734 |
| 09 | E³C: Video Generation with 3D Environmental Memory and Ego-Exo Human Pose Control | 2026 | https://e3c-videogen.github.io/ |

## 对 wiki 的映射

- [ego-9-papers-technology-map](../../wiki/overview/ego-9-papers-technology-map.md)（父节点 + Mermaid）
- [ego-category-01-data-collection](../../wiki/overview/ego-category-01-data-collection.md) … [ego-category-04-ego-exo-fusion](../../wiki/overview/ego-category-04-ego-exo-fusion.md)
- 论文实体：`wiki/entities/paper-ego-01-aoe.md` … `paper-ego-09-e3c.md`（06/07 与既有 `paper-hrl-stack-33`、`paper-wem` 互链）

## 可信度与使用边界

- 本文为 **微信公众号策展导读**，论文细节以 arXiv / 项目页为准。
- 部分 2026 工作可能尚无公开 PDF；链接以抓取日可访问页面为准。
