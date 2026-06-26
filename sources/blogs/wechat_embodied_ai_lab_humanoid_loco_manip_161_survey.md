# 重磅整理！161篇论文带你看人形机器人移动操作的十个方向和技术版图全景

> 来源归档（blog / 微信公众号）

- **标题：** 重磅整理！161篇论文带你看人形机器人移动操作的十个方向和技术版图全景
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A
- **发表日期：** 2026-06-26
- **入库日期：** 2026-06-26
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox）；正文约 8.0 万字 / 161 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **姊妹篇：** [运动小脑 64 篇](wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)、[Loco-Manip 8 篇数据入口](wechat_embodied_ai_lab_loco_manip_8_papers_survey.md)、[42 篇 humanoid RL 身体系统栈](wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- **一句话说明：** 将 **161 篇** 人形 loco-manip 论文按 **十个方向** 策展：运控底座 → 移动操作接口 → 视觉/生成/人类示范 → 接触任务 → 数据遥操作 → 硬件部署 → VLA/WM → ego 视频；核心判断：应按 **能力形成顺序** 串读，而非按时间堆摘要。

## 核心摘录（归纳，非全文）

### 十个大类（文内编号）

| # | 方向 | 篇数 | 核心问题 |
|---|------|------|----------|
| 01 | 运控基座与通用全身跟踪 | 31 | 身体如何稳住、跟踪、抗扰、通用执行？ |
| 02 | 上半身中心控制与移动操作接口 | 24 | 手–臂–腰–根如何协同接住任务？ |
| 03 | 视觉感知驱动的人形移动操作 | 37 | 视觉如何闭环定位、理解、执行？ |
| 04 | 生成式运动、语言控制与轨迹规划 | 16 | 语言/条件如何生成全身轨迹？ |
| 05 | 动捕、人类视频与交互动作规划 | 11 | 人类动作如何变成机器人先验？ |
| 06 | 特殊任务、接触规划与视觉闭环 | 8 | 开门/推物/搬运等接触任务 |
| 07 | 数据采集与遥操作系统 | 7 | 训练数据如何高效采集？ |
| 08 | 硬件平台、感知配置与部署扩展 | 11 | 本体、传感与真机部署 |
| 09 | 人形 VLA、世界模型与通用操作 | 15 | VLA/WM 如何接到执行层？ |
| 10 | 从人类第一视角视频学习 | 1 | ego 视频如何进入学习链？ |

### 读法（策展）

文内建议按 **能力形成顺序**：运控底座 → 全身控制接移动操作 → 视觉/语言/轨迹/人类示范 → 数据采集与硬件 → VLA 与世界模型 → ego 视频。

## 对 wiki 的映射

- [humanoid-loco-manip-161-papers-technology-map](../../wiki/overview/humanoid-loco-manip-161-papers-technology-map.md)（**父节点** + Mermaid）
- 子分类 hub：`wiki/overview/loco-manip-161-category-01-motion-base-wbt.md` … `10-ego-video.md`
- 全量索引：[humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)（**161/161** 各建 `paper-loco-manip-161-{NNN}-*` 独立实体）
- 与 [运动小脑 64 篇](../../wiki/overview/humanoid-motion-cerebellum-technology-map.md)、[Loco-Manip 8 篇](../../wiki/overview/loco-manip-8-papers-technology-map.md) **交叉覆盖、视角不同**

## 可信度与使用边界

- 本文为 **微信公众号策展导读**（含 LLM 风格「算法实现总结」模板句），论文细节以 arXiv / 原文为准。
- 161 图本地化体积大，完整正文见 `sources/raw/`；wiki 侧 **不复述逐篇摘要**。
- 原始抓取见 [wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)。
