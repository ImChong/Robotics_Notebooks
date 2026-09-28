# Humanoid Skills：从全身运动到操作任务

> 来源归档（blog / 微信公众号 · human five）

- **标题：** Humanoid Skills：从全身运动到操作任务
- **类型：** blog（ROBOTIS × NVIDIA 真机工程案例）
- **作者：** human five（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/JcaBxH1xBmRdEFjesOTC7A
- **发表日期：** 2026-09-28
- **入库日期：** 2026-09-28
- **抓取方式：** [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai) + Camoufox（`playwright==1.49.1`）
- **原始抓取落盘：** [`sources/raw/wechat_human_five_robotis_humanoid_skills_2026-09-28.md`](../raw/wechat_human_five_robotis_humanoid_skills_2026-09-28.md)
- **一句话说明：** ROBOTIS 在 **AI Sapiens K1**（全身 BeyondMimic + Isaac Lab + Newton 后端 + QDD 阻抗）与 **AI Worker**（268 条遥操作 + GR00T 1.7 + Cosmos Transfer 2.5 增广）两条 pipeline 上的 NVIDIA 物理 AI 栈落地复盘；核心判断：**全身靠可规模化仿真，操作靠可规模化真机数据（含合成视觉增广）**。

## 对 wiki 的映射

| 主题 | wiki |
|------|------|
| 阅读坐标 / 双 pipeline 地图 | [robotis-humanoid-skills-nvidia-stack-technology-map](../../wiki/overview/robotis-humanoid-skills-nvidia-stack-technology-map.md) |
| K1 软件入口 | [robotis-ai-sapiens](../../wiki/entities/robotis-ai-sapiens.md) |
| 半人形操作平台 | [robotis-ai-worker](../../wiki/entities/robotis-ai-worker.md) |
| 组织 hub | [robotis](../../wiki/entities/robotis.md) |
| 运动跟踪 | [BeyondMimic](../../wiki/methods/beyondmimic.md) |
| 视觉增广 | [Cosmos Transfer](../../wiki/entities/cosmos-transfer.md) |
| 物理后端 | [Newton Physics](../../wiki/entities/newton-physics.md) |

## 当前提炼状态

- [x] 公众号正文抓取
- [x] 技术地图 overview
- [x] ROBOTIS 实体页交叉更新（K1 / Worker / hub）
- [x] BeyondMimic / Cosmos Transfer / Newton 案例挂接
