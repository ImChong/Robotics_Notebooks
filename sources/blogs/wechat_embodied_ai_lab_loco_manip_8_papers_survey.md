# 人形机器人 Loco-Manip 这周都在卷啥？这 8 篇论文挺有意思

> 来源归档（blog / 微信公众号）

- **标题：** 人形机器人 Loco-Manip 这周都在卷啥？这 8 篇论文挺有意思
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/Ez87ljBYmCyIpLKjMjEyaQ
- **发表日期：** 2026-06-14
- **入库日期：** 2026-06-14
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install -e` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）；正文约 0.9 万字 / 14 图；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **关联姊妹篇：** [42 篇 humanoid RL 身体系统栈](wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)、[Ego 9 篇专题](wechat_embodied_ai_lab_ego_9_papers_survey.md)、[LEGS / 3DGS loco-manip VLA](wechat_embodied_ai_lab_legs_vla_3dgs_loco_manip.md)、[BFM 41 篇专题](wechat_embodied_ai_lab_bfm_41_papers_survey.md)
- **一句话说明：** 按 **四组数据入口**（第一视角、生成/仿真、命令空间/控制器、触觉/跨本体遥操作）串读 8 篇 2026-06 人形 loco-manip 论文；核心判断：数据入口正从单点真机遥操作扩展为 **混合数据生产链**——关键在来源、身体对齐、接触信息与控制器接口能否闭环。

## 核心摘录（归纳，非全文）

### 问题重框

- **Loco-manip ≠ 会走 + 会抓**：走、看、抓、搬、放、接触、恢复须同时发生；真机遥操作 **质量高但贵**，人类视频 **规模大但身体/动力学不对齐**，仿真 **可批量但视觉/接触有 gap**。
- **读法：** 不按时间堆摘要，而按 **四组数据入口** 组织：语义/动作来自哪里、如何对齐身体、如何进命令空间、如何补接触与跨平台复用。
- **收束判断：** 人形 loco-manip 数据入口正从单点采集变为混合链；真机遥操作仍重要，但人类视频、生成视频、仿真、触觉、跨本体遥操作与统一控制器接口均在并行试探。

### 四个分组（对应 8 篇）

| 组 | 篇数 | 核心问题 | 代表论文 |
|----|------|----------|----------|
| **01 第一视角数据** | 2 | 人类 ego 视频如何先补 **任务语义** 再补 **全身动作**？ | Ego-Pi、EgoPriMo |
| **02 生成与仿真数据** | 2 | 生成视频 / 仿真能否承担更重的 **可执行数据生产**？ | GenHOI、OASIS |
| **03 命令空间与控制器** | 2 | 多源数据最终如何落到 **解耦命令** 与 **统一 WBC**？ | VAIC、M3imic |
| **04 触觉与跨本体遥操作** | 2 | 接触/力与 **跨机器人** 遥操作如何提高数据复用？ | WT-UMI、X-OP |

## 8 篇论文索引

### 01 — 第一视角数据（2）

| # | 标题 | 机构 | 链接 |
|---|------|------|------|
| 01 | Ego-Pi: VLA Fine-Tuning for Ego-Centric Human and Robot Data | Stanford; Meta | https://egopipaper.github.io/ · arXiv:2606.08107 |
| 02 | EgoPriMo: Egocentric Motion Generation for Interactive Humanoid Control | 天大; 北航; DeepCybo 等 | arXiv:2606.08495 |

### 02 — 生成与仿真数据（2）

| # | 标题 | 机构 | 链接 |
|---|------|------|------|
| 03 | GenHOI: Contact-Aware Humanoid-Object Interaction by Imitating Generated Videos without Task-Specific Training | HKUST(GZ); 中科大; NUS 等 | arXiv:2606.12995 |
| 04 | OASIS: From Simulation Data Collection to Real-World Humanoid Loco-Manipulation | 中国电信 AI; 复旦; 上交等 | arXiv:2606.08548 |

### 03 — 命令空间与控制器（2）

| # | 标题 | 机构 | 链接 |
|---|------|------|------|
| 05 | VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands | 清华; HKUST(GZ); 小米机器人 | https://vaic-humanoid.github.io/ · arXiv:2606.09286 |
| 06 | M3imic: Learning a Versatile Whole-Body Controller for Multimodal Motion Mimicking | 东南; 清华; MBZUAI | https://github.com/Renforce-Dynamics/MultiModalWBC · arXiv:2606.04829 |

### 04 — 触觉与跨本体遥操作（2）

| # | 标题 | 机构 | 链接 |
|---|------|------|------|
| 07 | WT-UMI: Tactile-based Whole-Body Manipulation via Force-Supervised Contact-Aware Planning | Georgia Tech | https://wt-umi.github.io/WTUMI/ · arXiv:2606.13232 |
| 08 | X-OP: Cross-Morphology Whole-Body Teleoperation via MPC Retargeting | Amazon; UC Berkeley | arXiv:2606.07934 |

## 对 wiki 的映射

- [loco-manip-8-papers-technology-map](../../wiki/overview/loco-manip-8-papers-technology-map.md)（父节点 + Mermaid）
- [loco-manip-category-01-egocentric-data](../../wiki/overview/loco-manip-category-01-egocentric-data.md) … [loco-manip-category-04-contact-teleop](../../wiki/overview/loco-manip-category-04-contact-teleop.md)
- 论文实体：`wiki/entities/paper-loco-manip-01-ego-pi.md` … `paper-loco-manip-08-m3imic.md`（**GenHOI** 与既有 **SimGenHOI** 为不同工作，勿合并）

## 可信度与使用边界

- 本文为 **微信公众号策展导读**（「这周」周报体例），论文细节以 arXiv / 项目页为准。
- GenHOI（arXiv:2606.12995）≠ SimGenHOI（Paper Notebooks 待深读条目）。
- 原始抓取正文见 [wechat_loco_manip_8_papers_2026-06-14.md](../raw/wechat_loco_manip_8_papers_2026-06-14.md)。
