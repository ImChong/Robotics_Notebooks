# Simate 官网（Physical AI 平台）

> 来源归档

- **标题：** Simate — Intelligence, in motion
- **类型：** site（Physical AI 平台 + 模型 + AI Scientist 产品站）
- **链接：** <https://simate.ai/>（首页重定向至 <https://simate.ai/home/>）
- **产品页：** <https://simate.ai/research/sinfra/>（Sinfra）· <https://simate.ai/research/sipai/>（Sipai）· <https://simate.ai/research/RoboScientist/gallery/>（RoboScientist 实验画廊）
- **工作区入口：** <https://simate.ai/sifra/#/login>（Enterprise / Cloud workspace；页内 CTA 亦写「Explore the workspace」）
- **关于：** <https://simate.ai/about/> · 招聘 wildcard：`hr@simate.ai`
- **社媒：** [X @SimateAI](https://x.com/SimateAI) · [YouTube @simate_ai](https://www.youtube.com/@simate_ai) · Bilibili · 微信公众号（站内 QR）
- **演示视频 CDN：** `mate-robot.cn/open_videos/`（真机 demo 与 Sipai showcase 托管于此，非 Simate 自有 GitHub）
- **入库日期：** 2026-09-23
- **一句话说明：** Simate 面向 Physical AI 的 **Platform + Model + Scientist** 闭环：**Sinfra** 把任务从定义经训练/仿真/部署串成可审计流水线；**Sipai** 为具身动作模型栈（正进行 **RoboDojo** 评测）；**RoboScientist** 把实验与证据转为下一轮迭代输入。
- **沉淀到 wiki：** [`wiki/entities/simate.md`](../../wiki/entities/simate.md)

## 开源核查（步骤 2.5，2026-09-23）

| 资源 | 状态 | 说明 |
|------|------|------|
| **Sinfra / Sipai / RoboScientist 平台与模型** | **未开源** | 全站 **无** Simate 官方 GitHub / Hugging Face 模型仓；Sinfra 工作区需 **Request enterprise access** 或 **Sign in**（`/sifra/#/login`） |
| **Sipai 权重 / 训练代码** | **未公开** | Sipai 页标注 Training / Evaluation / Deployment 均为 **In development**；首页状态 **ROBODOJO EVALUATION** — 结果尚未公开 overstated |
| **RoboScientist** | **Research preview** | 实验画廊页可浏览叙事；无独立开源仓库链接 |
| **Sinfra 集成数据集** | **第三方已开源/开放** | 平台内链 **AgiBot World Beta**、**Daimon-Infinity**、**ABC-130k**、**Open X-Embodiment** 等公开数据（各依原许可） |
| **演示视频** | **公开托管** | `mate-robot.cn` 上 ring placement、tabletop sweeping 等 MP4；**不等于** 训推代码开放 |

## 产品三分法（官网叙事）

| 组件 | 状态（站内 badge） | 职责 |
|------|-------------------|------|
| **Sinfra** | IN DEVELOPMENT / PLATFORM | 任务定义 → 训练 → 仿真验证 → 真机部署 → 改进；连接数据、算力、实验记录与成本 |
| **Sipai** | ROBODOJO EVALUATION / MODEL | 具身动作模型：「One config. Everything.」— 数据、模型与训练配方可组合 |
| **RoboScientist** | RESEARCH PREVIEW / AI SCIENTIST | 实验 → 证据 → 洞察，支撑下一轮 Physical AI 任务 |

闭环：**Define & deploy（Sinfra）→ Learn how to act（Sipai）→ Experiment & discover（RoboScientist）→ Return evidence（↺ 改进下一任务）**。

## Sinfra 工作流摘录（2026-09-23）

1. **DEFINE** — 任务 brief、成功判据、可用数据；产出「Ready to train」门控。
2. **VALIDATE** — 训练 + 仿真 + 成本 + 指标并列对比；产出「Ready for robot trial」。
3. **DEPLOY** — 受控真机试跑、部署记录、Release or improve 决策。

**Task planner（站内交互）：** 可选任务类型（单臂 / 双手灵巧 / 移动操纵 / 视觉巡检等）、当前阶段与优先目标（首条证据 / 控成本 / 提部署信心），给出 **ILLUSTRATIVE** 算力组合（示例：**H100** 训练 · **RTX 5090** 仿真 · **H20** 推理）。

**公开数据集入口（Sinfra 页策展，非 Simate 自有）：**

| # | 数据集 | 许可（页内标注） | 规模摘要 |
|---|--------|------------------|----------|
| 01 | [AgiBot World Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | Non-commercial | 1M+ 轨迹 · 2976 h · 100 台机器人 |
| 02 | [Daimon-Infinity](https://github.com/dmrobot-admin/Daimon-Infinity) | CC BY-NC-SA 4.0 | 1209 h 视触语动作数据 |
| 03 | [ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) | Apache 2.0 | 130,703 episodes · 3590.7 h 双臂 YAM |
| 04 | [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) | 各源许可 | 1M+ 轨迹 · 22 本体 RLDS |

**算力展示（页内 DISPLAY FIGURES，非实时可用性报价）：** 仿真 **RTX 5090**（示例 ¥2.37/GPU·h · 1024 GPU）；训练 **H100**（1024 GPU · Availability on request）；推理 **H20**。

## 可公开证据（Evidence 区）

- **Ring placement** — Sinfra + Sipai 串联的仿真→真机示范（ring placement sim / real 图与视频）。
- **RoboDojo** — Sipai 评测进行中；首页写「Follow how Sipai is evaluated on physical AI tasks **without overstating results before they are public**」。
- 其它 demo 视频 ID（JSON）：tabletop sweeping、egg handling、plastic brick、tissue extraction 等（`mate-robot.cn`）。

## About 页组织文化摘录

- 「New SOTA reaches our robots **within a week**」— 强调真机快反馈。
- Wildcard 招聘：送 **一份最能证明判断力的作品**（论文 / 代码 / 系统 / 失败复盘），简历可选；`hr@simate.ai`。

## 对 wiki 的映射

- 主实体：[Simate](../../wiki/entities/simate.md)
- 评测交叉：[RoboDojo](../../wiki/entities/robodojo.md)（Sipai 当前评测基准，**非** Simate 运营）
- 数据集交叉：[Daimon-Infinity / 戴盟](../../wiki/entities/cn-os-daimon-infinity.md)
- 概念：[仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)、[VLA](../../wiki/methods/vla.md)
