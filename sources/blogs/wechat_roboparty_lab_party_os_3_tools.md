# RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源

> 来源归档（blog / 微信公众号）

- **标题：** RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源
- **类型：** blog
- **作者：** Roboparty萝博派对（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/DL-ypgpyLVnypxMwA5d5pw
- **发表日期：** 2026-07-14（frontmatter）
- **入库日期：** 2026-07-14
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；正文约 1.3 万字 / 6 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **姊妹关系：** 从 [Roboto Origin](../repos/roboto_origin.md)「开源一台人形」演进到「开源人形机器人基础设施」；与 [BFM 41 篇运控基座长文](wechat_embodied_ai_lab_bfm_41_papers_survey.md)、[训练栈分层解读](wechat_embodied_ai_lab_robot_training_stack_layers_2026.md) 互补
- **一句话说明：** RoboParty Lab 以 **Party OS** 为研发底座，首批开源 **MimicLite**（监督运动跟踪 infra）、**UFO**（无监督 RL 控制框架）、**human-humanoid-tools**（30 秒级动作重定向工具链）；核心判断：把人形研发中最耗时、最难复现的底层能力沉淀为可复用、可验证、可扩展的开放基础设施。

## 核心摘录（归纳，非全文）

### 问题重框

- **Lab 定位：** [RoboParty Lab](https://lab.roboparty.com) 面向「有能力、有技术、有想法」的开发者，提供 idea + 高自由度研究环境 + 开放平台，目标 3–6 个月内把想法变成可验证、可开源、可发表的成果。
- **痛点：** 好 idea 常输在底层基建——本体可靠性、数据来源、动作处理、训练框架稳定性、真机部署、实验复现、成果影响力，而非想法本身。
- **Party OS：** 连接本体、数据、训练、动作工具链、Sim2Real、真机验证、开源发布与技术影响力的 **研发底座**；GitHub：<https://github.com/Roboparty/Party_OS>。

### 三项开源工具（首批样本）

| 工具 | 类型 | 核心能力 | 仓库 |
|------|------|----------|------|
| **MimicLite** | 监督学习 / 运动跟踪 infra | 8×4090、约 3h 训练通用跟踪策略；any4hdmi + mjhub 统一数据与资产；跨 codebase 评测与 sim2real 部署层 | <https://github.com/Roboparty/MimicLite> |
| **UFO** | 无监督 RL 控制框架 | MJLab backend；8×4090 <12h 训 BFM-Zero；多机器人形态；TeCH 等新表征；首次开源无监督 RL 遥操作真机方案 | <https://github.com/Roboparty/UFO> |
| **hhtools** | Human-to-Humanoid 重定向 | Newton IK + Interaction-Mesh 双后端；30s 级 retarget；Any Motion / Any URDF / R2R；数据集分析与 3D 可视化 | <https://github.com/Roboparty/human-humanoid-tools> |

### MimicLite 要点

- **算力：** 约 24 GPU-hours（8×4090×3h），文称约为 SONIC 算力的 ~1/875，全局根部跟踪更好、局部身体跟踪相当。
- **Tracking Infra：** any4hdmi 统一 LAFAN、100STYLE、SONIC、真机数据；mjhub 保证训练/运动学/sim2sim 一致性。
- **遥操 + 高动态：** 同一 policy 支持 Pico/XR 低延迟遥操与高动态真机（虎跳、肩滚、侧手翻等）。
- **跨 codebase：** 模块化 observation interface，已接入 SONIC、HEFT、TeleopIT、Humanoid-GPT、BFM-Zero、TWIST2；YAML 定义 obs 顺序即可接入外部策略。

### UFO 要点

- **Fast Training：** MJLab backend；8×4090 <12h 完成 BFM-Zero；8×H200 6–8h；性能持续优于 BFM-Zero 原版。
- **通用可扩展：** 无缝适配不同机器人形态；多来源数据混合训练与灵活调度。
- **表征：** 集成 BFM-Zero（FB Representation）；探索 TeCH（Temporal Distance Modeling via Contrastive RL for Humanoid WBC）。
- **真机遥操：** 首次开源无监督 RL 控制遥操作代码与完整验证方案（深蹲、跪地、打滚、跌倒恢复、抗扰等）。

### hhtools 要点

- **Fast Retarget：** Newton IK（Warp 可并行）+ Interaction-Mesh（MPC solver）；单段复杂全身动作约 30s；支持批量并行。
- **Any Motion：** 自动识别 BVH/GLB/SMPL 及 AMASS、GVHMR、LAFAN1、OMOMO、PHUMA、Intermimic、Meshmimic 等数据集格式。
- **Any URDF：** 拖入 URDF + Mesh 即可，无需定制适配代码。
- **R2R：** 机器人到机器人动作互转，解决跨机型动作库迁移。
- **数据分析：** 关节轨迹、重心、接触热力图；按运动学指标筛选（如水平速度 3 m/s）。

### Party OS 四方向路线图（Lab 规划）

| 方向 | 英文 | 重点 |
|------|------|------|
| 通用基础运动 | Humanoid Locomotion | 数据 infra、Sim2Real/Real2Sim、BFM 通用运动模型 |
| 感知交互 | Humanoid Perceptive Interaction | 行走/奔跑/跳跃/攀爬；HSI、HOI |
| 全身操作 | Humanoid Whole-Body Manipulation | BFM 运控基座 + 可 scale 的 VLA / World Model |
| 智能体 | Agentic Humanoid | Agent + Skills 低成本高智能 |

## 对 wiki 的映射

- [roboparty-lab-party-os-technology-map](../../wiki/overview/roboparty-lab-party-os-technology-map.md)（**父节点** overview + Mermaid）
- 底座实体：[party-os](../../wiki/entities/party-os.md)（**父实体**，聚合三项工具）
- 子实体（**complete，非 stub**）：
  - [mimiclite](../../wiki/entities/mimiclite.md)
  - [roboparty-ufo](../../wiki/entities/roboparty-ufo.md)
  - [human-humanoid-tools](../../wiki/entities/human-humanoid-tools.md)
- 交叉更新：[roboto-origin](../../wiki/entities/roboto-origin.md)、[motion-retargeting](../../wiki/concepts/motion-retargeting.md)、[sonic-motion-tracking](../../wiki/methods/sonic-motion-tracking.md)、[mjlab](../../wiki/entities/mjlab.md)、[paper-bfm-zero](../../wiki/entities/paper-bfm-zero.md)

## 可信度与使用边界

- 本文为 **RoboParty 官方公众号发布**；性能数字（GPU-hours、1/875、训练时长）引用自文内自述，须以仓库 README / 复现实验为准。
- MimicLite 跨 codebase 接入列表以文内为准，具体 API 以 GitHub 为准。
- UFO 中 TeCH 为探索性表征，细节待论文/代码公开。
- 微信视频播放器 UI 残留已从 raw 抓取中剔除，CDN 图片未纳入 wiki 正文。

## 当前提炼状态

- [x] Agent Reach + Camoufox 正文抓取
- [x] 父/子节点 wiki 升格规划
- [ ] `make ci-preflight` 同步派生文件
