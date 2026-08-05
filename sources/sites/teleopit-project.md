# Teleopit 项目页（botrunner64.github.io/teleopit-page）

> 来源归档

- **标题：** TeleopIt — A Full-Embodiment Humanoid Teleoperation System
- **类型：** site（项目页 + 演示视频 + 多仓代码入口）
- **URL：** <https://botrunner64.github.io/teleopit-page/>
- **论文：** <https://arxiv.org/abs/2608.01834>
- **PDF（项目内）：** <https://botrunner64.github.io/teleopit-page/assets/paper.pdf>
- **视频：** [YouTube](https://youtu.be/MNDOi0vQFEc)、[Bilibili](https://www.bilibili.com/video/BV1KJuw66EPQ)
- **机构：** 西湖大学（Westlake University）、上海创智学院（Shanghai Innovation Institute）
- **作者：** Bingqian Wu、Zicheng Xu、Xianghui Fan、Dayu Li、Xiangru Huang（通讯 Xiangru Huang）
- **硬件：** Unitree G1（29 DoF）；LinkerHand 等灵巧手；OpenNeck 2-DoF 主动颈
- **入库日期：** 2026-08-05
- **一句话说明：** 以 **PICO VR** 统一提供身体 / 手 / 头意图，集成 **全身 RL 跟踪 + 优化灵巧手重定向 + 主动视觉 + 异步录制**，并开源五仓栈，支持演示采集后训练 ACT / GR00T。

## 开源状态（步骤 2.5，截至 2026-08-05）

**已开源（多仓）。** 项目页 hero 区明确列出五个 GitHub 入口，非「宣称将开源」：

| 模块 | 仓库 | 说明 |
|------|------|------|
| Whole-Body Control | [BotRunner64/Teleopit](https://github.com/BotRunner64/Teleopit) | 主框架；mjlab 训练、ONNX sim2sim、Pico→G1 sim2real；Apache-2.0 |
| Dexterous Hands | [BotRunner64/somehand](https://github.com/BotRunner64/somehand) | 跨形态灵巧手优化重定向；YAML 手型配置 |
| Pico Interface | [BotRunner64/pico-bridge](https://github.com/BotRunner64/pico-bridge) | PICO ↔ PC 身体/手/头流与可选回传视频 |
| Active Vision | [BotRunner64/OpenNeck](https://github.com/BotRunner64/OpenNeck) | 2-DoF 颈云台硬件/驱动/URDF/MuJoCo |
| Imitation Learning | [BotRunner64/lerobot-teleopit](https://github.com/BotRunner64/lerobot-teleopit) | 录制→LeRobot Dataset→ACT/GR00T→策略服务 |

文档站：<https://BotRunner64.github.io/Teleopit/>（含中文）。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Hero | Paper / arXiv / YouTube / Bilibili / Twitter；五仓 Code 导航 |
| Motion Tracking | 前走、侧走、单腿平衡、蹲坐立、双膝跪等跟踪演示 |
| Loco-Manipulation | 移动拾放、袋搬运、开门、货架取物 |
| VLA Policy | 遥操作采数 + ACT/GR00T 自主 rollout |
| Latency Test | 主动视觉 / 全身 / 手重定向 / 视频流时延演示 |

## 核心摘录（面向 wiki 编译）

### 1) 全身体遥操作定位

- 相对 **TWIST2**（全身 + 主动视觉，手多为离散夹爪）与 **HumDex**（连续灵巧手但依赖定制惯性衣/手套）：Teleopit 主张 **单一 VR 传感源** 同时驱动 **全身 + 连续灵巧手 + 视点**。
- 异步 runtime 按各流固有频率连接传感、控制、视觉反馈与录制。

### 2) 与下游学习闭环

- 项目页单独展示 **VLA Policy**：Teleopit 采数 → 策略训练 → 真机自主。
- 论文同任务上，**96** 条成功演示训练 ACT / GR00T N1.7，瓶放置任务 SR **90.0% / 95.0%**。

### 3) 时延量级（论文视频估计，与页内 Latency 演示一致）

| 路径 | 约时延 |
|------|--------|
| 全身响应 | ~0.10 s |
| 主动视觉（视点） | ~0.05 s |
| 视频回传头显 | ~0.10 s |
| 人手→灵巧手显示配置 | ~0.15 s |

## 对 wiki 的映射

- 主实体：[Teleopit（论文实体）](../../wiki/entities/paper-teleopit.md)
- 论文摘录：[teleopit_arxiv_2608_01834.md](../papers/teleopit_arxiv_2608_01834.md)
- 主仓归档：[teleopit.md](../repos/teleopit.md)
- 手重定向仓：[somehand.md](../repos/somehand.md)
- 任务交叉：[Teleoperation](../../wiki/tasks/teleoperation.md)、[TWIST2](../../wiki/entities/paper-twist2.md)、[HEFT](../../wiki/entities/paper-heft.md)、[OASIS](../../wiki/entities/paper-loco-manip-04-oasis.md)
