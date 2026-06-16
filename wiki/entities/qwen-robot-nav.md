---
type: entity
tags: [navigation, vln, qwen, agentic, mobile-robot, autonomous-driving]
status: complete
updated: 2026-06-16
related:
  - ./qwen-robot-suite.md
  - ../tasks/vision-language-navigation.md
  - ./qwen-vla.md
  - ../methods/vla.md
sources:
  - ../../sources/blogs/qwen_robot_nav.md
  - ../../sources/blogs/qwen_robot_suite.md
summary: "Qwen-RobotNav 是基于 Qwen3-VL 的可扩展导航模型：以任务模式与可控观测参数（token budget、时间衰减、相机权重、帧采样）统一 VLN/PointNav/ObjNav/跟踪/闭环驾驶，15.6M 样本单权重，并作为 Qwen3.7-Plus 等 agent 的导航原语。"
---

# Qwen-RobotNav

**Qwen-RobotNav**（[GitHub](https://github.com/QwenLM/Qwen-RobotNav) | [深度博客](https://qwen.ai/blog?id=qwen-robotnav)）将 **多域导航**  reframing 为 **上下文建模问题**：共享 **Qwen3-VL 感知–规划骨干**，但 **如何消费视觉历史** 因任务而异，故暴露 **类 MCP 的可配置观测协议**，供上层 agent **按 call 指定**。

## 一句话定义

**Qwen3-VL + 4 层 MLP 航点头** 输出 **8 路点（位姿+朝向）**；**任务模式** 与 **四轴观测参数** 在 **15.6M 样本** 上训练时随机化，推理时外部可控，**五类导航任务单权重**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLN | Vision-Language Navigation | 按自然语言指令在环境中导航 |
| VLN-CE | VLN in Continuous Environments | 连续环境 VLN 基准族（如 R2R/RxR） |
| ObjNav | Object-Goal Navigation | 按物体目标类别导航 |
| EQA | Embodied Question Answering | 探索后回答需物理证据的问题 |
| PDMS | Predictive Driver Model Score | NAVSIM 闭环驾驶综合评分 |
| MCP | Model Context Protocol | LLM 工具上下文协议；Nav 观测协议类比 |

## 为什么重要

- **统一多域而不牺牲记忆策略：** 指令跟随要 **长记忆**，跟踪要 **近帧**；固定上下文假设的 unified nav 模型易 **偏科**。
- **Agent 原生：** 与 **Qwen3.7-Plus** 组成 **两层系统** + **evidence notebook**，在 **EXPRESS-Bench** 等 **长时程 EQA** 上 **步数更少、分数更高**。
- **真机 zero-shot：** **Unitree Go2**（Jetson Thor **196ms**）仅 **内置低分相机** 即做多房间 verbal 导航与 **21.78m 折返**。

## 核心结构/机制

### 可控观测四轴

| 参数 | 作用 |
|------|------|
| Visual token budget | 跨相机/时间的总 visual token 上限 |
| Temporal decay | 近期帧相对历史帧的权重衰减 |
| Camera weights | 各相机重要性（如前向 > 后向） |
| Frame sample mode | `random`（全局覆盖）vs `latest`（紧 recency） |

训练 **每样本随机** 四轴 → 推理 **任意组合无需改架构**。

### Agentic 调用面

每次导航 call 指定：**子目标语言** + **task mode**（VLN / PointNav / ObjNav / Tracking 等）+ **观测配置**。上层可 **episode 内切换 mode**。

### 数据与扩展

- **15.6M** 五任务族 + VLM 推理 co-train 保 grounding。
- **T2V→轨迹** 管线额外 **+40K** 逼真样本（无 3D 重建）。
- **2B→8B** scaling；**8B：R2R 72.1% / RxR 76.5% SR**。

## 公开指标（博客）

| 设置 | 结果 |
|------|------|
| HM3Dv2 ObjNav RGB-only（4B） | **75.6% SR**，距目标 **1.72m** |
| EVT-Bench tracking | **90.0%** |
| NAVSIM PDMS（4B） | **91.4** |
| Agentic + EQA 三基准 | **新 SOTA** |

## 常见误区或局限

- **误区：Nav 模型 = 仅 VLN 榜。** 同一权重还覆盖 **ObjNav、跟踪、NAVSIM 驾驶**；评测需按 **任务 mode** 切换。
- **局限：** 上层 **Qwen3.7-Plus planner** 与 **RobotClaw** 细节 **部分待更文**；闭环真机除 Go2 demo 外 **以博客为准**。

## 参考来源

- [Qwen-RobotNav 博客归档](../../sources/blogs/qwen_robot_nav.md)
- [Qwen-Robot Suite 总览](../../sources/blogs/qwen_robot_suite.md)
- [Qwen-RobotNav 技术报告 PDF](https://qianwen-res.oss-accelerate.aliyuncs.com/qwenrobot/papers/Qwen_RobotNav.pdf)
- [QwenLM/Qwen-RobotNav](https://github.com/QwenLM/Qwen-RobotNav)

## 关联页面

- [Qwen-Robot Suite](./qwen-robot-suite.md) — 套件总览与 agent 闭环
- [Vision-Language Navigation](../tasks/vision-language-navigation.md) — VLN / RxR / EQA 语境
- [Qwen-VLA](./qwen-vla.md) — 通才 VLA 亦报告 R2R/RxR 的对照
- [VLA](../methods/vla.md) — 导航–操作统一策略方法纵览

## 推荐继续阅读

- [Qwen-RobotNav 深度博客](https://qwen.ai/blog?id=qwen-robotnav)
- [Qwen-RobotNav GitHub](https://github.com/QwenLM/Qwen-RobotNav)
