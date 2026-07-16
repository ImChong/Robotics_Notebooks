# Philia 官方项目页（Astribot）

> 来源归档（ingest）

- **标题：** Philia: The Physical AI Symbiotic Agent
- **类型：** site
- **官方入口：** <https://www.astribot.com/research/Philia>（英文：<https://www.astribot.com/en/research/Philia>）
- **技术报告：** <https://arxiv.org/abs/2607.11377>
- **机构：** Astribot Team（星尘智能）
- **入库日期：** 2026-07-16
- **一句话说明：** **PHILIA** 是面向 **长期人机物理共存** 的 **多机器人 Agent 运行时**：在 **OpenClaw** 控制平面上，通过 **Robot Gateway** 能力抽象解耦低频语义推理与高频真机执行；项目页聚合 **7 段代表性场景演示视频**。

## 三层架构（项目页）

| 层 | 职责 |
|----|------|
| **User Interfaces** | IM、语音对话、Web 等人机入口 |
| **Agent Control Plane** | 意图理解、长期记忆、任务规划与调度、多机器人编排、安全授权 |
| **Robot Gateways** | 感知、导航定位、操纵策略、语音播报、状态监控；封装平台 middleware |

## 核心设计原则（项目页）

- **Compositional Intelligence：** 模块化子系统可独立演进（UI、推理模型、记忆、导航、策略、新本体）。
- **Memory-Grounded Assistance：** 四类持久语义记忆（偏好、交互历史、任务历史、语义地点）；**只辅助规划，不直接驱动低层控制**。
- **One Agent, Multi-Robot：** 多机共享同一助手身份，语义空间共享、能力匹配与任务仲裁。
- **Safety by Design：** 物理动作须经授权、确认、就绪检查、任务仲裁与 actor 级 stop/cancel。

## 演示视频索引（OSS 全量）

**CDN 根路径：** `https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/philia/`

### Hero

| 文件 | URL |
|------|-----|
| hero.mp4 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/philia/hero.mp4> |

### Multi-Robot Coordination and Control（1）

| # | URL |
|---|-----|
| 01 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/philia/multi-robot-control/01.mp4> |

### Reasoning Grounding（4）

| # | URL |
|---|-----|
| 01 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/philia/reasoning-grounding/01.mp4> |
| 02 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/philia/reasoning-grounding/02.mp4> |
| 03 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/philia/reasoning-grounding/03.mp4> |
| 04 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/philia/reasoning-grounding/04.mp4> |

### Memory Grounding（1）

| # | URL |
|---|-----|
| 01 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/philia/memory-grounding/01.mp4> |

### Policy Execution（1）

| # | URL |
|---|-----|
| 01 | <https://astribot-website-shenzhen.oss-cn-shenzhen.aliyuncs.com/media/philia/policy-execution/01.mp4> |

## 核心摘录（面向 wiki 编译）

### 1) Agent–Runtime 解耦

- **摘录要点：** 助手维护会话、记忆与规划；机器人网关暴露 **观测 / 操纵 / 导航 / 语音 / 取消** 等能力契约；策略迭代不改变控制平面接口。
- **对 wiki 的映射：**
  - [Philia](../../wiki/entities/philia.md) — 架构图与五段式路由级联
  - [Hermes Agent](../../wiki/entities/hermes-agent.md) — 另一类开源 Agent 运行时对照

### 2) 策略作为可插拔能力

- **摘录要点：** 操纵策略以 gateway capability 暴露；Agent 负责语义 grounding、策略选择与 prompt 构造；支持 **Lumo 系基础模型零样本**、任务 SFT 与部署后 **advantage-conditioned 离线后训练**。
- **对 wiki 的映射：**
  - [Philia](../../wiki/entities/philia.md) — Policies as Capabilities 小节
  - [Lumo-2](../../wiki/entities/lumo-2.md) — 上游通才策略后端

### 3) 代表性 Playbook 场景

- **摘录要点：** 推理接地（观测/偏好/热量/食物类别语义）、记忆接地（早餐偏好→后续任务）、多机并行（Alice 收拾桌面 + Bob 提垃圾袋）、即插即用策略升级。
- **对 wiki 的映射：**
  - [Philia](../../wiki/entities/philia.md) — Playbook 与视频维度对照
  - [Manipulation](../../wiki/tasks/manipulation.md) — 长程家务与服务任务语境

## 当前提炼状态

- [x] 项目页文案 + 全量演示视频 URL 已归档
- [x] 与技术报告 arXiv:2607.11377 交叉核对架构与 dispatch 评测
