---
type: entity
project_id: murea-inworld
project: https://murea.ai/
code: https://github.com/Murea-Labs/InWorld
tags:
  - world-models
  - video-generation
  - audio-generation
  - interactive-agents
status: draft
updated: 2026-10-18
related:
  - ../methods/generative-world-models.md
  - ../concepts/video-as-simulation.md
  - ../concepts/latent-imagination.md
  - ./cosmos-3.md
sources:
  - ../../sources/sites/murea-inworld.md
  - ../../sources/repos/murea-inworld.md
summary: "Murea-InWorld 将显式世界状态 Agent（Murea-Mind）与视听生成器 Murea-Fast 组合成互动故事闭环；Mind 与 Fast 推理代码已开源，Fast 权重已发布，报告中的 Murea-Pro 开放状态未确认。"
---

# Murea-InWorld（互动视听世界模型）

**Murea-InWorld** 是 Murea Labs 提出的互动视听世界模型：Agent 维护角色、物体、地点与故事进度，把用户动作转成生成条件，再依据生成片段更新后续计划。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AV | Audio-Visual | 音视频联合生成与观察 |
| WM | World Model | 根据状态和动作生成后续世界状态或观测的模型 |
| API | Application Programming Interface | Murea-Mind 与 Murea-Fast 之间的 HTTP 服务接口 |
| HF | Hugging Face | Murea-Fast 权重发布平台 |

## 为什么重要

- **显式状态与生成模型分工：** Murea-Mind 记录实体和叙事状态并做计划；Murea-Fast 将计划条件转成带声音的视频。这提供了一种可检查的 Agent + 生成器组合方式。
- **跨片段延续：** 角色外观通过参考图像和描述锚定；续写使用先前视频片段的末尾帧，目标是让场景和角色在多轮互动中保持连续。
- **动作到视听结果闭环：** 系统观察生成帧后再规划，便于研究长时交互中“状态记忆—事件规划—视频 rollout—状态校验”的接口。

## 核心原理

1. **维护世界状态：** Murea-Mind 管理角色、物体、地点和叙事进展，并将用户输入解释为上下文中的动作。
2. **生成事件条件：** Agent 规划后，把事件编译为提示词与参考图像；仓库 README 描述了确定性检查和 critic 等组件。
3. **生成视听片段：** Murea-Fast 基于参考图像生成视频与同步音频，并支持片段续写。两者通过文档化 HTTP API 通信。
4. **回看并更新：** 生成帧返回 Agent，作为后续规划与选定状态更新的依据；分支历史可用于生成替代续篇，多人 co-op 可共享动作输入。

```mermaid
flowchart LR
    U["用户动作"] --> M["Murea-Mind：解释、记忆、计划"]
    M -->|提示词与参考图像| F["Murea-Fast：视听推理服务"]
    F -->|视频与声音| M
    M -->|片段与后续选择| U
```

## 工程实践

| 项目 | README 所述 |
|------|-------------|
| 生成分辨率 / 帧率 | 1280×704、24 fps |
| 快速采样 | 8 步蒸馏采样 |
| 性能线索 | 5 秒片段在 4 张 H20 上约 17 秒，使用 sequence parallelism |
| 权重 | Hugging Face 发布 `murea-fast.safetensors`；部署需准备 LTX-2.5 基础组件 |
| 许可 | 仓库代码 Apache-2.0；模型权重应另核对其发布许可 |

以上实现与性能信息来自仓库 README 和部署文档入口；本页未独立运行代码或复现报告评测。

## 实验与评测

以下数字摘自官方技术报告 §5（作者自报，未独立复核）：

| 基准 | 被测模型 | 主要结果 | 读法 |
|------|----------|----------|------|
| WBench Full-289（289 例 / 1,058 轮交互） | Murea-Pro（50 步、1344×768） | Overall **81.13**，高于对比中最高的 Kling 3.0（79.42）；Interaction 82.60 vs 73.08、Physical 74.42 vs 69.19 | 优势集中在指令执行与物理/因果一致性；Setting、Consistency 分别比 Kling 3.0 低 4.48 / 1.70 分 |
| OpenS2V-Eval Open-Domain | Murea-Fast（8 步） | Total **64.98**，高于所评开源 R2V 基线（Bernini +1.77、MiniMax-H3 +3.97）；Seedance 2.0 为 67.13 | Seedance 2.0 有 32 例拒答；在 148 个共有样例上 Murea-Fast 66.11 vs 67.13 |
| OS2V-Pro（报告自建扩展评测） | Murea-Fast | 聚合 **69.9**，接近 Kling V3 Omni（70.1） | 参考保真度与片内身份一致性是最强分项；评测由作者团队定义 |
| 生成延迟 | Murea-Fast | 5.04 秒片段在 8×H20 上 43.7 秒（MiniMax-H3 494.2 秒、Bernini 527.5 秒） | 各系统部署配置不同，只能视为所测部署的对比 |

报告 §6 自述的主要短板是 **导航、长程规划与场景设定一致性**（导航为任务级最低分项）。

## 与其他工作对比

| 对照对象 | 共同点 | 与 Murea-InWorld 的区别 |
|----------|--------|--------------------------|
| [生成式世界模型](../methods/generative-world-models.md)（一般路线） | 以生成模型预测后续观测 | InWorld 把世界状态放在生成器之外，由 Murea-Mind 显式维护与校验，而非全部隐含在视频潜变量中 |
| [Cosmos 3](./cosmos-3.md) | 都覆盖视频与音频生成，WBench 表中 Cosmos3-Super / Nano 是对比基线 | Cosmos 3 面向 Physical AI，含动作与正逆动力学接口；InWorld 面向互动故事，没有机器人动作接口 |
| [视频作为仿真](../concepts/video-as-simulation.md) | 用生成视频承担 rollout | InWorld 的“动作”是用户的文字指令，评测关注指令执行与视觉一致性，不涉及机器人控制闭环 |

## 结论

**Murea-InWorld 的主要价值是“显式状态 Agent + 视听生成器”的分工设计，以及作者报告中交互指令执行分数的提升；它是生成式互动环境的参考，不是机器人策略或物理仿真器。**

1. **架构要点：** Murea-Mind 在生成器外维护角色、物体、地点与叙事状态，再用生成帧回写，形成状态—动作—观测闭环。
2. **评测读法：** WBench 上的领先主要来自 Interaction 与 Physical 两项，Setting 和 Consistency 仍落后于部分商用系统。
3. **开放范围：** Murea-Mind 与 Murea-Fast 推理代码、Murea-Fast 权重可用；WBench 成绩对应的 Murea-Pro 未确认开放，复现报告最高分需另行核实。
4. **机器人研究外推：** 缺少动作接口与物理一致性保证，只适合作为长时交互、状态记忆与视频 rollout 接口设计的参考。

## 源码运行时序图

```mermaid
sequenceDiagram
    autonumber
    actor Player as 用户
    participant Mind as agent（Murea-Mind）
    participant API as inference（HTTP API）
    participant Fast as Murea-Fast 服务
    Player->>Mind: 输入动作
    Mind->>Mind: 解释上下文并规划事件
    Mind->>API: 提示词、参考图像、生成请求
    API->>Fast: 执行视听生成
    Fast-->>API: 视频片段与声音
    API-->>Mind: 返回生成结果
    Mind->>Mind: critic 检查并更新选定状态
    Mind-->>Player: 展示片段与后续选择
```

该时序图依据仓库 README 对组件边界与闭环的概述；部署入口及 API 文档位于仓库的 `inference/` 与 `agent/` 目录。

## 局限与风险

- **开放范围不等于整篇报告模型均开放：** README 声明仓库提供 Murea-Mind 和 Murea-Fast 服务，且发布 Murea-Fast 权重；报告评测 Murea-Pro，但当前开源说明没有确认 Pro 权重或训练代码已发布。
- **算力要求较高：** README 给出的速度依赖 4 张 H20 与 sequence parallelism，不宜外推到单卡或机器人实时控制。
- **不是机器人控制策略：** 项目面向互动故事与视频生成；现有资料没有证明其具备机器人动作接口、物理一致性保证或真机控制能力。用于机器人研究时应视作生成式环境/仿真研究参考。
- **复现前核对基础权重与许可：** Murea-Fast 依赖 LTX-2.5 基础组件；HF 权重及基础组件的许可需分别确认。

## 关联页面

- [生成式世界模型](../methods/generative-world-models.md)
- [视频作为仿真](../concepts/video-as-simulation.md)
- [Latent Imagination](../concepts/latent-imagination.md)
- [Cosmos 3](./cosmos-3.md) — WBench 对比基线所属的 Physical AI 世界模型家族

## 参考来源

- [Murea-InWorld 官方项目页归档](../../sources/sites/murea-inworld.md)
- [Murea-InWorld 官方仓库归档](../../sources/repos/murea-inworld.md)
- [官方技术报告 PDF](https://murea.ai/page/paper/InWorld-technical-report.pdf)
- [GitHub 仓库 README](https://github.com/Murea-Labs/InWorld)

## 推荐继续阅读

- [Murea-InWorld GitHub 仓库](https://github.com/Murea-Labs/InWorld)
- [Murea-Fast Hugging Face 权重](https://huggingface.co/MureaLabs/InWorld)
- [技术报告](https://murea.ai/page/paper/InWorld-technical-report.pdf)
