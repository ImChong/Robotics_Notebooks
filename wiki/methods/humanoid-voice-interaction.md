---
type: method
tags: [speech, hri, humanoid, asr, tts, llm, vln, interaction]
status: complete
updated: 2026-07-23
related:
  - ../overview/large-model-empowered-humanoids.md
  - ../tasks/vision-language-navigation.md
  - ../entities/paper-vln-10-navid.md
  - ../entities/humanoid-system-curriculum.md
  - ../entities/unitree-g1.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
summary: "人形智能语音交互系统：ASR→NLU/LLM→任务技能/导航→TTS 的闭环，是课程第 8.2 节与语音导航实践的工程骨架。"
---

# 人形机器人智能语音交互

## 一句话定义

**人形智能语音交互**是把 **语音识别、语言理解/大模型规划、技能或导航执行、语音合成** 串成可打断闭环，使人形能用自然语言接受任务——课程第 8.2 节与 Ch8 实践。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ASR | Automatic Speech Recognition | 语音→文本 |
| TTS | Text-to-Speech | 文本→语音 |
| NLU | Natural Language Understanding | 意图/槽位解析 |
| LLM | Large Language Model | 任务规划与对话 |
| VLN | Vision-Language Navigation | 视觉–语言导航下游 |
| VAD | Voice Activity Detection | 端点检测与打断 |

## 为什么重要

- 大模型赋能若只停在云端对话，没有 **ASR/TTS + 机器人动作接口** 就不能算系统课交付。
- 与 [VLN](../tasks/vision-language-navigation.md) / [NaVid](../entities/paper-vln-10-navid.md) 组合即「语音交互导航」实践。
- 方法选型总览见 [大模型赋能人形](../overview/large-model-empowered-humanoids.md)；导航落地仍遵守 [ROS 2 基础](../concepts/ros2-basics.md) 或厂商 SDK 速度接口。

## 主要技术路线

| 路线 | 理解层 | 执行层 |
|------|--------|--------|
| 技能表 + 规则 NLU | 意图/槽位 | 预注册 API（站立、挥手、导航点） |
| LLM agent | 多轮规划 | 工具调用白名单 |
| 语音 → VLN | 语言指令 | [VLN](../tasks/vision-language-navigation.md) / Nav2 目标 |
| 端到端语音导航（研究向） | 音频/文本条件策略 | 需额外安全层 |

## 核心原理

```mermaid
flowchart LR
  MIC["麦克风"] --> ASR["ASR"]
  ASR --> LLM["NLU / LLM 规划"]
  LLM --> SKILL["技能 / VLN / 运控"]
  SKILL --> TTS["TTS"]
  TTS --> SPK["扬声器"]
```

关键设计点：唤醒词、多轮上下文、技能白名单、失败重问、急停覆盖语音。

## 工程实践

- 教学可用云 ASR/TTS + 本地技能表；延迟敏感场景边缘化唤醒与 VAD。
- 导航类意图映射到 VLN 指令或 Nav2 目标点，而非直接关节角。
- 安全：移动中语音指令需确认或低速策略。

## 局限与风险

- 嘈杂赛场/工厂 ASR 掉字；误唤醒会导致危险运动。
- LLM 幻觉会生成不可执行技能——必须接地到已注册 API。

## 关联页面

- [大模型赋能人形](../overview/large-model-empowered-humanoids.md)
- [Vision-Language Navigation](../tasks/vision-language-navigation.md)
- [人形系统课程策展](../entities/humanoid-system-curriculum.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)

## 推荐继续阅读

- 开源语音栈示例：Whisper ASR + 任意 TTS + 机器人技能中间件（按部署约束选型）
