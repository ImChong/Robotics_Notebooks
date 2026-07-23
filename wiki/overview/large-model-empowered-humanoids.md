---
type: overview
tags: [humanoid, llm, vla, vln, multimodal, foundation-model]
status: complete
updated: 2026-07-23
related:
  - ../methods/vla.md
  - ../tasks/vision-language-navigation.md
  - ../methods/humanoid-voice-interaction.md
  - ../entities/paper-vln-10-navid.md
  - ../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md
  - ../entities/humanoid-system-curriculum.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
summary: "大模型赋能人形的实现方法总览：语音 agent、VLA 操作、VLN 导航与端到端视频导航（NaVid）等路径，对应课程第 8.1 节选型地图。"
---

# 大模型赋能人形机器人

## 一句话定义

**大模型赋能人形**泛指用 **LLM/VLM/VLA** 等预训练模型承接语义理解与任务规划，再通过技能库、导航栈或端到端策略驱动人形执行——课程第 8.1 节的方法地图。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LLM | Large Language Model | 语言规划与对话 |
| VLM | Vision-Language Model | 视觉问答/接地 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLN | Vision-Language Navigation | 语言引导导航 |
| NaVid | Video-based VLM Navigation | 视频流 VLM 导航代表 |
| API | Application Programming Interface | 技能落地接口 |

## 为什么重要

- 避免「上大模型」变成空泛口号：必须选 **交互 / 操作 / 导航** 哪条落地路径。
- 课程 Ch8 实践（语音交互导航）= [语音交互](../methods/humanoid-voice-interaction.md) + [VLN](../tasks/vision-language-navigation.md) / [NaVid](../entities/paper-vln-10-navid.md)。

## 核心原理（实现路径）

| 路径 | 输入 | 输出 | 本库入口 |
|------|------|------|----------|
| 语音技能 agent | 语音 | 技能 API / 对话 | [语音交互](../methods/humanoid-voice-interaction.md) |
| VLA 操作 | 图像+语言 | 关节/末端动作 | [VLA](../methods/vla.md) |
| VLN 导航 | 图像+语言指令 | 导航动作/路点 | [VLN 任务](../tasks/vision-language-navigation.md) |
| 视频 VLM 导航 | RGB 视频流 | 底层导航动作 | [NaVid](../entities/paper-vln-10-navid.md) |

分类细节见 [VLM/VLN/VLA taxonomy](../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md)。

## 工程实践

- 教学优先 **模块化**：ASR→LLM→Nav/技能，可解释、易调试。
- 研究向再尝试端到端 VLA/NaVid；注意真机算力与安全层。

## 局限与风险

- 语义成功 ≠ 运动可行；必须保留运控与碰撞安全。
- 数据与权重许可、云端隐私需单独评估。

## 关联页面

- [人形系统课程策展](../entities/humanoid-system-curriculum.md)
- [人形算法研究现状](./humanoid-algorithm-research-status.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)

## 推荐继续阅读

- [VLN 开源复现四范式](./vln-open-source-repro-paradigms.md)
