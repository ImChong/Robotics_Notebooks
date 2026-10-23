---
type: entity
tags: [paper, survey, affordance-learning, embodied-ai, robot-learning, manipulation, perception, reasoning]
status: complete
updated: 2026-10-23
project_id: affordance-learning-embodied-ai-survey
project: "https://jingliangli.com/affordance-survey/"
venue: "Preprints.org preprint, v1 (2026-10-07)"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../overview/affordance-awesome-technology-map.md
sources:
  - ../../sources/papers/affordance_learning_embodied_ai_survey_preprints_2026100392.md
summary: "综述以 affordance 感知、任务相关推理、引导动作三段组织具身交互文献，解释 affordance 如何从被动视觉区域预测发展为接地的规划与策略接口。"
---

# 具身 AI Affordance 学习综述

**From Passive Perception to Active Interaction: A Survey of Affordance Learning for Embodied AI**（[Preprints.org 202610.0392](https://www.preprints.org/manuscript/202610.0392)，v1，2026-10-07）由南洋理工大学、香港科技大学（广州）、上海交通大学、UIUC、悉尼大学、利物浦大学、爱丁堡大学、清华大学、MBZUAI 与 ETH Zurich 等机构的作者共同撰写。

## 一句话定义

Affordance 是连接「环境里有什么」与「当前 agent 能对它做什么」的功能性表征。本综述把学习流程组织为 **感知 → 推理 → 引导动作**，从可交互对象/区域的识别，推进到任务相关性选择，再到可执行的计划或策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Affordance | Action possibility | 环境 / 物体对 agent 提供的可交互机会 |
| VLM | Vision-Language Model | 视觉–语言模型，用于 affordance 推理 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| HOI | Human-Object Interaction | 人–物交互，常作监督来源 |

## 核心框架

| 角色 | 要回答的问题 | 常见产物 |
|------|--------------|----------|
| Affordance perception | 什么交互可行、位置在哪里？ | 语义类别、部件/区域、mask、heatmap、keypoint、3D grounding |
| Affordance reasoning | 哪种机会适合当前任务、上下文与 agent 状态？ | 关系、任务意图、可行性判断 |
| Affordance-guided action | 如何把 affordance 转成行为？ | 动作原语、规划、策略；显式条件、隐式表征或优化信号 |

论文强调 affordance 并非物体固定属性：它取决于任务、场景、交互对象与 embodiment。识别杯子只是起点，还需判断在当前目标和末端执行器条件下从何处、以何种方式操作。

## 方法

本文按 affordance perception、affordance reasoning 与 affordance-guided action 三个角色组织方法：先接地可行动作及位置，再依据任务和本体筛选相关机会，最后把表征转为动作原语、规划或策略。

## 评测

综述整理代表性方法、数据集与评测协议，但未提出新的统一基准或控制实验。比较结果应结合任务、模态、监督信号和评测设置阅读。

## 对比

相较只覆盖视觉感知或机器人交互的局部综述，本文将 perception、reasoning、action 纳入同一条具身交互链；它提供分类与查阅框架，不构成跨论文性能排名。

## 领域演进

作者归纳了五条趋势：监督信号从密集标签扩展到多样证据；表征从 2D 视角相关预测迈向 3D 度量空间；输出从静态功能区域扩展到接触点、方向、姿态和接触后轨迹；类别从封闭词表迈向开放词汇；affordance 模块逐渐进入 diffusion 与 VLA 策略。

## 怎么使用这篇综述

- **找方法：** 按感知、推理、动作三个角色定位研究，不要把区域检测和可执行策略混为一类。
- **做系统设计：** 检查中间 affordance 是否包含当前 embodiment 可执行的信息，以及后续策略是否真正使用了它。
- **找资源：** 使用配套 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning) 清单查论文、数据集与基准。
- **读性能表：** 注意任务、输入模态、监督和评测设置；不同设置的数字不可直接排序。

## 资源与边界

| 项目 | 信息 |
|------|------|
| 论文类型 | 综述预印本；Preprints.org 标记为未同行评审 |
| 项目主页 | [affordance-survey](https://jingliangli.com/affordance-survey/) |
| 资源列表 | [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)；站内逐篇节点见 [技术地图](../overview/affordance-awesome-technology-map.md) |
| 新算法 / 新实验 | 本文为综述，不提出新的控制策略或统一性能基准 |

## 结论

这篇综述的实用价值在于提供一条清晰的接口链：**看见可交互区域 → 判断它对任务和本体是否合适 → 将其转换为动作**。设计具身系统时，需验证 affordance 表征能否传到规划与控制，并以实际可行性和执行结果评估它。

## 关联页面

- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [原始预印本](https://www.preprints.org/manuscript/202610.0392)
- [项目主页](https://jingliangli.com/affordance-survey/)
- [论文与数据集清单](https://github.com/hq-King/Awesome-Affordance-Learning)
- [归档摘录](../../sources/papers/affordance_learning_embodied_ai_survey_preprints_2026100392.md)
