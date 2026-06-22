---
type: entity
tags: [entity, simulator, embodied-ai, interactive-3d, visual-ai, ai2]
status: complete
updated: 2026-06-22
related:
  - ./habitat-sim.md
  - ./igibson.md
  - ./matterport3d-simulator.md
  - ../tasks/vision-language-navigation.md
  - ../overview/sim-platforms-decade-technology-map.md
sources:
  - ../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md
summary: "艾伦人工智能研究所 2017 年推出的交互式 3D 室内环境：高质量视觉渲染 + 细粒度物体状态交互，推动从被动视觉识别到主动具身交互的范式转变。"
---

# AI2-THOR

**AI2-THOR**（An Interactive 3D Environment for Visual AI）是艾伦人工智能研究所（AI2）于 2017 年推出的早期代表性 **交互式 3D 室内仿真环境**，面向视觉 AI 与具身交互研究。

## 一句话定义

> AI2-THOR 把「能看」推进到「能改环境状态」：不仅渲染高质量室内场景，还允许智能体打开微波炉、切苹果、改变物体温度等 **细粒度状态交互**，为指令跟随与视觉问答提供关键测试床。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AI2 | Allen Institute for AI | 美国艾伦人工智能研究所 |
| VQA | Visual Question Answering | 视觉问答任务 |
| IL | Imitation Learning | 从演示学习策略 |
| RGB | Red Green Blue | 三通道彩色图像观测 |

## 为什么重要

在 CV 与 NLP 交汇的早期，社区需要 **可重复、可交互** 的 3D 环境来评估「理解指令并在环境中行动」的智能体。AI2-THOR 的贡献在于：

1. **状态可变交互**：物体不仅有几何，还有开关、温度、切片等 **语义状态**——超越静态场景分类。
2. **室内视觉基准**：为 embodied AI 提供标准化房间布局与物体库，降低环境搭建摩擦。
3. **范式转折锚点**：与 [Matterport3D Simulator](./matterport3d-simulator.md)（真实扫描导航）、[Habitat](./habitat-sim.md)（高速渲染）形成 **视觉交互 → 真实感导航 → 吞吐** 的不同演进支路。

## 核心结构/机制

- **场景与物体库**：多房间室内布局 + 可交互日常物体。
- **状态机式交互**：操作触发物体状态转移（开/关、加热、切割等）。
- **Agent 接口**：支持第一/第三人称 RGB 观测与离散/连续动作（随版本演进）。

## 常见误区或局限

- **误区：AI2-THOR = VLN 平台** — 它更偏 **室内操作与视觉交互**；大规模语言导航基准见 [Matterport3D](./matterport3d-simulator.md) / [Habitat](./habitat-sim.md) 线。
- **局限：物理保真** — 早期版本物理简化；高保真操作与接触见 [SAPIEN](./sapien.md)、[ManiSkill2](./maniskill2.md) 等后继栈。

## 关联页面

- [十年仿真平台技术地图](../overview/sim-platforms-decade-technology-map.md)
- [Habitat-Sim](./habitat-sim.md) — 高速具身渲染对照
- [iGibson](./igibson.md) — 真实感场景 + 物理交互融合
- [视觉–语言导航](../tasks/vision-language-navigation.md)

## 参考来源

- [sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md](../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md)
- Kolve et al., *AI2-THOR: An Interactive 3D Environment for Visual AI* — [arXiv](https://arxiv.org/abs/1712.05474)

## 推荐继续阅读

- [AI2-THOR 项目页](https://ai2thor.allenai.org/)
- [Matterport3D Simulator](./matterport3d-simulator.md) — VLN 真实感基准姊妹线
