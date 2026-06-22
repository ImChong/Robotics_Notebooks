---
type: entity
tags: [entity, simulator, embodied-ai, navigation, meta, habitat, gpu-rendering]
status: complete
updated: 2026-06-22
related:
  - ./matterport3d-simulator.md
  - ./igibson.md
  - ./paper-vln-02-vln-ce.md
  - ../tasks/vision-language-navigation.md
  - ../overview/sim-platforms-decade-technology-map.md
  - ../overview/vln-10-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md
summary: "Meta AI 2019 年推出的高速具身 AI 仿真平台：单 GPU 数千至上万 FPS 渲染，使大规模真实扫描场景上的端到端 RL 训练可行；Habitat 2.0/3.0 扩展可交互物体与人类化身。"
---

# Habitat-Sim

**Habitat**（仿真核心常称 **Habitat-Sim**）是 Meta AI（原 Facebook AI Research）2019 年发布的 **具身 AI 研究平台**，以 **极致渲染吞吐** 为核心设计目标。

## 一句话定义

> Habitat 把仿真瓶颈从「能不能渲染」变成「能不能 **足够快** 地渲染」：在 Gibson、Matterport3D 等真实扫描场景上实现单 GPU **数千–上万 FPS**，让亿级探索步数的 RL 训练在工程上可行。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Habitat | Habitat Embodied AI Platform | Meta 具身 AI 仿真与研究栈 |
| VLN-CE | VLN in Continuous Environments | 连续动作空间 VLN 设定 |
| RL | Reinforcement Learning | 强化学习 |
| FPS | Frames Per Second | 每秒渲染帧数，吞吐核心指标 |

## 为什么重要

当智能体需要 **数十亿步** 环境交互时，渲染速度成为硬约束。Habitat 的贡献：

1. **吞吐突破**：高度优化的 C++ 渲染管线 + GPU 批处理，远超同期室内仿真器。
2. **真实场景资产**：原生支持加载 [Matterport3D](./matterport3d-simulator.md)、Gibson 等扫描场景。
3. **任务扩展**：Habitat 2.0 引入可交互物体；Habitat 3.0 引入人类化身与社交导航——从 **纯导航** 走向 **交互具身**。

## 核心结构/机制

- **Habitat-Sim**：底层渲染与物理（轻量接触）引擎。
- **Habitat-Lab**：上层任务定义、数据集接口与 baseline（VLN-CE、ObjectNav 等）。
- **资产管线**：MP3D / Gibson / HM3D 等场景网格 + episode 数据集。

## 常见误区或局限

- **误区：Habitat 只做 VLN** — Lab 层覆盖 ObjectNav、Rearrangement、Social Navigation 等多任务。
- **局限：操作物理** — 精细关节操作与接触仍以 [SAPIEN](./sapien.md)、[ManiSkill2](./maniskill2.md)、[iGibson](./igibson.md) 见长。

## 关联页面

- [Matterport3D Simulator](./matterport3d-simulator.md)
- [paper-vln-02-vln-ce](./paper-vln-02-vln-ce.md)
- [VLN 10 篇技术地图](../overview/vln-10-papers-technology-map.md)
- [十年仿真平台技术地图](../overview/sim-platforms-decade-technology-map.md)

## 参考来源

- [sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md](../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md)
- Savva et al., *Habitat: A Platform for Embodied AI Research* — [arXiv](https://arxiv.org/abs/1904.11121)

## 推荐继续阅读

- [Habitat 官方文档](https://aihabitat.org/)
- [VLN 任务页](../tasks/vision-language-navigation.md)
