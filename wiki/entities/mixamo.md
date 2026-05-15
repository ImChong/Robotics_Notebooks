---
type: entity
title: Mixamo（Adobe 在线角色与动画服务）
tags: [commercial, animation, mocap-derived, character-rigging, game-pipeline, adobe]
summary: "Mixamo 是 Adobe 运营的在线角色与动画库：提供预设绑定角色、自动骨骼绑定与大量全身动作下载；面向影视/游戏管线，属商业服务。机器人研究中多作快速视觉原型源，不等同于可自由再分发的原始 MoCap 档案。"
updated: 2026-05-15
status: complete
related:
  - ../concepts/motion-retargeting.md
  - ./motioncode.md
  - ./amass.md
  - ./lafan1-dataset.md
  - ../methods/zest.md
sources:
  - ../../sources/sites/mixamo.md
---

# Mixamo

**Mixamo** 是 **Adobe** 旗下的 **Web 端角色动画服务**：浏览并下载带骨骼的 3D 角色与 **大量全身动作**（站点描述为专业演员动捕后迁移到角色），也支持上传自定义人形网格做 **Auto-Rigging**，并导出到常见 DCC 与游戏引擎工作流。

## 为什么重要？

- **极低摩擦的「人类动作占位」**：在预研阶段需要人形参考轨迹驱动渲染或交互 Demo 时，比自建 MoCap 室更快。
- **与科研数据源的差异**：Mixamo 资产经过 **产品化绑定与美术封装**，与 **[AMASS](./amass.md)** 或 **[LaFAN1](./lafan1-dataset.md)** 这类「研究向原始/半原始运动档案」在 **可追溯性、参数化表示、许可粒度** 上都不等价；写论文或开源发布时应分开声明数据来源。
- **异构监督讨论中的对照组**：在讨论「动画 / 非物理真实轨迹能否作为 RL 人类先验」时，可与 **[ZEST](../methods/zest.md)** 等对异构人类运动源的论述对照。

## 常见误区或局限

- **许可≠科研默认可用**：企业项目、数据集论文或开源权重若包含 Mixamo 导出资产，需核对 **Adobe 使用条款** 与是否允许 **再分发**。
- **动力学可信度**：动画为视觉效果优化，**不保证**满足特定机器人或地面接触物理；不宜直接当作 **辨识级 MoCap** 使用。
- **骨架与单位**：导出骨架拓扑、骨骼命名与单位可能与目标仿真不一致，仍需 **[Motion Retargeting](../concepts/motion-retargeting.md)** 或引擎侧重定向。

## 与其他页面的关系

- **[Motion Retargeting](../concepts/motion-retargeting.md)**：把 Mixamo 动作接到机器人或物理仿真时的必经概念页。
- **[MotionCode](./motioncode.md)**：同属「产业侧人体运动资产」叙事，可对照商业数据供应商与自采 MoCap。
- **[ZEST](../methods/zest.md)**：讨论异构人类运动数据（含非物理真实动画）作为 RL 输入时的框架参照。

## 参考来源

- [Mixamo 站点归档](../../sources/sites/mixamo.md)
- Mixamo 官网：<https://www.mixamo.com>

## 关联页面

- [Motion Retargeting](../concepts/motion-retargeting.md)
- [MotionCode](./motioncode.md)
- [AMASS](./amass.md)
- [LaFAN1](./lafan1-dataset.md)
- [ZEST](../methods/zest.md)

## 推荐继续阅读

- Adobe 官方 Mixamo 帮助与许可说明（以当前站点文档为准）
- [AMASS](./amass.md) 与 [LaFAN1](./lafan1-dataset.md) — 研究向动捕档案的对照入口
