---
type: entity
title: AMASS（统一 SMPL 表示的大规模人体动捕档案）
tags: [dataset, mocap, smpl, human-motion, deep-learning, animation, mpi-tuebingen]
summary: "AMASS 将多份光学标记动捕数据通过 MoSh++ 拟合到 SMPL 参数序列，形成可合并训练的人体运动档案；站点提供注册下载与教程代码，是机器人学习里常见的人体参考运动来源之一。"
updated: 2026-05-15
status: complete
related:
  - ../concepts/motion-retargeting.md
  - ./mimickit.md
  - ./protomotions.md
  - ./kimodo.md
  - ../methods/amp-reward.md
sources:
  - ../../sources/sites/amass-dataset.md
---

# AMASS（Archive of Motion Capture as Surface Shapes）

**AMASS** 是 MPI-IS Perceiving Systems 维护的 **人体运动元数据集**：把多份独立 **光学标记动捕** 序列转换到统一的 **SMPL**（及网格）参数化上，使动画、可视化与机器学习可以在同一表示下吃「合并后的」人类动作分布。

## 为什么重要？

- **打破数据集孤岛**：不同实验室 MoCap 的标记布局与后处理各异；统一到 SMPL 后，便于与 **重定向**、**生成模型**、**判别式运动先验（如 AMP）** 等下游共享同一接口。
- **规模与覆盖**：官网摘要对规模给出 **40+ 小时**、**300+ 被试**、**11000+ 动作** 等叙述量级（细节以论文与数据版本为准）。
- **与 SMPL 生态绑定**：人体形状与姿态参数可直接对接大量视觉与图形学工具链，降低从「原始标记」到「网络张量」的摩擦。

## 流程总览（概念级）

```mermaid
flowchart LR
  subgraph In[输入]
    M[多源光学标记 MoCap]
  end
  subgraph Fit[拟合]
    P[MoSh++]
  end
  subgraph Out[输出]
    S[SMPL 姿态/形状参数序列]
    U[网格与骨架<br/>便于渲染与导出]
  end
  M --> P --> S --> U
```

## 常见误区或局限

- **不是「开箱即用的机器人关节角」**：SMPL 系人体轨迹通常仍需 **[Motion Retargeting](../concepts/motion-retargeting.md)** 与动力学修正才能上机；否则易出现脚滑、穿透与力矩不可行。
- **许可与用途**：下载与使用受站点 **License** 约束；商业或再分发场景必须自行核对条款与引用要求。
- **与「视频估计人体」不同**：AMASS 主体来自 **标记动捕** 拟合，分布与单目视频估计的 SMPL 轨迹在噪声与偏差特性上并不相同。

## 与其他页面的关系

- **[ProtoMotions](./protomotions.md)**：官方文档把 AMASS 作为大规模并行训练的典型数据来源之一。
- **[MimicKit](./mimickit.md)**：研究管线中常出现从 AMASS（SMPL）到目标骨架的重定向工具链叙述。
- **[AMP 奖励与运动先验](../methods/amp-reward.md)**：MoCap 风格先验训练常以 AMASS 类统一表示为输入。

## 参考来源

- [AMASS 站点与论文索引归档](../../sources/sites/amass-dataset.md)
- Mahmood et al., *AMASS: Archive of Motion Capture as Surface Shapes* (ICCV 2019) — 论文 PDF 见站点链接
- AMASS 官网：<https://amass.is.tue.mpg.de/>
- 教程代码仓库：<https://github.com/nghorbani/amass>

## 关联页面

- [Motion Retargeting](../concepts/motion-retargeting.md)
- [ProtoMotions](./protomotions.md)
- [MimicKit](./mimickit.md)
- [Kimodo](./kimodo.md)
- [LaFAN1 动捕数据集](./lafan1-dataset.md)

## 推荐继续阅读

- SMPL 官方项目与文档（与 AMASS 参数字段对齐）
- [MoSh++ 与 AMASS 论文 PDF](http://files.is.tue.mpg.de/black/papers/amass.pdf)（方法细节与评测）
