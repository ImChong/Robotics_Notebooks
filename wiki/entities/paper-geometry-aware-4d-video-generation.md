---
type: entity
tags: [paper, stanford, realab, manipulation]
status: complete
updated: 2026-08-18
arxiv: "2507.01099"
venue: "ICLR 2026"
code: https://robot4dgen.github.io/
related:
  - ../overview/robot-world-models-training-loop-taxonomy.md
  - ../tasks/manipulation.md
  - ../methods/diffusion-policy.md
  - ../overview/realab-14-papers-technology-map-2026.md
sources:
  - ../../sources/papers/geometry_aware_4d_video_arxiv_2507_01099.md
  - ../../sources/blogs/wechat_shenlan_realab_14_papers_2026.md
summary: "Geometry-Aware 4D Video（ICLR 2026）：跨视角点图对齐监督的多视角一致 4D RGB-D 视频；无推理期相机位姿；位姿追踪恢复 EE 轨迹训策略。"
---

# Geometry-Aware 4D Video Generation for Robot Manipulation（arXiv:2507.01099）

**Geometry-Aware 4D Video Generation for Robot Manipulation**（Zeyi Liu, Shuang Li, Eric Cousineau, Siyuan Feng, Benjamin Burchfiel, Shuran Song；Stanford University; Toyota Research Institute；[arXiv:2507.01099](https://arxiv.org/abs/2507.01099)，[项目页](https://robot4dgen.github.io/)）— 用跨视角点图对齐监督 4D 视频生成，使多相机 RGB-D 未来帧时空几何一致，再经位姿追踪器提取末端轨迹训练操作策略。

## 一句话定义

用跨视角点图对齐监督 4D 视频生成，使多相机 RGB-D 未来帧时空几何一致，再经位姿追踪器提取末端轨迹训练操作策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| 4D | 3D space + time | 时空一致场景表征 |
| RGB-D | RGB + Depth | 彩色与深度观测 |
| EE | End-Effector | 末端执行器 |
| 6DoF | Six Degrees of Freedom | 位姿追踪输出 |

## 为什么重要

视频生成模型多视角几何不一致会误导下游操作；机器人需要可执行的时空一致预测。

## 核心原理（方法）

双视角 RGB-D 输入 → U-Net 预测未来点图与 RGB；训练期 cross-view pointmap 对齐；推理不需相机外参。

## 实验与评测

仿真操作任务新视角泛化；长时程双臂任务时空对齐优于基线。

## 结论

几何监督的 4D 视频是连接世界模型预测与可执行操作轨迹的中间表示。

- 跨视角点图对齐保证 3D 一致
- 推理无需相机位姿输入
- FoundationPose 等追踪器可恢复 EE 轨迹
- 支持 sim 策略迁移到真机 RGB-D

## 源码运行时序图

**不适用**（截至 2026-08-18：无统一公开可运行代码仓库，或本文为综述/控制器论文以项目页演示为主）。

## 局限与风险

依赖初始 RGB-D 质量；动态遮挡与快速接触仍挑战生成模型。

## 与其他工作对比

相对纯 2D 视频预测，显式 3D 几何约束；相对直接 VLA，多一步预测–追踪链。

## 关联页面

- [robot-world-models-training-loop-taxonomy](../overview/robot-world-models-training-loop-taxonomy.md)
- [manipulation](../tasks/manipulation.md)
- [diffusion-policy](../methods/diffusion-policy.md)
- [REALab 14 篇技术地图](../overview/realab-14-papers-technology-map-2026.md)

## 参考来源

- [geometry_aware_4d_video_arxiv_2507_01099.md](../../sources/papers/geometry_aware_4d_video_arxiv_2507_01099.md)
- [wechat_shenlan_realab_14_papers_2026.md](../../sources/blogs/wechat_shenlan_realab_14_papers_2026.md)

## 推荐继续阅读

- 论文：<https://arxiv.org/abs/2507.01099>
- 项目页：<https://robot4dgen.github.io/>
