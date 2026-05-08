---
type: task
tags: [vln, navigation, embodied-ai, vision-language, matterport]
summary: "视觉–语言导航（VLN）要求智能体在三维环境中依据自然语言指令执行一系列离散或连续动作到达目标，是连接语言理解与空间运动规划的基准任务。"
updated: 2026-05-07
status: complete
related:
  - ../entities/sceneverse-pp.md
  - ../concepts/3d-spatial-vqa.md
  - ../methods/vla.md
  - locomotion.md
sources:
  - ../../sources/repos/sceneverse-pp.md
---

# 视觉–语言导航（Vision-and-Language Navigation, VLN）

**VLN**：智能体接收 **自然语言导航指令** 与 **第一人称（egocentric）视觉观测**（渲染视图或真实相机图像），在离散或连续动作空间中决策，最终到达指令描述的目标位置或物体。**语言–视觉接地** 与 **路径效率** 是核心评价维度。

## 为什么重要？

- **机器人场景**：家庭或服务机器人需要理解「穿过客厅，在冰箱左侧停下」这类指令；VLN 提供了可复现的 **语言–几何–动作** 闭环基准。
- **与纯导航的区别**：传统导航多依赖地图与坐标目标；VLN 强调 **语义描述**（地标、相对运动），更贴近人类口头指路。
- **与 VLA 的衔接**：高层策略可将 VLN 视作「语言条件下的路径生成」子问题；仿真基准（如 Matterport3D 上的 R2R）与真实视频蒸馏数据（如室内 tour）常混合使用以缓解 **sim–real** 与 **轨迹分布** 差异。

## 核心要素

| 要素 | 说明 |
|------|------|
| 环境 | 常用大规模室内扫描数据集（如 Matterport3D）构建可导航网格 |
| 观测 | 全景图序列或 pinhole 渲染视图；近年也引入真实行走视频 |
| 动作 | 常见为离散前向/转向步长；需与数据集标注一致 |
| 监督 | 专家轨迹模仿、强化学习、或从网页视频重建的伪轨迹 + VLM 生成指令 |

**分布差异**：仿真中最短路径、朝前行走居多；真实 Room-tour 视频存在停顿、回头与冗余旋转，直接用作监督需要 **轨迹清洗与动作离散化**（SceneVerse++ 论文中描述了面向 R2R 的三阶段管线）。

## 常见误区

- **误区**：「VLN 做得好就等于机器人能走。」仿真离散动作与真实连续控制、动力学约束仍有鸿沟，通常需要低层控制与碰撞规避模块。
- **误区**：「只用仿真轨迹训练就能覆盖真实室内。」真实视频的引入（含自动指令生成）是为了丰富 **语言风格与行走模式**，但仍需评估在标准基准上的可迁移性。

## 与其他页面的关系

- **数据**：[SceneVerse++](../entities/sceneverse-pp.md) 将室内漫游视频转为 R2R 兼容的离散导航数据，并报告在相关基准上的增益。
- **空间推理**：[3D 空间 VQA](../concepts/3d-spatial-vqa.md) 侧重问答；VLN 侧重 **时序决策**，二者常共享场景表示与 VLM 骨干。
- **运动基础**：[Locomotion](locomotion.md) 提供低层移动能力；VLN 更多占据 **任务规划与语义接地** 层，可与 VLA 分层结合。
- **模型**：[VLA](../methods/vla.md) 可作为统一骨架，在导航子任务上接入离散动作头或目标点输出。

## 参考来源

- [SceneVerse++ 原始资料归档](../../sources/repos/sceneverse-pp.md)
- Chen et al., *Lifting Unlabeled Internet-level Data for 3D Scene Understanding* (arXiv:2604.01907) — VLN 数据生成与 R2R 实验
- Anderson et al., *Vision-and-Language Navigation* — R2R 任务经典定义（如需溯源基准起源可查阅原文）

## 关联页面

- [SceneVerse++](../entities/sceneverse-pp.md)
- [3D 空间 VQA](../concepts/3d-spatial-vqa.md)
- [Locomotion](locomotion.md)
- [VLA](../methods/vla.md)

## 推荐继续阅读

- Matterport3D / R2R、RxR 等官方基准说明
- NaVILA、RoomTour3D 等「真实视频 + 导航」相关工作（与互联网视频蒸馏路线对照）
