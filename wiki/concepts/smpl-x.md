---
type: concept
tags: [human-body-model, motion-retargeting, parametric-model, humanoid, mocap, world-models]
status: complete
updated: 2026-09-05
related:
  - ./motion-retargeting.md
  - ./motion-retargeting-pipeline.md
  - ../methods/motion-retargeting-gmr.md
  - ../entities/paper-mamma-markerless-motion-capture.md
  - ../entities/paper-dimos-human-scene-motion-synthesis.md
  - ../entities/gen2humanoid.md
  - ../entities/paper-uma.md
  - ../entities/paper-luna-universal-3d-human-animation.md
sources:
  - ../../sources/papers/mamma_arxiv_2506_13040.md
  - ../../sources/papers/dimos_arxiv_2305_12411.md
  - ../../sources/papers/coins_arxiv_2207_12824.md
  - ../../sources/papers/uma_arxiv_2506_01802.md
  - ../../sources/papers/luna_arxiv_2606_31981.md
summary: "SMPL-X（SMPL eXpressive）是马普所提出的参数化全身人体模型，用一组低维形状/姿态参数驱动约 10475 顶点的可微 mesh，统一身体 + 手 + 脸。它是人体动作捕捉、人-场景交互与「人类动作→人形机器人」重定向链路里事实上的中间人体表征。"
---

# SMPL-X（参数化全身人体模型）

**SMPL-X**（*SMPL eXpressive*）是在 [SMPL](https://smpl.is.tue.mpg.de/) 身体模型基础上扩展出的 **参数化全身人体模型**：用一组低维的 **形状（shape）** 与 **姿态（pose）** 参数，驱动一张约 **10475 顶点** 的可微三角网格（mesh），并在同一模型里统一了 **身体 + 双手（MANO）+ 面部（FLAME）** 的自由度。在本知识库里，它反复以 **人体动作的中间表征** 出现——动作捕捉先估计出 SMPL-X 序列，再由重定向前端把它映射到具体人形机器人。

## 一句话定义

**用少量形状/姿态参数生成一张可微全身人体 mesh 的参数化模型，作为「视频/mocap → 人形机器人」链路中统一的人体表征。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SMPL | Skinned Multi-Person Linear model | 只含身体的参数化人体模型，SMPL-X 的基础 |
| SMPL-X | SMPL eXpressive | 身体 + 手 + 脸的统一扩展，约 10475 顶点 |
| MANO | hand Model with Articulated and Non-rigid defOrmations | 参数化手部模型，被 SMPL-X 并入手部自由度 |
| FLAME | Faces Learned with an Articulated Model and Expressions | 参数化人脸模型，被 SMPL-X 并入表情自由度 |
| LBS | Linear Blend Skinning | 骨骼驱动顶点变形的线性混合蒙皮 |
| mocap | Motion Capture | 动作捕捉，SMPL-X 是其常见输出格式 |
| GMR | General Motion Retargeting | 把 SMPL-X 人体动作重定向到人形的方法 |

## 为什么重要

- **统一的人体「中间格式」：** 视频重建、marker/markerless mocap、生成式动作模型输出五花八门，但普遍收敛到 **SMPL-X 参数序列** 作为交换格式；下游只需对接一种人体表征即可接入重定向。
- **重定向链路的上游锚点：** [运动重定向管线](./motion-retargeting-pipeline.md) 通常是「生成/重建 → **SMPL-X** → 运动学重定向 → 人形关节轨迹」。把接缝显式暴露在 SMPL-X 上，便于定位 **脚滑、全局漂移** 等伪影是在上游生成阶段还是在重定向阶段被放大（见 [gen2humanoid](../entities/gen2humanoid.md) 中 `convert_smpl.py → GMR` 的分工）。
- **手 + 脸一体：** 相比只含身体的 SMPL，SMPL-X 的手部/表情自由度对 **灵巧操作、人-物交互、社交/HRI** 场景更贴近，接近人形机器人上肢与手的自由度上限（见 [DIMOS 人-场景动作合成](../entities/paper-dimos-human-scene-motion-synthesis.md) 类工作对人体自由度的沿用）。
- **可微 + 低维：** mesh 由参数可微生成，便于把「拟合观测」写成优化问题（拟合视频关键点、marker、接触），也便于在潜空间做生成与编辑。

## 参数结构速览

| 参数组 | 含义 | 典型维度量级 |
|--------|------|--------------|
| **形状 $\beta$** | 体型（高矮胖瘦），跨帧固定 | ~10–300 |
| **身体姿态 $\theta_b$** | 身体关节的轴角旋转 | ~21 关节 × 3 |
| **手部姿态 $\theta_h$** | 左右手（MANO）关节 | 两手，常用 PCA 压缩 |
| **表情/下颌 $\psi,\theta_f$** | 面部表情与下颌（FLAME） | ~10–100 |
| **全局 $R, t$** | 根朝向与全局平移 | 3 + 3 |

顶点由 **LBS**（线性混合蒙皮）从模板 mesh、形状/姿态相关的形变（blend shapes）与关节旋转共同生成，因此给定参数即可确定一张全身 mesh。

## 在本库中的典型用法

| 场景 | SMPL-X 的角色 | 代表页面 |
|------|----------------|----------|
| **markerless 动捕** | 多视角视频逐帧回归 SMPL-X 姿态/体型 | [MAMMA](../entities/paper-mamma-markerless-motion-capture.md) |
| **人-场景交互合成** | 用 SMPL-X + 体表 marker/顶点表示人体，联合物体接触 | [DIMOS](../entities/paper-dimos-human-scene-motion-synthesis.md)、[COINS](../entities/paper-coins-compositional-human-scene-interaction.md) |
| **生成→重定向接缝** | 生成式动作统一到 SMPL-X 再进 GMR | [gen2humanoid](../entities/gen2humanoid.md)、[GMR](../methods/motion-retargeting-gmr.md) |
| **多视角数字人元数据** | 与 DDC 角色并列提供 SMPL-X 拟合 mesh/姿态 | [UMA](../entities/paper-uma.md)（40×6K 着装 avatar 数据集） |
| **LBS-free 外观动画对照** | 推理不走 LBS，只在训练蒸馏软结构；输出是 3DGS 不是本页参数序列 | [LUNA](../entities/paper-luna-universal-3d-human-animation.md) |

## 与重定向的关系

SMPL-X 只是 **人体侧** 的表征；要驱动一台真实人形机器人，还需 [运动重定向](./motion-retargeting.md) 把人体骨架与形态映射到目标机器人的关节拓扑与限位。SMPL-X 的价值在于把「人体长什么样、怎么动」这件事标准化，让重定向前端（如 [GMR](../methods/motion-retargeting-gmr.md)）只需面对一种规整输入，而不必为每种 mocap/生成来源单独适配。

## 关联页面

- [运动重定向（Motion Retargeting）](./motion-retargeting.md) — SMPL-X 之后把人体动作映射到人形关节的环节
- [运动重定向管线](./motion-retargeting-pipeline.md) — 「生成/重建 → SMPL-X → 重定向」的整体链路
- [GMR（General Motion Retargeting）](../methods/motion-retargeting-gmr.md) — 以 SMPL-X 为统一输入的重定向方法
- [MAMMA（markerless 动捕）](../entities/paper-mamma-markerless-motion-capture.md) — 从多视角视频回归 SMPL-X
- [DIMOS（人-场景动作合成）](../entities/paper-dimos-human-scene-motion-synthesis.md) — 以 SMPL-X + 体表 marker 表示人体
- [gen2humanoid](../entities/gen2humanoid.md) — `convert_smpl.py` 统一到 SMPL-X 再进 GMR 的接缝
- [UMA（超精细可驱动 avatar）](../entities/paper-uma.md) — 数据集元数据同时发布 SMPL-X 与 DDC 角色
- [4DAnyone](../entities/paper-4danyone.md) — 仓内下载 SMPL-X 资产并经 GVHMR 出骨架；下游是多视角外观，不是本页的重定向中间格式用法
- [LUNA](../entities/paper-luna-universal-3d-human-animation.md) — LBS-free 的隐式 2D 驱动 3DGS 动画；对照「参数体中间格式」路线，不能当重定向输入

## 参考来源

- [MAMMA 论文归档（arXiv:2506.13040）](../../sources/papers/mamma_arxiv_2506_13040.md) — 逐帧回归 SMPL-X 姿态/体型
- [DIMOS 论文归档（arXiv:2305.12411）](../../sources/papers/dimos_arxiv_2305_12411.md) — SMPL-X + 67 体表 marker 表示
- [COINS 论文归档（arXiv:2207.12824）](../../sources/papers/coins_arxiv_2207_12824.md) — BodyVAE 回归 SMPL-X 与接触
- [UMA 论文归档（arXiv:2506.01802）](../../sources/papers/uma_arxiv_2506_01802.md) — 多视角数字人元数据中的 SMPL-X 角色
- [LUNA 论文归档（arXiv:2606.31981）](../../sources/papers/luna_arxiv_2606_31981.md) — 推理期丢掉 LBS、训练期仍蒸馏参数体结构
- SMPL-X 官方项目页 <https://smpl-x.is.tue.mpg.de/>（Pavlakos et al., CVPR 2019，Max Planck Institute for Intelligent Systems）
