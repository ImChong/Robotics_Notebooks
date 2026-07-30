---
type: entity
tags: [paper, quadruped, vision, locomotion, perception, mit, exploration]
status: complete
updated: 2026-07-25
venue: "ICRA 2020"
related:
  - ../queries/robot-perception-stack-selection-loop.md
  - ./mit-mini-cheetah.md
  - ./paper-wbic-mpc-mini-cheetah.md
  - ./paper-robust-autonomous-navigation-mini-cheetah-vision.md
  - ../concepts/footstep-planning.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/vision_aided_dynamic_exploration_mini_cheetah_icra_2020.md
  - ../../sources/blogs/robot_daycare_mini_cheetah_2019.md
summary: "Kim et al. ICRA 2020：Mini Cheetah 双 RealSense + 轻量落脚/避障评估，动态 trot 与跳跃探索非结构地形。"
---

# Vision Aided Dynamic Exploration of Unstructured Terrain

## 一句话定义

**Kim et al.（MIT，ICRA 2020，[DOI:10.1109/ICRA40945.2020.9196777](https://doi.org/10.1109/ICRA40945.2020.9196777)）** 在 **Mini Cheetah** 上集成两台 **Intel RealSense**，用简单滤波与落脚评估做落脚调整与避障，并结合**动态 trot 与跳跃**探索高度不规则地形。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RGB-D | RGB + Depth | RealSense 类深度相机模态 |
| FOV | Field of View | 小型机身限制传感器视场与安装 |
| WBC | Whole-Body Control | 底层全身/步态控制依赖 |
| MPC | Model Predictive Control | 常与动态步态联用 |
| ICRA | International Conference on Robotics and Automation | 发表会议 |

## 为什么重要

- 正面回答「小四足怎么塞视觉还保持动态」：传感空间、越障净空、速度缩放三重约束。
- 证明不必上极重感知栈，**轻量几何评估 + 动态运动**也能探索非结构地形。
- 为后续 [Mini-Cheetah Vision 导航](./paper-robust-autonomous-navigation-mini-cheetah-vision.md) 铺系统集成路。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 麻省理工（MIT） |
| **传感** | 双 Intel RealSense |
| **运动** | 动态 trot + jumping |
| **开源** | **感知栈未单独开源**；OA：https://hdl.handle.net/1721.1/138841 |

## 核心原理

- 小型平台约束 → 传感器选型与安装紧凑化。
- 点云/深度经简单过滤 → 落脚点可通行性与障碍评估。
- 动态步态与跳跃提供「感知分辨率不够时用运动换通过性」的互补。

## 源码运行时序图

**不适用**（截至入库日无官方可运行感知+控制一体仓库；底层 loco 可参考 Cheetah-Software）。

## 工程实践

| 项 | 建议 |
|----|------|
| 安装 | 优先保证前向与足端相关视野，避免自遮挡 |
| 算法 | 先可靠的高度/占用滤波，再上复杂建图 |
| 运动 | 跳跃用于净空不足，而非替代落脚规划 |

## 评测

| 维度 | 要点 |
|------|------|
| 平台 | 全传感 Mini Cheetah |
| 地形 | 高度不规则非结构地形探索 |
| 运动 | 动态 trot + jump |

## 结论

**总判：** 本文是 Mini Cheetah **外感知入门系统论文**——强调集成与动态运动，而非重型 SLAM。

- 真影响：双 RealSense + 轻量落脚评估可行。
- 次要代价：感知与规划仍偏启发式，泛化环境有限。
- 部署：与 [IROS 2020 导航](./paper-robust-autonomous-navigation-mini-cheetah-vision.md) 连读看系统化。

## 与其他工作对比

| 对照对象 | 差异要点 |
|----------|----------|
| 重型 SLAM / 建图感知栈 | 本文用双 RealSense + 轻量落脚评估过非结构地形，而非上重型建图 |
| [Mini-Cheetah Vision 导航](./paper-robust-autonomous-navigation-mini-cheetah-vision.md) | 本文是外感知入门系统；导航文进一步接入分层估计做无绳自主航点跟踪 |
| [Learning to Jump from Pixels](./paper-learning-to-jump-from-pixels.md) | 跳跃文用像素级学习执行敏捷跳跃；本文用几何评估 + 动态 trot/jump 探索 |

## 局限与风险

- 小平台算力/视角限制高速度下的感知质量。
- 无官方完整开源复现包。

## 关联页面

- [MIT Mini Cheetah](./mit-mini-cheetah.md)
- [Robust navigation](./paper-robust-autonomous-navigation-mini-cheetah-vision.md)
- [Footstep planning](../concepts/footstep-planning.md)
- [WBIC+MPC](./paper-wbic-mpc-mini-cheetah.md)

## 参考来源

- [论文归档](../../sources/papers/vision_aided_dynamic_exploration_mini_cheetah_icra_2020.md)
- [博文清单](../../sources/blogs/robot_daycare_mini_cheetah_2019.md)

## 推荐继续阅读

- OA：<https://hdl.handle.net/1721.1/138841>
