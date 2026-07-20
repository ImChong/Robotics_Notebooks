# Niantic Spatial — Real-world Foundation Models for Physical AI

> 来源归档（ingest）

- **标题：** Niantic Spatial | Real-world foundation models for physical AI
- **类型：** site（企业官网）
- **机构：** Niantic Spatial, Inc.
- **原始链接：** <https://nianticspatial.com>
- **入库日期：** 2026-07-20
- **一句话说明：** 面向 **Physical AI** 的 **真实世界基础模型** 公司：从手机/360°/无人机/卫星/LiDAR 等采集 **几何精确数字孪生**（mesh + Gaussian splats），提供 **视觉定位（VPS）** 与 **空间理解** 能力；与 Flexion、NVIDIA 联合发布 **人形 RGB 导航 Real2Sim→Sim2Real** 管线（2026-07）。

## 三大能力（官网叙事）

| 能力 | 说明 |
|------|------|
| **Reconstruct** | 几何精确数字孪生；输出 mesh 与 Gaussian splats；创建开源 **SPZ** splat 文件格式（行业采用） |
| **Localize** | Visual Positioning System：空中与地面采集设备上的精确定位与朝向；GPS 失效场景可用 |
| **Understand** | 空间智能：可对话的物理世界 AI 接口，理解、测量与推理 3D 空间 |

## 与机器人 Sim2Real 相关摘录（2026-07 联合发布）

- **采集：** 商用 **360° 相机** 单次 walkthrough 数分钟即可；**米制尺度**重建。
- **表示：** **3D Gaussian Splat** 作照片级视觉层；**MVSAnywhere** 派生对齐 **碰撞 mesh**；二者同源对齐。
- **导出：** **NuRec volume USDZ**，直接载入 **NVIDIA Isaac Sim / Isaac Lab**（splat RTX 渲染 + mesh 物理代理）。
- **产品入口：** [Scaniverse](https://scaniverse.com)（消费/企业扫描）；企业级 geospatial AI 服务。

## 论文 / 代码状态

- **SPZ 格式：** 官网称 **开源** Gaussian splat 文件格式（行业采用）；**非**完整重建训练/导出管线代码。
- **重建与 VPS 栈：** 截至入库日官网 **未列出** 面向第三方的完整开源仓库；机器人数孪生工作流以 **企业服务 / 合作发布** 为主（见 [flexion_niantic_nvidia_sim2real_rgb_2026-07-20.md](../blogs/flexion_niantic_nvidia_sim2real_rgb_2026-07-20.md)）。
- **研究：** 官网列举 ICCV/CVPR 等论文（如 ACE-G、Cross-View Splatter、PlaceIt3D）；以研究页与出版物为主，非单一机器人策略 repo。

## 对 wiki 的映射

- [flexion-niantic-nvidia-rgb-sim2real-pipeline](../../wiki/entities/flexion-niantic-nvidia-rgb-sim2real-pipeline.md)
- [Sim2Real](../../wiki/concepts/sim2real.md)
- [SimFoundry](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md) — 另一路真机视频→数字孪生（NVIDIA GEAR）
