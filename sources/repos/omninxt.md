# OmniNxt

> 来源归档

- **标题：** OmniNxt: A Fully Open-source and Compact Aerial Robot with Omnidirectional Visual Perception
- **类型：** repo / hardware / UAV / omnidirectional vision
- **链接：** https://github.com/HKUST-Aerial-Robotics/OmniNxt
- **项目页：** https://hkust-aerial-robotics.github.io/OmniNxt/
- **论文：** IROS 2024 Oral — [IEEE Xplore](https://ieeexplore.ieee.org/document/10802134)
- **Stars：** ~570（2026-09-16）
- **机构：** 香港科技大学（HKUST）Aerial Robotics Group
- **入库日期：** 2026-09-16
- **一句话说明：** HKUST 开源紧凑型四旋翼机体：**360° 鱼眼全向感知** + **Jetson Orin NX** GPU 算力，为 SwarmNxt 等蜂群/视觉自主栈提供硬件基座。
- **沉淀到 wiki：** [paper-swarmnxt](../../wiki/entities/paper-swarmnxt.md)、[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

**OmniNxt** 是面向 **机载视觉自主** 的开源硬件平台（非仅飞控固件）：

- 紧凑机体 + 全向鱼眼相机阵列
- NVIDIA Jetson Orin NX 承担感知与规划
- PX4 飞控 + 开源 BOM/组装文档

[SwarmNxt](swarm_nxt.md) 在其上叠加 **ROS 2 蜂群编排** 与 **HDSM + MPC + S2M2** 集成栈。

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [swarm_nxt.md](swarm_nxt.md) | SwarmNxt 默认硬件平台 |
| [px4_autopilot.md](px4_autopilot.md) | 飞控执行层 |
| [vins_fusion.md](vins_fusion.md) | 同属 HKUST Aerial Robotics 生态 |
