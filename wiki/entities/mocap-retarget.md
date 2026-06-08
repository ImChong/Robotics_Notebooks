---
type: entity
tags: [repo, motion-retargeting, mocap, humanoid, engineering]
status: complete
updated: 2026-06-08
summary: "ccrpRepo/mocap_retarget 是工程向动捕→机器人运动空间转换的参考管线，适合作为几何重定向脚本与配置的对照样本。"
related:
  - ../concepts/motion-retargeting.md
  - ../methods/motion-retargeting-gmr.md
  - ./legged-gym.md
sources:
  - ../../sources/repos/mocap_retarget.md
---

# mocap_retarget

**mocap_retarget**（<https://github.com/ccrpRepo/mocap_retarget>）是社区维护的 **动捕数据重定向到机器人** 的工程向示例仓库，侧重脚本化地把光学/文件格式 MoCap 映射到目标 URDF/MJCF 关节空间，而非提供完整 RL 训练栈。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MoCap | Motion Capture | 光学或惯性动捕得到的参考动作 |
| IK | Inverse Kinematics | 满足末端/姿态约束的关节解算 |
| URDF | Unified Robot Description Format | 机器人连杆与关节描述格式 |
| Retargeting | Motion Retargeting | 跨骨架/跨体型动作映射 |

## 为什么重要

- **低门槛对照**：理解「源骨架关节角 → 目标机器人关节角」需要哪些标定与坐标对齐时，可作为比 GMR 更小的阅读样本。
- **导航锚点**：本知识库 [retarget-tools](../../references/repos/retarget-tools.md) 与历史资源地图均收录此外链，补齐实体页可避免详情页空链。

> **一手性说明：** 非论文官方实现，而是 `ccrpRepo` 个人维护的工程示例；与 [GMR](../methods/motion-retargeting-gmr.md) 等论文作者仓相比，更适合作脚本参考而非方法复现基线。

## 定位与局限

| 维度 | 说明 |
|------|------|
| 强项 | 工程脚本、README 级管线说明 |
| 弱项 | 多机型维护、实时性能、物理可行性筛选不如 GMR / PHC / holosoma |
| 典型下游 | 导出关节轨迹后接模仿学习或 WBC |

## 关联页面

- [Motion Retargeting](../concepts/motion-retargeting.md)
- [GMR](../methods/motion-retargeting-gmr.md)
- [Retarget Tools](../../references/repos/retarget-tools.md)

## 参考来源

- [mocap_retarget 仓库归档](../../sources/repos/mocap_retarget.md)

## 推荐继续阅读

- GitHub：<https://github.com/ccrpRepo/mocap_retarget>
