---
type: query
tags: [humanoid, quadruped, deployment, troubleshooting, field-robotics]
status: complete
updated: 2026-04-21
related:
  - ../overview/humanoid-motion-control-know-how.md
  - ./robot-policy-debug-playbook.md
  - ./real-time-control-middleware-guide.md
  - ../concepts/terrain-adaptation.md
sources:
  - ../overview/humanoid-motion-control-know-how.md
  - ../../sources/papers/humanoid_motion_control_know_how.md
summary: "野外机器人排障指南：总结了机器人在非结构化地形、极端光照及不确定天气下常见的失效模式，如激光雷达过曝、IMU 累计漂移及接触估计误报。"
---

# 野外机器人排障指南

> **Query 产物**：本页由以下问题触发：「机器人在实验室跑得很稳，一带到户外草地或碎石路就各种摔，传感器也频繁报错，怎么排查？」
> 综合来源：[Know-how](../overview/humanoid-motion-control-know-how.md)、[System ID](../concepts/system-identification.md)

---

实验室环境是“匀质”的，而野外环境具有高随机性和干扰。

## 1. 感知层：当“眼睛”失效时

### 激光雷达 (LiDAR)
- **失效模式**：草地、积水产生的镜面反射或漫反射会导致点云极度嘈杂。
- **排障**：
  - 检查强度 (Intensity) 过滤。
  - 在 [Terrain Adaptation](../concepts/terrain-adaptation.md) 中增加地平面拟合的鲁棒性（如 RANSAC 算法）。

### 摄像头
- **失效模式**：户外强光直射导致图像过曝，物体检测失败。
- **排障**：
  - 启用硬件级的自动曝光 (Auto-exposure) 与自动白平衡。
  - 训练 VLA 或感知模型时必须加入大量的亮度增强（Data Augmentation）。

## 2. 状态估计层：当“平衡感”产生幻觉

### IMU 累积漂移
- **现象**：机器人走着走着，Base 姿态开始向一边倾斜，最终摔倒。
- **原因**：电机大电流产生的电磁干扰（EMI）或机身剧烈振动导致的加速度计饱和。
- **对策**：
  - 使用双 IMU 冗余校验。
  - 在卡尔曼滤波中动态调高振动状态下的 R（观测噪声）矩阵。

### 接触估计误报 (False Contact)
- **现象**：踩到深草或软泥时，足端压力传感器未达到阈值，但控制层认为已触地。
- **对策**：
  - 结合关节电流（扭矩）进行多源融合判定。
  - 引入 [Contact Estimation](../concepts/contact-estimation.md) 概率模型，而不是简单的阈值判定。

## 3. 动力学层：Sim2Real 的野外显现

### 地面硬度不匹配
- **现象**：机器人在实验室硬地走得很干脆，在草地表现得像“深陷泥潭”。
- **原因**：草地具有阻尼 and 弹性，而仿真通常假设刚性地面。
- **调教技巧**：
  - 在 WBC 或 RL 中降低足端 PD 的 $K_p$。
  - 在 [Domain Randomization](./domain-randomization-guide.md) 中加入地面刚度随机化。

## 关联页面
- [人形机器人运动控制 Know-How](../overview/humanoid-motion-control-know-how.md)
- [机器人策略排障手册](./robot-policy-debug-playbook.md)
- [实时运控中间件配置指南](./real-time-control-middleware-guide.md)
- [地形自适应 (Terrain Adaptation)](../concepts/terrain-adaptation.md)

## 参考来源
- [humanoid-motion-control-know-how.md](../overview/humanoid-motion-control-know-how.md)
- [sources/papers/humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md)
- ETH Zurich RSL, *Learning to Walk in Minutes* 户外测试部分。
