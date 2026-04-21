---
type: overview
tags: [humanoid, control, know-how, engineering, deployment]
status: complete
updated: 2026-04-21
summary: "人形机器人运动控制工程实践经验汇总：涵盖了从传感器噪声处理、电机热管理到非结构化地形适应的闭坑指南。"
---

# 人形机器人运动控制 Know-How

> **说明**：本文档沉淀了在真实人形机器人研发与调试过程中的硬核工程经验。不同于教科书上的理论，这里的每一条建议往往都对应着一次“炸机”或烧毁电机的惨痛教训。

## 1. 传感器与状态估计

### 1.1 IMU 的“幻觉”
- **高频振动**：足端撞击地面的瞬时振动会通过骨架传导至 IMU。如果固定不牢，加速度计会因共振产生巨大偏置。
- **解决**：使用硅胶减震垫；在软件层使用 200Hz 以上的采样率并配合 Butterworth 低通滤波。

### 1.2 接触估计 (Contact Estimation)
- **阈值陷阱**：仅靠足底压力传感器（Force Sensor）判断触地极不靠谱，因为地面的反弹和草地的虚位会产生误报。
- **融合逻辑**：结合关节编码器的速度突变和高度图信息进行多源判定。

## 2. 动力学与控制 (WBC/RL)

### 2.1 摩擦锥的线性化
- 在 QP 求解器中，摩擦锥通常被线性化为四棱锥。**注意**：摩擦系数 $\mu$ 在实机上通常比仿真里要小（建议取 0.4-0.6 进行保守估计），以防止脚底打滑。

### 2.2 奇异点保护
- 当腿部完全伸直（膝关节奇异点）时，雅可比矩阵会退化，导致控制力矩瞬间爆炸。
- **对策**：在控制律中加入虚拟弹簧力，阻止关节进入极限位置前 2-5 度的区域。

## 3. 硬件与电气工程

### 3.1 电池压降
- 高动态动作（如跳跃）会导致母线电压从 48V 瞬间降至 40V 以下。
- 必须确保逻辑计算板（IPC）拥有独立的稳定供电，否则主控会因欠压重启，导致机器人失去平衡。

### 3.2 热管理
- 连续行走 15 分钟后，膝关节电机通常会达到 70°C 以上。
- **经验**：在 WBC 中加入能效项，减少不必要的对抗出力。

## 关联页面
- [人形机器人电池与热管理指南](../queries/humanoid-battery-thermal-management.md)
- [野外机器人排障指南](../queries/field-robotics-troubleshooting.md) — 应对非结构化地形下的感知与平衡失效
- [Sim2Real 真机部署检查清单](../queries/sim2real-deployment-checklist.md)

## 参考来源
- [sources/papers/sim2real.md](../../sources/papers/sim2real.md)
- [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md)

