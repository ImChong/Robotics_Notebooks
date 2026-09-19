# OmniCalib（arXiv:2609.19582）

> 来源归档（paper）

- **标题：** OmniCalib: Target-Free, Task-Structured Self-Calibration for Humanoid Robots
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.19582>
- **PDF：** <https://arxiv.org/pdf/2609.19582>
- **入库日期：** 2026-09-19
- **一句话说明：** 免标定板、任务结构化人形自校准：上肢 14 关节零位+双腕/胸 RGB-D 外参（深度 ICP）、下肢 12 关节零位（四姿态双支撑）、头部多相机 rig（平面行走+腿式里程计+动态 tf）；AGIBOT A3 Ultra 真机验证。

## 开源状态

- **确认未开源**（步骤 2.5，2026-09-19）：arXiv 与 HTML 版无 GitHub/项目页链接。

## 核心摘录

1. **任务–参数块匹配：** 每个模块对应机器人原生动作（臂工作空间扫描、双支撑、平面行走），可观性检查后只写回支持的 CAD 修正量。
2. **上肢：** 胸 RGB-D 观测手–臂工作空间，深度 ICP 联合恢复 14 关节零位与腕/胸外参；点–面残差 **2.09 mm**；注入偏移恢复 max **0.006°**。
3. **下肢：** 四静态双支撑姿态恢复 12 维注入偏置 RMS **0.063°**；MuJoCo 固定基座回放足位 RMS 28.5→**2.8 mm**。
4. **头部：** 多相机 VO + 腿式里程计 + 实时 ROS tf 补偿；三序列 SO(3) 均值 **1.061°**，最佳 **0.775°**（对照 iKalibr 0.902°）。
5. **平台：** AGIBOT A3 Ultra；作者含 Enyu Li、Yehao Lu 等（AgiBot 关联作者）。

**对 wiki 的映射**

- [paper-omnicalib](../../wiki/entities/paper-omnicalib.md)
- [paper-contact-constrained-joint-offset-calibration](../../wiki/entities/paper-contact-constrained-joint-offset-calibration.md)
- [hub-state-estimation](../../wiki/overview/hub-state-estimation.md)
