# XRoboToolkit: A Cross-Platform Framework for Robot Teleoperation（arXiv:2508.00097）

> 来源归档（ingest）

- **标题：** XRoboToolkit: A Cross-Platform Framework for Robot Teleoperation
- **类型：** paper / teleoperation / XR / OpenXR / VLA-data / inverse-kinematics
- **arXiv abs：** <https://arxiv.org/abs/2508.00097>
- **PDF：** <https://arxiv.org/pdf/2508.00097>
- **HTML：** <https://arxiv.org/html/2508.00097v1>
- **项目页：** <https://xr-robotics.github.io/>
- **代码组织：** <https://github.com/XR-Robotics>（已开源；见 [repos/xrobotoolkit.md](../repos/xrobotoolkit.md)）
- **机构：** 字节跳动 PICO（ByteDance PICO）；佐治亚理工学院（Georgia Tech / IRIM）；乔治梅森大学（George Mason University）
- **奖项：** SII 2026 Best Paper Award（项目页标注）
- **入库日期：** 2026-07-22
- **最后更新：** 2026-07-22
- **一句话说明：** 基于 **OpenXR** 的跨平台 XR 遥操作套件：低延迟立体视觉回传、**QP-IK**（PlaCo/Pinocchio）、头/手柄/手/辅助 tracker 多模态追踪；验证于 UR5 / ARX R5 / Galaxea R1-Lite / Shadow Hand / MuJoCo，并用采集数据 LoRA 微调 π₀。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [xr-robotics.github.io](https://xr-robotics.github.io/) | 摘要、应用图、视频、BibTeX；页头链到 GitHub org |
| 代码 org | [XR-Robotics](https://github.com/XR-Robotics) | Unity Client / PC-Service / Pybind / Python&C++ Sample / ROS / Vision |
| 视频 | [YouTube 补充视频](https://youtu.be/g6QJX2s-RCo) | 论文补充演示 |
| 延迟对照 | [Open-TeleVision](https://robot-tv.github.io/) | 同硬件下视频流延迟基线（论文 Table II） |
| 下游使用 | [TWIST2](https://yanjieze.com/projects/TWIST2/) | 人形全身遥操作栈以 XRoboToolkit 统一视觉+姿态流 |

## 开源状态（项目页核查，2026-07-22）

- **判定：已开源。** 项目页 Footer / 页头提供 **GitHub** 链至 [XR-Robotics](https://github.com/XR-Robotics)；arXiv HTML 摘要亦写明 Github org。
- 公开组件覆盖：头显 Unity Client（PICO / Quest）、PC Service（C++/Qt，Apache-2.0）、Pybind SDK（MIT）、Python/C++ 遥操作 sample（MIT）、ROS2 桥、立体视觉 PC/Orin 发送端等。
- **未宣称开放**：π₀ 微调用的 100 条折毯示范权重/数据集未见独立 HF/Zenodo 发布；复现依赖自行采数 + 外部 openpi 转换脚本（sample README 指向）。

## 摘要级要点

- **问题：** VLA 需要大规模高质量示范；现有遥操作在可扩展性、配置复杂度与数据质量上受限——leader–follower 绑死硬件、纯视觉跟踪不稳、既有 XR 方案常依赖单机 Unity SDK / WebXR 且缺 XR↔机器人标准化数据格式。
- **回答：** **XRoboToolkit** 以 **OpenXR** 统一 XR 侧坐标系与追踪字段（头/手柄/26 关节手/全身 24 关节/辅助 motion tracker，JSON @90 Hz），机器人侧用模块化 Python/C++ 接口 + **QP-IK** + dex_retargeting；立体视觉支持 PICO 机载相机与 ZED Mini。
- **延迟：** 同硬件（ZED Mini→Quest 3）相对 Open-TeleVision **121.5 ms → 94.5 ms**（约 −22%）；ZED Mini→PICO 4 Ultra 均值 **82 ms**（Table II）。
- **数据质量：** ARX R5 双臂折毯 **100** 条示范 → π₀ LoRA 80k step；连续 30 min 自主成功率 **100%**，出现自主 regrasping / 重定位。
- **局限（论文结论）：** 全身追踪依赖 PICO 24-joint（OpenXR 无标准全身模型）；全身→人形重定向未验证；耦合关节灵巧手（如 INSPIRE）retargeting 不准；仿真目前主推 MuJoCo。

## 核心摘录（面向 wiki 编译）

### 1) 系统架构

| 模块 | 职责 |
|------|------|
| Unity Client（XR） | 追踪采集 + 立体 UI；Network / Tracking / Remote Vision / Data Collection / Log 五面板 |
| PC Service（C++） | 异步回调流；SDK 连接头显；Pybind 暴露 Python |
| Robot Vision | PICO 机载或 ZED Mini → 低延迟立体流 |
| Teleop 控制 | QP-IK（PlaCo）、灵巧手 retargeting、头跟踪、移动基座摇杆 |

### 2) 控制模式

- **相对运动 IK：** grip 按下时末端跟踪手柄相对位移，改善奇异附近稳定性；可加 elbow tracker 作 QP 额外位姿约束。
- **灵巧手：** OpenXR 26 关节 → `dex_retargeting`（AnyTeleop）映射 Shadow Hand 等。
- **移动基座：** 左摇杆线速度、右摇杆 X 角速度（全向底盘）。

### 3) 验证平台

双臂 ARX R5（长程折毯）、Galaxea R1-Lite 移动操作、双 UR5 + 2-DoF 主动头（3 mm 螺丝刀入 4 mm 孔）、MeshCat 中 G1 上身 elbow tracker、MuJoCo Shadow Hand。

## 对 wiki 的映射

- 沉淀实体页：[XRoboToolkit（论文实体）](../../wiki/entities/paper-xrobotoolkit.md)
- 项目页归档：[xr-robotics-github-io.md](../sites/xr-robotics-github-io.md)
- 代码归档：[xrobotoolkit.md](../repos/xrobotoolkit.md)
- 交叉补强：[Teleoperation](../../wiki/tasks/teleoperation.md)、[Open-TeleVision](../../wiki/entities/paper-loco-manip-161-131-open-television.md)、[TWIST2](../../wiki/entities/paper-twist2.md)、[Isaac Teleop](../../wiki/entities/isaac-teleop.md)、[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[数据手套 vs 视觉遥操作](../../wiki/comparisons/data-gloves-vs-vision-teleop.md)

## 当前提炼状态

- [x] arXiv / 项目页 / GitHub org 开源核查
- [x] 延迟表、VLA 微调、架构与局限摘录
- [x] wiki 实体页与 teleoperation / TWIST2 / Isaac Teleop 交叉链接规划
- [x] 关联 wiki 参考来源将随本次 ingest 同步
