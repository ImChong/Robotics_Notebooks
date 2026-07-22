# XRoboToolkit（XR-Robotics 开源组织）

> 来源归档

- **标题：** XR-Robotics / XRoboToolkit
- **类型：** repo（GitHub organization；多仓套件）
- **来源：** ByteDance PICO 等（论文作者归属）
- **组织主页：** <https://github.com/XR-Robotics>
- **项目页：** <https://xr-robotics.github.io/>
- **论文：** <https://arxiv.org/abs/2508.00097>
- **入库日期：** 2026-07-22
- **一句话说明：** XRoboToolkit 官方开源组织：XR Unity Client、PC Service / Pybind、Python&C++ 遥操作 sample、立体视觉与 ROS2 桥等，支撑 OpenXR 跨平台机器人遥操作与数据采集。
- **沉淀到 wiki：** [`wiki/entities/paper-xrobotoolkit.md`](../../wiki/entities/paper-xrobotoolkit.md)

---

## 开源状态

**已开源（截至 2026-07-22）**：项目页与 arXiv 均指向本组织；核心遥操作链路可从 sample README 跑通（需先安装 PC Service）。

| 仓库 | 许可证（API） | 角色 |
|------|---------------|------|
| [XRoboToolkit-PC-Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) | Apache-2.0 | C++/Qt PC 端服务与 Robot SDK |
| [XRoboToolkit-PC-Service-Pybind](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind) | MIT | Python SDK（`xrobotoolkit_sdk`） |
| [XRoboToolkit-Unity-Client](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client) | （仓库 license 字段未断言） | PICO 等 Unity XR Client |
| [XRoboToolkit-Unity-Client-Quest](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client-Quest) | MIT | Meta Quest 客户端 |
| [XRoboToolkit-Teleop-Sample-Python](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python) | MIT | MuJoCo / 真机遥操作与采数 demo（主复现入口） |
| [XRoboToolkit-Teleop-Sample-Cpp](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Cpp) | MIT | C++ sample |
| [XRoboToolkit-Teleop-ROS](https://github.com/XR-Robotics/XRoboToolkit-Teleop-ROS) | MIT | ROS2 遥操作桥 |
| [XRoboToolkit-RobotVision-PC](https://github.com/XR-Robotics/XRoboToolkit-RobotVision-PC) | GPL-3.0 | PC 侧立体视觉 |
| [XRoboToolkit-Orin-Video-Sender](https://github.com/XR-Robotics/XRoboToolkit-Orin-Video-Sender) | MIT | Jetson Orin 视频编码发送 |

---

## 推荐复现入口（Python Sample）

官方推荐路径（Ubuntu 22.04）：

1. 安装并启动 [XRoboToolkit-PC-Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)
2. `git clone` [Teleop-Sample-Python](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python) → `setup_conda.sh`
3. 仿真：`python scripts/simulation/teleop_dual_ur5e_mujoco.py`
4. 真机示例：`teleop_dual_ur5e_hardware.py` / `teleop_dual_arx_r5_hardware.py` / `teleop_r1lite_hardware.py`
5. 采数：手柄 **B** 键启停；日志 `.pkl` → 可选转 LeRobot（README 链到 openpi 转换示例）

依赖要点：PlaCo IK、Pinocchio、`dex_retargeting`、UR RTDE / ARX / Galaxea ROS 等按平台启用。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-xrobotoolkit](../../wiki/entities/paper-xrobotoolkit.md) | 本套件对应论文实体 |
| [paper-twist2](../../wiki/entities/paper-twist2.md) | 下游人形栈使用 XRoboToolkit 统一 egocentric 视觉与全身姿态流 |
| [paper-loco-manip-161-131-open-television](../../wiki/entities/paper-loco-manip-161-131-open-television.md) | 论文延迟对比基线 Open-TeleVision |
| [isaac-teleop](../../wiki/entities/isaac-teleop.md) | NVIDIA 仿真/真机 XR 遥操作对照栈 |
| [teleoperation](../../wiki/tasks/teleoperation.md) | XR 数据采集任务语境 |
