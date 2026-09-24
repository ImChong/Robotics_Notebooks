# hunter130_collection

> 来源归档

- **标题：** Hunter 130 开源项目合集（hunter130_collection）
- **类型：** repo（组织聚合入口）
- **机构：** 南京因克斯智能科技有限公司（EncosTech / Encos）
- **链接：** <https://github.com/EncosTech/hunter130_collection>
- **官网：** <https://www.encos.cn>
- **入库日期：** 2026-09-24
- **代码：** **已开源** — 聚合仓本身为 MIT 风格索引 README；子项目独立仓库见下表（2026-09-24 步骤 2.5：GitHub 组织页可访问，无独立 `*.github.io` 项目页）
- **一句话说明：** EncosTech 对 Hunter V2（EC H130-V2 / Hunter 130）人形的全栈开源导航：硬件 URDF、C++ 驱动与关节 SDK、调试 CLI、Isaac Lab 行走训练、ROS 2 Jazzy 实机部署与整机校准工具链。
- **沉淀到 wiki：** 是 → [`wiki/entities/encos-hunter130.md`](../../wiki/entities/encos-hunter130.md)

## README 要点（编译自上游 main，2026-09-24）

- **定位：** 本仓仅做**索引与整体说明**；各组件源码、版本、构建与 Issue 在对应仓库独立维护。
- **机器人：** Hunter V2（硬件型号 **EC H130-V2**），README 与 [`hunter130_hardware`](https://github.com/EncosTech/hunter130_hardware) 称 **130 cm / 约 32 kg / 25 DoF**；训练仓 [`hunter130_train`](https://github.com/EncosTech/hunter130_train) 针对 **23 个受控关节** 平地行走（PPO + AMP）。
- **典型环境：** Ubuntu 24.04、**ROS 2 Jazzy**、CMake/C++17；训练侧 **Isaac Sim 5.1 + Isaac Lab**（pip/Conda 安装）。

## 子仓库导航

| 仓库 | 定位 | 主要技术 | 许可证（README 声明） |
| --- | --- | --- | --- |
| [encos_driver](https://github.com/EncosTech/encos_driver) | Encos 电机、电池、PMS 的 C++17 驱动；可扩展通信适配器插件 | EtherCAT、SocketCAN、WASM 等 | 主体 MIT；EtherCAT 插件及第三方各自许可 |
| [joint_sdk](https://github.com/EncosTech/joint_sdk) | 基于 `encos_driver` 的关节控制 SDK | 旋转 / 连续旋转 / 双电机耦合关节抽象；含 TS/WASM | 主体 MIT |
| [encos_cli](https://github.com/EncosTech/encos_cli) | 电机与外围设备 CLI / TUI 调试、监控、基准与轨迹播放 | C++17、FTXUI | MIT |
| [hunter130_hardware](https://github.com/EncosTech/hunter130_hardware) | 机械 / 电气 / URDF / 安装手册 | STEP、Parasolid、STL | **CERN-OHL-S-2.0** |
| [hunter130_train](https://github.com/EncosTech/hunter130_train) | RL 行走训练、回放与 **ONNX** 导出 | Python、Isaac Lab、PPO、AMP、RSL-RL；Fork 自 [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab) | BSD-3-Clause |
| [hunter130_deploy](https://github.com/EncosTech/hunter130_deploy) | EC130 实机 **ROS 2** 控制系统 | ros2_control、IBUS 遥控、站立 / RL 行走、ONNX Runtime | **GPL-3.0** |
| [robot_setup](https://github.com/EncosTech/robot_setup) | 连接检查、通信验证、电机 ID、关节校准、IMU、整机运动验证 | Python、Web UI、`emcli` | 仓库暂未声明统一许可证 |

## 推荐阅读路径（上游）

1. 结构 / 仿真模型 → `hunter130_hardware`
2. 电机与总线 → `encos_driver` + `encos_cli`
3. 关节级应用 → `joint_sdk`
4. 行走策略 → `hunter130_train`（导出 ONNX）
5. 实机 → `hunter130_deploy`
6. 交付前校准 → `robot_setup`

## 协作分工（hardware README）

- **硬件设计与制造：** 南京因克斯智能科技有限公司
- **部署与训练相关软件：** 桥介数物（与 V1 开源合作延续）

## 安全提示（上游）

可运动实体机器人；上电、校准、轨迹或策略部署前需固定机体、有效急停、限制首次运行速度与输出。

## 与知识库其他条目的关系

| 条目 | 关系 |
|------|------|
| [`sources/sites/encos.cn.md`](../sites/encos.cn.md) | 厂商官网归档 |
| [`wiki/entities/encos-hunter130.md`](../../wiki/entities/encos-hunter130.md) | 提炼页 |
| [`wiki/entities/tienkung-lab.md`](../../wiki/entities/tienkung-lab.md) | `hunter130_train` 上游训练框架 |
| [`wiki/entities/tienkung-humanoid-open-source.md`](../../wiki/entities/tienkung-humanoid-open-source.md) | 同源 Open-X-Humanoid / 天工 Isaac Lab 生态对照 |
