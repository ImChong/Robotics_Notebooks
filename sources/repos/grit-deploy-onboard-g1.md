# KurokoWeidong/GRIT_Deploy_Onboard_G1（Unitree G1 机载部署）

- **标题**: GRIT Deploy Onboard G1 — Onboard C++ deployment for Unitree G1
- **类型**: repo / humanoid / unitree-g1 / locomotion / teleoperation / sim2real
- **作者**: KurokoWeidong
- **链接**: <https://github.com/KurokoWeidong/GRIT_Deploy_Onboard_G1>
- **许可证**: MIT（仓库代码；模型与第三方依赖另按上游条款）
- **收录日期**: 2026-10-13

## 一句话摘要

面向 Unitree G1 的机载 C++ 控制与部署仓库，将 GRIT 全身遥操作跟踪、AMP locomotion 与 NPZ 轨迹播放集成到统一运行时；PICO/工作站侧另提供 VR 重定向服务。

## 项目核查

以仓库 README、代码结构及许可证为准：项目代码公开，采用 MIT 许可证。WBC ONNX 模型来自 [mrzuang/GRIT_teleop_deploy](https://github.com/mrzuang/GRIT_teleop_deploy)，locomotion ONNX 模型来自 [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm)；机器人桥接与 VR 重定向设计参考 [Axellwppr/motion_tracking 的 sim2real 分支](https://github.com/Axellwppr/motion_tracking/tree/sim2real)。README 声明已确认随项目分发两个模型的许可；复用者仍应逐项核对当前模型、依赖版本与上游许可，不应把项目代码的 MIT 许可等同于所有模型与依赖许可。

## 值得保留

- 将机载推理、Unitree SDK2 低层控制、VR 数据桥接与运动文件播放整理成可部署工程，便于研究者理解策略到真机执行之间的集成边界。
- Loco、VR Tracking、NPZ 三种模式可在统一程序中切换；切换包含受控站立过渡，并提供超时、阻尼退出等保护逻辑。
- 明确区分 PICO、PC Server 与 G1 onboard 的职责，对照部署架构比单独展示策略效果更有复现价值。

## 公开结构与运行入口

| 路径 / 组件 | 职责 |
|-------------|------|
| `g1_sim2real/src/g1_udp_bridge.cpp` | G1 侧运行桥与控制模式协调 |
| `g1_sim2real/src/g1_npz_policy.cpp` | C++ 推理 / 轨迹控制逻辑 |
| `g1_sim2real/scripts/run_npz_onboard.sh` | Loco + NPZ 入口 |
| `g1_sim2real/scripts/run_vr_onboard.sh` | Loco + VR（可选 NPZ）入口 |
| `sim2real/teleop/` | 工作站侧 PICO/XRoboToolkit 重定向服务 |
| `motions/` | 示例 NPZ 轨迹 |

README 描述 G1 机载侧以 50 Hz 进行 C++ 推理和低层控制。VR 模式需 PICO、PC Server、G1 位于可互通的同一局域网；单独使用 Loco + NPZ 不要求 PC Server。控制模式及接口细节以仓库 README 与脚本为准。

## 数据与接口边界

NPZ 轨迹约定为 50 Hz，包含 `joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w` 与 `body_ang_vel_w` 等数组；关节排列、帧率、机器人配置及模型必须配套。VR 模式则由工作站服务将 XRoboToolkit/PICO 数据重定向后通过网络送入 G1 运行栈。两条输入链路最终由机载桥统一协调模式与控制输出。

## 上游关系

- [mrzuang/GRIT_teleop_deploy](https://github.com/mrzuang/GRIT_teleop_deploy)：WBC 模型与 GRIT 遥操作部署基础。
- [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm)：AMP locomotion ONNX 模型来源。
- [Axellwppr/motion_tracking sim2real](https://github.com/Axellwppr/motion_tracking/tree/sim2real)：机器人桥与 VR 重定向设计参考；该知识库已有 [HEFT motion_tracking 页面](../../wiki/entities/axellwppr-motion-tracking.md)。

## 使用边界

这是部署与集成仓库，不是完整策略训练仓库；README 仅声明两个 ONNX 模型的来源，不能据此推断训练代码、训练数据或全部策略权重都由本仓库提供。上机前必须先核对适用硬件和固件、网络接口、关节顺序、策略与配置配套性，并在仿真和保护条件下逐步验证。项目自身建议精确模型 / 轨迹先仿真测试，使用支撑保护，并由操作者保持可触达的物理急停；软件阻尼退出、LED 或 SSH 都不能替代物理急停。

## 对 Wiki 的映射

- **wiki/entities/grit-deploy-onboard-g1.md**：部署范围、模式架构、运行链路与安全边界。
- **wiki/entities/axellwppr-motion-tracking.md**：上游 sim2real/重定向参考。
- **wiki/tasks/locomotion.md**：机载 locomotion 策略部署。
- **wiki/tasks/teleoperation.md**：PICO 遥操作与在线重定向。

## 推荐继续阅读

- [项目仓库 README](https://github.com/KurokoWeidong/GRIT_Deploy_Onboard_G1)
- [GRIT_teleop_deploy](https://github.com/mrzuang/GRIT_teleop_deploy)
- [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm)
