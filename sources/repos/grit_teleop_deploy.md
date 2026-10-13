# mrzuang/GRIT_teleop_deploy

> 来源归档

- **标题：** GRIT v0.0.1 — Whole-Body Motion-Tracking Control and VR Teleoperation Deployment for Unitree G1
- **类型：** repo（全身运动跟踪策略部署）
- **作者 / 所有者：** mrzuang
- **链接：** <https://github.com/mrzuang/GRIT_teleop_deploy>
- **机器人：** Unitree G1
- **许可证：** 以仓库 THIRD_PARTY.md 与各组件声明为准
- **开源状态：** GRIT ONNX 策略与部署实现公开；训练及数据生成代码不在该仓库范围内
- **核查日期：** 2026-10-13
- **一句话说明：** GRIT v0.0.1 的 Unitree G1 部署仓库，提供 50 Hz Python 推理与控制、MuJoCo sim2sim、PICO/XRoboToolkit 动作重定向和 Unitree SDK2 原生桥接。

## 公开实现与边界

- 策略契约和观测构造位于 sim2real/src/runtime/grit_policy.py、grit_observation.py。
- 部署入口为 sim2real/src/deploy.py；MuJoCo 仿真入口为 sim2real/src/sim2sim.py。
- 提供示例参考动作、G1 配置、VR 实时动作流及 native SDK2 硬件桥；具有过期指令 watchdog。
- README 明确说明训练与数据生成不属于该仓库，因此该仓库支持复现部署与评测，不等于完整训练复现。
- README 仍将 v0.0.2 和 v1.0.0 描述为计划发布项；这些计划不能视为已发布的事实，应以当前 release / commit 为准。

## 对 wiki 的映射

- [GRIT v0.0.1](../../wiki/entities/grit.md)
- [GRIT Deploy Onboard G1 集成工程](../../wiki/entities/grit-deploy-onboard-g1.md)
