# GRIT v0.0.1 G1 deployment repository

> 来源归档

- **类型：** repo
- **链接：** https://github.com/mrzuang/GRIT_teleop_deploy
- **访问日期：** 2026-10-13
- **一句话说明：** GRIT v0.0.1 的 Unitree G1 部署实现，提供 ONNX 跟踪 runtime、MuJoCo sim2sim、PICO 遥操作和 SDK2 真机桥接。

## 仓库职责

README 将仓库范围限定为 GRIT v0.0.1 的部署，不包含训练或数据生成。策略通过 ONNX 发布，50 Hz Python runtime 使用统一 UDP 接口；仓库另含 MuJoCo headless/interactive sim2sim、PICO/XRoboToolkit 在线 retargeting，以及带 stale-command watchdog 的 Unitree SDK2 硬件桥。

## 复现与接口

仓库给出 G1 tracking 与 PICO teleoperation 配置，运行要求 Ubuntu 22.04/24.04、x86_64 或 aarch64、Python 3.10/uv、MuJoCo 图形环境；PICO 流还需要 XRoboToolkit 和低延迟共享 Wi-Fi。训练和数据生成明确不在仓库范围内。

## 与其他 GRIT 节点的区别

本节点对应榜单的 GRIT v0.0.1 策略/该作者的部署仓库。另一个 [GRIT Deploy Onboard G1](../../wiki/entities/grit-deploy-onboard-g1.md) 是不同的开源部署实现；此处采用独立项目 ID 以明确 repo 来源，不把两套代码混成同一仓库。
