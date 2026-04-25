---
type: entity
tags: [humanoid, hardware, open-source, ros2, isaac-lab]
status: complete
updated: 2026-04-25
related:
  - ./open-source-humanoid-hardware.md
  - ./humanoid-robot.md
  - ./robot-lab.md
  - ./unitree.md
  - ./atom01-hardware.md
  - ./atom01-description.md
  - ./atom01-train.md
  - ./atom01-deploy.md
  - ./atom01-firmware.md
sources:
  - ../../sources/repos/roboto_origin.md
  - ../../sources/repos/atom01_hardware.md
  - ../../sources/repos/atom01_deploy.md
  - ../../sources/repos/atom01_train.md
  - ../../sources/repos/atom01_description.md
  - ../../sources/repos/atom01_firmware.md
summary: "Roboto Origin 是 Roboparty 发布的开源人形机器人基线项目，通过聚合 hardware/deploy/train/description/firmware 五个子仓库，提供可复现的 DIY 人形机器人研发路径。"
---

# Roboto Origin（开源人形机器人基线）

**Roboto Origin** 是 Roboparty 发布的“全链路开源”人形机器人项目入口页，目标不是只给一个仓库，而是提供从硬件到训练再到部署的可复现工程路径。

## 为什么重要

1. **低门槛复现路径**：将 DIY 人形机器人的关键模块显式拆分，降低单人/小团队上手门槛。
2. **工程边界清晰**：官方明确聚合仓库用于导航，贡献回流到子仓库，避免 monorepo 混乱。
3. **覆盖全栈闭环**：硬件设计、URDF 描述、RL 训练、ROS2 部署、固件链路全部公开。

## 核心结构

Roboto Origin 的仓库体系可理解为“五段流水线”：

- **Atom01_hardware**：机械与电子设计（CAD/PCB/BOM），见 [sources/repos/atom01_hardware.md](../../sources/repos/atom01_hardware.md)
- **atom01_description**：机器人模型描述（URDF + 网格），见 [sources/repos/atom01_description.md](../../sources/repos/atom01_description.md)
- **atom01_train**：IsaacLab 训练与 Sim2Sim，见 [sources/repos/atom01_train.md](../../sources/repos/atom01_train.md)
- **atom01_deploy**：ROS2 真机驱动与部署，见 [sources/repos/atom01_deploy.md](../../sources/repos/atom01_deploy.md)
- **atom01_firmware**：底层固件与板端支持，见 [sources/repos/atom01_firmware.md](../../sources/repos/atom01_firmware.md)

这种拆分方式适合把“研究迭代”和“工程落地”分离维护。

## 常见误区 / 局限

- **误区 1：把 `roboto_origin` 当成可直接开发主仓库。**
  实际上它更像索引入口，真正开发需进入各模块仓库。
- **误区 2：认为开源即开箱即用。**
  项目仍需要较强的 Linux/ROS2/硬件调试能力。
- **局限：** 当前公开资料更偏“原型复现与社区协作”，对工业级可靠性（长期运行、认证、安全冗余）覆盖有限。

## 关联页面

- [开源人形机器人硬件方案对比](./open-source-humanoid-hardware.md)
- [人形机器人（Humanoid Robot）](./humanoid-robot.md)
- [robot_lab (IsaacLab 扩展框架)](./robot-lab.md)
- [Unitree](./unitree.md)
- [Atom01 Hardware](./atom01-hardware.md)
- [Atom01 Description](./atom01-description.md)
- [Atom01 Train](./atom01-train.md)
- [Atom01 Deploy](./atom01-deploy.md)
- [Atom01 Firmware](./atom01-firmware.md)

## 推荐继续阅读

- [Roboparty / roboto_origin](https://github.com/Roboparty/roboto_origin)
- [Humanoid Robot Know-How Documentation](https://roboparty.com/roboto_origin/doc)

## 参考来源

- [sources/repos/roboto_origin.md](../../sources/repos/roboto_origin.md)
- [sources/repos/atom01_hardware.md](../../sources/repos/atom01_hardware.md)
- [sources/repos/atom01_deploy.md](../../sources/repos/atom01_deploy.md)
- [sources/repos/atom01_train.md](../../sources/repos/atom01_train.md)
- [sources/repos/atom01_description.md](../../sources/repos/atom01_description.md)
- [sources/repos/atom01_firmware.md](../../sources/repos/atom01_firmware.md)
- [Roboparty/roboto_origin README](https://github.com/Roboparty/roboto_origin)
