# roboto_origin

> 来源归档

- **标题：** ROBOTO_ORIGIN - Fully Open-Source DIY Humanoid Robot
- **类型：** repo（聚合仓库）
- **来源：** Roboparty（GitHub 组织）
- **链接：** https://github.com/Roboparty/roboto_origin
- **入库日期：** 2026-04-25
- **一句话说明：** Roboparty 发布的人形机器人开源基线仓库，聚合硬件、部署、训练、描述、固件五个子仓库，目标是可复现的 DIY 人形机器人研发链路。
- **沉淀到 wiki：** 是（`wiki/entities/roboto-origin.md`）

---

## 核心定位

`roboto_origin` 官方声明为 **snapshot aggregation only**：
- 用于汇总项目整体说明与入口
- 实际开发与贡献建议进入各子仓库
- 覆盖结构、电子、训练、部署全栈流程

该仓库强调：可通过公开供应链与开源代码复现一个可跑可跳的人形机器人原型。

---

## 关联仓库（官方模块）

| 模块 | 主要职责 | 仓库链接 |
|------|---------|---------|
| Atom01_hardware | 机械结构、CAD、PCB、BOM | https://github.com/Roboparty/Atom01_hardware |
| atom01_deploy | ROS2 驱动、中间件、部署配置、IMU/电机集成 | https://github.com/Roboparty/atom01_deploy |
| atom01_train | IsaacLab 训练、仿真配置、Sim2Sim（含 MuJoCo） | https://github.com/Roboparty/atom01_train |
| atom01_description | URDF 运动学/动力学模型与网格资源 | https://github.com/Roboparty/atom01_description |
| atom01_firmware | 固件、USB2CAN、OrangePi 构建与守护进程 | https://github.com/Roboparty/atom01_firmware |

---

## 关联资料（文档与知识库）

- 官方文档：<https://roboparty.com/roboto_origin/doc>
- 中文 README：<https://github.com/Roboparty/roboto_origin/blob/main/README_cn.md>
- 中文贡献指南：<https://github.com/Roboparty/roboto_origin/blob/main/CONTRIBUTING_CN.md>
- 人形机器人运动控制 Know-How（飞书）：<https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87>

---

## 与本仓库现有资料的关系

| 资料 | 关系 |
|------|------|
| [unitree.md](unitree.md) | 同属人形硬件与控制生态参考对象 |
| [robot_lab.md](robot_lab.md) | 同样强调 IsaacLab 训练工作流，但定位不同（通用训练扩展 vs 原型机全栈） |
| [isaac_gym_isaac_lab.md](isaac_gym_isaac_lab.md) | `atom01_train` 的上游训练基础设施 |
| [sources/notes/know-how.md](../notes/know-how.md) | 早期条目已有 Roboto 文档入口，这里补齐结构化来源归档 |
