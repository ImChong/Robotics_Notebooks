# atom01_deploy

> 来源归档

- **标题：** atom01_deploy
- **类型：** repo
- **来源：** Roboparty（GitHub 组织）
- **链接：** https://github.com/Roboparty/atom01_deploy
- **入库日期：** 2026-04-25
- **一句话说明：** Atom01 的部署与运行仓库，聚焦 ROS2 驱动、中间件、设备接入与上机部署流程。
- **复核（2026-10-27）：** `git ls-remote` 显示 HEAD 与 [roboparty_deploy](roboparty_deploy.md) 同为 `c4d8b87`，即同一仓库更名；当前面向 RPO 与 RP1。
- **沉淀到 wiki：** 是（[`wiki/entities/roboparty-deploy.md`](../../wiki/entities/roboparty-deploy.md)）

---

## 为什么值得保留

- 承接训练策略到真实机器人执行的工程落地路径
- 涵盖 IMU/电机/通信等部署关键环节
- 对 Sim2Real 与可复现部署流程有直接参考价值

## 与本仓库现有资料的关系

- [roboto_origin.md](roboto_origin.md) 的部署子模块
- 可与 [robot_lab.md](robot_lab.md) 对比“训练框架”与“部署执行”分层
