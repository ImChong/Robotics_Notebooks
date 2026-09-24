# cyclo_solution（ROBOTIS）

- **标题:** Cyclo Solution — Physical AI 集成与部署工作区
- **链接:** [https://github.com/ROBOTIS-GIT/cyclo_solution](https://github.com/ROBOTIS-GIT/cyclo_solution)
- **类型:** repo
- **摘要:** ROBOTIS **Physical AI** 侧的集成解决方案仓库（含 Docker `cyclo_solution` 环境）；[AI Worker × Isaac ROS cuMotion](https://docs.robotis.com/docs/systems/aiworker/resources/technical_story/isaac_cumotion/) 文档在此提供 **`cyclo_cumotion_bringup`**（`cumotion_moveit.launch.py`）、MoveIt 配置与静态场景等资源，将 **AI Worker** 深度/状态与工作站侧 **cuMotion + Nvblox** 对接。
- **许可:** 以仓库 LICENSE 为准（通常 Apache-2.0 系，以 GitHub 为准）

## 为什么值得保留

- 是 **ROBOTIS 官方 cuMotion 集成** 的可复现入口，补全 [ai_worker](ai_worker.md) 硬件栈之外的 **GPU 碰撞感知规划** 路径。

## 对 wiki 的映射

- [wiki/entities/robotis-ai-worker-isaac-cumotion.md](../../wiki/entities/robotis-ai-worker-isaac-cumotion.md)
- [wiki/entities/robotis-ai-worker.md](../../wiki/entities/robotis-ai-worker.md)

---
- **录入日期:** 2026-09-24
