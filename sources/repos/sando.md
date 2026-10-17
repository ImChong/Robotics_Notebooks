# mit-acl/sando

> 官方代码仓来源归档

- **标题：** SANDO: Safe Autonomous Trajectory Planning for Dynamic Unknown Environments
- **类型：** repo
- **链接：** https://github.com/mit-acl/sando
- **组织：** MIT Aerospace Controls Laboratory
- **语言 / 许可：** C++ / BSD 3-Clause
- **仓库状态：** 公开；核查日期 2026-10-17
- **开发环境：** Ubuntu 22.04、ROS 2 Humble
- **入库日期：** 2026-10-17
- **一句话说明：** SANDO 的官方 ROS 2 规划栈，实现动态未知环境的感知、障碍跟踪、全局搜索、时空安全走廊与硬约束轨迹优化。
- **论文：** https://arxiv.org/abs/2604.07599（HTML v3：<https://arxiv.org/html/2604.07599v3>）
- **项目页：** https://kotakondo.github.io/projects/sando/
- **视频：** https://youtu.be/_T10DJiLQXg
- **沉淀到 wiki：** [SANDO](../../wiki/entities/paper-sando.md)

## 开源范围与复现入口

仓库 README 提供 Docker 与原生 Linux 两种安装方式，列出四种仿真模式：RViz 交互规划、静态障碍、已知动态障碍、基于点云感知的未知动态障碍；场景难度覆盖 50 / 100 / 200 个障碍。运行脚本入口包括 `src/sando/scripts/run_sim.py`，Docker 演示由 `docker/Makefile` 提供。

- **依赖：** ROS 2 Humble；Docker 镜像同时集成 Gazebo / RViz。仓库推荐 Ubuntu 22.04。
- **优化器许可：** Docker 运行说明要求 Gurobi WLS Academic license；计划在自有机器构建前应确认 Gurobi 许可可用。
- **真机：** 论文展示 PX4 四旋翼硬件实验；硬件部署仍需匹配飞控、传感器、定位与机载计算配置。
- **开放结论：** 规划与仿真代码已公开；仓库 README 未列独立训练数据或模型权重下载入口。

## 运行流程要点

点云进入地图管理与障碍追踪模块；追踪器输出动态障碍状态和预测，静态/动态风险热图供 A* 全局规划使用；路径和障碍可达集生成 STSFC；MIQP 优化器输出轨迹供低层控制执行。动态场景中重规划失败时沿用旧轨迹，旧轨迹失效时进入悬停回退。

## 版本与论文状态

作者主页（2026 年 7 月新闻）称论文有条件接收至 T-RO；仓库 README 在 2026-10-17 核查时使用“Accepted to the IEEE Transactions on Robotics (T-RO)”。该状态差异作为来源快照记录；引用时优先注明来源和核查日期。

## 关联原始来源

- [SANDO 论文来源归档](../papers/sando_arxiv_2604_07599.md)
- [SANDO 项目页归档](../sites/sando-project.md)
- [mit-acl/sando](https://github.com/mit-acl/sando)
