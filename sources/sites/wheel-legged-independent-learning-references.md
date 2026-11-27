# 轮足独立学习资料：StackForce、RC WheelLeg、Roly、Go2W SuperMimic

> 对照目录，不是 CHRIC 或 RW-02OP 的资料组成、依赖或官方镜像。

- **核查日期：** 2026-11-27
- **配套知识页：** [RW-02OP 与独立对照资料](../../wiki/entities/rw-02op.md)
- **通读：** 已完成
- **核对范围：** StackForce 的 Gitee README 与文件目录、RC WheelLeg 的 Tags 和 README、SteveZC8537 的 Roly README 与自平衡仿真入口、Go2W SuperMimic 的 README 和仓库说明。未在这些平台上跑训练或验证真机。

| 维护方来源 | 阅读切入点 | 不应等同于 |
|---|---|---|
| [StackForce 双足轮机器人课程](https://gitee.com/StackForce/bipedal_wheeled_robot) | 课程附 3D 模型、BOM 和逐课 PlatformIO 工程；README 标注 Apache-2.0 | RW-02OP 硬件缩比版或控制器 |
| [RC WheelLeg Tags](https://github.com/zeitvex/RC_WheelLeg/tags) · [主仓](https://github.com/zeitvex/RC_WheelLeg) | 教学可按标签定位版本；v0.4.1 的标签说明是 ROS 2/C++ 初版与导航原型，主分支 README 还区分后续部署分支目录 | 双轮足；上游主要平台为 16 DoF 四轮足 |
| [SteveZC8537/AI_Self-Balance_Robot](https://github.com/SteveZC8537/AI_Self-Balance_Robot) · [自平衡仿真入口](https://github.com/SteveZC8537/AI_Self-Balance_Robot/blob/7a587b09b98a2876ae4231a24fe1bf598f7ecd95/software/s8-control/balance_sim.py) | README 将项目称为 Roly；适合并行理解小型双轮平衡的线性化和 LQR | CHRIC 的控制参数生成脚本或可互换的增益 |
| [Go2W SuperMimic](https://github.com/uwvwko-zzz/go2w-supermimic) | mjlab 中从仿真回放拟合世界模型，再由想象轨迹优化策略；仓库说明自报可上下 20 cm 台阶，README 验证范围为仿真 | RW-02OP 越障指标、标准化评测或真机结果 |
