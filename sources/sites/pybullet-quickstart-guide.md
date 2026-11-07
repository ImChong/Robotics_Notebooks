# PyBullet Quickstart Guide（官方 API 文档）

> 来源归档（官方文档通读）

- **标题：** PyBullet Quickstart Guide
- **作者：** Erwin Coumans, Yunfei Bai（2016–2023）
- **类型：** site / 官方文档（Google Docs）
- **链接：** <https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit>（2026-11-07 通过 `/export?format=txt` 取得纯文本，约 16 万字符、87 页目录）
- **配套：** [`../repos/bullet3.md`](../repos/bullet3.md)（C++ SDK 与 `pybullet_envs`）、[`pybullet-org.md`](pybullet-org.md)（官方站）
- **入库日期：** 2026-11-07
- **一句话说明：** PyBullet 的官方 API 手册：连接方式、加载模型、步进、关节电机控制、状态读取、碰撞查询、逆运动学 / 动力学、Gym 环境与安装。
- **核对范围：** 读完目录与 Introduction；逐段读了「connect / disconnect」「stepSimulation、performCollisionDetection、setRealTimeSimulation」「setJointMotorControl2 / Array」「getLinkState」；读了「Reinforcement Learning Gym Envs」整章、「Build and install」开头与「Tips, Citation」FAQ（阻尼、速度钳制、控制模式建议、Colab）。**没有读**：可变形体 / 布料、相机渲染、碰撞查询细节、约束、VR、调试 GUI、插件、逆运动学细节等章节的正文；没有运行任何脚本。
- **通读：** 未完成
- **沉淀到 wiki：** [`wiki/entities/pybullet.md`](../../wiki/entities/pybullet.md)

## 核心摘录（均出自原文）

- **定位（Introduction）：** 面向机器人仿真与机器学习、强调 sim-to-real 的 Python 模块；可从 URDF、SDF、MJCF 加载关节体；提供正向动力学仿真、逆动力学、正逆运动学、碰撞检测与射线查询；可用 pip 安装，同时暴露 `pybullet_envs` Gym 环境。简介称降维坐标多体、刚体与可变形体由统一的 LCP 约束求解器处理，并用 Articulated Body Algorithm 做线性时间正向动力学。默认使用 CPU 上的 Bullet 2.x API，OpenCL GPU 的 Bullet 3.x 将来放在同一 C-API 之后。
- **客户端–服务器：** PyBullet 是客户端发命令、物理服务器返回状态的架构；内置服务器 `DIRECT`（无图形、同进程）与 `GUI`（OpenGL 窗口、同进程；Linux / Windows 另起线程）；还可用 `SHARED_MEMORY`、UDP、TCP 连接别的进程或远程机器。`DIRECT` 下不可用 OpenGL / VR / 调试 GUI，但可用软件渲染的 `getCameraImage`，适合无 GPU 的云服务器。
- **步进（stepSimulation）：** 一次调用完成碰撞检测、约束求解与积分；**默认时间步 1/240 s**，可用 `setTimeStep` / `setPhysicsEngineParameter` 改；服务器默认**不自动步进**，除非开启 `setRealTimeSimulation`，这样便于保持确定性。
- **关节电机（setJointMotorControl2）：** 模式有 `POSITION_CONTROL`（实质为 position-velocity PD）、`VELOCITY_CONTROL`、`TORQUE_CONTROL`、`PD_CONTROL`，另有实验性的 `STABLE_PD_CONTROL`。**默认每个转动 / 移动关节都带一个速度电机**，要做力矩控制必须先用 `VELOCITY_CONTROL` 且 `force=0` 关掉它；`TORQUE_CONTROL` 立即施加力矩，仅在显式调用 `stepSimulation` 时有效；PD 的目标速度公式由 `kp`、`erp`、`dt` 与当前速度决定。文档 FAQ 建议先用位置或速度控制，力矩控制要靠更多工作才可靠。
- **状态读取：** `getLinkState` 返回连杆质心的世界位置、四元数朝向 [x,y,z,w]、局部惯性系偏移等，可选计算速度与重算正运动学；`getJointState` 读关节状态。
- **稳定性相关默认值（FAQ）：** PyBullet 默认施加一定的线 / 角阻尼，速度上限钳制在 100；可用 `changeDynamics` 把 `linearDamping`、`angularDamping` 设 0。
- **Gym 环境（RL 章）：** `pip install pybullet` 同时安装 `pybullet_envs` 与 `pybullet_data`；导入后自动注册 Gym 环境。包括 Roboschool 风格的 Ant、HalfCheetah、Hopper、Walker2D、Humanoid、倒立摆系列，以及 MinitaurBulletEnv-v0（RSS 2018 sim-to-real 四足论文所用）、DeepMimic 重实现（Backflip / Walk，需 TensorFlow 1.x，带 MPI 并行 PPO 训练）、MIT Racecar、Kuka 抓取环境（文档注明 Kuka 环境训练存在问题）。文档称可用 DQN、PPO、TRPO、DDPG 等训练，并提及 Stable Baselines。
- **Colab：** 提供预编译 manylinux wheel，可在 Colab 使用，EGL 渲染可用。
- **引用：** Coumans & Bai, *PyBullet, a Python module for physics simulation for games, robotics and machine learning*, 2016–2023。
