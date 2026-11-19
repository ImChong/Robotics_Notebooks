# PyBullet Quickstart Guide（官方 API 文档）

> 来源归档（官方文档与 `pybullet_envs` 源码通读）

- **标题：** PyBullet Quickstart Guide
- **作者：** Erwin Coumans, Yunfei Bai（2016–2023）
- **类型：** site / 官方文档（Google Docs）+ 核心 Gym 封装源码
- **链接：** <https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit>（纯文本导出 4629 行，约 16 万字符）
- **配套：** [`../repos/bullet3.md`](../repos/bullet3.md)（C++ SDK 与 `pybullet_envs`）、[`pybullet-org.md`](pybullet-org.md)（官方站）
- **入库日期：** 2026-11-07（2026-11-19 补完全文与底层封装源码通读）
- **一句话说明：** PyBullet 官方 API 与架构全景手册：覆盖客户端–服务器连接、多格式加载与状态快照、关节驱动与阻尼控制、正逆动力学/IK/Jacobian、FEM/PBD 软体、合成视觉渲染、碰撞查询及 `pybullet_envs` Gym 运行时机制。
- **核对范围：** 2026-11-19 通读 Quickstart 导出文本全文（4629 行，涵盖 Introduction/Hello World、Client-Server、URDF/SDF/MJCF 与状态序列化、Multi-body builder、stepping/transforms、Joint/Motor 控制与传感器、Dynamics/Constraints、Deformables/Soft Body、Rendering 渲染与摄像头、Collision Detection 与射线查询、IK/ID 与 Jacobian/MassMatrix、Shared Memory/VR、`pybullet_envs` Gym 环境及 Build/FAQ），并精读 `bullet3/examples/pybullet/gym/pybullet_envs/` 核心基类 `env_bases.py`、`robot_bases.py`、`robot_locomotors.py`；未重跑训练脚本。
- **通读：** 已完成
- **沉淀到 wiki：** [`wiki/entities/pybullet.md`](../../wiki/entities/pybullet.md)

## 核心摘录（均出自官方文档与配套源码）

### 1) 客户端–服务器连接与多实例管理

- **五种连接模式：** `DIRECT`（同进程无头软件仿真，适合服务器无 GPU 训练，仍支持 CPU 软件渲染）、`GUI`（带 OpenGL 3D 调试窗口）、`SHARED_MEMORY`（跨进程连 Bullet VR / C++ 示例服务器）、`UDP` 与 `TCP` 远程物理服务器。
- **面向对象客户端：`pybullet_utils.bullet_client.BulletClient`**：在单 Python 进程内管理多个相互独立的物理仿真实例，每次方法调用自动绑定对应的 `physicsClientId`，避免全局状态混淆。

### 2) 模型加载、程序化建模与状态快照

- **文件格式：** 支持 `loadURDF`、`loadSDF`、`loadMJCF`；`createCollisionShape` / `createVisualShape` + `createMultiBody` 支持运行时无 XML 拼装刚体/多体树。
- **状态保存与恢复：** `saveState`（内存快照）、`saveBullet` / `restoreState`（磁盘二进制状态读写，用于 RL 分支 rollout 与复位加速）、`saveWorld`（导出当前场景初始化脚本）。

### 3) 物理步进、求解器参数与坐标变换

- **显式步进优先：** 默认时间步 `1/240 s`，`stepSimulation()` 单次执行宽相/窄相碰撞检测、约束求解与积分；`performCollisionDetection()` 可在不推进时间的情况下单独刷新接触结果；默认不开启实时异步步进（`setRealTimeSimulation(0)`）以保障步进确定性。
- **确定性与求解器：`setPhysicsEngineParameter`** 支持配置 `deterministicOverlappingPairs=1`、`numSolverIterations`、`erp`、`contactERP`、`frictionERP` 等；所有朝向与旋转变换统一使用四元数 `[x, y, z, w]`（与 `euler` 转换 API 配合）。

### 4) 关节电机控制、传感器与默认动力学陷阱

- **控制模式（`setJointMotorControl2` / `setJointMotorControlArray`）：** 提供 `POSITION_CONTROL`（底层按位置–速度 PD 计算目标力矩/速度）、`VELOCITY_CONTROL`、`TORQUE_CONTROL`、`PD_CONTROL`，以及实验性 `STABLE_PD_CONTROL`。
- **力矩控制前置步骤：** URDF/MJCF 载入的旋转/平移关节默认挂有速度电机（带隐藏阻尼力）；使用 `TORQUE_CONTROL` 前**必须**先以 `setJointMotorControl2(..., VELOCITY_CONTROL, force=0)`（或 `POSITION_CONTROL` 且 `force=0`）显式关闭默认电机，且 `TORQUE_CONTROL` 仅在显式 `stepSimulation` 模式下生效。
- **传感器与状态：** `enableJointForceTorqueSensor` 开启六维关节反力矩测量并由 `getJointState` 读取；`getLinkState` 默认返回**连杆质心（CoM）**世界位姿与惯性系偏移，而非 URDF 连杆坐标系原点；`getBasePositionAndOrientation` 与 `getBaseVelocity` 读取基座（root link）状态。
- **动力学参数与默认阻尼：** `getDynamicsInfo` / `changeDynamics` 管理质量、局部惯性、摩擦系数、恢复系数；默认存在非零线/角阻尼并将关节/刚体线速度上限钳制在 `100`，高保真或高速动力学场景需显式调零。
- **通用约束与软体：** `createConstraint` 可运行时构建闭链、固定架或抓取辅助约束（`JOINT_FIXED` / `JOINT_POINT2POINT` 等）；`loadSoftBody` 支持基于 FEM（有限元）和 PBD（基于位置的动力学）的可变形体及布料仿真与刚体双向耦合（常配合 `RESET_USE_DEFORMABLE_WORLD` 启用）。

### 5) 合成视觉、碰撞查询、IK/ID 与动力学矩阵

- **视觉渲染（`getCameraImage`）：** 支持同时输出 RGBA、深度缓冲（depth buffer）与实例/连杆分割掩码（segmentation mask）；切换 `ER_TINYRENDERER`（纯 CPU 软件光栅化，`DIRECT` 下可用）与 `ER_BULLET_HARDWARE_OPENGL`（需 `GUI` 或通过 `PYBULLET_EGL` 加载 `eglRendererPlugin` 实现无头 GPU 离屏渲染）。
- **几何与接触查询：** `getContactPoints`（实际接触力、法向量、穿透深度）、`getClosestPoints`（给定安全距离阈值内的最近点，无需步进即可做碰撞距离监测）、`rayTestBatch` 批量激光雷达射线查询，以及内置 `vhacd` 对非凸网格进行凸分解。
- **运动学与动力学算法：**
  - **逆运动学：** `calculateInverseKinematics` 与 `calculateInverseKinematics2`（多末端同步求解），支持 Damped Least Squares（DLS）与 Selective Damped Least Squares（SDLS）及零空间姿态引导（null-space rest poses）。
  - **解析动力学（基于 ABA / RNEA / CRBA）：** `calculateInverseDynamics` 计算给定 `(q, qdot, qddot)` 所需广义关节力矩（RNEA）；`calculateJacobian` 返回质心或局部点线/角速度雅可比；`calculateMassMatrix` 计算广义质量矩阵 $M(q)$（CRBA）。

### 6) `pybullet_envs` Gym 运行时代码结构（源码实核）

- **`MJCFBaseBulletEnv`（`env_bases.py`）：**
  - `reset()` 在首次调用时根据 `isRender` 初始化 `BulletClient(pybullet.GUI)` 或无参 `DIRECT` 客户端，随后调用 `resetSimulation()` 与 `setPhysicsEngineParameter(deterministicOverlappingPairs=1)`，并在设置 `PYBULLET_EGL` 且为 `DIRECT` 时尝试 `loadPlugin(..., "_eglRendererPlugin")`；然后创建单人场景 `create_single_player_scene`、调用 `scene.episode_restart(self._p)` 与 `self.robot.reset(self._p)`，最后以 `calc_potential()` 初始化势能基线。
- **`XmlBasedRobot` / `MJCFBasedRobot` / `URDFBasedRobot`（`robot_bases.py`）：**
  - `reset(bullet_client)` 通过 `loadMJCF` / `loadURDF` 加载模型并调用 `addToScene` 遍历所有关节；`addToScene` 对每个关节立即执行 `setJointMotorControl2(..., POSITION_CONTROL, positionGain=0.1, velocityGain=0.1, force=0)` **主动关闭默认电机**，随后调用 `robot_specific_reset` 与 `calc_state()`。
- **`WalkerBase`（`robot_locomotors.py`）：**
  - `apply_action(a)` 将裁剪到 `[-1, 1]` 的动作乘以 `power * power_coef`，调用 `Joint.set_motor_torque` 转为 `setJointMotorControl2(..., controlMode=pybullet.TORQUE_CONTROL, force=torque)`。
  - `calc_state()` 汇总 `Joint.current_relative_position()`、`robot_body.pose()` / `speed()`、目标方向偏航角及 `feet_contact` 标志并裁剪到 `[-5, +5]` 作为观测向量返回。

## 引用

- Coumans, E., & Bai, Y. (2016–2023). *PyBullet, a Python module for physics simulation for games, robotics and machine learning*. <http://pybullet.org>.
