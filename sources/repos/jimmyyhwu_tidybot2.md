# GitHub 仓库：jimmyyhwu/tidybot2

- **仓库：** https://github.com/jimmyyhwu/tidybot2
- **关联项目：** TidyBot++
- **开源状态：** 已开源（包含全套机械 CAD/BOM 引用、250 Hz 舵轮 CAN 底层控制、1 kHz Kinova 柔顺关节控制、MuJoCo 仿真镜像环境、WebXR 手机遥操服务端及 Diffusion Policy ZMQ 远程推理客户端/服务端代码）
- **核对范围：** 已通读官方仓库 `tidybot2` 中的 `README.md`、`constants.py`、`main.py`、`real_env.py`、`mujoco_env.py`、`base_server.py`、`base_controller.py`、`arm_server.py`、`arm_controller.py`、`ik_solver.py`、`policies.py`、`policy_server.py` 与 `convert_to_robomimic_hdf5.py`
- **通读：** 已完成

## 代码结构与核心模块

1. **进程隔离与实时底层控制（`base_server.py` / `base_controller.py` & `arm_server.py` / `arm_controller.py`）**
   - 顶层 `main.py` 在 **10 Hz**（`POLICY_CONTROL_PERIOD = 0.1s`）主循环内统一调度 `RealEnv` 或 `MujocoEnv`，通过 Python `multiprocessing.managers`（端口 `50000` / `50001`）向各自独立高优进程发送控制指令，避免相机编解码/GC 抢占底层控制周期。
   - **底盘进程（`base_controller.Vehicle`）**：绑定 `SCHED_FIFO` 实时线程运行 **250 Hz**（`dt = 4 ms`），读取 Phoenix 6 8 台 TalonFX 电机与 4 个 CANcoder 编码器，构建操作空间运动学映射 `C` 与伪逆 `C_pinv` 完成全局里程计积分与速度分配，内置 **Ruckig** 在线时间最优平滑插补与防甩尾限速（转向速度 > 720 deg/s 自动减速），超时 `> 0.25s` 自动置 `set_neutral`。
   - **机械臂进程（`arm_controller.JointCompliantController` + `ik_solver.IKSolver`）**：`Arm.execute_action` 通过 `mujoco.mj_jacSite` 阻尼伪逆求解 7-DOF Kinova 逆运动学（含回缩零位 `retract` 零空间约束，单次求解 ~0.59 ms），交由 **1 kHz** 力矩回调 `control_callback` 经 **Ruckig** 平滑与名义电机模型阻尼柔顺控制律输出 `tau_c`。

2. **仿真-真机同构环境与 WebXR 遥操作（`real_env.py` / `mujoco_env.py` / `policies.py`）**
   - `RealEnv` 与 `MujocoEnv` 共享一致字典观测（`base_pose`、`arm_pos`、`arm_quat`、`gripper_pos`、底座/手腕图像）与动作接口；`MujocoEnv` 使用 `ShmState`/`ShmImage` 共享内存与独立 `mjpython` 渲染进程同步状态。
   - `TeleopPolicy` 启动本地 Flask + SocketIO HTTPS WebXR 服务端并结合 `TeleopController` 解析手机 VIO 增量位姿并过滤 `> 250 ms` 过期包。

3. **Diffusion Policy 异步 ZMQ 推理与数据转换（`policy_server.py` / `convert_to_robomimic_hdf5.py`）**
   - `RemotePolicy` 经 `ZMQ REQ/REP`（端口 `5555`，图像 84×84 JPEG）向运行于 GPU 主机的 `PolicyServer`（`PolicyWrapper` + `DiffusionPolicy`）异步请求 8 步动作块；`PolicyWrapper` 利用 `LATENCY_STEPS = 2`（200 ms 预算）与异步线程重叠推理耗时（~115 ms RTX 4080 Laptop）实现无卡顿闭环，同时提供 `EpisodeWriter` 与 `convert_to_robomimic_hdf5.py` 无缝导出 Robomimic HDF5 数据集。
