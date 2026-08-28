# NVIDIA：Getting Started With Isaac Lab（官方四模块课）

> 来源归档

- **标题：** Getting Started With Isaac Lab
- **类型：** course（厂商官方自学课，四模块）
- **来源：** NVIDIA Physical AI Learning
- **链接：** https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/index.html
- **上级门户：** https://docs.nvidia.com/learning/physical-ai/
- **入库日期：** 2026-08-28
- **一句话说明：** 官方 Isaac Lab 入门主线：机器人学习与 Sim/Lab 分工 → Cartpole 并行 PPO → UR10+夹爪 reach 自定义 MDP → sim-to-real 三类桥接（仿真增强 / Real2Sim / 策略鲁棒）。
- **开源状态：** 课程本身是文档；可运行代码走 [Isaac Lab](https://github.com/isaac-sim/IsaacLab)（已开源）。无独立课程仓。云端入口为 NVIDIA Brev / Isaac Launchable。
- **沉淀到 wiki：** 是 → [`wiki/entities/nvidia-getting-started-isaac-lab.md`](../../wiki/entities/nvidia-getting-started-isaac-lab.md)

---

## 为什么值得保留

这是 Physical AI 门户上 **Isaac Lab 动手入门** 路径（官方标注 Intermediate · 3–4 h），与已入库的 [SO-101 Sim2Real 课](./nvidia_sim_to_real_so101_isaac.md) 互补：本课走 **RL + manager-based 任务设计**，SO-101 课走 **VLA + 四类 gap 策略**。第四模块虽无动手实验，但把 reality gap 拆成三类桥接，可直接对照本库 [Sim2Real](../../wiki/concepts/sim2real.md) / [Privileged Training](../../wiki/concepts/privileged-training.md) 概念页。

---

## 课程结构（四模块）

| 模块 | 主题 | 入口 |
|------|------|------|
| 1 | An Introduction to Robot Learning and Isaac Lab | [index](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/an-introduction-to-robot-learning-and-isaac-lab/index.html) |
| 2 | Train Your First Robot（Cartpole） | [index](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-your-first-robot-with-isaac-lab/index.html) |
| 3 | Train Your Second Robot（UR10 + Robotiq 2F-140 reach） | [index](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-your-second-robot-with-isaac-lab/index.html) |
| 4 | Transferring Policies From Simulation to Reality | [index](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/transferring-robot-learning-policies-from-simulation-to-reality/index.html) |

官方门户标注 **Intermediate · 3–4 h**。模块 4 明确 **不含动手实验**。

---

## 核心摘录

### 1) 模块 1：Robot Learning 与 Sim / Lab 分工

- **Robot learning：** 用数据与环境交互让机器人自主学习；对照「从第一性原理写控制律」。收益：自治、适应、减少手工编程、跨情境泛化。
- **IL：** 轨迹采集（遥操作 / 视频 / mocap / oracle / 轨迹优化）→ 策略训练（BC 或 IRL）→ 评测部署。点名 DAgger、AMP。局限：难泛化到演示外、不优化长期后果。
- **RL：** 无需预存演示；agent 得 observation / reward / done。强调 **state ≠ observation**（完整物理态 vs 可测信号）。过程：初始化策略 → rollout → 策略更新。Isaac Lab 默认算法叙事是 **PPO**；支持 model-free / model-based。视觉 RL 走 Isaac Sim 渲染 + **tiled rendering**。
- **产品谱系：** Isaac Gym Standalone 把数据生成与训练留在 GPU，但不是通用仿真器（无刚柔耦合、高保真渲染、ROS，也不基于 Omniverse）。Isaac Lab 合并 OIGE + Orbit。
- **三分法：** 数据生成与渲染 → Isaac Sim；任务与环境 Python API → Isaac Lab；训练库 → RSL-RL / RL-Games / SKRL / Stable-Baselines。
- **七步闭环：** Sim 资产与物理态 → Lab 处理 → 加噪声（sim2real）→ 观测进策略 → 动作回 Sim → 循环。产物：`.pt` / `.onnx`（可选 `.jit`）。
- **环境族（课内口径，入库时为「26+」预置任务）：** 灵巧操作、足式 locomotion、多智能体、导航（到点，不同于跟速度指令）、tiled 视觉 RL、遥操作 + IL（GR00T-Mimic 增广 + RoboMimic）。全量注册 ID 以 [Isaac Lab 默认环境](../../wiki/entities/isaac-lab-default-environments.md) 为准。
- **Tiled rendering：** 多相机输出拼成一张大图一次渲染；`TiledCamera` 支持 RGB / depth 等 annotator。课内举例 Cartpole 与 ShadowHand。

### 2) 模块 2：Cartpole 第一台机器人

- 外部工程：`./isaaclab.sh --new` → External / Manager-based / **skrl + PPO**；`pip install -e source/Cartpole`；任务名 `Template-Cartpole-v0`。
- 默认 **4096** 并行；可用 `--num_envs` 降载；`--headless` 关视口；云端流式加 `--livestream 2`。
- Manager 映射 MDP：`TerminationsCfg`（超时 + 滑轨越界 ±3 m）、`ActionsCfg`（`JointEffortActionCfg`，scale 100）、`ObservationsCfg`（`joint_pos_rel` / `joint_vel_rel`）、`RewardsCfg`（alive +1、终止 −2、杆角 L2、车速/杆速 shaping）。
- `CartpoleEnvCfg`：`num_envs=4096`，`decimation=2`，`episode_length_s=5`，`sim.dt=1/120`。奖励函数对 **整批环境的 tensor** 一次算完。
- **版本坑：** 课内警告当前 Isaac Launchable 用 **Isaac Lab 3.0**，可能与课测版本不兼容。

### 3) 模块 3：UR10 + 2F-140 reach

- 资产：Isaac Sim 里把 `ur10_instanceable.usd` 与 Robotiq 2F-140 以 **USD reference** 组合；去掉夹爪多余 Articulation Root；`Fixed Joint` 接到 `ee_link`；保存 `UR-with-gripper.usd`。口诀：**Sim 配机器人 → Lab 训练 → 回 Sim 验证**。
- 工程选择：推荐 **external template**，不要把任务写进 Isaac Lab 仓库内部（升级困难）。Workflow：Manager-based single-agent；库：skrl PPO。
- `ArticulationCfg`：`ImplicitActuatorCfg`（臂 stiffness 800 / damping 40；夹爪 280 / 28）；执行器与场景道具放 Lab 配置，不写进核心 USD。
- **Manager vs Direct：** Direct 单脚本、上手快、适合难拆分逻辑与 JIT，接近旧 Isaac Gym 心智；Manager 模块化、可复用、团队协作，**课内推荐新人**。
- MDP 组件：`ActionsCfg`（六轴 `JointPositionActionCfg`）≠ `CommandsCfg`（`UniformPoseCommandCfg` 目标位姿，每 4 s 重采样）；观测含关节 pos/vel + 命令 + last action，`enable_corruption=True` 加 `Unoise`；终止仅 timeout；reset 用 `reset_joints_by_scale`。
- 奖励：末端位置 L2（负权）+ tanh 细粒度（近目标梯度更大）+ 动作变化率 / 关节速度惩罚；课程在 4500 step 加大后两项权。**第一次 play 只到位姿、不到姿态** → 再加 `orientation_command_error`（最短路径四元数误差）。
- 调试：`zero_agent` / `random_agent` 先确认加载与关节能动；`Template-Reach-Play-v0` 回放。
- **社区坑（Lab 3.x / skrl 2.x）：** 模板 `skrl_ppo_cfg.yaml` 若仍写 `input: STATES`，训练会 `NoneType.shape`；改为 `OBSERVATIONS`（见 [IsaacLab#5416](https://github.com/isaac-sim/IsaacLab/issues/5416)）。课内 YAML 原文仍是 `STATES`。

### 4) 模块 4：Sim-to-Real 理论（无动手）

- **为何仿真：** RL 样本 1e7–1e9、真机不安全、重置劳动密集。课内数字：Unitree G1 粗糙地形 locomotion，RTX 4090 上 **1 s 仿真 ≈ 27 min 真机经验**。
- **Reality gap 三源：** 近似误差（离散化穿透、不守恒）、模型误差（质量/摩擦/公差/磨损）、未建模动力学（SEA、吸盘接触、传感器噪声、网络/执行器延迟）。
- **三类桥接：**
  1. **仿真增强：** 物理/形状/任务 DR；视觉 DR（纹理/色/光/相机，**不追求照片级**）；深度相机噪声（边缘孔洞、非均匀噪声）；点云扰动（位移、孔、干扰点）。
  2. **Real-World Data Integration：** SysID；**Actuator Network**（指令+关节历史 → 弹簧偏转测得力矩，冻结后替换仿真 PID）；数字孪生扫描网格；NeRF 作训练期渲染并叠动态物体；世界模型跨 sim/real 学共享 latent。
  3. **策略鲁棒：** 正则（action rate、关节速度、接触力，防仿真里「能完成但会砸硬件」）；特权信息：非对称 actor-critic vs teacher–student（课内 DextrAH-G 三阶段叙事）。
- **权衡：** DR 扩大仿真圆 → 更 generic、专项变弱；Real2Sim 平移仿真圆 → 更 specialist、要采数。建议最小必要 DR + 有针对的 Real2Sim。
- **Tips：** 先测再随机化；从最小 DR 集开始；首次真机部署极度谨慎。

---

## 对 wiki 的映射

| 知识点 | wiki 页 |
|--------|---------|
| 本课主线（四模块 workflow） | [`wiki/entities/nvidia-getting-started-isaac-lab.md`](../../wiki/entities/nvidia-getting-started-isaac-lab.md) |
| 门户选型 | [`wiki/entities/nvidia-physical-ai-learning.md`](../../wiki/entities/nvidia-physical-ai-learning.md) |
| Isaac Lab / Sim / Gym 分工 | [`wiki/entities/isaac-lab.md`](../../wiki/entities/isaac-lab.md)、[`wiki/entities/isaac-sim.md`](../../wiki/entities/isaac-sim.md)、[`wiki/entities/isaac-gym-isaac-lab.md`](../../wiki/entities/isaac-gym-isaac-lab.md) |
| Cartpole 教学任务 | [`wiki/concepts/cartpole.md`](../../wiki/concepts/cartpole.md) |
| Manager vs Direct、skrl PPO | [`wiki/entities/isaac-lab.md`](../../wiki/entities/isaac-lab.md)、[`wiki/entities/skrl.md`](../../wiki/entities/skrl.md) |
| Implicit 执行器 | [`wiki/concepts/implicit-explicit-actuator-modeling.md`](../../wiki/concepts/implicit-explicit-actuator-modeling.md) |
| Sim2Real / DR / SysID / 特权 | [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)、[`wiki/concepts/domain-randomization.md`](../../wiki/concepts/domain-randomization.md)、[`wiki/concepts/system-identification.md`](../../wiki/concepts/system-identification.md)、[`wiki/concepts/privileged-training.md`](../../wiki/concepts/privileged-training.md) |
| Actuator Network | [`wiki/methods/actuator-network.md`](../../wiki/methods/actuator-network.md) |
| 操作臂 reach / 默认任务 | [`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md)、[`wiki/entities/isaac-lab-default-environments.md`](../../wiki/entities/isaac-lab-default-environments.md) |
| 对照：VLA 动手课 | [`wiki/entities/nvidia-so101-sim2real-lab-workflow.md`](../../wiki/entities/nvidia-so101-sim2real-lab-workflow.md) |

---

## 推荐继续阅读（外部）

- 课程首页：https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/index.html
- Isaac Lab 文档：https://isaac-sim.github.io/IsaacLab/
- Task Design Workflows：https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html
- Tiled Rendering：https://isaac-sim.github.io/IsaacLab/main/source/overview/sensors/tiled_camera.html
- Isaac Lab 仓库：https://github.com/isaac-sim/IsaacLab
- 社区修复（skrl `STATES`→`OBSERVATIONS`）：https://github.com/isaac-sim/IsaacLab/issues/5416
