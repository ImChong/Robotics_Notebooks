---
type: entity
tags: [paper, humanoid, hardware, open-source, berkeley, reinforcement-learning, qdd, humanoid-paper-notebooks, cycloidal, actuator]
status: complete
updated: 2026-11-21
arxiv: "2504.17249"
original:
  - ../../sources/papers/humanoid_pnb_berkeley-humanoid-lite-an-open-source-accessible.md
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ./humanoid-robot.md
  - ./open-source-humanoid-hardware.md
  - ../comparisons/open-source-qdd-actuator-projects.md
  - ./odri-solo-and-bolt.md
  - ./internal-cycloidal-actuator.md
  - ../overview/robot-open-source-wechat-issue01-curator.md
  - ../methods/reinforcement-learning.md
  - ../../roadmap/depth-torque-motor-design.md
sources:
  - ../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md
  - ../../sources/papers/humanoid_pnb_berkeley-humanoid-lite-an-open-source-accessible.md
  - ../../sources/repos/berkeley_humanoid_lite.md
  - ../../sources/sites/berkeley_humanoid_lite.md
  - ../../sources/personal/open_source_qdd_actuator_learning_curator.md
summary: "Berkeley Humanoid Lite（BHL）：UC Berkeley 低成本准直驱开源人形；~15:1 3D 打印摆线关节、电机/控制参数、BOM、底层控制、Isaac Lab 训练与实机部署——学「执行器如何装进人形」的优先整机参考。"
---

# Berkeley Humanoid Lite（BHL）

## 一句话定义

**Berkeley Humanoid Lite** 是 **UC Berkeley Hybrid Robotics** 的 **轻量人形** 开源方案：门户 **[lite.berkeley-humanoid.org](https://lite.berkeley-humanoid.org/)**，源码与仿真/学习脚本在 **[Berkeley-Humanoid-Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite)**（MIT）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| QDD | Quasi-Direct Drive | 准直驱，低减速比、高背驱动性的作动方案 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| IK | Inverse Kinematics | 满足末端/姿态约束求解关节角的运动学逆解 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| BOM | Bill of Materials | 物料清单，硬件零部件列表 |

## 为什么重要

- **低成本 QDD 叙事**：在 [开源人形硬件对比](./open-source-humanoid-hardware.md) 中常与商业整机对照，作为 **动力学透明 + RL 友好** 的参考轴。
- **最接近「人形关节需求」的开源整机**：含整机 CAD、3D 打印摆线、电机与关节参数、控制器配置、BOM、底层控制、**Isaac Lab** 训练与实机部署。
- 在 [开源 QDD 学习路线](../comparisons/open-source-qdd-actuator-projects.md) 中排在 ODRI 之后：学完力控关节后，看执行器如何进人形。

## 关节与电机要点（原文核对）

| 项 | 内容 |
|----|------|
| 整机规格 | 高 **0.8 m**、重 **16 kg**，22 DoF（**10× 6512** + **12× 5010** 摆线执行器），6S 4000 mAh LiPo（~30 min） |
| 6512 执行器 | MAD M6C12 150KV 外转子（14 对极，$K_T\approx0.1176\,\text{N}\cdot\text{m/A}$）+ B-G431B-ESC1 + AS5600 + 6811ZZ，PLA 摆线减速箱；BOM 美区 **$188** / 中国 **$157** |
| 5010 执行器 | 5010 无刷外转子 + B-G431B-ESC1 + AS5600 + 3D 打印摆线；BOM 美区 **$136** / 中国 **$94** |
| 整机 BOM | 美区 **$4,312** / 中国 **$3,236**（所有打印件适配 200×200×200 mm 桌面 FDM 打印机） |
| 通信与计算 | Intel N95 mini PC + 4 路 1 Mbps USB-CAN 2.0 + BNO085 IMU，底层读写与状态更新 **250 Hz**，PPO 行走策略推理 **25 Hz**（仅消耗极限力矩 30%） |
| 台架实测（§IV） | 摆线机械效率 **~90%**；PLA 传输刚度 **319.49 N·m/rad**；新打印 6 只 6512 最大背隙 **0.0229 rad**（$\sigma=0.0042\,\text{rad}$）；力矩误差 **±0.5 N·m**；5-DoF 臂末端重复定位标准差 **3.433 mm**；60 h @ 0.5 Hz 耐久无故障 |

## 核心机制与硬件系统总览

```mermaid
flowchart LR
  Motor["M6C12 / 5010 外转子电机<br/>+ B-G431B-ESC1 + AS5600"] --> Cyc["FDM-PLA 3D 打印摆线减速<br/>黄铜六角嵌件+贯通螺钉防分层"]
  Cyc --> CAN["4 路 1 Mbps USB-CAN<br/>250 Hz 底层读写 + BNO085 IMU"]
  CAN --> N95["Intel N95 Mini PC<br/>25 Hz PPO 行走策略 / Pinocchio+Pink IK"]
  VR["SteamVR / OptiTrack 遥操作"] --> N95
```

- **用摆线替代行星以容忍桌面 FDM 精度**：相比行星细齿对层间结合与印制公差的严苛要求，摆线多齿同时啮合分散载荷，并借助贯通螺钉与嵌入黄铜六角柱抑制 PLA 层间剥离（§III-C2）。
- **整机架构去耦合与重构**：每关节自包含电源与 CAN（单总线理论挂载上限 64 节点），可把下肢更换碳纤维管扩展为成人尺度或重组成四足/桌面基座（§III-D）。

## 论文与深读状态

| 字段 | 内容 |
|------|------|
| 论文 | *Berkeley Humanoid Lite: An Open-source, Accessible, and Customizable 3D-printed Humanoid Robot*，<https://arxiv.org/abs/2504.17249> |
| Paper Notebooks 分类 | 12_Hardware_Design（[分类父节点](../overview/paper-notebook-category-12-hardware-design.md)） |
| 深读状态 | 已在本库完成 `arXiv:2504.17249v1` 全文核对与核心源码入口校验 |

## 工程实践

| 环节 | 实证取舍与操作规范 |
|---|---|
| 打印与结构强化 | 全部结构件均用 PLA 适配 200 mm³ 构建盒；高载输入轴需嵌入黄铜六角柱并用贯通螺钉压紧层间 |
| 刚度与背隙预期 | PLA 摆线刚度（**319.49 N·m/rad**）明显低于 PA-CF（文献 1468 N·m/rad），高频高载荷下需控制力矩带宽与控制增益避免振荡 |
| 控制器选型 | B-G431B-ESC1 + M6C12/5010 为低成本量产妥协方案，高动态需求可平替 moteus / ODrive / VESC |
| 热边界与降额 | 原论文明确指出尚未系统评测塑料结构长时热软化（§VI），连续高负载测试须监测壳体与绕组温升 |

## 开源入口

| 类型 | 链接 | 状态 |
|------|------|------|
| 项目门户 | [lite.berkeley-humanoid.org](https://lite.berkeley-humanoid.org/) | 已发布 |
| GitBook 文档 | [berkeley-humanoid-lite.gitbook.io/docs](https://berkeley-humanoid-lite.gitbook.io/docs) | 已发布 |
| 主仓库 | [HybridRobotics/Berkeley-Humanoid-Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite) | **已开源**（MIT） |

## 复现入口：硬件与软件

原论文平台以模块化 3D 打印摆线减速器和通用采购件降低门槛；官方按美国市场价格估算硬件成本低于 5000 美元，实际成本依地区与时间变化。复现前先核对[官方发布页](https://berkeley-humanoid-lite.gitbook.io/docs/releases)的 BOM、整机与 6512 / 5010 执行器 CAD、打印文件及修订记录，再按[硬件文档](https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-hardware)装配和校准。具体入口见[项目页归档](../../sources/sites/berkeley_humanoid_lite.md)。

软件链路包括 Isaac Lab 环境与策略训练、sim2sim、实机底层控制及 SteamVR 遥操作；[软件归档](../../sources/repos/berkeley_humanoid_lite.md)列出描述资产（URDF / MJCF / USD）、固件与各仓库。官方组织还发布 Lite V1.5 双臂和 Lite V2 相关资源，不能默认与原论文版本直接兼容。

## 局限与风险

- 官方已指出：当前 **3D 打印摆线** 在高性能运动中偏脆弱，后续版本可能改为成品关节。
- **适合学习与验证，不适合原样用于重型人形**。
- 可与 [Internal Cycloidal](./internal-cycloidal-actuator.md)（电机—减速一体）及 [Urs 等打印 QDD 论文](./paper-3d-printed-open-source-actuators-legged.md)（系统热/寿命评测）对照。

## 源码运行时序图

仓库按功能分三条入口（均在 `scripts/` 下）。`source/berkeley_humanoid_lite_lowlevel/` 是 git 子模块（`.gitmodules` 指向 Berkeley-Humanoid-Lite-Lowlevel），本次浅克隆中为空，下图涉及该子模块处只标注为黑盒，未读其内部。

### 训练：Isaac Lab + RSL-RL

入口：`python scripts/rsl_rl/train.py --task=Velocity-Berkeley-Humanoid-Lite-v0`（任务 id 来自 `tasks/locomotion/velocity/config/humanoid/__init__.py` 的 `gym.register`；`--num_envs`、`--max_iterations`、`--video` 等参数见脚本 argparse）

```mermaid
sequenceDiagram
    participant U as 用户 train.py
    participant AL as isaaclab.app.AppLauncher
    participant H as hydra_task_config
    participant T as berkeley_humanoid_lite.tasks（gym.register）
    participant E as gym.make → ManagerBasedRLEnv
    participant W as RslRlVecEnvWrapper
    participant R as rsl_rl.runners.OnPolicyRunner

    U->>U: argparse，cli_args.add_rsl_rl_args，AppLauncher.add_app_launcher_args
    U->>AL: AppLauncher(args_cli)，得到 simulation_app
    U->>T: import berkeley_humanoid_lite.tasks 触发任务注册
    U->>H: main() 由 @hydra_task_config(task, "rsl_rl_cfg_entry_point") 装饰
    H-->>U: env_cfg（BerkeleyHumanoidLiteEnvCfg）、agent_cfg（BerkeleyHumanoidLitePPORunnerCfg）
    U->>U: 用 CLI 覆盖 num_envs、max_iterations、seed，确定 logs/rsl_rl/... 日志目录
    U->>E: gym.make(task, cfg=env_cfg)
    opt --video
        U->>E: gym.wrappers.RecordVideo
    end
    U->>W: RslRlVecEnvWrapper(env)
    U->>R: OnPolicyRunner(env, agent_cfg.to_dict(), log_dir)
    opt agent_cfg.resume
        U->>R: runner.load(get_checkpoint_path(...))
    end
    U->>U: dump_yaml / dump_pickle 写 params/env 与 agent
    U->>R: runner.learn(num_learning_iterations=max_iterations)
    U->>E: env.close()，simulation_app.close()
```

### 回放与导出，以及 MuJoCo sim2sim

入口：`python scripts/rsl_rl/play.py ...` 加载检查点后导出 `policy.pt` / `policy.onnx` 并写部署用 yaml；再用 `python scripts/sim2sim/play_mujoco.py`（配置经 `Cfg.from_arguments()` 读取，`configs/` 下有 `policy_humanoid.yaml` 等）在 MuJoCo 中加载配置里的策略检查点（配置中为 .onnx 路径；加载逻辑在未读的 RlController 内）。

```mermaid
sequenceDiagram
    participant U as 用户 play_mujoco.py
    participant C as Cfg（lowlevel 子模块，未读）
    participant S as environments/mujoco.py MujocoSimulator
    participant MJ as mujoco（MjModel / mj_step / viewer）
    participant RC as RlController（lowlevel 子模块，未读）

    U->>C: Cfg.from_arguments()
    U->>S: MujocoSimulator(cfg)
    S->>MJ: MjModel.from_xml_path(bhl_scene.xml 或 bhl_biped_scene.xml)，launch_passive 打开 viewer
    U->>S: robot.reset()
    S-->>U: obs（_get_observations）
    U->>RC: RlController(cfg)，load_policy()
    loop while True
        U->>RC: controller.update(obs.numpy())
        RC-->>U: actions（None 时回退到 default_joint_positions）
        U->>S: robot.step(actions)
        loop physics_substeps
            S->>S: _apply_actions（PD 力矩、EMA 滤波、力矩限幅）
            S->>MJ: mj_step
        end
        S->>MJ: mj_viewer.sync()
        S-->>U: 新 obs，并按 policy_dt 休眠对齐实时
    end
```

图依据：HybridRobotics/Berkeley-Humanoid-Lite@984741a，读过 README.md、.gitmodules、scripts/rsl_rl/train.py、scripts/rsl_rl/play.py（仅导出相关行）、scripts/sim2sim/play_mujoco.py、configs/policy_humanoid.yaml（前段）、source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/humanoid/__init__.py、source/berkeley_humanoid_lite/berkeley_humanoid_lite/environments/mujoco.py（reset / step / _apply_actions）。

## 关联页面

- [开源 QDD 执行器项目对比](../comparisons/open-source-qdd-actuator-projects.md)
- [执行器驱动链选型闭环](../queries/actuator-drive-chain-selection-loop.md)
- [ODRI Solo / Bolt](./odri-solo-and-bolt.md)
- [开源人形硬件方案对比](./open-source-humanoid-hardware.md)
- [Paper Notebooks 分类父节点：12 Hardware Design](../overview/paper-notebook-category-12-hardware-design.md)
- [强化学习](../methods/reinforcement-learning.md)
- [力矩电机设计纵深](../../roadmap/depth-torque-motor-design.md)
- [Humanoid Locomotion (人形机器人移动)](../tasks/humanoid-locomotion.md) — 把低成本关节、训练配置与真机部署放到双足行走的稳定性和热限制下评估。

## 推荐继续阅读

- [开源人形硬件方案对比 · Berkeley Humanoid 段](./open-source-humanoid-hardware.md#1-berkeley-humanoid-准直接驱动派)
- 主仓 README 与 GitBook 关节/控制参数章节

## 参考来源

- [berkeley_humanoid_lite.md](../../sources/repos/berkeley_humanoid_lite.md)
- [berkeley_humanoid_lite 项目页](../../sources/sites/berkeley_humanoid_lite.md)
- [开源 QDD 执行器学习策展](../../sources/personal/open_source_qdd_actuator_learning_curator.md)
- [wechat_jixie_robot_open_source_treasury_issue01_10_robots.md](../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md)
- [humanoid_pnb_berkeley-humanoid-lite-an-open-source-accessible.md](../../sources/papers/humanoid_pnb_berkeley-humanoid-lite-an-open-source-accessible.md)
- 论文：<https://arxiv.org/abs/2504.17249>
- 源码核对（2026-11-02）：HybridRobotics/Berkeley-Humanoid-Lite@984741a，读过 README.md、.gitmodules、scripts/rsl_rl/{train,play}.py、scripts/sim2sim/play_mujoco.py、configs/policy_humanoid.yaml、tasks 注册 __init__.py、environments/mujoco.py；lowlevel 子模块（RlController、Cfg）未克隆未读；时序图依据以上文件，未运行代码。
