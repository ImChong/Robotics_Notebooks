---
type: entity
project_id: locoleggedwheel
tags:
  - repo
  - independent
  - wheel-legged
  - rl
  - isaac-lab
  - sim2real
  - go2w
status: complete
updated: 2026-11-01
code: https://github.com/zhaozijie2022/LocoLeggedWheel
related:
  - ../entities/isaac-lab.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
  - ../tasks/locomotion.md
  - ../overview/quadruped-open-source-projects-technology-map.md
sources:
  - ../../sources/repos/locoleggedwheel.md
  - ../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md
summary: "LocoLeggedWheel（LocoWheeledLegged）是基于 NVIDIA Isaac Lab 的轮足机器人 RL 运动控制项目，重点优化奖励函数与训练稳定性，并提供宇树 Go2W 的 sim2real 部署接口与配置。"
---

# LocoLeggedWheel

**LocoLeggedWheel**（[GitHub](https://github.com/zhaozijie2022/LocoLeggedWheel)）来自 [古月居转载的四足机器人开源项目综述](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)，属「轮足机器人」组。

## 一句话定义

**LocoLeggedWheel（LocoWheeledLegged）是基于 NVIDIA Isaac Lab 的轮足机器人 RL 运动控制项目，重点优化奖励函数与训练稳定性，并提供宇树 Go2W 的 sim2real 部署接口与配置。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Go2W | Unitree Go2-W | 宇树轮足版 Go2（README 所称） |

## 为什么重要

- 综述把它放在「轮足机器人」这条技术路线下，作为可读源码的参照项目；横向对比见 [四足机器人开源项目技术地图：从极简运动学到研究级硬件（36 个 GitHub 项目）](../overview/quadruped-open-source-projects-technology-map.md)。
- 文中星标：115（文章表述，入库日未复核）。

## 核心信息

| 项 | 内容 |
|----|------|
| **仓库** | [zhaozijie2022/LocoLeggedWheel](https://github.com/zhaozijie2022/LocoLeggedWheel) |
| **最近提交** | 2026-07-09 |
| **可达性** | `git ls-remote` 可达（2026-11-01） |
| **文章归类** | 轮足机器人 |

## 文章说法核对

- **文章：** 文章称基于 Isaac Lab，强调 Sim-to-Real 迁移。
- **README / 核查：** README 指出 Isaac Gym 版本见 DreamWaQ_Go2W；安装需 Isaac Sim 5.x（README 同时出现 v5.5.1 与 5.1.0 两种写法）；无 LICENSE 文件。
- **差异与保留：** README 中 Isaac Sim 版本写法不一致，部署前以官方文档为准。

## 源码运行时序图

### 训练（仿真）

入口：`python locowheeledlegged/scripts/train.py --task Isaac-LocomotionGo2W-v1 --num_envs=4096 --headless`

```mermaid
sequenceDiagram
    participant U as 用户 CLI
    participant T as scripts/train.py main()
    participant G as gym 注册表 Isaac-LocomotionGo2W-v1
    participant E as ManagerBasedRLEnv（LocomotionEnvCfg）
    participant W as RslRlVecEnvWrapper
    participant R as loco_rl OnPolicyRunner
    participant A as loco_rl PPO
    participant AC as actor_critic 网络
    U->>T: 运行 train.py（先 AppLauncher 启动 Isaac Sim）
    T->>T: import locowheeledlegged 触发 gym.register
    T->>T: hydra_task_config 注入 env_cfg 与 LocomotionPPORunnerCfg
    T->>G: gym.make(task, cfg=env_cfg)
    G-->>E: 创建仿真环境
    T->>W: RslRlVecEnvWrapper(env)
    T->>R: OnPolicyRunner(env, agent_cfg, log_dir)
    R->>AC: 构建 actor_critic
    R->>A: alg_class 即 PPO(actor_critic)，init_storage
    T->>R: runner.learn(max_iterations)
    loop 每次迭代，num_steps_per_env 步
        R->>A: act(obs, critic_obs)
        A->>AC: actor_critic.act(obs)
        R->>W: env.step(actions)
        W->>E: 仿真推进一步
        E-->>R: obs, rewards, dones, infos
        R->>A: process_env_step(rewards, dones, infos)
    end
    R->>A: compute_returns(critic_obs)
    R->>A: update() 计算 surrogate、value 损失并 Adam 优化
    R->>R: log() 写 TensorBoard；按 save_interval 调用 save()
```

图依据：LocoLeggedWheel@015b823，实际读过 README.md、locowheeledlegged/scripts/train.py、locowheeledlegged/config/go2w/__init__.py、loco_rl/runners/on_policy_runner.py、loco_rl/algorithms/ppo.py；未运行代码。

### 实机部署（Go2-W，sim2real）

入口（工作目录为 `deploy_real/`）：`python deploy_real/deploy_real.py <网卡名> go2w.yaml`

```mermaid
sequenceDiagram
    participant O as 操作者与遥控器
    participant D as deploy_real.py main()
    participant C as Controller
    participant P as actor 网络（policy_path）
    participant SDK as unitree_sdk2py 通道
    participant RB as Go2-W 底层 LowState/LowCmd
    O->>D: 运行 deploy_real.py（网卡名，go2w.yaml）
    D->>D: Config(./configs/go2w.yaml)
    D->>SDK: ChannelFactoryInitialize(0, 网卡)
    D->>C: Controller(config)
    C->>P: torch.load 后仅载入 actor.* 权重
    C->>SDK: 创建 LowCmd 发布者与 LowState 订阅者
    D->>C: zero_torque_state() 等待 start 键，发零力矩
    D->>C: move_to_default_pos() 插值到默认姿态
    D->>C: default_pos_state() 等待 A 键
    loop 每个控制周期 control_dt
        D->>C: run()
        C->>C: 摇杆写 cmd；LowState 得到 ang_vel、gravity、pos、vel，拼接历史窗口
        C->>P: policy(obs_tensor)
        P-->>C: action
        C->>C: 腿 q = 默认角 + action 缩放；轮 dq = action 缩放
        C->>SDK: send_cmd(low_cmd) 计算 CRC 后发布
    end
    O->>D: 按 select 键退出
    D->>C: create_damping_cmd，发送阻尼指令
```

图依据：LocoLeggedWheel@015b823，实际读过 README.md、deploy_real/deploy_real.py、deploy_real/configs/go2w.yaml（检索关键字段）；未运行代码。

## 关联页面

- [isaac-lab](../entities/isaac-lab.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [sim2real](../concepts/sim2real.md)
- [locomotion](../tasks/locomotion.md)
- [quadruped-open-source-projects-technology-map](../overview/quadruped-open-source-projects-technology-map.md)

## 参考来源

- [locoleggedwheel.md](../../sources/repos/locoleggedwheel.md)
- [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)
- 源码核对（2026-11-02）：LocoLeggedWheel@015b823，读过 README.md、locowheeledlegged/scripts/train.py、locowheeledlegged/scripts/play.py、locowheeledlegged/config/go2w/__init__.py、locowheeledlegged/config/go2w/locomotion_env_cfg.py、locowheeledlegged/config/go2w/agents/rsl_rl_ppo_cfg.py、loco_rl/runners/on_policy_runner.py、loco_rl/algorithms/ppo.py、deploy_real/deploy_real.py、deploy_real/configs/go2w.yaml；时序图依据以上文件，未运行代码。
