---
type: entity
project_id: rex-gym
tags:
  - repo
  - independent
  - quadruped
  - rl
  - openai-gym
  - ppo
  - 3d-printed
status: complete
updated: 2026-11-01
code: https://github.com/nicrusso7/rex-gym
related:
  - ../methods/reinforcement-learning.md
  - ../entities/quadruped-robot.md
  - ../tasks/locomotion.md
  - ../concepts/reward-design.md
  - ../overview/quadruped-open-source-projects-technology-map.md
sources:
  - ../../sources/repos/rex_gym.md
  - ../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md
summary: "Rex-Gym 是 3D 打印开源四足机器人 Rex 的 OpenAI Gym 训练环境集合：含 Rex URDF、PPO 学习智能体与训练 / 回放脚本，目标是仿真学到的策略无需再手调即可迁移到真机。"
---

# Rex-Gym

**Rex-Gym**（[GitHub](https://github.com/nicrusso7/rex-gym)）来自 [古月居转载的四足机器人开源项目综述](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)，属「强化学习」组。

## 一句话定义

**Rex-Gym 是 3D 打印开源四足机器人 Rex 的 OpenAI Gym 训练环境集合：含 Rex URDF、PPO 学习智能体与训练 / 回放脚本，目标是仿真学到的策略无需再手调即可迁移到真机。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PPO | Proximal Policy Optimization | 近端策略优化 |
| URDF | Unified Robot Description Format | 机器人描述格式 |

## 为什么重要

- 综述把它放在「强化学习」这条技术路线下，作为可读源码的参照项目；横向对比见 [四足机器人开源项目技术地图：从极简运动学到研究级硬件（36 个 GitHub 项目）](../overview/quadruped-open-source-projects-technology-map.md)。
- 文中星标：1089（文章表述，入库日未复核）。

## 核心信息

| 项 | 内容 |
|----|------|
| **仓库** | [nicrusso7/rex-gym](https://github.com/nicrusso7/rex-gym) |
| **最近提交** | 2021-05-15 |
| **可达性** | `git ls-remote` 可达（2026-11-01） |
| **文章归类** | 强化学习 |

## 文章说法核对

- **文章：** 文章称其提供多个训练任务与精心设计的奖励函数（前进速度、能量、冲击力、姿态偏差、关节速度等惩罚），并使用课程学习。
- **README / 核查：** README 称仓库包含用于训练 Rex 的 Gym 环境、Rex URDF 模型、PPO 学习智能体与批量训练 / 策略回放 CLI；相关仓库 rexctl（控制 CLI）、rex-models（URDF 可视化）。
- **差异与保留：** 奖励函数项与课程学习在 README 前段未见，来自文章，未核实。

## 源码运行时序图

入口：`rex-gym`（`setup.py` 的 console script，指向 `rex_gym.cli.entry_point:cli`）。两个主要子命令：`rex-gym policy --env walk`（播放预训练策略）与 `rex-gym train --env walk --log-dir LOG_DIR`（PPO 批量训练）。`--open-loop` / `--inverse-kinematics` 决定信号类型，未指定时取 `flag_mapper.DEFAULT_SIGNAL`。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant CLI as rex_gym/cli/entry_point.py<br/>click 命令 policy / train
    participant PP as rex_gym/playground/policy_player.py<br/>PolicyPlayer
    participant TR as rex_gym/playground/trainer.py<br/>Trainer
    participant PPO as rex_gym/agents/ppo/<br/>simple_ppo_agent / algorithm (PPOAlgorithm)
    participant BE as rex_gym/agents/tools/batch_env.py<br/>BatchEnv
    participant GYM as rex_gym/envs/gym/walk_env.py<br/>RexWalkEnv (gym.make 注册 RexWalk-v0)
    participant ENV as rex_gym/envs/rex_gym_env.py<br/>RexGymEnv 基类
    participant REX as rex_gym/model/rex.py<br/>Rex (pybullet 机器人)

    U->>CLI: rex-gym policy 或 rex-gym train（含 --env、--open-loop/--inverse-kinematics、--mark）
    alt policy 子命令
        CLI->>PP: PolicyPlayer(env, args, signal_type).play()
        PP->>PP: 由 flag_mapper.ENV_ID_TO_POLICY 取 checkpoint 与 config
        PP->>PPO: SimplePPOPolicy(sess, env, network, ..., checkpoint) 恢复策略网络
        PP->>GYM: config.env(render=True, **args) 创建带渲染的环境
        PP->>GYM: reset()
        loop 直到 done
            PP->>PPO: get_action([observation])
            PP->>GYM: step(action)
        end
    else train 子命令
        CLI->>TR: Trainer(env, args, playground, log_dir, agents, signal_type).start_training()
        TR->>TR: configs 读取 walk_ik 等配置，保存配置到日志目录
        TR->>BE: define_batch_env：gym.make(config.env) 并套 LimitDuration、RangeNormalize、ClipAction、ConvertTo32Bit
        TR->>PPO: define_simulation_graph(batch_env, PPOAlgorithm, config)
        TR->>TR: Loop 注册 train 与 eval 阶段并运行
        loop 每个训练步
            PPO->>BE: 批量 action（由策略网络输出）
            BE->>GYM: step(action)
        end
    end

    Note over GYM,REX: 环境单步（两种子命令共用）
    GYM->>ENV: step(action)，walk 环境先 _transform_action_to_motor_command 经 GaitPlanner 与 Kinematics（ik）
    ENV->>REX: Step(action)，按 action_repeat 循环 ApplyAction 与 pybullet stepSimulation
    REX-->>ENV: ReceiveObservation 更新关节与基座状态
    ENV->>ENV: _reward() 与 _termination()（is_fallen 等）
    ENV-->>GYM: observation, reward, done, info
```

图依据：nicrusso7/rex-gym@2666304，实际读过 `README.md`、`setup.py`（`entry_points`）、`rex_gym/cli/entry_point.py`（`policy`、`train`、`_parse_input`）、`rex_gym/playground/policy_player.py`（`play`）、`rex_gym/playground/trainer.py`（`start_training`、`_create_environment`、`_train` 片段）、`rex_gym/playground/__init__.py`（`gym` 注册 `RexWalk-v0` 等环境）、`rex_gym/agents/scripts/configs.py`（前段）、`rex_gym/agents/ppo/simple_ppo_agent.py`（构造与 get_action 签名）、`rex_gym/agents/tools/batch_env.py`（`step` 签名）、`rex_gym/agents/ppo/algorithm.py`（类与方法列表）、`rex_gym/envs/rex_gym_env.py`（`step`）、`rex_gym/envs/gym/walk_env.py`（`reset`、`_transform_action_to_motor_command` 签名）、`rex_gym/model/rex.py`（`Step`）、`rex_gym/util/flag_mapper.py`；训练循环 `Loop` 的内部与 `PPOAlgorithm` 的更新细节未逐行核对；未运行代码。

## 关联页面

- [reinforcement-learning](../methods/reinforcement-learning.md)
- [quadruped-robot](../entities/quadruped-robot.md)
- [locomotion](../tasks/locomotion.md)
- [reward-design](../concepts/reward-design.md)
- [quadruped-open-source-projects-technology-map](../overview/quadruped-open-source-projects-technology-map.md)

## 参考来源

- [rex_gym.md](../../sources/repos/rex_gym.md)
- [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)
- 源码核对（2026-11-02）：nicrusso7/rex-gym@2666304，读过 README.md、setup.py、rex_gym/cli/entry_point.py、rex_gym/playground/policy_player.py、rex_gym/playground/trainer.py、rex_gym/playground/__init__.py、rex_gym/agents/scripts/configs.py（前段）、rex_gym/agents/ppo/simple_ppo_agent.py、rex_gym/agents/ppo/algorithm.py、rex_gym/agents/tools/batch_env.py、rex_gym/envs/rex_gym_env.py、rex_gym/envs/gym/walk_env.py、rex_gym/model/rex.py、rex_gym/util/flag_mapper.py；时序图依据以上文件，未运行代码。
