---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: draft
updated: 2026-09-06
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/livelybot_pi_rl_baseline.md
summary: "高擎机电 开源项目 livelybot_pi_rl_baseline（运动RL/技能训练）：并行 RL 运动训练框架（含仿真任务与策略导出链路）…"
institutions:
  - hightorque
---

# livelybot_pi_rl_baseline

## 一句话定义

**livelybot_pi_rl_baseline** 是 [高擎机电](https://github.com/HighTorque-Robotics) 公开的 **运动RL/技能训练** 开源项目：并行 RL 运动训练框架（含仿真任务与策略导出链路）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第一层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 高擎机电 |
| 类别 | 运动RL/技能训练 |
| 官方组织 | https://github.com/HighTorque-Robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `livelybot_pi_rl_baseline` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python humanoid/scripts/train.py --task=pai_ppo --run_name v1 --headless --num_envs 4096`（`auto_train4096.bash` 同此命令；README 写作 `scripts/train.py`，仓库内实际路径为 `humanoid/scripts/`），随后 `python humanoid/scripts/play.py --task=pai_ppo --run_name v1` 与 `python humanoid/scripts/sim2sim.py --load_model <policy.pt>`。

```mermaid
sequenceDiagram
    participant U as 用户 / auto_*.bash
    participant TR as humanoid/scripts/train.py
    participant REG as humanoid/utils/task_registry.py<br/>task_registry
    participant ENV as humanoid/envs/pai/pai_env.py<br/>PaiFreeEnv(LeggedRobot)
    participant GYM as Isaac Gym (gym.simulate)
    participant RUN as humanoid/algo/ppo/on_policy_runner.py<br/>OnPolicyRunner
    participant PPO as algo/ppo/ppo.py + actor_critic.py
    participant PL as humanoid/scripts/play.py
    participant S2S as humanoid/scripts/sim2sim.py<br/>(MuJoCo)

    U->>TR: --task=pai_ppo --run_name --num_envs
    TR->>REG: from humanoid.envs import *（register "pai_ppo": PaiFreeEnv, PaiCfg, PaiCfgPPO）
    TR->>REG: make_env(name="pai_ppo", args)
    REG->>ENV: PaiFreeEnv(cfg, sim_params, physics_engine, sim_device, headless)
    TR->>REG: make_alg_runner(env, name, args)
    REG->>RUN: OnPolicyRunner(env, train_cfg, log_dir)（ActorCritic + PPO，init_storage）
    TR->>RUN: learn(max_iterations, init_at_random_ep_len=True)
    loop 每个 iteration × num_steps_per_env
        RUN->>PPO: alg.act(obs, critic_obs)
        RUN->>ENV: env.step(actions)
        ENV->>ENV: PaiFreeEnv.step：参考动作叠加 + 随机延迟/动态随机化
        ENV->>ENV: LeggedRobot.step：clip；decimation 次 _compute_torques → set_dof_actuation_force_tensor
        ENV->>GYM: gym.simulate；refresh_dof_state_tensor
        ENV->>ENV: post_physics_step：compute_reward / compute_observations / check_termination
        ENV-->>RUN: obs, privileged_obs, rewards, dones, infos
        RUN->>PPO: alg.process_env_step
    end
    RUN->>PPO: compute_returns；alg.update()
    RUN-->>U: logs/Pai_ppo/<time>_<run_name>/model_<it>.pt

    U->>PL: --task=pai_ppo --run_name
    PL->>REG: make_env(env_cfg 缩小到 ≤64 环境、plane 地形)
    PL->>REG: make_alg_runner（resume=True，载入最新 model_*.pt）
    PL->>RUN: get_inference_policy
    PL->>PL: export_policy_as_jit / export_policy_to_onnx → exported/policies
    loop stop_state_log 步
        PL->>ENV: env.step(policy(obs))
    end

    U->>S2S: --load_model policy_*.pt
    S2S->>S2S: torch.jit.load；mujoco.MjModel.from_xml_path
    loop MuJoCo 步进
        S2S->>S2S: 每 decimation 步：构造 obs（sin/cos 步态相位、cmd、q、dq、IMU）× frame_stack
        S2S->>S2S: action = policy(policy_input)；target_q = action*action_scale
        S2S->>S2S: pd_control → tau（clip）→ mujoco.mj_step
    end
```

图依据：HighTorque-Robotics/livelybot_pi_rl_baseline@e8b787a，读过 README.md、auto_train4096.bash、auto_sim2sim.bash、humanoid/envs/__init__.py、humanoid/envs/pai/pai_env.py（step 与类方法清单）、humanoid/envs/base/legged_robot.py（step）、humanoid/utils/task_registry.py、humanoid/algo/ppo/on_policy_runner.py（learn、save、get_inference_policy）、humanoid/scripts/{train,play,sim2sim}.py。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [livelybot_pi_rl_baseline 源码归档](../../sources/repos/livelybot_pi_rl_baseline.md)（<https://github.com/HighTorque-Robotics/livelybot_pi_rl_baseline>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：HighTorque-Robotics/livelybot_pi_rl_baseline@e8b787a，读过 README.md、auto_train4096.bash、humanoid/envs/__init__.py、humanoid/envs/pai/pai_env.py、humanoid/envs/base/legged_robot.py（step）、humanoid/utils/task_registry.py、humanoid/algo/ppo/on_policy_runner.py、humanoid/scripts/{train,play,sim2sim}.py；ppo.py / actor_critic.py 仅按 runner 的调用关系推断，未读；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [高擎机电 官方组织](https://github.com/HighTorque-Robotics)
