---
type: entity
tags: [repo, quadruped, amp, unitree-a1, legged-gym]
status: complete
updated: 2026-06-08
summary: "inspirai/MetalHead 在 Unitree A1 上用 AMP 实现自然行走、跳跃与跌倒恢复，基于 AMP_for_hardware + Isaac Gym + legged_gym。"
related:
  - ./amp-for-hardware.md
  - ./legged-gym.md
  - ../methods/amp-reward.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/repos/metalhead.md
---

# MetalHead

**MetalHead**（<https://github.com/inspirai/MetalHead>）在 **Unitree A1** 四足上实现 **walk / run / jump / recovery** 等自然运动，核心算法为 **AMP**，代码基于 [AMP_for_hardware](./amp-for-hardware.md) 与 [legged_gym](./legged-gym.md)，仿真后端为 **Isaac Gym**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AMP | Adversarial Motion Prior | 从参考运动库学习风格约束 |
| A1 | Unitree A1 Quadruped | 宇树四足科研平台 |
| RL | Reinforcement Learning | PPO + 风格判别器 |
| Sim2Real | Simulation to Real | 论文/项目强调真机可迁移调参 |

## 为什么重要

- **四足 AMP 工程样板**：展示如何把动物/参考运动风格迁移到 A1 并处理高动态技能（跳跃、恢复）的超参与奖励权衡。
- 与纯几何重定向不同，重点在 **仿真内可跟踪 + 判别器风格匹配**。

## 源码运行时序图

入口：按 AMP_for_hardware 配置好 Isaac Gym 后，运行 `python legged_gym/scripts/train.py --task=a1_amp_jump_cmd --headless True`（推理演示为 `legged_gym/scripts/play.py`）

```mermaid
sequenceDiagram
    participant U as train.py train()
    participant TR as task_registry make_env 与 make_alg_runner
    participant LR as LeggedRobot 环境 legged_robot.py
    participant R as AMPOnPolicyRunner
    participant P as AMP_PPO amp_ppo.py
    participant D as AMPDiscriminator
    participant L as AMPLoader 专家运动数据
    U->>TR: make_env(name a1_amp_jump_cmd, args)
    TR->>LR: 按 A1AMPJumpCMDCfg 构造环境
    U->>TR: make_alg_runner(env, name, args)
    TR->>R: eval runner_class_name 即 AMPOnPolicyRunner
    R->>L: 载入 amp_motion_files
    R->>D: 构造判别器
    R->>P: 构造 AMP_PPO(actor_critic, discriminator, amp_data, normalizer)
    U->>R: learn(max_iterations, init_at_random_ep_len)
    loop 每次迭代的每个 rollout 步
        R->>P: act(obs, critic_obs, amp_obs)
        R->>LR: env.step(actions)
        LR->>LR: simulate 后 fetch_results，进入 post_physics_step
        LR->>LR: check_termination、compute_reward、get_amp_observations、reset_idx、compute_observations
        LR-->>R: obs、rewards、dones、infos、reset_env_ids、terminal_amp_states
        R->>LR: get_amp_observations 取 next_amp_obs
        R->>D: predict_amp_reward(amp_obs, next_amp_obs, rewards, normalizer)
        D-->>R: 返回用于 PPO 的组合奖励
        R->>P: process_env_step(rewards, dones, infos, next_amp_obs)
    end
    R->>P: compute_returns(critic_obs)
    R->>P: update()
    P->>L: feed_forward_generator 取专家样本
    P->>D: 判别器前向，MSE 对抗损失与梯度惩罚
    P->>P: PPO surrogate 与 value 损失，Adam 更新策略与判别器
    R->>R: log，save model_迭代数.pt
```

图依据：inspirai/MetalHead@1f55679，实际读过 README.md、legged_gym/scripts/train.py、legged_gym/envs/__init__.py（任务注册）、legged_gym/envs/a1/a1_amp_jump_cmd_config.py（runner_class_name）、legged_gym/envs/base/legged_robot.py（step 与 post_physics_step 调用顺序）、legged_gym/utils/task_registry.py（make_env、make_alg_runner）、rsl_rl/rsl_rl/runners/amp_on_policy_runner.py、rsl_rl/rsl_rl/algorithms/amp_ppo.py（act、update 结构）、rsl_rl/rsl_rl/algorithms/amp_discriminator.py（predict_amp_reward 签名）；未运行代码。

## 关联页面

- [AMP_for_hardware](./amp-for-hardware.md)
- [legged_gym](./legged-gym.md)
- [AMP 奖励设计](../methods/amp-reward.md)
- [Locomotion](../tasks/locomotion.md)

## 参考来源

- [MetalHead 仓库归档](../../sources/repos/metalhead.md)
- 源码核对（2026-11-02）：inspirai/MetalHead@1f55679，读过 README.md、legged_gym/scripts/train.py、legged_gym/envs/__init__.py、legged_gym/envs/a1/a1_amp_jump_cmd_config.py、legged_gym/envs/base/legged_robot.py、legged_gym/utils/task_registry.py、rsl_rl/rsl_rl/runners/amp_on_policy_runner.py、rsl_rl/rsl_rl/algorithms/amp_ppo.py、rsl_rl/rsl_rl/algorithms/amp_discriminator.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- GitHub：<https://github.com/inspirai/MetalHead>
