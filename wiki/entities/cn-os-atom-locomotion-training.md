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
  - ../../sources/repos/atom-locomotion-training.md
summary: "越疆科技 开源项目 atom-locomotion-training（运动RL/技能训练）：提供ATOM人形机器人Locomotion强化学习训练入口，连接机器人模型、任务环境和策略配置。…"
institutions:
  - dobot
---

# atom-locomotion-training

## 一句话定义

**atom-locomotion-training** 是 [越疆科技](https://github.com/embodied-dobot) 公开的 **运动RL/技能训练** 开源项目：提供ATOM人形机器人Locomotion强化学习训练入口，连接机器人模型、任务环境和策略配置。

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
| 机构 | 越疆科技 |
| 类别 | 运动RL/技能训练 |
| 官方组织 | https://github.com/embodied-dobot、https://github.com/Dobot-Arm |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `atom-locomotion-training` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python humanoid/scripts/train.py --task ATOM_p2_ppo --run_name test --headless --num_envs 4096`（需先按 README 安装 Isaac Gym Preview 4 与本仓库 `pip install -e .`）。

```mermaid
sequenceDiagram
    participant U as 用户命令行
    participant T as humanoid/scripts/train.py<br/>train
    participant G as humanoid/utils/task_registry.py<br/>task_registry
    participant E as humanoid/envs/custom/humanoid_env.py<br/>ATOM 继承 base/legged_robot.py
    participant N as humanoid/algo/ppo/on_policy_runner.py<br/>OnPolicyRunner
    participant P as humanoid/algo/ppo/ppo.py<br/>PPO 与 actor_critic
    participant I as NVIDIA Isaac Gym 仿真
    U->>T: python train.py --task ATOM_p2_ppo --headless --num_envs
    T->>T: get_args 解析命令行参数
    T->>G: make_env(name=args.task, args)
    G->>E: 取注册的 ATOM 类与 ATOM2Cfg 并构造环境
    E->>I: create_sim 与 _create_envs 创建并行环境
    T->>G: make_alg_runner(env, name, args)
    G->>N: eval runner_class_name 构造 OnPolicyRunner
    N->>P: 构造 actor_critic 与 PPO 并 init_storage
    T->>N: learn(max_iterations, init_at_random_ep_len=True)
    loop 每次迭代 it
        loop num_steps_per_env 步
            N->>P: act(obs, critic_obs) 采样动作
            N->>E: env.step(actions)
            E->>E: 裁剪动作 加入 action_delay 与 action_noise 后调用 base step
            loop decimation 次
                E->>I: _compute_torques 后 set_dof_actuation_force_tensor 与 simulate
                I-->>E: refresh_dof_state_tensor
            end
            E->>E: post_physics_step 中 compute_reward 与 compute_observations
            E-->>N: obs rewards dones infos
            N->>P: process_env_step 写入 rollout storage
        end
        N->>P: compute_returns
        N->>P: update 多轮小批量 clip 代理损失 与值函数损失 Adam 更新
        opt 到达保存间隔
            N->>N: save 写入 model_it.pt
        end
    end
```

后续流程（未在图中展开）：`humanoid/scripts/play.py` 调用 `get_inference_policy` 并导出 JIT 与 ONNX 策略；`humanoid/scripts/sim2sim.py` 以 `torch.jit.load` 加载策略，在 MuJoCo 中用 `mj_step` 与 PD 控制做跨引擎验证。

图依据：embodied-dobot/atom-locomotion-training@f1f5aae，读过 README.md、humanoid/scripts/train.py、humanoid/envs/__init__.py、humanoid/utils/task_registry.py（register、make_env、make_alg_runner 函数段）、humanoid/algo/ppo/on_policy_runner.py（learn 循环）、humanoid/algo/ppo/ppo.py（update 段）、humanoid/envs/base/legged_robot.py（step 段）、humanoid/envs/custom/humanoid_env.py（step 段）、humanoid/scripts/play.py 与 sim2sim.py（函数列表与导出段）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [atom-locomotion-training 源码归档](../../sources/repos/atom-locomotion-training.md)（<https://github.com/embodied-dobot/atom-locomotion-training>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：embodied-dobot/atom-locomotion-training@f1f5aae，读过 README.md、humanoid/scripts/train.py、humanoid/utils/task_registry.py、humanoid/algo/ppo/on_policy_runner.py、humanoid/algo/ppo/ppo.py、humanoid/envs/base/legged_robot.py、humanoid/envs/custom/humanoid_env.py、humanoid/envs/__init__.py、humanoid/scripts/play.py、humanoid/scripts/sim2sim.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [越疆科技 官方组织](https://github.com/embodied-dobot)
