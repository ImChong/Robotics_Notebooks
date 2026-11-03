---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: draft
updated: 2026-10-12
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/demograsp.md
summary: "智在无界 开源项目 DemoGrasp（具身Agent/规划）：移动操作（Loco-Manip）策略接入…"
institutions:
  - beingbeyond
---

# DemoGrasp

## 一句话定义

**DemoGrasp** 是 [智在无界](https://github.com/BeingBeyond) 公开的 **具身Agent/规划** 开源项目：移动操作（Loco-Manip）策略接入

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第二层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 智在无界 |
| 类别 | 具身Agent/规划 |
| 官方组织 | https://github.com/BeingBeyond |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `DemoGrasp` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

训练入口：`python -u run_rl_grasp.py task=grasp train=PPOOneStep hand=fr3_inspire_tac num_envs=7000 ...`（完整参数见 `train.sh`）；回放 / 评测入口：同一脚本加 `test=True checkpoint='ckpt/inspire.pt'`（见 `play_policy.sh`）；演示回放：`+debug=test_demo_replay`

```mermaid
sequenceDiagram
    participant RUN as run_rl_grasp.py main (hydra, tasks/config.yaml)
    participant IGE as isaacgymenvs.make → tasks/grasp.py Grasp(VecTask)
    participant RN as build_runner → algo/ppo_onestep PPO
    participant AC as ActorCritic (algo/ppo_onestep/module.py, PointNet 骨干)
    participant ST as RolloutStorage (storage.py)
    participant RW as tasks/reward.py reward_binary

    RUN->>IGE: create_isaacgym_env()，env.reset_idx(全部 env)
    RUN->>RN: build_runner：断言 randomize_tracking_reference，act_dim = 6 (+ 活动手关节数)
    RN-->>RUN: ppo_onestep.PPO(vec_env, ActorCritic, ...)
    RUN->>RN: runner.run()
    loop 每次学习迭代
        RN->>IGE: reset_idx(全部 env)，取 obs 与 get_state()
        RN->>AC: actor_critic(obs, states) 采样 actions（单步：整条抓取轨迹只出一个动作）
        RN->>IGE: generate_reaching_plan_idx(actions)：用动作的 6 维随机化演示腕部参考轨迹（旋转 + 平移），其余维调整抓取手型
        loop episode_length 步
            RN->>IGE: compute_reference_actions()（接近段线性插值 + 参考轨迹跟踪）
            RN->>IGE: step(env_action)：pre_physics_step、decimation 次 gym.simulate、post_physics_step
            IGE->>RW: compute_reward → reward_function = REWARD_DICT["binary"]（抬升 > 0.1 且手靠近物体为成功）
        end
        RN->>RN: 终步 successes 作为奖励（碰桌 has_hit_table 则置 0）
        RN->>ST: add_transitions(obs, states, actions, rews, ...)
        RN->>ST: compute_returns(gamma, lam)
        RN->>RN: update()：mini-batch 多 epoch PPO 更新
        RN->>RN: 每 save_interval 保存 runs_ppo/.../model_N.pt
    end
```

图依据：BeingBeyond/DemoGrasp@1255dc5，读过 `README.md`、`train.sh`（头部）、`run_rl_grasp.py`、`algo/ppo_onestep/ppo.py`（run / update 开头）、`algo/ppo_onestep/module.py`（类列表）、`tasks/grasp.py`（generate_reaching_plan_idx、compute_reference_actions、step、pre_physics_step）、`tasks/reward.py`。IsaacGymEnvs 本身、`compute_observations` / `reset_idx` 细节、`data/dataset_utils.py` 与点云编码细节未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [DemoGrasp 源码归档](../../sources/repos/demograsp.md)（<https://github.com/BeingBeyond/DemoGrasp>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：BeingBeyond/DemoGrasp@1255dc5，读过 README.md、train.sh（头部）、run_rl_grasp.py、algo/ppo_onestep/ppo.py、algo/ppo_onestep/module.py（类列表）、algo/ppo_onestep/storage.py（类名）、tasks/grasp.py（generate_reaching_plan_idx、compute_reference_actions、step、pre_physics_step、compute_reward）、tasks/reward.py、tasks/train（log_dir）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [智在无界 官方组织](https://github.com/BeingBeyond)
