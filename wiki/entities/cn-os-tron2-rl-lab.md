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
  - ../../sources/repos/tron2_rl_lab.md
summary: "逐际动力 开源项目 tron2_rl_lab（运动RL/技能训练）：并行 RL 运动训练框架（含仿真任务与策略导出链路）…"
institutions:
  - limx
---

# tron2_rl_lab

## 一句话定义

**tron2_rl_lab** 是 [逐际动力](https://github.com/limxdynamics) 公开的 **运动RL/技能训练** 开源项目：并行 RL 运动训练框架（含仿真任务与策略导出链路）

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
| 机构 | 逐际动力 |
| 类别 | 运动RL/技能训练 |
| 官方组织 | https://github.com/limxdynamics、https://github.com/FluxVLA |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `tron2_rl_lab` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python scripts/rsl_rl/train.py --task Isaac-Limx-SF-TRON2A-Blind-Flat-v0 --num_envs 4096 --headless`（需 Isaac Sim 5.1 与 Isaac Lab 2.3.1；推理导出见 scripts/rsl_rl/play.py）

```mermaid
sequenceDiagram
    autonumber
    participant U as 操作者（shell）
    participant TR as scripts/rsl_rl/train.py main
    participant AL as AppLauncher（Isaac Sim，仓库外依赖）
    participant GY as gym.register（tasks/locomotion/robots/__init__.py）
    participant EV as ManagerBasedRLEnv 与 limx_base_env_cfg（Isaac Lab，仓库外依赖）
    participant RW as RslRlVecEnvWrapper
    participant OR as OnPolicyRunner（rsl_rl/runner/on_policy_runner.py）
    participant AG as PPO 算法对象（rsl_rl/algorithm）
    participant LG as logs/rsl_rl/实验名/时间戳目录

    U->>TR: python scripts/rsl_rl/train.py --task ... --num_envs 4096 --headless
    TR->>AL: AppLauncher(args_cli)，启动仿真应用
    TR->>GY: gym.make(args_cli.task, cfg=env_cfg)，按注册项取 env_cfg_entry_point
    GY->>EV: 构造环境（SF_TRON2A_BlindFlatEnvCfg）
    TR->>RW: RslRlVecEnvWrapper(env)
    TR->>OR: OnPolicyRunner(env, agent_cfg.to_dict(), log_dir)
    opt --resume True
        TR->>OR: load(resume_path)
    end
    TR->>OR: learn(max_iterations, init_at_random_ep_len=True)
    OR->>EV: get_observations()
    loop 每个学习迭代
        loop num_steps_per_env 次 rollout（inference_mode）
            OR->>AG: act(obs, obs_history, commands, critic_obs)
            OR->>RW: env.step(actions)
            RW->>EV: 仿真推进，返回 policy、critic、obsHistory、commands 观测与 rewards、dones
            OR->>AG: process_env_step(rewards, dones, infos, obs)
        end
        OR->>AG: compute_returns，然后 update() 执行 PPO 更新
        opt it 满足 save_interval
            OR->>LG: save model_it.pt
        end
    end
    OR->>LG: 保存最终 model_N.pt
    TR->>EV: env.close()
```

图依据：limxdynamics/tron2_rl_lab@0527456，scripts/rsl_rl/train.py（全文）、scripts/rsl_rl/play.py（关键段）、rsl_rl/rsl_rl/runner/on_policy_runner.py（learn 与 save 段）、exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py（gym.register 段）、exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/agents/dasf_rsl_rl_ppo_cfg.py（关键段）、README.md（训练运行段）

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [tron2_rl_lab 源码归档](../../sources/repos/tron2_rl_lab.md)（<https://github.com/limxdynamics/tron2_rl_lab>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：limxdynamics/tron2_rl_lab@0527456，读过 scripts/rsl_rl/train.py、scripts/rsl_rl/play.py（关键段）、rsl_rl/rsl_rl/runner/on_policy_runner.py（learn 与 save 段）、exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py（gym.register 段）、exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/agents/dasf_rsl_rl_ppo_cfg.py（关键段）、README.md（训练与 play 命令段，经检索）；未读 PPO 算法实现 rsl_rl/rsl_rl/algorithm；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [逐际动力 官方组织](https://github.com/limxdynamics)
