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
  - ../../sources/repos/lejulab-train.md
summary: "乐聚机器人 开源项目 LejuLab-Train（运动RL/技能训练）：并行 RL 运动训练框架（含仿真任务与策略导出链路）…"
institutions:
  - leju
---

# LejuLab-Train

## 一句话定义

**LejuLab-Train** 是 [乐聚机器人](https://github.com/LejuRobotics) 公开的 **运动RL/技能训练** 开源项目：并行 RL 运动训练框架（含仿真任务与策略导出链路）

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
| 机构 | 乐聚机器人 |
| 类别 | 运动RL/技能训练 |
| 官方组织 | https://github.com/LejuRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `LejuLab-Train` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python scripts/reinforcement_learning/rsl_rl/train.py --task Tracking-Dance-Flat-RobanS14 --motion_file path/to/motion.npz --num_envs 8192 --headless`（README「Train RL Agent」；需 Isaac Sim 4.5 / Isaac Lab 2.1）。

```mermaid
sequenceDiagram
    participant U as 用户 CLI
    participant T as scripts/reinforcement_learning/rsl_rl/train.py
    participant AL as isaaclab.app.AppLauncher
    participant REG as leju_robot.tasks（gym.register）
    participant H as hydra_task_config
    participant E as ManagerBasedRLEnv（gym.make）
    participant W as RslRlVecEnvWrapper
    participant R as RobotOnPolicyRunner（rsl_rl_extensions/utils/on_policy_runner.py）

    U->>T: python train.py --task ... --motion_file ...
    T->>AL: AppLauncher(args_cli)，启动 Omniverse 应用
    T->>REG: import leju_robot.tasks，import_packages 注册 Tracking-* 任务
    T->>H: hydra_task_config(task, "rsl_rl_cfg_entry_point")
    H-->>T: env_cfg（tracking_env_cfg.RobanS14FlatEnvCfg）与 agent_cfg（RobanS14DancePPORunnerCfg）
    T->>T: 覆盖 num_envs、max_iterations、seed、commands.motion.motion_file
    T->>E: gym.make(task, cfg=env_cfg)
    E->>E: MotionCommand 载入 npz 参考动作，Observation / Action(JointPositionResiduals) / Reward 管理器
    T->>W: RslRlVecEnvWrapper(env)
    T->>R: eval(agent_cfg.runner_class_name)(env, agent_cfg.to_dict(), log_dir, device)
    T->>R: learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
    loop PPO 迭代（rsl_rl OnPolicyRunner，位于 Isaac Lab 依赖）
        R->>W: step(actions) 并行 rollout
        W->>E: 物理仿真、命令更新、奖励与终止
        E-->>R: obs, rewards, dones
    end
    R->>R: save()：super().save 后 export_policy_as_onnx，attach_onnx_metadata
    T->>E: env.close()，simulation_app.close()
```

图依据：LejuLab-Train@845f373，读过 `README.md`、`scripts/reinforcement_learning/rsl_rl/train.py`、`source/leju_robot/leju_robot/tasks/__init__.py`、`tasks/tracking/config/robanS14/dance/__init__.py`（gym.register）、`tasks/tracking/config/robanS14/dance/tracking_env_cfg.py`（Actions、Commands、Observations 片段）、`tasks/tracking/agents/rsl_rl_ppo_cfg.py`、`rsl_rl_extensions/utils/on_policy_runner.py`、`tasks/tracking/mdp/commands.py`（类与方法名）。`OnPolicyRunner.learn` 内部在 Isaac Lab / rsl_rl 依赖中，未读；奖励、终止、课程细节与 AMP / skrl 路径未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [LejuLab-Train 源码归档](../../sources/repos/lejulab-train.md)（<https://github.com/LejuRobotics/LejuLab-Train>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：LejuRobotics/LejuLab-Train@845f373，读过 README.md、scripts/reinforcement_learning/rsl_rl/train.py、source/leju_robot/leju_robot/tasks/{__init__.py,tracking/config/robanS14/dance/*,tracking/agents/rsl_rl_ppo_cfg.py}、rsl_rl_extensions/utils/on_policy_runner.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [乐聚机器人 官方组织](https://github.com/LejuRobotics)
