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
  - ../../sources/repos/wheeldog_rl.md
summary: "仙工智能 开源项目 wheelDog_RL（运动RL/技能训练）：仙工智能公开的第一阶段轮足机器人强化学习项目，为其轮足平台提供训练环境与策略实验入口。…"
institutions:
  - seer
---

# wheelDog_RL

## 一句话定义

**wheelDog_RL** 是 [仙工智能](https://github.com/seer-robotics) 公开的 **运动RL/技能训练** 开源项目：仙工智能公开的第一阶段轮足机器人强化学习项目，为其轮足平台提供训练环境与策略实验入口。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第五层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 仙工智能 |
| 类别 | 运动RL/技能训练 |
| 官方组织 | https://github.com/seer-robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `wheelDog_RL` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python scripts/rsl_rl/train.py --task=Wheeldog-Rl-v0 --num_envs 4096 --headless`（亦可运行 `bash scripts/rsl_rl/bashBlindWheelDog.sh`；需 Isaac Lab release/2.3.2 环境，`pip install -e source/wheelDog_RL`）

```mermaid
sequenceDiagram
    participant U as 用户 shell
    participant S as scripts/rsl_rl/train.py main
    participant G as tasks/wheeldog_rl __init__ gym.register
    participant C as robotCfg 与 wheeldog_rl_env_cfg
    participant E as envEntry WheelDog_BlindLocomotionEnv
    participant I as Isaac Lab ManagerBasedRLEnv
    participant R as rsl_rl OnPolicyRunner
    participant W as watchDogs 与 mdp 管理器
    U->>S: python scripts/rsl_rl/train.py --task Wheeldog-Rl-v0
    S->>S: AppLauncher 启动 Isaac Sim，检查 rsl-rl-lib 版本不低于 3.0.1
    S->>G: hydra_task_config 按 task 名取入口
    G->>C: env_cfg_entry_point 得到 WheelDog_BlindLocomotionEnvCfg
    G->>S: rsl_rl_cfg_entry_point 得到 PPORunnerCfg（1600 次迭代，每环境 32 步）
    S->>E: gym.make 实例化环境
    E->>I: super().__init__ 建场景与各 MDP 管理器
    E->>W: 创建 TiltDetectionManager、VelocityErrorRecorder、ZeroDriftManager
    S->>S: RslRlVecEnvWrapper 包装环境
    S->>R: OnPolicyRunner(env, agent_cfg, log_dir)
    S->>S: dump_yaml 写 params/env.yaml 与 agent.yaml
    S->>R: learn 开始训练
    loop 每个学习迭代的 rollout
        R->>E: step(actions)，动作为髋、膝关节位置与轮毂速度
        E->>I: super().step 执行 decimation=4 个物理步（控制周期约 0.02 s）
        E->>W: 管理器 step（倾倒检测、速度误差、零漂）
        E->>W: check_observations 与 check_rewards 检查 NaN 或 Inf 并裁剪奖励
        E-->>R: 观测（actor 与 critic）、奖励、终止标志
        E->>E: 终止环境时调用 _reset_idx 重置
    end
    R->>R: PPO 更新策略（rsl_rl 库内部，不在本仓库）
```

图依据：wheelDog_RL@3c65bed，README.md、scripts/rsl_rl/train.py（主流程，读过）、scripts/rsl_rl/bashBlindWheelDog.sh（读过）、source/wheelDog_RL/wheelDog_RL/tasks/manager_based/wheeldog_rl/__init__.py（gym.register，读过）、envEntry.py（读过）、robotCfg.py（读过关键段）、wheeldog_rl_env_cfg.py（仅 grep 类结构与 decimation）、agents/rsl_rl_ppo_cfg.py（仅 grep）、actionSpaceCfg.py 与 observationSpaceCfg.py（仅 grep）。play.py 与导出流程未画。未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [wheelDog_RL 源码归档](../../sources/repos/wheeldog_rl.md)（<https://github.com/seer-robotics/wheelDog_RL>）

- 源码核对（2026-11-02）：seer-robotics/wheelDog_RL@3c65bed，读过 README.md、scripts/rsl_rl/train.py、scripts/rsl_rl/bashBlindWheelDog.sh、source/wheelDog_RL/wheelDog_RL/tasks/manager_based/wheeldog_rl/__init__.py、envEntry.py、robotCfg.py；时序图依据以上文件，未运行代码。

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [仙工智能 官方组织](https://github.com/seer-robotics)
