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
  - ../../sources/repos/engineai_amp.md
summary: "众擎机器人 开源项目 engineai_amp（运动RL/技能训练）：并行 RL 运动训练框架（含仿真任务与策略导出链路）…"
institutions:
  - engineai
---

# engineai_amp

## 一句话定义

**engineai_amp** 是 [众擎机器人](https://github.com/engineai-robotics) 公开的 **运动RL/技能训练** 开源项目：并行 RL 运动训练框架（含仿真任务与策略导出链路）

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
| 机构 | 众擎机器人 |
| 类别 | 运动RL/技能训练 |
| 官方组织 | https://github.com/engineai-robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `engineai_amp` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 源码运行时序图

入口：`python scripts/train.py --task=Flat-AMP-PM01-v0 --num_envs 4096 --headless --run_name <name>`（训练）；评估与导出为 `python scripts/play.py --task=Flat-AMP-PM01-v0 --num_envs 128 --load_run <name>`（play 会调用 `export_policy_to_onnx`）。

```mermaid
sequenceDiagram
    autonumber
    participant CLI as scripts/train.py main
    participant APP as AppLauncher Isaac Sim
    participant REG as engineai_lab.tasks gym.register
    participant ENV as ManagerBasedRLEnv Flat-AMP-PM01-v0
    participant RSL as RslRlVecEnvWrapper OnPolicyRunner
    participant AMP as AMPPPO algorithms/amp_ppo.py
    participant DISC as Discriminator utils/AMP_discriminator.py
    participant DL as AMPDataLoader utils/AMP_data_loader.py

    CLI->>APP: AppLauncher(args_cli) 启动仿真
    CLI->>REG: import engineai_lab.tasks 注册任务
    CLI->>CLI: hydra_task_config 读取 flat_amp_env_cfg 与 amp_ppo_cfg
    CLI->>ENV: gym.make(task, cfg=env_cfg)
    CLI->>RSL: RslRlVecEnvWrapper(env) 与 OnPolicyRunner(env, agent_cfg)
    RSL->>AMP: construct_algorithm
    AMP->>DISC: 构造 Discriminator(frame_dim, frame_length, hidden dims)
    AMP->>DL: 构造 AMPDataLoader(dataset_path)
    DL->>DL: 读取 dataset/config/dataset.yaml 中的 motions 列表并加载 npz
    CLI->>RSL: runner.learn(max_iterations)

    loop 每轮迭代 rollout 与更新
        RSL->>ENV: step(actions)
        ENV-->>RSL: obs 含 amp 观测 与 rewards
        RSL->>AMP: process_env_step(obs, rewards, dones)
        AMP->>DISC: get_amp_reward(obs amp)
        DISC-->>AMP: style reward
        AMP->>AMP: total = task reward + style_weight 乘 amp reward
        RSL->>AMP: update()
        opt ppo_update_counter 为 4 的倍数
            AMP->>DL: mini_batch_generator 取专家样本
            AMP->>DISC: 专家与策略得分的 MSE 判别损失加梯度惩罚
            AMP->>AMP: disc_optimizer 更新判别器
        end
        AMP->>AMP: PPO.update 更新策略与价值网络
    end
    CLI->>CLI: 保存 checkpoint 到 logs/rsl_rl/<experiment_name>
```

图依据：engineai-robotics/engineai_amp@83ba64b，读过 `README.md`（全文）、`scripts/train.py`（全文）、`scripts/play.py`（grep 与 main 段）、`source/engineai_lab/tasks/velocity/config/pm01/__init__.py`（gym.register 段）、`source/engineai_lab/tasks/velocity/config/pm01/agents/amp_ppo_cfg.py`（grep）、`source/engineai_lab/tasks/velocity/config/pm01/flat_amp_env_cfg.py`（grep）、`source/engineai_lab/algorithms/amp_ppo.py`（`construct_algorithm`、`process_env_step`、`update` 段与 grep）、`source/engineai_lab/utils/AMP_data_loader.py` 与 `AMP_discriminator.py`（grep）、`dataset/config/dataset.yaml`。`rsl_rl` 与 `isaaclab` 为外部依赖，未读。未运行代码。时序图依据以上文件。

## 参考来源

- [engineai_amp 源码归档](../../sources/repos/engineai_amp.md)（<https://github.com/engineai-robotics/engineai_amp>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：engineai-robotics/engineai_amp@83ba64b，读过 `README.md`、`scripts/train.py`、`scripts/play.py`（部分）、`source/engineai_lab/tasks/velocity/config/pm01/__init__.py`、`agents/amp_ppo_cfg.py`、`flat_amp_env_cfg.py`（grep）、`source/engineai_lab/algorithms/amp_ppo.py`（部分）、`utils/AMP_data_loader.py` 与 `AMP_discriminator.py`（grep）、`dataset/config/dataset.yaml`；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [众擎机器人 官方组织](https://github.com/engineai-robotics)
