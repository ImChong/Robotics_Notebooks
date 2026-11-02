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
  - ../../sources/repos/fourier_lab.md
summary: "傅利叶智能 开源项目 fourier_lab（运动RL/技能训练）：并行 RL 运动训练框架（含仿真任务与策略导出链路）…"
institutions:
  - fourier
---

# fourier_lab

## 一句话定义

**fourier_lab** 是 [傅利叶智能](https://github.com/FFTAI) 公开的 **运动RL/技能训练** 开源项目：并行 RL 运动训练框架（含仿真任务与策略导出链路）

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
| 机构 | 傅利叶智能 |
| 类别 | 运动RL/技能训练 |
| 官方组织 | https://github.com/FFTAI |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `fourier_lab` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`../IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task PPV211HumanoidRoughEnvCfg_WBC_LOWER --headless`（README 给出的训练命令；回放用 `scripts/rsl_rl/play.py --task ..._Play`）。依赖 IsaacLab v2.3.0 / Isaac Sim 5.1.0，`rsl_rl/` 为仓库内置副本。

```mermaid
sequenceDiagram
    participant T as scripts/rsl_rl/train.py
    participant L as isaaclab.app.AppLauncher
    participant G as gym.make 与 GRX_humanoid.tasks 注册表
    participant E as MultiStageManagerBasedRLEnv (env/multi_stage_manager_based_rl_env.py)
    participant W as RslRlVecEnvWrapper (isaaclab_rl)
    participant R as rsl_rl.runners.OnPolicyRunner
    T->>L: AppLauncher(args_cli) 启动 Isaac Sim，simulation_app
    T->>T: import GRX_humanoid.tasks，各 config/*/__init__.py 执行 gym.register
    T->>G: parse_env_cfg(task) 与 cli_args.parse_rsl_rl_cfg(task)（取 env_cfg_entry_point / rsl_rl_cfg_entry_point）
    T->>G: gym.make(task, cfg=env_cfg)
    G->>E: 以注册的 entry_point 构造环境，load_managers()
    T->>W: RslRlVecEnvWrapper(env)
    T->>R: runner_class(env, agent_cfg.to_dict(), log_dir, device)
    opt agent_cfg.resume
        T->>R: load(resume_path)
    end
    T->>R: learn(num_learning_iterations, init_at_random_ep_len=True)
    loop 每次迭代
        loop num_steps_per_env 步
            R->>R: alg.act(obs)
            R->>W: env.step(actions)
            W-->>R: obs, rewards, dones, extras
            R->>R: alg.process_env_step(...)
        end
        R->>R: alg.compute_returns(obs) 后 alg.update()
        R->>R: 每 save_interval 保存 model_{it}.pt
    end
    T->>W: env.close()，simulation_app.close()
```

图依据：FFTAI/fourier_lab@74249f7，读过 README.md、scripts/rsl_rl/train.py、scripts/rsl_rl/play.py（仅 runner/策略导出部分）、exts/GRX_humanoid/GRX_humanoid/tasks/locomotion/velocity/config/ppv211_humanoid/__init__.py、env/multi_stage_manager_based_rl_env.py（仅类声明）、rsl_rl/rsl_rl/runners/on_policy_runner.py（learn 部分）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [fourier_lab 源码归档](../../sources/repos/fourier_lab.md)（<https://github.com/FFTAI/fourier_lab>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：FFTAI/fourier_lab@74249f7，读过 README.md、scripts/rsl_rl/train.py、scripts/rsl_rl/play.py、tasks/locomotion/velocity/config/ppv211_humanoid/__init__.py、env/multi_stage_manager_based_rl_env.py（仅类声明）、rsl_rl/rsl_rl/runners/on_policy_runner.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [傅利叶智能 官方组织](https://github.com/FFTAI)
