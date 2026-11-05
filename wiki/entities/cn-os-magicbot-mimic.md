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
  - ../../sources/repos/magicbot-mimic.md
summary: "魔法原子 开源项目 magicbot-mimic（全身动作跟踪/技能训练）：面向MagicBot训练动作模仿与全身跟踪策略，连接参考动作、策略和机器人关节目标。…"
institutions:
  - magiclab
---

# magicbot-mimic

## 一句话定义

**magicbot-mimic** 是 [魔法原子](https://github.com/MagiclabRobotics) 公开的 **全身动作跟踪/技能训练** 开源项目：面向MagicBot训练动作模仿与全身跟踪策略，连接参考动作、策略和机器人关节目标。

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
| 机构 | 魔法原子 |
| 类别 | 全身动作跟踪/技能训练 |
| 官方组织 | https://github.com/MagiclabRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `magicbot-mimic` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：先 `python scripts/csv_to_npz_z1.py --input_file motion.csv --input_fps 30 --output_name <name> --headless` 生成动作 `.npz`，再 `python scripts/rsl_rl/train.py --task=Tracking-Flat-Z1-v0 --motion_file <npz> --num_envs 4096 --headless` 训练（均为 README 所列命令；需要 Isaac Lab 2.1.0 环境）。

```mermaid
sequenceDiagram
    participant U as 用户 / CLI
    participant C2N as scripts/csv_to_npz_z1.py
    participant Tr as scripts/rsl_rl/train.py main()
    participant Reg as tasks/tracking/config/z1/__init__.py (gym.register)
    participant Env as ManagerBasedRLEnv (Z1FlatEnvCfg + mdp/commands.py MotionCommand)
    participant Run as utils/my_on_policy_runner.py MotionOnPolicyRunner
    participant W as wandb（可选）
    U->>C2N: CSV（Z1 关节角 + 根位姿）
    C2N->>C2N: MotionLoader 插值（lerp/slerp）并计算速度
    loop run_simulator 每帧
        C2N->>C2N: robot.write_root_state_to_sim / write_joint_state_to_sim（只渲染，不 sim.step）
        C2N->>C2N: 记录 joint_pos / body_pos_w / body_quat_w 等到 log
    end
    C2N->>C2N: np.savez("/tmp/motion.npz")
    C2N->>W: log_artifact 并 link 到 wandb-registry-motions
    U->>Tr: --task Tracking-Flat-Z1-v0 --motion_file / --registry_name
    Tr->>Tr: AppLauncher 启动 Isaac Sim
    Tr->>Reg: import whole_body_tracking.tasks（注册 Tracking-Flat-Z1-v0）
    Tr->>Tr: hydra_task_config 取 env_cfg 与 Z1FlatPPORunnerCfg
    alt --motion_file
        Tr->>Env: env_cfg.commands.motion.motion_file = 本地路径
    else --registry_name
        Tr->>W: api.artifact(name).download()，取 motion.npz
    end
    Tr->>Env: gym.make(task, cfg=env_cfg) → RslRlVecEnvWrapper
    Tr->>Run: MotionOnPolicyRunner(env, agent_cfg, log_dir, registry_name)
    Tr->>Run: learn(num_learning_iterations, init_at_random_ep_len=True)
    Run->>Run: save()：写 model_*.pt；logger 为 wandb 时导出 ONNX 并上传
```

图依据：magicbot-mimic@8eb9c55，读过 README.md、scripts/csv_to_npz_z1.py、scripts/rsl_rl/train.py、source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/z1/__init__.py、source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py、tasks/tracking/mdp/commands.py（仅类/方法列表）。`play.py`、`replay_npz_z1.py`、`sim2sim_mujoco.py` 未读，不在图中。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [magicbot-mimic 源码归档](../../sources/repos/magicbot-mimic.md)（<https://github.com/MagiclabRobotics/magicbot-mimic>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：MagiclabRobotics/magicbot-mimic@8eb9c55，读过 README.md、scripts/csv_to_npz_z1.py、scripts/rsl_rl/train.py、tasks/tracking/config/z1/__init__.py、utils/my_on_policy_runner.py、tasks/tracking/mdp/commands.py（仅类/方法列表）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [魔法原子 官方组织](https://github.com/MagiclabRobotics)
