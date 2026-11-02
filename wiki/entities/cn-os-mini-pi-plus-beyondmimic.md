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
  - ../../sources/repos/mini-pi-plus_beyondmimic.md
summary: "高擎机电 开源项目 Mini-Pi-Plus_BeyondMimic（全身动作跟踪/技能训练）：全身动作跟踪策略（带仿真/真机验证）…"
institutions:
  - hightorque
---

# Mini-Pi-Plus_BeyondMimic

## 一句话定义

**Mini-Pi-Plus_BeyondMimic** 是 [高擎机电](https://github.com/HighTorque-Robotics) 公开的 **全身动作跟踪/技能训练** 开源项目：全身动作跟踪策略（带仿真/真机验证）

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
| 类别 | 全身动作跟踪/技能训练 |
| 官方组织 | https://github.com/HighTorque-Robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Mini-Pi-Plus_BeyondMimic` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python scripts/csv_to_npz.py --robot pi_plus ...` → `python scripts/rsl_rl/train.py --task=Tracking-Flat-PI-Plus-Wo-v0 --motion_file <npz> --headless` → `python scripts/rsl_rl/play.py --checkpoint <pt>` → `python scripts/sim2sim.py --robot pi_plus --policy_path <onnx>`（README 给出的四段命令）。

```mermaid
sequenceDiagram
    participant U as 用户 / README 命令
    participant C2N as scripts/csv_to_npz.py<br/>(MotionLoader + run_simulator)
    participant IS as Isaac Sim / Isaac Lab<br/>(AppLauncher, InteractiveScene)
    participant TR as scripts/rsl_rl/train.py<br/>(@hydra_task_config)
    participant ENV as ManagerBasedRLEnv<br/>(Tracking-Flat-PI-Plus-Wo-v0)
    participant MC as mdp/commands.py<br/>MotionCommand
    participant RUN as utils/my_on_policy_runner.py<br/>MotionOnPolicyRunner(rsl_rl)
    participant PL as scripts/rsl_rl/play.py
    participant EXP as utils/exporter.py
    participant S2S as scripts/sim2sim.py<br/>(MuJoCo + onnxruntime)

    U->>C2N: csv（GMR 重定向后的 pi_plus 动作）+ input_fps
    C2N->>IS: AppLauncher；SimulationContext(dt=1/output_fps)；InteractiveScene(robot_config["cfg"])
    loop 逐帧回放（sim.render，不调用 sim.step）
        C2N->>C2N: MotionLoader 插值（_lerp/_slerp）并 _compute_velocities
        C2N->>IS: 写入 root / joint 状态，scene.update
        IS-->>C2N: body_pos_w / body_quat_w 等
    end
    C2N-->>U: np.savez → motion.npz（可选 wandb 上传）

    U->>TR: --task / --motion_file
    TR->>TR: import whole_body_tracking.tasks（gym.register）
    TR->>TR: env_cfg.commands.motion.motion_file = motion_file
    TR->>ENV: gym.make(task, cfg=env_cfg)
    ENV->>MC: 构造 MotionCommand，MotionLoader 读 npz
    TR->>RUN: RslRlVecEnvWrapper(env)；MotionOnPolicyRunner(env, agent_cfg)
    TR->>RUN: runner.learn(max_iterations)
    loop PPO 迭代
        RUN->>ENV: env.step(actions)
        ENV->>MC: _update_command（time_steps+=1，播完则 _resample_command）
        ENV-->>RUN: obs, reward（motion_* 误差项）, done
    end
    RUN-->>U: logs/rsl_rl/... model_*.pt（wandb 时另存 .onnx）

    U->>PL: --checkpoint model_xxx.pt
    PL->>RUN: OnPolicyRunner.load；get_inference_policy
    PL->>EXP: export_motion_policy_as_onnx；attach_onnx_metadata
    EXP-->>U: exported/model_xxx.onnx
    PL->>ENV: while simulation_app.is_running: policy(obs) → env.step

    U->>S2S: --motion_file --xml_path pi_20dof.xml --policy_path onnx
    S2S->>S2S: np.load(npz)；读 onnx metadata（关节顺序/刚度/阻尼/action_scale）
    S2S->>S2S: mujoco.MjModel.from_xml_path；onnxruntime.InferenceSession
    loop viewer 循环
        S2S->>S2S: mj_step；pd_control → d.ctrl
        S2S->>S2S: 每 control_decimation 步：create_observation_hi_pi
        S2S->>S2S: policy.run(obs, time_step) → action
        S2S->>S2S: target_dof_pos = action*action_scale + 默认关节位
    end
```

图依据：HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic@df3fcf1，读过 README.md、scripts/csv_to_npz.py、scripts/rsl_rl/train.py、scripts/rsl_rl/play.py、scripts/sim2sim.py、scripts/bvh_to_robot.py（仅入口）、source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py、tasks/tracking/config/pi_plus/__init__.py 与 flat_env_cfg.py、tasks/tracking/tracking_env_cfg.py、tasks/tracking/mdp/commands.py（仅类/函数签名）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Mini-Pi-Plus_BeyondMimic 源码归档](../../sources/repos/mini-pi-plus_beyondmimic.md)（<https://github.com/HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic@df3fcf1，读过 README.md、scripts/csv_to_npz.py、scripts/rsl_rl/train.py、scripts/rsl_rl/play.py、scripts/sim2sim.py、scripts/bvh_to_robot.py、utils/my_on_policy_runner.py、tasks/tracking/config/pi_plus/{__init__,flat_env_cfg}.py、tasks/tracking/tracking_env_cfg.py、mdp/commands.py（签名）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [高擎机电 官方组织](https://github.com/HighTorque-Robotics)
