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
  - ../../sources/repos/mini-pi-plus_pbhc.md
summary: "高擎机电 开源项目 Mini-Pi-Plus_PBHC（全身动作跟踪/技能训练）：全身动作跟踪策略（带仿真/真机验证）…"
institutions:
  - hightorque
---

# Mini-Pi-Plus_PBHC

## 一句话定义

**Mini-Pi-Plus_PBHC** 是 [高擎机电](https://github.com/HighTorque-Robotics) 公开的 **全身动作跟踪/技能训练** 开源项目：全身动作跟踪策略（带仿真/真机验证）

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

1. 从官方 GitHub/Gitee 组织检索 `Mini-Pi-Plus_PBHC` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python GMR/scripts/bvh_to_robot.py` → `python motion_source/count_pkl_contact_mask.py` → `python robot_motion_process/motion_clip_interpolation.py` → `python humanoidverse/train_agent.py +exp=motion_tracking_pi ...` → `python humanoidverse/eval_agent_save_json.py` → `python humanoidverse/sim2sim_pi20dof.py` → `python onnx2rknn.py`（README 给出的顺序）。

```mermaid
sequenceDiagram
    participant U as 用户 / README 命令
    participant GMR as GMR/scripts/bvh_to_robot.py<br/>GeneralMotionRetargeting (mink IK)
    participant CM as motion_source/<br/>count_pkl_contact_mask.py
    participant CL as robot_motion_process/<br/>motion_clip_interpolation.py
    participant TA as humanoidverse/train_agent.py<br/>(@hydra.main)
    participant ENV as LeggedRobotMotionTracking<br/>(envs/motion_tracking/motion_tracking_pi20dof.py)
    participant ML as utils/motion_lib/<br/>MotionLibRobot
    participant ALG as agents/mh_ppo/mh_ppo.py<br/>MHPPO
    participant EV as humanoidverse/<br/>eval_agent_save_json.py
    participant S2S as humanoidverse/<br/>sim2sim_pi20dof.py
    participant RK as onnx2rknn.py (RKNN)

    U->>GMR: --bvh_file lafan1/*.bvh --robot pi_football --save_path *.pkl
    loop 每帧
        GMR->>GMR: load_lafan1_file → retargeter.retarget(frame)
        GMR->>GMR: mink.solve_ik + integrate_inplace → qpos
    end
    GMR-->>U: pkl（root_pos / root_rot / dof_pos）

    U->>CM: +input_folder +output_folder robot=pi+_20dof
    CM->>CM: process_motion：Humanoid_Batch.fk_batch + foot_detect
    CM-->>U: *_cont_mask_fixed.pkl（含 contact_mask）
    U->>CL: --origin_file_name --start --end --end_inter_frame
    CL-->>U: 裁剪并插值首尾过渡的 pkl（joblib.dump）

    U->>TA: +exp=motion_tracking_pi +env=motion_tracking_pi20dof robot.motion.motion_file=pkl
    TA->>ENV: instantiate(config.env)
    ENV->>ML: _init_motion_lib → load_motions
    TA->>ALG: instantiate(config.algo)；algo.setup()
    TA->>ALG: algo.learn()
    loop 每个 learning iteration
        ALG->>ALG: _rollout_step（num_steps_per_env 步）
        ALG->>ENV: env.step(actor_state["actions"])
        ENV->>ENV: _pre_physics_step → _physics_step → _post_physics_step
        ENV-->>ALG: obs_dict, rewards, dones
        ALG->>ALG: _training_step（_update_ppo）
        ALG->>ALG: 每 save_interval 存 model_N.pt
    end

    U->>EV: +checkpoint=model_xxx.pt
    EV->>EV: 读取 checkpoint 旁 config.yaml 合并 eval 配置
    EV->>ALG: algo.load(ckpt)；export_policy_as_onnx → exported/*.onnx
    loop motion_len*50 步
        EV->>ENV: algo.env_step(actor_state)
        EV->>EV: 记录 pri_rel_fut_ref_motion_state_flat
    end
    EV-->>U: rel_fut_ref_motion_state_flat_data.json

    U->>S2S: --checkpoint onnx --xml_path --json_file
    S2S->>S2S: ONNXPolicyWrapper(onnxruntime)；读取 json 未来参考帧
    loop MuJoCo 仿真循环
        S2S->>S2S: get_obs + get_rel_fut_ref_obs_from_json
        S2S->>S2S: action = policy(obs_buf)；clip；target_q = default + action*scale
        S2S->>S2S: pd_control → tau → mujoco.mj_step
    end

    U->>RK: 修改路径后运行 onnx2rknn.py
    RK-->>U: policy_from_onnx.rknn（target_platform=rk3588s）
    Note over U: 真机：README 用 scp 把 rknn/json 拷到机器人<br/>并在 sim2real_master（catkin/roslaunch）里运行，<br/>该工作区不在本仓库内，未读
```

图依据：HighTorque-Robotics/Mini-Pi-Plus_PBHC@3ef902f，读过 README.md、GMR/scripts/bvh_to_robot.py、GMR/general_motion_retargeting/motion_retarget.py、motion_source/count_pkl_contact_mask.py、robot_motion_process/motion_clip_interpolation.py（入口与函数签名）、humanoidverse/train_agent.py、config/exp/motion_tracking_pi.yaml、config/env/motion_tracking_pi20dof.yaml、envs/motion_tracking/motion_tracking_pi20dof.py、envs/legged_base_task/legged_robot_base.py、agents/mh_ppo/mh_ppo.py、humanoidverse/eval_agent_save_json.py、humanoidverse/sim2sim_pi20dof.py、onnx2rknn.py。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Mini-Pi-Plus_PBHC 源码归档](../../sources/repos/mini-pi-plus_pbhc.md)（<https://github.com/HighTorque-Robotics/Mini-Pi-Plus_PBHC>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：HighTorque-Robotics/Mini-Pi-Plus_PBHC@3ef902f，读过 README.md、GMR/scripts/bvh_to_robot.py、GMR/general_motion_retargeting/motion_retarget.py、motion_source/count_pkl_contact_mask.py、robot_motion_process/motion_clip_interpolation.py、humanoidverse/{train_agent,eval_agent_save_json,sim2sim_pi20dof}.py、envs/motion_tracking/motion_tracking_pi20dof.py、envs/legged_base_task/legged_robot_base.py、agents/mh_ppo/mh_ppo.py、config/exp/motion_tracking_pi.yaml、config/env/motion_tracking_pi20dof.yaml、onnx2rknn.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [高擎机电 官方组织](https://github.com/HighTorque-Robotics)
