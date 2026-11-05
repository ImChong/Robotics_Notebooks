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
  - ../../sources/repos/galaxeamanipsim.md
summary: "星海图 开源项目 GalaxeaManipSim（仿真环境）：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证…"
institutions:
  - galaxea
---

# GalaxeaManipSim

## 一句话定义

**GalaxeaManipSim** 是 [星海图](https://github.com/OpenGalaxea) 公开的 **仿真环境** 开源项目：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

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
| 机构 | 星海图 |
| 类别 | 仿真环境 |
| 官方组织 | https://github.com/OpenGalaxea |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `GalaxeaManipSim` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python -m galaxea_sim.scripts.collect_demos --env-name R1ProBlocksStackEasy --num-demos 100`，随后 `python -m galaxea_sim.scripts.replay_demos --env-name ... --target_controller_type bimanual_relaxed_ik`，再 `python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_galaxea_lerobot --task ... --tag replayed --robot r1_pro --use_eef`（README 另给出 `convert_single_galaxea_sim_to_lerobot`、`train_lerobot_dp_policy`、`eval_lerobot_dp_policy` 入口，未画入图中）。

```mermaid
sequenceDiagram
    participant C as scripts/collect_demos.py
    participant ENV as gym.make(env_name) 即 BimanualManipulationEnv 子类
    participant PL as planners/bimanual.py BimanualPlanner (mplib)
    participant RP as scripts/replay_demos.py
    participant CV as scripts/convert_single_galaxea_sim_to_galaxea_lerobot.py
    C->>ENV: gym.make(env_name, control_freq, obs_mode)
    C->>PL: BimanualPlanner(urdf_path, left/right move group, active_joint_names)
    loop 直到收集 num_demos 个成功 episode
        C->>ENV: reset()
        loop for substep in env.unwrapped.solution()
            C->>PL: solve(substep, robot.get_qpos(), last_gripper_cmd)
            PL-->>C: 动作序列
            C->>ENV: env.step(action)，轨迹追加 obs
        end
        C->>C: info["success"] 为真则 save_dict_list_to_hdf5 到 dataset_dir/env/collected/demo_N.h5
    end
    RP->>ENV: gym.make(env_name, controller_type=target_controller_type)
    loop 遍历 collected 下的 h5
        RP->>ENV: reset() 与 reset_world(reset_info)
        RP->>RP: 从 h5 读 ee_pose 或关节指令拼成 actions
        RP->>ENV: 逐步 env.step(actions[i])，再重复末尾动作 5 步
        RP->>RP: 成功则写入 dataset_dir/env/replayed/demo_N.h5
    end
    CV->>CV: 读 h5，LeRobotDataset.create 后 add_frame 与 save_episode
```

图依据：OpenGalaxea/GalaxeaManipSim@abe7f51，读过 `README.md`、`galaxea_sim/scripts/collect_demos.py`、`replay_demos.py`、`convert_single_galaxea_sim_to_galaxea_lerobot.py`（仅函数骨架）、`galaxea_sim/envs/base/bimanual_manipulation.py`（骨架）、`galaxea_sim/planners/bimanual.py`（骨架）、`envs/robotwin/blocks_stack_easy.py`（骨架）；`solution()` 内部、控制器与 `train/eval_lerobot_dp_policy.py` 未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [GalaxeaManipSim 源码归档](../../sources/repos/galaxeamanipsim.md)（<https://github.com/OpenGalaxea/GalaxeaManipSim>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：OpenGalaxea/GalaxeaManipSim@abe7f51，读过 README.md、galaxea_sim/scripts/collect_demos.py、replay_demos.py、convert_single_galaxea_sim_to_galaxea_lerobot.py、galaxea_sim/envs/base/bimanual_manipulation.py、galaxea_sim/planners/bimanual.py、galaxea_sim/envs/robotwin/blocks_stack_easy.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [星海图 官方组织](https://github.com/OpenGalaxea)
