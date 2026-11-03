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
  - ../../sources/repos/magiclab_gmr.md
summary: "魔法原子 开源项目 Magiclab_GMR（动作重定向）：把人体或其他动作源重定向到MagicBot Z1本体，为Mimic训练构建目标机器人参考动作。…"
institutions:
  - magiclab
---

# Magiclab_GMR

## 一句话定义

**Magiclab_GMR** 是 [魔法原子](https://github.com/MagiclabRobotics) 公开的 **动作重定向** 开源项目：把人体或其他动作源重定向到MagicBot Z1本体，为Mimic训练构建目标机器人参考动作。

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
| 类别 | 动作重定向 |
| 官方组织 | https://github.com/MagiclabRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Magiclab_GMR` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python scripts/bvh_to_robot.py --bvh_file path/to/motion.bvh --robot dreame_z1 --format lafan1 --motion_fps 120 --save_path output/motion.pkl --rate_limit`（SMPL-X 路径另有 `smplx_to_robot.py`，本图只覆盖已读的 BVH 入口）。

```mermaid
sequenceDiagram
    participant CLI as scripts/bvh_to_robot.py (__main__)
    participant L as utils/lafan1.py load_bvh_file
    participant G as GeneralMotionRetargeting (motion_retarget.py)
    participant P as params.py ROBOT_XML_DICT / IK_CONFIG_DICT
    participant M as mink + MuJoCo
    participant V as RobotMotionViewer (robot_motion_viewer.py)
    participant K as motion_pkl.build_motion_data
    CLI->>L: load_bvh_file(bvh_file, format)
    L-->>CLI: 逐帧人体数据 + actual_human_height
    CLI->>G: GMR(src_human="bvh_lafan1", tgt_robot="dreame_z1", actual_human_height)
    G->>P: 取 MJCF 路径与 ik_configs/bvh_lafan1_to_z1.json
    G->>M: MjModel.from_xml_path；按 ik_match_table1/2 建 mink.FrameTask
    CLI->>V: RobotMotionViewer(robot_type, motion_fps, record_video...)
    loop 每帧 i（--loop 则循环）
        CLI->>G: retarget(lafan1_data_frames[i])
        G->>G: update_targets：to_numpy / scale_human_data / offset_human_data / apply_ground_offset
        G->>M: mink.solve_ik + integrate_inplace（table1，误差下降>0.001 且 <10 次则重复）
        G->>M: 同上，针对 table2
        G-->>CLI: qpos（根位置3 + 四元数4 + 关节角）
        CLI->>V: step(root_pos, root_rot, dof_pos, human_motion_data, rate_limit)
        CLI->>CLI: --save_path 时 qpos_list.append(qpos)
    end
    CLI->>K: build_motion_data(qpos_list, retargeter.xml_file, motion_fps)
    K-->>CLI: 含 fps / root_pos / root_rot / dof_pos / local_body_pos 的字典
    CLI->>CLI: pickle.dump 写出 .pkl
    CLI->>V: close()
```

图依据：Magiclab_GMR@011f5d9，读过 README.md、scripts/bvh_to_robot.py、general_motion_retargeting/motion_retarget.py、general_motion_retargeting/motion_pkl.py（头部）、general_motion_retargeting/utils/lafan1.py（函数列表）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Magiclab_GMR 源码归档](../../sources/repos/magiclab_gmr.md)（<https://github.com/MagiclabRobotics/Magiclab_GMR>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：MagiclabRobotics/Magiclab_GMR@011f5d9，读过 README.md、scripts/bvh_to_robot.py、general_motion_retargeting/motion_retarget.py、general_motion_retargeting/motion_pkl.py、general_motion_retargeting/utils/lafan1.py（仅函数列表）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [魔法原子 官方组织](https://github.com/MagiclabRobotics)
