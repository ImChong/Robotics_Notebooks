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
  - ../../sources/repos/gmr.md
  - ../../sources/repos/cn_os_engineai_gmr.md
summary: "众擎机器人 开源项目 EngineAI GMR（动作重定向）：动作重定向：人体/MoCap 动作映射为目标本体训练参考…"
institutions:
  - engineai
---

# EngineAI GMR

## 一句话定义

**EngineAI GMR** 是 [众擎机器人](https://github.com/engineai-robotics) 公开的 **动作重定向** 开源项目：动作重定向：人体/MoCap 动作映射为目标本体训练参考

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
| 类别 | 动作重定向 |
| 官方组织 | https://github.com/engineai-robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `EngineAI GMR` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python scripts/bvh_to_robot.py --bvh_file path/to/motion.bvh --format lafan1 --robot engineai_t800 --save_path outputs/t800_motion.pkl --rate_limit`

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户 CLI
    participant S as scripts/bvh_to_robot.py
    participant L as utils/lafan1.py load_bvh_file
    participant G as motion_retarget.py GeneralMotionRetargeting
    participant P as params.py 注册表
    participant M as MuJoCo 模型与 mink IK
    participant V as robot_motion_viewer.py RobotMotionViewer
    U->>S: python scripts/bvh_to_robot.py --bvh_file --robot engineai_t800
    S->>L: load_bvh_file(bvh_file, format="lafan1")
    L-->>S: frames 列表 (每帧为 骨骼名 到 pos 和 quat), human_height = 1.75
    S->>G: GMR(src_human="bvh_lafan1", tgt_robot, actual_human_height)
    G->>P: ROBOT_XML_DICT[tgt_robot] 与 IK_CONFIG_DICT[src][tgt]
    G->>M: MjModel.from_xml_path(xml), 登记 DoF / body / actuator 名
    G->>G: 读取 IK JSON, 按 actual_human_height 比例缩放 human_scale_table
    G->>M: setup_retarget_configuration, 为 ik_match_table1/2 建 mink.FrameTask
    S->>V: RobotMotionViewer(robot_type, motion_fps, record_video, ...)
    loop 每一帧 (由 --loop 决定是否回绕)
        S->>G: retarget(frame)
        G->>G: update_targets: scale_human_data, offset_human_data, apply_ground_offset
        G->>M: task.set_target(SE3) 设定人体目标位姿
        loop 最多 max_iter=10 次, 且误差下降超过 0.001
            G->>M: mink.solve_ik(configuration, tasks, dt, solver="daqp", damping, limits)
            G->>M: configuration.integrate_inplace(vel, dt)
            G->>G: error1() / error2() 重新计算任务误差
        end
        G-->>S: configuration.data.qpos.copy() (root_pos, wxyz 根姿态, dof_pos)
        S->>V: step(root_pos, root_rot, dof_pos, human_motion_data, rate_limit)
        opt 指定 --save_path
            S->>S: qpos_list.append(qpos)
        end
    end
    opt 指定 --save_path
        S->>S: root_rot 由 wxyz 转 xyzw, pickle.dump({fps, root_pos, root_rot, dof_pos, ...})
    end
    S->>V: close()
```

图依据：engineai-robotics/GMR@71e02c0，README.md、scripts/bvh_to_robot.py、general_motion_retargeting/__init__.py、motion_retarget.py、params.py、data_loader.py、utils/lafan1.py（scripts/smplx_to_robot.py 与 robot_motion_viewer.py 仅检索了关键行）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [EngineAI GMR 源码归档](../../sources/repos/cn_os_engineai_gmr.md)（<https://github.com/engineai-robotics/GMR>）

- [EngineAI GMR 源码归档](../../sources/repos/gmr.md)（<https://github.com/engineai-robotics/GMR>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：engineai-robotics/GMR@71e02c0，读过 README.md、scripts/bvh_to_robot.py、general_motion_retargeting/{__init__.py,motion_retarget.py,params.py,data_loader.py,utils/lafan1.py}；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [众擎机器人 官方组织](https://github.com/engineai-robotics)
