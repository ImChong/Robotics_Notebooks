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
  - ../../sources/repos/leju-gmr.md
summary: "乐聚机器人 开源项目 Leju-GMR（动作重定向）：动作重定向：人体/MoCap 动作映射为目标本体训练参考…"
institutions:
  - leju
---

# Leju-GMR

## 一句话定义

**Leju-GMR** 是 [乐聚机器人](https://github.com/LejuRobotics) 公开的 **动作重定向** 开源项目：动作重定向：人体/MoCap 动作映射为目标本体训练参考

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
| 类别 | 动作重定向 |
| 官方组织 | https://github.com/LejuRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Leju-GMR` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python scripts/bvh_to_pkl.py --bvh_file output/BVH/leju/dance_bj_01_Skeleton_002.bvh --bvh_fps 100 --robot roban_s17 --motion_fps 50 --rate_limit --format leju`（README「快速使用」）。

```mermaid
sequenceDiagram
    participant U as 用户 CLI
    participant S as scripts/bvh_to_pkl.py
    participant AD as RobotAdapter（roban_s17 的 load / build / validate / postprocess）
    participant L as utils/lafan_vendor（read_bvh，quat_fk）
    participant G as motion_retarget.GeneralMotionRetargeting
    participant IK as mink（solve_ik，integrate_inplace）
    participant V as robot_motion_viewer.RobotMotionViewer（MuJoCo）

    U->>S: 解析参数，validate_robot_joint_layout，查 IK_CONFIG_DICT
    S->>AD: load_motion_data(format, bvh_file, 单位换算)
    AD->>L: read_bvh，quat_fk 求全局位姿
    AD-->>S: 人体帧列表（骨骼名到位置与四元数）与 human_height
    AD->>AD: ensure_required_human_keys_roban_s17 补齐/规范关键点
    S->>AD: build_preprocessor(format, ..., human_height)
    AD->>G: GeneralMotionRetargeting(src_human="bvh_leju", tgt_robot="roban_s17", solver="daqp")
    G->>G: 按 ROBOT_XML_DICT 载入 MuJoCo 模型，按 IK_CONFIG_DICT 载入 JSON 匹配表，建 FrameTask 与限位
    S->>V: RobotMotionViewer(...)，注册键盘回调
    loop 逐帧（空格暂停，N 单步，Q 退出）
        S->>G: retarget(human_frame)
        G->>G: update_targets：缩放、偏移人体数据，设各 task 目标位姿
        loop 误差仍下降且未超 max_iter
            G->>IK: solve_ik(configuration, tasks, dt, solver, damping, limits)
            IK-->>G: 关节速度，configuration.integrate_inplace
        end
        G-->>S: qpos（含基座位姿与关节角）
        S->>AD: validate_qpos，postprocess_qpos
        S->>V: step(root_pos, root_rot, dof_pos, rate_limit)
    end
    S->>S: 组装 root_pos / root_rot / dof_pos，pickle.dump 写入 pkl
```

图依据：Leju-GMR@0b1394f，读过 `README.md`、`scripts/bvh_to_pkl.py`（参数、load_bvh_leju_common、load_motion_data_roban_s17、_build_gmr_preprocessor、主循环、pickle.dump 段）、`general_motion_retargeting/motion_retarget.py`（__init__ 局部、update_targets、retarget）、`robot_motion_viewer.py`（step 签名）。`pkl_to_csv.py`、`smplx_to_robot.py`、`optitrack_to_robot.py` 等其它脚本未读，未画。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Leju-GMR 源码归档](../../sources/repos/leju-gmr.md)（<https://github.com/LejuRobotics/Leju-GMR>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：LejuRobotics/Leju-GMR@0b1394f，读过 README.md、scripts/bvh_to_pkl.py、general_motion_retargeting/motion_retarget.py、general_motion_retargeting/robot_motion_viewer.py（step 签名）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [乐聚机器人 官方组织](https://github.com/LejuRobotics)
