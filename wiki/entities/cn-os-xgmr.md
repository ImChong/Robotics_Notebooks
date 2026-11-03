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
  - ../../sources/repos/xgmr.md
summary: "北京人形机器人创新中心 开源项目 xGMR（动作重定向）：动作重定向：人体/MoCap 动作映射为目标本体训练参考…"
institutions:
  - x-humanoid
---

# xGMR

## 一句话定义

**xGMR** 是 [北京人形机器人创新中心](https://github.com/Open-X-Humanoid) 公开的 **动作重定向** 开源项目：动作重定向：人体/MoCap 动作映射为目标本体训练参考

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
| 机构 | 北京人形机器人创新中心 |
| 类别 | 动作重定向 |
| 官方组织 | https://github.com/Open-X-Humanoid |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `xGMR` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python scripts/bvh_to_robot.py --bvh_file <file.bvh> --robot tienkung2 --format noitom --save_path out.pkl`（离线 BVH 重定向并在 MuJoCo 窗口预览；`scripts/smplx_to_robot.py` 主循环结构相同，只是数据源换成 SMPL-X）

```mermaid
sequenceDiagram
    participant S as scripts/bvh_to_robot.py
    participant L as utils/lafan1.py load_bvh_file
    participant G as GeneralMotionRetargeting (motion_retarget.py)
    participant P as params.py (ROBOT_XML_DICT / IK_CONFIG_DICT)
    participant M as mink + MuJoCo
    participant V as RobotMotionViewer (robot_motion_viewer.py)
    S->>L: load_bvh_file(bvh_file, format)
    L-->>S: 逐帧 {骨骼名: [pos, quat]} + 人体身高
    S->>G: GMR(src_human="bvh", tgt_robot, format, actual_human_height)
    G->>P: 查机器人 XML 与 IK 配置 JSON
    G->>M: MjModel.from_xml_path 与 setup_retarget_configuration 建 mink.FrameTask
    S->>V: RobotMotionViewer(robot_type, motion_fps)
    loop 每一帧
        S->>G: retarget(frame_data)
        G->>G: update_targets 内依次 scale_human_data / offset_human_data / apply_ground_offset
        G->>M: solve_ik(tasks1, limits) + integrate_inplace（误差下降 > 0.001 且 < 10 次则重复）
        G->>M: 同上，对 tasks2 再求解一次
        G-->>S: configuration.data.qpos 副本
        S->>V: step(root_pos, root_rot, dof_pos, human_motion_data)
        V->>M: mj_forward 后 viewer.sync()
    end
    S->>S: 打包 fps/root_pos/root_rot(wxyz→xyzw)/dof_pos，pickle.dump 到 save_path
    S->>V: close()
```

图依据：Open-X-Humanoid/xGMR@70f08d7，读过 `README.md`、`CLAUDE.md`、`scripts/bvh_to_robot.py`、`scripts/smplx_to_robot.py`、`general_motion_retargeting/motion_retarget.py`、`params.py`、`utils/lafan1.py`、`robot_motion_viewer.py`。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [xGMR 源码归档](../../sources/repos/xgmr.md)（<https://github.com/Open-X-Humanoid/xGMR>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Open-X-Humanoid/xGMR@70f08d7，读过 README.md、CLAUDE.md、scripts/bvh_to_robot.py、scripts/smplx_to_robot.py、general_motion_retargeting/motion_retarget.py、params.py、utils/lafan1.py、robot_motion_viewer.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [北京人形机器人创新中心 官方组织](https://github.com/Open-X-Humanoid)
