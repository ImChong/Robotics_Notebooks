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
  - ../../sources/repos/robomind-dataset-utils.md
summary: "北京人形机器人创新中心 开源项目 RoboMIND-dataset-utils（数据集/Benchmark）：数据采集/转换工具：同步记录并可转入 LeRobot/RLDS 训练栈…"
institutions:
  - x-humanoid
---

# RoboMIND-dataset-utils

## 一句话定义

**RoboMIND-dataset-utils** 是 [北京人形机器人创新中心](https://github.com/Open-X-Humanoid) 公开的 **数据集/Benchmark** 开源项目：数据采集/转换工具：同步记录并可转入 LeRobot/RLDS 训练栈

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
| 类别 | 数据集/Benchmark |
| 官方组织 | https://github.com/Open-X-Humanoid |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `RoboMIND-dataset-utils` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

仓库是两个独立的离线小工具，无训练/服务进程。入口：质量扫描 `python robomind_check/scan_robomind_quality.py -o full_report.csv`（可加 `--benchmarks` / `--robot-types` / `--task-names` / `--ee-tolerance`）；末端位姿重算 `python eepose/example.py`（需 `pinocchio`，调用 `eepose/joint2pose.py` 的 `get_kinematics`）。

```mermaid
sequenceDiagram
    participant U as 用户
    participant S as scan_robomind_quality.py
    participant P as multiprocessing.Pool
    participant C as robomind_data_check.check_episode
    participant H as h5py / trajectory.hdf5
    participant E as eepose/example.py
    participant K as joint2pose.get_kinematics → JointToEePose

    U->>S: main() 解析 --base-dirs/--robot-types/--ee-tolerance 等
    S->>S: discover_files：os.walk 找 trajectory.hdf5，按 success_episodes 前的路径层级过滤
    S->>P: scan_parallel：imap_unordered(check_episode, files)
    P->>C: check_episode(file_path, config)
    C->>C: _parse_path_into 解析 benchmark/robot_type/task/split/episode
    C->>H: detect_keys 在 puppet/end_effector、puppet/joint_position、master/joint_position 候选键中找首个存在者
    C->>C: check_array_quality：逐维 max-min 与容差比较判 is_stuck，并统计 unique 比例
    C-->>S: EpisodeQualityResult
    S->>S: write_csv(CSV_COLUMNS) 与 print_summary
    S-->>U: robomind_quality_report.csv / full_report.csv
    U->>E: python eepose/example.py（MODEL_NAME）
    E->>K: get_kinematics(model_name) → _find_urdf_for_model(eepose/urdf/<机型>)
    K->>K: 改写 URDF 中 package:// 路径，pin.buildModelFromUrdf，按 _MODEL_FRAME_CONFIG 选 base/TCP 帧与臂关节
    E->>K: get_left_tcp_pose(q_left) / get_right_tcp_pose / get_tcp_pose(q)
    K->>K: 写入 self.q，pin.forwardKinematics + updateFramePlacements，取 base→TCP 相对位姿
    K-->>E: Pose(position, quat_xyzw)
```

图依据：Open-X-Humanoid/RoboMIND-dataset-utils@77d7b31，读过 `README.md`、`robomind_check/scan_robomind_quality.py`、`robomind_check/robomind_data_check.py`、`eepose/example.py`、`eepose/joint2pose.py`。`full_report.csv` 与各 URDF 文件内容未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [RoboMIND-dataset-utils 源码归档](../../sources/repos/robomind-dataset-utils.md)（<https://github.com/Open-X-Humanoid/RoboMIND-dataset-utils>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Open-X-Humanoid/RoboMIND-dataset-utils@77d7b31，读过 README.md、robomind_check/scan_robomind_quality.py、robomind_check/robomind_data_check.py、eepose/example.py、eepose/joint2pose.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [北京人形机器人创新中心 官方组织](https://github.com/Open-X-Humanoid)
