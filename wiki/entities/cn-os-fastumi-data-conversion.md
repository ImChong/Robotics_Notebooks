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
  - ../../sources/repos/fastumi_data_conversion.md
summary: "鹿明机器人 开源项目 FastUMI_Data_Conversion（数据集/Benchmark）：把FastUMI采集结果转换成下游训练或分析所需格式，连接原始记录与策略训练数据。…"
institutions:
  - lumos
---

# FastUMI_Data_Conversion

## 一句话定义

**FastUMI_Data_Conversion** 是 [鹿明机器人](https://github.com/LumosRobot) 公开的 **数据集/Benchmark** 开源项目：把FastUMI采集结果转换成下游训练或分析所需格式，连接原始记录与策略训练数据。

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
| 机构 | 鹿明机器人 |
| 类别 | 数据集/Benchmark |
| 官方组织 | https://github.com/LumosRobot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `FastUMI_Data_Conversion` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python raw_hdf5/raw2hdf5.py --task_root <raw> --output <dir> --frequency 20`（原始 session → HDF5）；`python raw_lerobot_v4.0/raw_2_lerobot_V4.0.py --task-root <raw> --repo-id <org/name> ...`（原始 session → LeRobot v4.0 数据集；`raw_lerobot_v2.1/raw_2_lerobot_V2.1.py` 为 v2.1 版本，另用 `ProcessPoolExecutor`，本图未画）。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant H as raw2hdf5.py convert_task
    participant POOL as ProcessPoolExecutor
    participant S as process_session_auto
    participant L as raw_2_lerobot_V4.0.py convert_raw_to_lerobot
    participant LR as LeRobotDataset
    participant FS as session_* 目录（RGB_Images / Clamp_Data / Merged_Trajectory）

    U->>H: --task_root --output --frequency
    H->>FS: find_all_sessions（os.walk，名字以 session 开头）
    H->>POOL: executor.map(process_session_auto, sessions)
    POOL->>S: 每个 session 一个任务
    S->>FS: detect_layout（left_hand*/right_hand* → dual，有 RGB_Images → single）
    S->>FS: load_arm_data：ensure_clamp_csv、ffprobe 校验视频、读轨迹/clamp/时间戳
    S->>S: 以 60fps/target_fps 为步长取主时间轴，argmin 最近邻对齐位姿与 clamp
    S->>FS: cv2.VideoCapture 按 frame_index 读帧
    S->>H: write_hdf5_single / write_hdf5_dual → episode_XXXXXX.hdf5
    U->>L: --task-root --repo-id --target-fps --mode
    L->>FS: find_all_sessions + detect_layout
    L->>LR: create_empty_dataset（LeRobotDataset.create）
    loop 每个 session
        L->>FS: load_arm_data(traj_source)
        L->>L: prepare_single/bimanual_alignment（最近邻对齐并顺序抽帧）
        L->>L: _to_rgb_and_resize(224x224)
        L->>LR: add_frame(observation.state / action / observation.images.robot_N)
        L->>LR: save_episode()
    end
    L->>L: repo_output 非空时把 LEROBOT_HOME/repo_id 复制到输出目录
```

图依据：LumosRobot/FastUMI_Data_Conversion@4ddbf90，读过 readme.md、raw_hdf5/readme.md、raw_hdf5/raw2hdf5.py、raw_lerobot_v4.0/readme.md、raw_lerobot_v4.0/raw_2_lerobot_V4.0.py（convert_raw_to_lerobot 与函数清单）；raw_lerobot_v2.1 仅读函数清单与参数。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [FastUMI_Data_Conversion 源码归档](../../sources/repos/fastumi_data_conversion.md)（<https://github.com/LumosRobot/FastUMI_Data_Conversion>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：LumosRobot/FastUMI_Data_Conversion@4ddbf90，读过 readme.md、raw_hdf5/readme.md、raw_hdf5/raw2hdf5.py、raw_lerobot_v4.0/readme.md、raw_lerobot_v4.0/raw_2_lerobot_V4.0.py（raw_lerobot_v2.1 仅读函数清单）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [鹿明机器人 官方组织](https://github.com/LumosRobot)
