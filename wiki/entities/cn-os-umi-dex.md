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
  - ../../sources/repos/umi-dex.md
summary: "灵心巧手 开源项目 umi-dex（数据集/Benchmark）：面向灵巧手操作采集高质量示范数据，连接人类操作输入、灵巧手状态与训练数据转换。…"
institutions:
  - linkerbot
---

# umi-dex

## 一句话定义

**umi-dex** 是 [灵心巧手](https://github.com/linker-bot) 公开的 **数据集/Benchmark** 开源项目：面向灵巧手操作采集高质量示范数据，连接人类操作输入、灵巧手状态与训练数据转换。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第三层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 灵心巧手 |
| 类别 | 数据集/Benchmark |
| 官方组织 | https://github.com/linker-bot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `umi-dex` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`uv run umi-process <bag> --vocab ./config/ORBvoc.txt --settings ./config/intel_d455.yaml --out sessions/<session_id>/`（`src/umi_dex/cli/process.py` 的 `main`，由 `pyproject.toml` 的 `[project.scripts]` 注册）

```mermaid
sequenceDiagram
    participant U as 用户 shell
    participant P as cli/process.py main
    participant BR as bag_reader.BagReader
    participant CD as controllers/can_decode CanDecoder
    participant UD as controllers/usart_decode UsartDecoder
    participant CA as controllers/calibrate Calibrator
    participant VE as video/encode encode_d405_from_messages
    participant EP as episodes extract_episodes
    participant RP as slam/replay replay_bag
    participant OR as orbslam3 System
    participant EX as slam/exporter export
    participant AS as dataset/assemble assemble
    participant SM as session_meta SessionMeta
    Note over U,P: ROS2 录制侧（ros2/umi_dex_bringup）不在本图内
    U->>P: umi-process bag 与 --vocab --settings --out
    P->>SM: compute_bag_sha256 与 write_bag_sha256
    P->>BR: 打开 bag 并 available_topics
    alt 存在 /hand/can_raw
        loop 每个 CAN 帧
            P->>CD: feed_can_frame
            P->>CA: map_counts 得到关节角
        end
    else 存在 /hand/usart_raw
        loop 每个 USART 帧
            P->>UD: feed_usart_frame
            P->>CA: map_counts 得到关节角
        end
    end
    P->>VE: encode_d405_from_messages 写 d405_color.mp4
    P->>EP: extract_episodes 并写 episodes.csv
    P->>RP: replay_bag 运行 SLAM 回放
    RP->>BR: read_messages 读取 IR1 IR2 IMU
    RP->>OR: System 与 initialize
    loop 每个 IR1 帧（50 ms 内配对最近 IR2）
        RP->>OR: process_stereo_inertial_enhanced（stereo_only 时为 process_stereo_enhanced）
    end
    RP->>OR: get_trajectory_points get_tracked_mappoints get_map_info shutdown
    RP-->>P: ReplayResult
    P->>EX: export 写 trajectory 与 map 等文件
    P->>AS: assemble 写 aligned_dataset 与可选 episodes
    P->>SM: meta.write
```

图依据：umi-dex@b5c88d0，README.md、pyproject.toml、docs/processing.md、src/umi_dex/cli/process.py（完整读过）、cli/extract.py、cli/slam.py、cli/inspect.py（读过入口与参数）、src/umi_dex/bag_reader.py（类与方法定义）、slam/replay.py（replay_bag 全函数）、slam/exporter.py 与 dataset/assemble.py（仅 grep 函数定义与输出文件名）、controllers/can_decode.py（仅 grep 类与方法定义）；usart_decode.py、calibrate.py、video/encode.py、episodes.py 仅见 process.py 中的调用。未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [umi-dex 源码归档](../../sources/repos/umi-dex.md)（<https://github.com/linker-bot/umi-dex>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：linker-bot/umi-dex@b5c88d0，读过 README.md、pyproject.toml、docs/processing.md、src/umi_dex/cli/{process,extract,slam,inspect}.py、src/umi_dex/slam/replay.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [灵心巧手 官方组织](https://github.com/linker-bot)
