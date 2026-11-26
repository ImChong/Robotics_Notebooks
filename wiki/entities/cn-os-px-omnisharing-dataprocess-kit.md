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
  - ../../sources/repos/px_omnisharing_dataprocess_kit.md
summary: "帕西尼感知科技 开源项目 px_omnisharing_dataprocess_kit（数据集/Benchmark）：触觉数据产线后处理工具链（官方首个开源项目）：把 Super EID 产线原始数据（DF-1）经双手/物体位姿估计（PX Pose）加工为 DF-2/DF-2R，并转出可直接训练的 DF-3（LeRobot 格式）；配套发布 Omnisha…"
institutions:
  - paxini
---

# px_omnisharing_dataprocess_kit

## 一句话定义

**px_omnisharing_dataprocess_kit** 是 [帕西尼感知科技](https://github.com/px-DataCollection) 公开的 **数据集/Benchmark** 开源项目：触觉数据产线后处理工具链（官方首个开源项目）：把 Super EID 产线原始数据（DF-1）经双手/物体位姿估计（PX Pose）加工为 DF-2/DF-2R，并转出可直接训练的 DF-3（LeRobot 格式）；配套发布 Omnisharing 采样数据（HF 与官方数据门户）

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
| 机构 | 帕西尼感知科技 |
| 类别 | 数据集/Benchmark |
| 官方组织 | https://github.com/px-DataCollection |

### 架构流程图

```mermaid
flowchart LR
  IN["Super EID 产线原始数据 DF-1 (HDF5)"] --> POSE["Part 1: PX Pose (多 GPU 估计双手与可选物体 6D 位姿)"]
  IN --> POST["Part 2: PX Post-Process (CPU/RAM 解析编码器与触觉)"]
  POSE --> POST
  POST --> DF2["DF-2 (HDF5: 观测/动作 + 位姿)"]
  POST --> DF2R["DF-2R (HDF5: 重定向至灵巧手模型)"]
  POST --> DF3["DF-3 (LeRobot 数据集格式)"]
  DF2 --> VIS["Omnisharing 采样发布 (Hugging Face / PXCap Viewer)"]
  DF3 --> TRAIN["大规模预训练与仿真下游应用"]
```

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `px_omnisharing_dataprocess_kit` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 源码运行时序图

以下时序图基于仓库唯一可执行入口脚本 `entry/first_run.sh` 编写；该脚本仅负责宿主机工作区校验、清理旧容器并拉起 `px_pipeline:0.1.3` 容器交互式 Shell（实际后处理脚本位于容器镜像 `/home/app/pipeline` 内，未在当前源码仓库展开）：

```mermaid
sequenceDiagram
  participant User as 宿主机用户
  participant Script as entry/first_run.sh
  participant Docker as Docker Engine
  participant Cont as px_pipeline 容器 (/home/app)

  User->>Script: bash entry/first_run.sh <WORKDIR>
  Script-->>User: 打印 ver 0.1.3 横幅并 read 等待 Enter 确认
  User->>Script: 按 Enter 确认
  Script->>Script: 3 秒倒计时并解析 WORKDIR 绝对路径 (为空则 exit 1)
  Script->>Docker: docker rm -f px_pipeline (移除同名旧容器)
  Script->>Docker: docker run -it --gpus all --network=host -v WORKDIR:/ws -w /home/app --name px_pipeline px_pipeline:0.1.3
  Docker->>Cont: 启动容器并运行 bash -lc
  Cont->>Cont: 输出 Initialization finished 并执行 exec bash -i
  Cont-->>User: 交付容器内交互式 Shell (工作目录 /home/app, 挂载目录 /ws)
```

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [px_omnisharing_dataprocess_kit 源码归档](../../sources/repos/px_omnisharing_dataprocess_kit.md)（<https://github.com/px-DataCollection/px_omnisharing_dataprocess_kit>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [帕西尼感知科技 官方组织](https://github.com/px-DataCollection)
