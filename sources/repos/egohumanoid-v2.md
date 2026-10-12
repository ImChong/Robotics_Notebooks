# EgoHumanoid-V2（OpenDriveLab）源码核查

> 来源归档（repo / 官方实现）

- **仓库：** <https://github.com/OpenDriveLab/EgoHumanoid-V2>
- **组织：** OpenDriveLab
- **默认分支：** `main`
- **可见性：** public；GitHub repo API 核查仓库存在且可公开读取
- **许可证：** Apache-2.0（根目录 `LICENSE`）
- **项目页：** <https://opendrivelab.com/EgoHumanoid-V2/>
- **论文：** <https://arxiv.org/abs/2609.37181>
- **项目页归档：** [egohumanoid-v2-opendrivelab.md](../sites/egohumanoid-v2-opendrivelab.md)
- **Wiki 主实体：** [EgoHumanoid-V2](../../wiki/entities/paper-egohumanoid-v2.md)
- **核查日期：** 2026-10-12

## 仓库覆盖范围

官方 README 提供 Ubuntu/Python/CUDA/SONIC/PICO/G1/Revo2 等硬件与环境配置，并描述人类示范采集、同步、动作对齐与回放操作。已公开的重要入口如下：

| 用途 | README 入口 / 路径 | 状态 |
|------|-------------------|------|
| 人类数据采集 | `eh2-collect-human` | README 有操作步骤 |
| 同步与质量检查 | `eh2-sync-human --root data/{task}` | README 有操作步骤；数据输出为同步后的 LeRobot 格式 |
| 粗到细动作对齐 | `data_alignment/action_alignment`；`eh2-align --input ... --endpoints hands|feet --output ...` | 有源码与四项任务样例 |
| 前后动作仿真回放 | `eh2-replay --mode smpl|token` | 有命令和输出说明，结果写 `replay.mp4` |
| 视觉对齐 | README 的 Vision Alignment | Coming soon |
| VLA 策略训练 | Policy Training & Serving 环境有安装说明；Training pipeline checklist 未完成 | 完整项目训练命令/流水线尚待发布 |
| 策略部署 | Policy Deployment checklist | 尚待发布 |

README 声明 action alignment 示例输入放在 `{task}/input.parquet`，对齐输出目录含 `data.parquet`（人体 SMPL 动作及对齐 token）和 `run.json`（逐轮指标与被选轮次）。其 `H→K→D` 与论文一致：SONIC rollout 建立初始化、IK 修正参考、执行响应反馈驱动离线迭代。这里的结构概括结合论文方法与 README 命令；不是独立运行验收报告。

## 开源与复现边界

- **部分开源，有可运行的数据对齐入口。** 论文所需的动作对齐流程不是占位 README，仓库有对应实现目录、CLI 和示例，因此可对这一段绘制运行时序图。
- **端到端复现仍不完整。** README 将视觉对齐、VLA 模型训练、部署列为未完成；未见完整论文数据集或 checkpoint 下载入口。能重跑示例对齐不等于可以从原始示范重训出论文策略并部署到 G1。
- **硬件和依赖成本显著。** README 指定多套主机/机器人环境，并依赖 G1、BrainCo Revo2、SONIC 控制栈、PICO 追踪和 GoPro。复现者应以当前 README 为准检查版本、设备和 checkpoint 来源。
- **许可证范围。** Apache-2.0 适用于仓库所发布内容；不自动覆盖未发布的人体采集数据、模型权重、上游依赖和仓库中引用的第三方媒体。

## 上下游关系

- 项目页：<https://opendrivelab.com/EgoHumanoid-V2/>
- 论文来源：[egohumanoid_v2_arxiv_2609_37181.md](../papers/egohumanoid_v2_arxiv_2609_37181.md)
- 项目页来源：[egohumanoid-v2-opendrivelab.md](../sites/egohumanoid-v2-opendrivelab.md)
- Wiki 实体：[paper-egohumanoid-v2.md](../../wiki/entities/paper-egohumanoid-v2.md)
