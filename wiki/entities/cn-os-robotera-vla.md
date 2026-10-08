---
type: entity
tags: [robotera, repo, china-embodied-opensource, open-source, project]
status: complete
updated: 2026-10-08
related:
  - ./robotera.md
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/robotera_vla.md
summary: "以 M7 为默认本体、π₀.₅ / openpi 为基础的示例工程，公开数据规范与训练/推理接入。"
institutions:
  - roboterax
project_id: robotera-vla
code: https://github.com/roboterax/robotera_vla
---

# M7 VLA：采集、训练与推理基线

## 一句话定义

以 M7 为默认本体、π₀.₅ / openpi 为基础的示例工程，公开数据规范与训练/推理接入。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
| --- | --- | --- |
| ROS | Robot Operating System | 机器人通信与软件中间件 |
| SDK | Software Development Kit | 对接机器人运行环境的开发工具 |
| VLA | Vision-Language-Action | 由视觉和语言生成动作的策略 |
| MPC | Model Predictive Control | 基于预测模型优化控制目标 |

## 为什么重要

这一入口补足[星动纪元](./robotera.md)研究与产品之间的工程接口；读者可核对公开代码和厂商运行环境的边界，避免仅凭仓库名推断完整复现能力。

## 核心结构

| 机构 | 星动纪元（ROBOTERA） |
| --- | --- |
| 官方仓 | [roboterax/robotera_vla](https://github.com/roboterax/robotera_vla) |

| 模块 | 当前公开内容与依赖 |
| --- | --- |
| data_collection/ | XOS / Meta Quest 采集操作与数据字段契约；机器人侧 recorder 和遥操作服务需已有环境 |
| training/ | π₀.₅ 调整基线、归一化和微调命令；README 的训练细节 Status 仍待补充 |
| inference/ | Docker 推理部署示例、ROS 2 观测与动作消息契约 |

训练 README 链接 `roboterax/M7_pickplace_example` 样例数据与 `M7_pickplace_example_ckpt` 权重，基础模型取自 openpi。根 README 的 `release_1.0` 指当前基线，不代表 ERA-42 的版本或完整训练配方。

## 工程实践

先核对 `docs/HARDWARE_SOFTWARE_REQUIREMENTS.md`，按数据契约完成采集导出，再从 `training/README.md` 的 `compute_norm_stats.py` 与 `train.py` 进入微调，最后按 `inference/interfaces/ros2_api_contract.md` 对接观测和动作。训练报告 8 张 A800、约 24 小时为 README 自述；本页未重跑。

## 局限与风险

截至 2026-10-08，仓库按 MIT 许可提供公开基线，但采集服务不在仓内实现，训练文档仍有待完善项。公开样例不证明 ERA-42 完整模型、预训练池和部署服务开放。未确认单一首发日，当前维护日期不充当模型发布时间。

## 关联页面

- [星动纪元](./robotera.md)
- [控制 SDK](./cn-os-xbot-sdk-api.md)、[遥操作入口](./cn-os-teleop-client.md)
- [VLA](../methods/vla.md)、[Humanoid-Gym](./humanoid-gym.md)

## 参考来源

- [官方 README 补核](../../sources/repos/robotera_vla.md)
- [既有国内具身开源策展](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [官方 README](https://github.com/roboterax/robotera_vla/blob/main/README.md)
