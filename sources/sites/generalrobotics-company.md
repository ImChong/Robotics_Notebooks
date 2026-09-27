# General Robotics 官网（GRID / Auto-Engineering）

> 来源归档

- **标题：** General Robotics — Deployment ready intelligence for real world robotics
- **类型：** site（公司产品站 + 博客）
- **URL：** <https://www.generalrobotics.company/>
- **博客：** <https://www.generalrobotics.company/post/introducing-auto-engineering-for-robotics>
- **入库日期：** 2026-09-10
- **一句话说明：** General Robotics 商业 Physical AI 平台 **GRID** 官网；主推 **Auto-Engineering** 闭环与四类 robotics harness，面向仓储物流、制造、能源、政府等部署场景。

## 开源核查（步骤 2.5，2026-09-27 复核）

| 资源 | 状态 | 说明 |
|------|------|------|
| GRID Enterprise / 完整 monorepo | **未开源** | 企业版为私有可扩展部署；无完整平台公开仓库 |
| **Open GRID** | **Web/CLI 产品** | <https://grid.generalrobotics.dev>；文档 v2.1：<https://docs.generalrobotics.dev/> |
| **GRID-playground** | **部分开源** | [GenRobo/GRID-playground](https://github.com/GenRobo/GRID-playground) — notebook + 仿真 JSON |
| Auto-Engineering harness | **未开源（实现）** | 四类 harness 以博客/产品描述为主 |
| 技术报告 | **公开** | [arXiv:2310.00887](https://arxiv.org/abs/2310.00887) |
| 第三方依赖 | **部分可独立获取** | Isaac Sim、AirGen、MuJoCo、Warp、GELLO 等为外部组件 |

**结论：** **Open GRID + Playground** 可审计入门；**Enterprise 训练/auto-engineering 闭环** 仍需 PoC。详见 [generalrobotics-grid-product.md](./generalrobotics-grid-product.md) 与 [paper-grid 实体](../../wiki/entities/paper-grid-general-robot-intelligence-development.md)。

## 页面要点（策展）

- **产品定位：** 「Put robots to work」— 模块化 intelligence 加速部署，随 auto-engineering **复利**。
- **原则：** Sovereign（数据与 IP 归客户）、Accessible、Evolving、General（任意机器人 + 任意 AI 模型）。
- **场景：** Warehouse & Logistics、Government、Manufacturing、Energy。
- **相关新闻（站内）：** Accenture 投资（2026-04-14）、Microsoft Pegasus（2025-11-17）、Auto-Engineering 发布（2026-09-09）。

## 对 wiki 的映射

- [GRID（General Robotics）](../../wiki/entities/grid-general-robotics.md)
- [Introducing Auto Engineering for Robotics（博客归档）](../blogs/generalrobotics_auto_engineering_2026-09-09.md)
