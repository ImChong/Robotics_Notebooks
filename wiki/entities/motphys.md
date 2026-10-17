---
type: entity
tags: [company, motphys, simulation, physics-engine, robot-learning, infrastructure, mjcf]
title: Motphys（谋先飞）
status: complete
updated: 2026-10-17
institutions: [motphys]
summary: "Motphys 以自研实时物理引擎起家，向上做 MotrixSim 仿真、MotrixLab 训练框架与数据/资产工具链，属具身仿真与训练基础设施公司；不训练机器人基础模型，公开研究为与高校联合的 GS-Playground 与 UniLab。"
related:
  - ./motrix.md
  - ./unilab.md
  - ./gs-playground.md
  - ./lightwheel.md
  - ../comparisons/robot-foundation-model-company-paths-2026.md
sources:
  - ../../sources/sites/motphys.md
  - ../../sources/repos/motphys-motrix.md
---

# Motphys（谋先飞）

## 一句话定义

**Motphys**（法人主体北京谋先飞技术有限公司）是具身智能**仿真与训练基础设施**公司：自研实时物理引擎 **MotrixSim**，上接训练框架 **MotrixLab**、数据合成 **MotrixGen**、场景编辑器 **MotrixStudio** 与 SimReady 资产工具链，主张「仿真中大规模习得、真实世界零样本部署」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MJCF | MuJoCo XML Format | MuJoCo 的模型描述格式；MotrixSim 声称原生兼容 |
| RL | Reinforcement Learning | 强化学习，MotrixLab 的主要训练范式 |
| 3DGS | 3D Gaussian Splatting | 三维高斯泼溅渲染；MotrixSim v0.9.0 加入混合渲染 |
| IK | Inverse Kinematics | 逆运动学；v0.3.0 起提供 IK 模块 |
| SimReady | Simulation-Ready | 可直接用于仿真的物理资产；Motphys 提供资产库与转换工具 |

## 为什么重要

- **不同于 Isaac 系的路线：** [Motrix](./motrix.md) 强调 Rust 实现的 CPU 后端与 MJCF 兼容；[UniLab](./unilab.md) 进一步论证「物理放 CPU、学习放 GPU」的异构架构（论文自报端到端提速 3–10 倍）。
- **与模型公司互补：** 读公司路线时，它回答的是「策略在哪里训练、资产从哪来」，而非「谁的基础模型更强」；同类基础设施公司见 [Lightwheel](./lightwheel.md)。

## 公司概况

| 项 | 内容 | 来源 |
|----|------|------|
| 成立时间 | **2020 年 10 月**（36氪 2021-08-16 报道原文）；官网关于页只写「自 2021 年起专注自研实时物理引擎」，两者口径不同，路线以 2020 年排序；CB Insights 亦记 2020 年 | [36氪](https://www.36kr.com/p/1355856359654023) |
| 地点 | 北京总部；杭州研发中心；武汉训练中心 | 官网关于页 |
| 产品 | MotrixSim、MotrixLab、MotrixGen、MotrixStudio、SimReady 资产与工具链 | 官网首页 |

## 公开时间线

| 日期 | 事件 | 本库入口 |
|------|------|----------|
| 2025-09 | 版本日志最早列出的 MotrixSim v0.2.0 / v0.3.0（**不证明首发**） | [Motrix](./motrix.md) |
| 2026-04 | GS-Playground：3DGS 高吞吐照片级仿真器（多机构联合，RSS 2026） | [GS-Playground](./gs-playground.md) |
| 2026-05 | UniLab：CPU 仿真 + GPU 学习的异构 RL 架构，MotrixSim 作后端（多机构联合） | [UniLab](./unilab.md) |

版本日志另列 v0.6.0（2026-03 macOS）、v0.7.0（2026-04 浏览器 Web 平台）、v0.9.0（2026-07 3DGS 混合渲染）、v0.10.0（2026-08 多束 LiDAR），见[官网归档](../../sources/sites/motphys.md)。

## 工程实践

| 场景 | 建议 |
|------|------|
| 想快速验模 | 用浏览器 [MotrixSim Web Viewer](./motrix.md)，无需安装 |
| 想训练 | 从 MotrixLab 入口起步（50+ 环境、SKRL / RSL-RL / FastSAC），见 [Motrix](./motrix.md) |
| 引用论文 | UniLab、GS-Playground 为多机构联合署名，不当作 Motphys 单独成果 |
| 开源状态 | 官网只给 GitHub 组织链接，未说明许可证；MotrixGen / MotrixStudio / 增值资产服务需联系商务 |

## 局限与风险

- 「已在 Unitree GO1 / G1、ANYmal-C 真机迁移」是应用案例描述，不代表合作关系。
- 版本日志只列部分亮点且只写年月。
- 论文数字（3–10 倍、约 10,000 FPS、约 90% 抓取成功率）来自官网转述，本次未重读论文全文。

## 关联页面

- [Motrix（MotrixSim / MotrixLab）](./motrix.md)
- [UniLab](./unilab.md)
- [GS-Playground](./gs-playground.md)
- [Lightwheel（光轮智能）](./lightwheel.md) — 另一家仿真 / 数据基础设施公司
- [公司技术路线对照（2026）](../comparisons/robot-foundation-model-company-paths-2026.md)

## 参考来源

- [Motphys 官网首页、关于、研究与版本日志核查（2026-10-17）](../../sources/sites/motphys.md)
- [Motrix 仓库归档](../../sources/repos/motphys-motrix.md)
- 官网：<https://www.motphys.com/zh/>

## 推荐继续阅读

- [Motphys 研究页](https://www.motphys.com/zh/research/)
- [MotrixSim 版本日志](https://www.motphys.com/zh/release-notes/)
