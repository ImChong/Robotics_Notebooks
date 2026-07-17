---
type: entity
tags: [edge-compute, automotive, asil-d, rhino-auto, embodied-ai]
status: complete
updated: 2026-07-17
related:
  - ../concepts/state-estimation.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ./humanoid-robot.md
  - ../comparisons/lidar-slam-lio-vio-selection.md
sources:
  - ../../sources/sites/rhino-auto.md
summary: "辉羲智能：北京亦庄车载高阶智驾芯片与全栈方案商；光至 R1（7nm、>500 TOPS、ASIL-D）与无图城区方案 RINA；公开材料将具身智能机器人列为边缘算力应用场景，但无机器人开源 SDK。"
---

# 辉羲智能（Rhino Auto）

**辉羲智能**（[rhino.auto](https://rhino.auto/)）是专注 **车载智能计算平台** 的芯片与方案公司：2024 年世界智能网联汽车大会发布首款高阶智驾芯片 **光至 R1**，并推出城区 **无图** 参考方案 **RINA（Rhino Intelligent Navigation Assistant）**。公开报道将 **智能驾驶与具身智能机器人** 并列为高算力 Transformer 的应用场景——本库从 **边缘大算力 + 车规安全** 角度索引，供移动机器人感知/规划栈与无人车选型对照。

## 一句话定义

**车规级大算力 SoC + 无图智驾参考方案供应商：为高阶算法迭代提供 >500 TOPS 边缘算力与安全框架，并宣称延伸至服务/工业机器人等具身场景。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SoC | System on Chip | 片上系统；集成 NPU/CPU/安全岛等 |
| NPU | Neural Processing Unit | 深度学习加速器；R1 为 8 核 SIMT 架构 |
| TOPS | Tera Operations Per Second | 深度学习算力常用量级单位 |
| ASIL-D | Automotive Safety Integrity Level D | 最高车规功能安全等级之一 |
| FSD | Full Self-Driving | 高阶自动驾驶能力的市场称谓 |
| RINA | Rhino Intelligent Navigation Assistant | 辉羲城区无图智驾参考方案 |

## 为什么重要（机器人/wiki 语境）

- **算力边界上移：** 人形/移动机器人上的 **端到端 VLA、多相机 BEV、大模型规划** 开始碰到与智驾类似的 **边缘算力 + 实时性 + 安全** 约束；R1 代表 **车规认证的大算力 Transformer 载体**。
- **无图城区方案可类比：** RINA 的 **无高精地图、端到端感知规划** 叙事与移动机器人 **未知室内/园区导航** 在工程上同族（虽传感器栈不同）。
- **非整机玩家：** 与 [Unitree](./unitree.md)、[乐聚](./leju-robotics.md) 不同，辉羲提供 **芯片与方案**，不发布机器人 URDF 或 RL 环境。

## 核心产品（公开资料策展）

| 组件 | 要点 |
|------|------|
| **光至 R1** | 7nm 车规工艺；约 **450 亿晶体管**；**>500 TOPS** NPU + **24× Cortex-A78AE**；宣称 **原生适配 Transformer**；**ASIL-D / EVITA Full** 认证 |
| **RIF** | Risk Immune Framework：分级分域错误诊断与安全策略 |
| **RINA** | 城区无图方案；公开称 **~60% 算力裕量**、系统成本 **~40%** 降幅；目标 **30 天迁移 / 12 个月量产** |
| **数据闭环** | 采集—标注—智驾算法工具链（官网方案叙事） |

> 性能与量产时间表来自 2024–2025 公开报道；落地车型与 2026 出货节奏需以官方最新发布为准。

## 开源状态

- **截至 2026-07-17：** 官网与公开报道 **未见** GitHub / Hugging Face 等机器人或智驾 **开源 SDK 仓库**；定位为 **商业芯片 + 方案交付**。
- **本页不收录** 未在官方站点列出的第三方镜像或传闻驱动。

## 常见误区或局限

- **误区：辉羲 = 人形机器人公司。** 主业是 **车载智驾计算**；机器人仅为公开材料中的 **算力应用场景** 之一。
- **误区：R1 可直接替代机器人 GPU 训练卡。** R1 面向 **车规部署推理**；研究训练仍在数据中心 GPU，边缘侧讨论的是 **部署算力与安全**。
- **局限：** 缺少可复现的开源机器人栈；与 wiki 内 VLA/控制方法页多为 **概念级交叉**，而非代码级集成。

## 关联页面

- [State Estimation](../concepts/state-estimation.md) — 车载/机器人感知融合共性
- [Navigation / SLAM / Autonomy 技术栈](../overview/navigation-slam-autonomy-stack.md) — 无图城区方案与移动机器人/无人车感知–规划栈对照
- [人形机器人](./humanoid-robot.md) — 具身整机语境（辉羲非整机方）

## 推荐继续阅读

- 辉羲官网：<https://rhino.auto/>
- 新华网：光至 R1 发布报道（2024-10）<http://www.news.cn/auto/20241018/330f965781c1444d924227397c6c42df/c.html>

## 参考来源

- [rhino-auto.md](../../sources/sites/rhino-auto.md) — 官网入口与公开产品策展
