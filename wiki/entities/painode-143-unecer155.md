---
type: entity
tags: [curated-index, physical-ai, awesome-physical-ai, standard]
status: complete
updated: 2026-09-20
summary: "Cybersecurity requirements for connected and automated road vehicles."
related:
  - ../entities/awesome-physical-ai-natnew.md
  - ../overview/awesome-physical-ai-technology-map.md
  - ../concepts/sim2real.md
  - ../methods/vla.md
sources:
  - ../../sources/repos/pai_awesome_standard_143_unece-r155.md
  - ../../sources/repos/awesome-physical-ai-union-catalog.md
  - ../../sources/repos/awesome-physical-ai-natnew.md
  - ../../sources/repos/awesome-physical-ai-aichr.md
---

# UNECE R155

**UNECE R155** 收录于 awesome-physical-ai（natnew）**第 143/384** 条，分组 **Governance & Policy**。本页是 **清单索引**：给出它在清单中的位置与官方入口，细节以官方文档 / 原文为准。

## 一句话定义

Cybersecurity requirements for connected and automated road vehicles.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PAI | Physical AI | 具身/物理智能策展主题 |
| RMF | Risk Management Framework | AI 风险管理框架 |
| ISO | International Organization for Standardization | 机器人/功能安全标准族 |

## 为什么重要

- Cybersecurity requirements for connected and automated road vehicles.
- 想横向对照同一分组的其他工作，可以从 [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md) 逐条展开。
- 两份清单合并去重：同一条目只有这一页，出处见下方「参考来源」。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 143/384 |
| 分组 | Governance & Policy |
| 来源清单 | natnew |
| 主链接 | <https://unece.org/transport/vehicle-regulations/un-regulation-no-155-cyber-security-and-cyber-security> |

## 核心原理

Cybersecurity requirements for connected and automated road vehicles.

该条目在 Physical AI 清单中的角色是 **standard**，分组 **Governance & Policy**。本页只给出清单里的问题设定与入口链接，不转载外部营销页或课程大纲。

这一页的用处是从清单跳到可核对的官方入口；机制细节、API 与版本以官方文档为准。

## 工程实践

| 字段 | 内容 |
|------|------|
| 官方入口 | <https://unece.org/transport/vehicle-regulations/un-regulation-no-155-cyber-security-and-cyber-security> |
| 开源核查 | 以项目页 / GitHub 实际链接为准（清单可能滞后） |
| 源码运行时序图 | **不适用**（非论文可运行训练仓，或未核 README 入口） |

使用前先确认链接指向的是官方仓 / 文档，而不是镜像或过期 fork。

## 局限与风险

- 不要把 Awesome 摘要当成完整方法证明或合规结论。
- 同名 GitHub 仓（natnew vs aichr）条目链接可能不同；以本页主链接与技术地图为准。
- 清单中的实验室 / 硬件 / 人物条目偶发链到错误 org，复现或引用前先打开官方页核对。

## 机制与架构图

```mermaid
flowchart LR
    IDX["awesome-physical-ai (natnew 143/384)"] --> GRP["Governance & Policy (standard)"]
    GRP --> ENT["UNECE R155"]
    ENT --> CS["网络安全合规要求 (Cybersecurity Requirements)"]
    CS --> VEH["网联与自动驾驶道路车辆 (Connected & Automated Vehicles)"]
    VEH --> ALIGN["法规与安全标准对齐 (ISO / RMF / PAI)"]
    ENT -.->|"核对 UNECE 法规原文"| DOC["官方入口 (unece.org R155)"]
```

## 关联页面

- [awesome-physical-ai（natnew）](../entities/awesome-physical-ai-natnew.md)
- [awesome-physical-ai（aichr）](../entities/awesome-physical-ai-aichr.md)
- [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md)
- [Physical AI 策展清单对比](../comparisons/awesome-physical-ai-curated-lists.md)
- [Robot Software System Integration · 机器人软件系统集成](../methods/robot-software-integration.md)：规定网联与自动驾驶系统的全生命周期网络安全要求，直接约束车载通信与软件系统集成中的安全防护。
- [Robot Control System Architecture · 机器人控制系统架构](../methods/robot-control-architecture.md)：要求在车辆及自主移动平台的控制系统架构中落实网络隔离、认证与安全边界保护机制。

## 参考来源

- [`sources/repos/pai_awesome_standard_143_unece-r155.md`](../../sources/repos/pai_awesome_standard_143_unece-r155.md) — 本条目策展摘录
- [`sources/repos/awesome-physical-ai-union-catalog.md`](../../sources/repos/awesome-physical-ai-union-catalog.md) — 双清单并集目录
- [sources/repos/awesome-physical-ai-natnew.md](../../sources/repos/awesome-physical-ai-natnew.md)
- [sources/repos/awesome-physical-ai-aichr.md](../../sources/repos/awesome-physical-ai-aichr.md)
- 主链接：<https://unece.org/transport/vehicle-regulations/un-regulation-no-155-cyber-security-and-cyber-security>

## 推荐继续阅读

- [natnew/awesome-physical-ai](https://github.com/natnew/awesome-physical-ai)
- [aichr/awesome-physical-ai](https://github.com/aichr/awesome-physical-ai)
- [原文 / 官方入口](https://unece.org/transport/vehicle-regulations/un-regulation-no-155-cyber-security-and-cyber-security)
