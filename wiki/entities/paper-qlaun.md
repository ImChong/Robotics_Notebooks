---
type: entity
tags: ['paper', 'quadruped', 'hardware', 'open-source', 'quasi-direct-drive']
status: complete
updated: 2026-09-07
arxiv: "2609.03623"
summary: "黎巴嫩美国大学（arXiv:2609.03623）：15 kg/12 DoF 全 3D 打印 QDD 四足；RAPID 可换腿；宣称将开源；截至入库日未见仓库。"
related:
  - ../tasks/locomotion.md
  - ./paper-ebert-nonlinear-normal-modes.md
  - ./odri-solo-and-bolt.md
sources:
  - ../../sources/papers/qlaun_arxiv_2609_03623.md
---

# QLAUN：模块化准直驱 3D 打印四足

**QLAUN**（[arXiv:2609.03623](https://arxiv.org/abs/2609.03623)）由 **黎巴嫩美国大学（Lebanese American University）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_papers_2026-09-04.md)）。

## 一句话定义

QLAUN 用 **电子-free 可换腿 + 准直驱 3D 打印** 在低成本下同时追求 **鲁棒与敏捷**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QDD | Quasi-Direct Drive | 准直驱低减速执行器 |
| RAPID | Robust & Affordable Prototyping of an Inertial Driven Leg | 本文腿模块名 |
| HAA/HFE/KFE | Hip Abduction / Flexion / Knee Flexion | 三关节命名 |

## 为什么重要

MENA 等地区缺 **可负担** 力矩控制四足；SOLO 等虽开源但载荷比低。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 黎巴嫩美国大学（Lebanese American University） |
| **开源** | 见 [工程实践](#工程实践) |

## 核心原理

MJ5208 + 8:1 行星 + 4:1 皮带；腿 PLA 打印、TPU 足端；髋屈伸连续旋转；底盘模块化可挂传感器/臂。

### 流程总览

```mermaid
flowchart LR
  motor[BLDC QDD] --> belt[皮带 4:1]
  belt --> leg[RAPID 3-DoF 腿]
  leg --> chassis[3D 打印机身]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-07** 无可运行官方代码（或本文为硬件/协议类工作）。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 见论文摘录与项目页核查结论 |
| 复现入口 | 以 arXiv 为准 |

## 实验与评测

| 规格 | 数值 |
|------|------|
| 质量 | **~15 kg** |
| 尺寸 | 38×77×38 cm |
| 关节 | **12 DoF**（4:1×8:1≈32:1） |

## 结论

QLAUN 定位 **科研向低价力矩四足**；开源承诺待落地后更新本页。

1. ICRA@40 Extended Abstract 起源。
2. 对标 Raibert 早期四足 **演化叙事**。
3. 腿三螺栓快换。
4. 宣称 **将开源** CAD/软件。
5. 入库日 **无 URL**。

## 与其他工作对比

低价力矩控制四足这条线上，各家在 **成本 / 载荷 / 可得性** 三者间取舍不同：

| 平台 | 传动 | 结构工艺 | 开源状态 | 与 QLAUN |
|------|------|----------|----------|----------|
| **QLAUN** | QDD，8:1 行星 + 4:1 皮带 ≈ **32:1** | **全 PLA 打印** + TPU 足端；三螺栓快换腿 | **宣称将开源**，入库日 **无 URL** | 本页；~15 kg / 12 DoF |
| [ODRI SOLO / Bolt](./odri-solo-and-bolt.md) | QDD | 打印件 + 碳杆 | **已开源**（CAD + 固件 + 控制） | 页首动机直指其 **载荷比低**；但 SOLO 的开源是 **已兑现的**，这是 QLAUN 目前最大的差距 |
| [开源 QDD 执行器项目](../comparisons/open-source-qdd-actuator-projects.md) | QDD 各式 | 视项目 | 多数已开源 | 选型时应先看这张对照，QLAUN 未落地前不构成可复现选项 |
| [eBert](./paper-ebert-nonlinear-normal-modes.md) | **SEA 低刚度** | 常规加工 | 无 | 硬件哲学相反：QLAUN 追求 **高带宽力矩可控**，eBert 追求 **共振涌现** |
| 商用四足（如 Unitree 系） | 一体化关节 | 量产 | 闭源 | 载荷/耐久更强、可得性更好；QLAUN 的位置是 **MENA 等地区的可负担科研平台** |

**选型读法：** 现在要一台能跑力矩控制实验的开源四足，**候选是 SOLO 一类已开源项目**，不是 QLAUN。QLAUN 的价值待其 CAD/软件真正发布后再评估——本页在此之前应视为 **扩展摘要级的设计公告**。

## 局限与风险

扩展摘要级细节；长跑耐久与控制器栈未在本页展开。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [paper-ebert-nonlinear-normal-modes.md](./paper-ebert-nonlinear-normal-modes.md)
- [ODRI SOLO/Bolt](./odri-solo-and-bolt.md)
- [开源 QDD 执行器项目](../comparisons/open-source-qdd-actuator-projects.md) — 已兑现开源的同类对照

## 参考来源

- [qlaun_arxiv_2609_03623.md](../../sources/papers/qlaun_arxiv_2609_03623.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_papers_2026-09-04.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.03623](https://arxiv.org/abs/2609.03623)
