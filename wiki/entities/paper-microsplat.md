---
type: entity
project_id: microsplat
tags:
  - paper
  - perception
  - planning
status: complete
updated: 2026-11-01
arxiv: "2610.23899"
project: https://microsplat.github.io/
related:
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/papers/microsplat_arxiv_2610_23899.md
  - ../../sources/sites/microsplat.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "MicroSplat（arXiv:2610.23899）：MicroSplat用机械臂搭载消费级显微镜，自动规划视点、焦点堆叠与分割后重建高斯场；真实和合成实验显示主动获取清晰图像的价值，流程仍较耗时。"
---

# MicroSplat（arXiv:2610.23899）

**MicroSplat**（*MicroSplat: Towards Automating 3D Small-Scale Reconstruction*，[arXiv:2610.23899](https://arxiv.org/abs/2610.23899)，[项目页](https://microsplat.github.io/)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「感知、规划与系统」组。

## 一句话定义

**MicroSplat用机械臂搭载消费级显微镜，自动规划视点、焦点堆叠与分割后重建高斯场；真实和合成实验显示主动获取清晰图像的价值，流程仍较耗时。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MicroSplat | MicroSplat: Towards Automating 3D Small-Scale Reconstruction | 本文方法 / 基准的简称 |
| 3D | 3D | 三维 |

## 为什么重要

- 显微重建先拍清楚，再生成三维。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-11-01）。项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
  A["Setup（唯一人在回路阶段）：一次性标定 + 每个样本的人工居中"] --> A1["棋盘格标定：内参 + 手眼标定，机械臂人工微调对焦"]
  A --> A2["5 个姿态人工点动对准十字线，最小二乘估计样本中心"]
  A1 --> B
  A2 --> B["数据采集离线规划：半球 Fibonacci 视点，look-at 朝向，最近邻排序，图像中心射线朝向修正"]
  B --> C["SEW 解析 IK（7 自由度）枚举解，过滤碰撞与关节限位，保留约 134–135 个可达视点"]
  C --> D["在线采集：RRT* 无碰撞运动到各视点"]
  D --> E["每视点沿光轴推进共 40 mm，拍 30 张焦点堆栈"]
  E --> F["预处理-聚焦堆叠：中值滤波，Tenengrad 清晰度，高斯拟合，取 μ±σ 内图像做小波 EDOF 融合"]
  F --> G["预处理-掩码：首张图 Canny 边缘采样点提示 SAM 2，视频模式传播到全部视点"]
  G --> H["COLMAP：SIFT 特征匹配，以机器人位姿为初值做光束法平差，输出稀疏点云"]
  H --> I["gsplat 3DGS：调高致密化阈值 + 背景 alpha 正则 L_bg，训练高斯场"]
```

图依据：论文 §III（Setup、数据采集、预处理、重建）/ Fig.3。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.23899](https://arxiv.org/abs/2610.23899) |
| **开源** | **项目页未列 GitHub** |
| **要点** | MicroSplat用机械臂搭载消费级显微镜，自动规划视点、焦点堆叠与分割后重建高斯场；真实和合成实验显示主动获取清晰图像的价值，流程仍较耗时。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**显微重建先拍清楚，再生成三维：MicroSplat用机械臂搭载消费级显微镜，自动规划视点、焦点堆叠与分割后重建高斯场；真实和合成实验显示主动获取清晰图像的价值，流程仍较耗时。**

1. 开源边界：**项目页未列 GitHub** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
2. 核心机制：MicroSplat用机械臂搭载消费级显微镜，自动规划视点、焦点堆叠与分割后重建高斯场；真实和合成实验显示主动获取清晰图像的价值，流程仍较耗时。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [locomotion](../tasks/locomotion.md)
- [sim2real](../concepts/sim2real.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — 本页感知能力在机器人视觉感知栈中的位置与选型

## 参考来源

- [microsplat_arxiv_2610_23899.md](../../sources/papers/microsplat_arxiv_2610_23899.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.23899](https://arxiv.org/abs/2610.23899)
- [arXiv PDF](https://arxiv.org/pdf/2610.23899)：已通读方法部分（实验与附录以略读为准）：硬件、标定与居中、视点规划与采集、聚焦堆叠与掩码、COLMAP + 3DGS 重建（§III）

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.23899)
- [项目页](https://microsplat.github.io/)
