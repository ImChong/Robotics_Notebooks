# mimic robotics（mimicrobotics.com）

> 来源归档（ingest）

- **标题：** mimic robotics — Frontier AI for robot dexterity
- **类型：** project site / company
- **官方入口：** https://www.mimicrobotics.com/
- **总部：** 瑞士（硬件标注 made in Switzerland）
- **入库日期：** 2026-07-17
- **核对日期：** 2026-11-19
- **核对范围：** 官网首页全文（`mimic_home.txt`，6,679 chars，含任务场景、部署工位、商业合作流程、技术/新闻时间线及开源边界核查）
- **通读：** 已完成
- **一句话说明：** 欧洲 Physical AI 公司，专注 **通用人形尺度灵巧操作**：自研 **mimic hand M1**、**wearable U1**、**mimic-ipc** 中间件与 **Video-Action Models（VAM）** 研究；面向非结构化工业任务提供单/双臂工作站与全栈商业部署方案。

## 开源 / 代码核查（截至 2026-11-19）

| 组件 | 状态 | 链接 / 说明 |
|------|------|-------------|
| **Smooth Operator / SBR 重定向实验室快照** | **已开源**（研究/基准代码） | https://github.com/mimicrobotics/mimic_retargeter_lab |
| **mimic-video / FLUX-mimic / LAD / MAPLE 论文线** | **论文 + 项目页公开**（部分研究快照） | [mimic-video.github.io](https://mimic-video.github.io/) |
| **mimic hand M1 / wearable U1 硬件与固件** | **未开源**（商业自研硬件） | 博客产品发布，无 CAD / 固件 / 驱动源码仓库 |
| **mimic-ipc / 机队遥测 / 商用部署栈** | **未开源**（自研闭源中间件） | 仅博客性能自报，随商业交付提供 |

## 站点要点（全文核对）

- **典型工业任务：** 官网主打三类非结构化双臂/灵巧任务——**Assembly（装配）**、**Packaging & Sorting（包装与分拣）**、**Complex Manual Tasks（复杂手工作业）**。
- **部署形态与商业流程：** 基于 1 臂或 2 臂的固定工作台（stationary）或移动平台（AMR）机器人工作站，硬件平台无关；企业导入路径为 **Intro call → Use-case analysis → Deploy & scale**。
- **公开时间线（Highlights）：**
  - **2026-09-23 / 2026-07-23：** `FLUX-mimic`（与 Black Forest Labs 合作的视觉-动作生成更新）
  - **2026-07-16：** [Solving Dexterity: A Full-Stack Approach](https://www.mimicrobotics.com/blog/solving-dexterity-a-full-stack-approach) — 发布 **M1** + **U1** + **mimic-ipc** + **Smooth Operator**
  - **2026-01-06 / 2025-12-17：** `mimic-video`（Video-Action Models）
  - **2025-07-17：** `LAD`（CoRL 2025 跨形态工作）
  - **2025-07-01：** `Form Follows Data`
  - **2025-06-13：** `mimic-one`（CoRL 2025，16-DoF 灵巧手达成 **93.3%** 成功率）
  - **2025-04-08：** `MAPLE`（arXiv，与 ETH Zürich / MSR 合作）

## 对 wiki 的映射

- [mimic hand M1](../../wiki/entities/mimic-hand-m1.md)
- [mimic wearable U1](../../wiki/entities/mimic-wearable-u1.md)
- [mimic-video（VAM）](../../wiki/methods/mimic-video.md)
- [Smooth Operator 项目页](./mimicrobotics-smooth-operator.md)
- [mimic_retargeter_lab 仓库](../repos/mimicrobotics_mimic_retargeter_lab.md)
