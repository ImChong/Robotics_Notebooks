---
type: entity
tags:
  - paper
  - embodied-ai
  - long-horizon
  - navigation
  - manipulation
  - bit
status: complete
updated: 2026-08-31
arxiv: "2608.27371"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../overview/clap-cross-embodiment-vla-wm-9-papers-technology-map.md
sources:
  - ../../sources/papers/esrp_arxiv_2608_27371.md
  - ../../sources/blogs/wechat_embodied_station_clap_9_papers_open_source_2026-08-31.md
  - ../../sources/sites/pie-lab-esrp.md
summary: "ESRP（arXiv:2608.27371，BIT）：仅 egocentric 观察 + 俯视目标布局的三维家具重排；ESRP-Bench 5400+ 场景对；现有 TAMP/VLM/IL/RL 基线仍难高效完成。"
---

# ESRP：具身场景家具重排规划

**ESRP**（*Embodied Scene Rearrangement Planning*，[arXiv:2608.27371](https://arxiv.org/abs/2608.27371)，IEEE RA-L，[项目页](https://pie-lab.cn/ESRP/)）由 **北京理工大学（BIT）** 提出：agent 在 **无全局状态** 条件下，仅凭 **第一视角观察** 与 **俯视目标布局图**，将三维室内家具重排至目标构型。

## 一句话定义

**场景重排的难点是持续维护局部观察与全局目标布局的对应，而非识别单件家具。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ESRP | Embodied Scene Rearrangement Planning | 本文任务名 |
| TAMP | Task and Motion Planning | 分层任务—运动规划基线 |
| IL | Imitation Learning | 模仿学习基线 |
| RL | Reinforcement Learning | 强化学习基线 |

## 为什么重要

- 纳入 [2026-08-31 九篇盘点](../../sources/blogs/wechat_embodied_station_clap_9_papers_open_source_2026-08-31.md) 的「长时程规划」支线。
- 相对桌面重排与二维全局布局任务，引入 **三维遮挡、相互依赖与动态场景演化**。
- ESRP-Bench 规模：**5400+ 场景对、8200+ 物体**（OmniGibson + 3D-FRONT）。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 北京理工大学（BIT） |
| **基准** | ESRP-Bench（三级指标 + 难度分级） |
| **基线** | 分层 TAMP、VLM、IL、RL 四类 |
| **开源** | **未开源** 训练仓；项目页提供任务/基准说明 |

### 流程总览

```mermaid
flowchart TB
  ego[第一视角观察] --> align[局部—全局布局对齐]
  top[俯视目标布局] --> align
  align --> plan[长时程重排规划]
  plan --> nav[导航 + 操作]
  nav --> scene[动态场景更新]
  scene --> ego
```

## 评测

- 实验显示现有方法 **难以高效完成** 重排，凸显局部可观测下的长时程对齐难题。
- 数据出处：[ingest 摘录](../../sources/papers/esrp_arxiv_2608_27371.md)。

## 与其他工作对比

重排类任务的难度不由「物体多少」决定，而由 **允许 agent 看到多少全局状态** 决定——这正是 ESRP 相对既有基准抬高的那一维：

| 基准 / 任务 | 观测假设 | 物体尺度 | 主要难点 | 相对 ESRP |
|-------------|----------|----------|----------|-----------|
| **ESRP** | **仅 egocentric + 俯视目标布局图，禁全局状态** | 三维家具 | 局部—全局持续对齐、遮挡与相互依赖 | — |
| 桌面重排 | 通常单视角可覆盖全场 | 桌面小物 | 抓取与摆放精度 | 尺度与遮挡结构不同，导航—操作不耦合 |
| 二维全局布局重排 | **给全局状态** | 平面布局 | 组合搜索 | 去掉了本文最核心的可观测性约束 |
| [MS-HAB（ManiSkill-HAB）](./paper-notebook-maniskill-hab-a-benchmark-for-low-level-manipula.md) | 仿真家务重排，强调**真实低层控制**（替代「魔法抓取」） | 家居物件 | 低层操作 + GPU 高吞吐训练 | 难点落在**执行保真度**；ESRP 难点落在**规划期的信息缺失**，两者可视为互补的两端 |

- **基线的信息量**：分层 TAMP / VLM / IL / RL 四类基线**均未高效解决**，说明瓶颈不在某一类方法的容量，而在任务本身的局部可观测结构。
- **同批次分工**：在 [CLAP 九篇地图](../overview/clap-cross-embodiment-vla-wm-9-papers-technology-map.md) 的「长时程规划与安全」组里，ESRP 压的是 **规划难度**，[TrapVLA](./paper-trapvla.md) 压的是 **安全威胁模型**，二者共享「现有 VLA 栈尚未覆盖」的判断。
- **选型提醒**：ESRP 与 MS-HAB 的成功率不可横比——观测假设、物体尺度与仿真后端（OmniGibson + 3D-FRONT vs ManiSkill）均不同；且本文截至入库日**无公开训练仓**（见「局限与风险」）。

## 结论

**ESRP 把「看不见全局」的三维家具重排推成具身长时程规划的前沿考题。**

- 仅 egocentric + 俯视目标，禁止全局状态
- 物体遮挡与相互依赖制造物理死锁风险
- 大规模 ESRP-Bench 与四类基线便于横向对比
- 当前 VLM / IL / RL 均未高效解决
- 复现时区分项目页说明与可下载代码（截至入库日无公开仓）

## 源码运行时序图

源码运行时序图 | **不适用**（截至 2026-08-31 项目页未提供可运行训练/评测仓库）。

## 局限与风险

- **无代码：** 复现依赖后续发布或自行按论文重建 OmniGibson 环境。
- **任务难度：** 家具尺度 + 导航—操作耦合，sim-to-real 成本高。
- **观测受限：** 俯视目标图与 egocentric 对齐本身易累积误差。

## 关联页面

- [Manipulation](../tasks/manipulation.md)
- [VLA](../methods/vla.md)
- [CLAP / 跨本体 9 篇技术地图](../overview/clap-cross-embodiment-vla-wm-9-papers-technology-map.md)

## 参考来源

- [esrp_arxiv_2608_27371](../../sources/papers/esrp_arxiv_2608_27371.md)
- [pie-lab-esrp](../../sources/sites/pie-lab-esrp.md)
- [wechat_embodied_station_clap_9_papers_open_source_2026-08-31](../../sources/blogs/wechat_embodied_station_clap_9_papers_open_source_2026-08-31.md)

## 推荐继续阅读

- [arXiv:2608.27371](https://arxiv.org/abs/2608.27371)
- [ESRP 项目页](https://pie-lab.cn/ESRP/)
