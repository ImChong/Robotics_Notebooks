---
type: entity
project_id: contactbench-1-0
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-11-01
arxiv: "2610.24512"
project: https://jingyi-xiang.github.io/contactbench/
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/papers/contactbench_1_0_arxiv_2610_24512.md
  - ../../sources/sites/contactbench-1-0.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "ContactBench 1.0（arXiv:2610.24512）：ContactBench 1.0用可打印物体与可调阻力仿真覆盖日常接触技能；六项实测任务中，加入触觉未带来明显提升，提示传感器与策略利用能力需分别评估。"
---

# ContactBench 1.0（arXiv:2610.24512）

**ContactBench 1.0**（*ContactBench 1.0: A Benchmark for Everyday Contact-Rich Robotic Manipulation*，[arXiv:2610.24512](https://arxiv.org/abs/2610.24512)，[项目页](https://jingyi-xiang.github.io/contactbench/)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「评测与基准」组。

## 一句话定义

**ContactBench 1.0用可打印物体与可调阻力仿真覆盖日常接触技能；六项实测任务中，加入触觉未带来明显提升，提示传感器与策略利用能力需分别评估。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ContactBench 1.0 | ContactBench 1.0: A Benchmark for Everyday Contact-Rich Robotic Manipulation | 本文方法 / 基准的简称 |

## 为什么重要

- 把日常接触技能做成可复现任务。
- 开源结论：**待发布**（步骤 2.5，2026-11-01）。项目页 HTTP 200，仿真代码入口标 Simulation code (coming soon)。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
    OBS["观察日常人类操作 → 归纳 8 项接触丰富技能 Fig. 2<br/>抓取薄或小物体 / 抓取多个物体 / 杂乱中抓取 / 受环境约束的放置<br/>手内重定位 / 外部接触控制 / 关节物体操作 / 插入与对准"] --> TASKS["36 个参考任务 Fig. 3: 31 个短时域 + 5 个长时域<br/>明确任务定义与成功判据, 并录制人类完成任务的视频"]
    OBS --> ASSETS["40 多个 3D 打印资产: PLA 材料, M3 螺钉装配<br/>用 PLA 柔性弹簧设定所需力, 减少对打印精度的依赖"]
    ASSETS --> VAL["跨打印机验证: F/T 传感器测门把手按压力与充电器插入力<br/>Bambu Lab P1S 与 Prusa XL, 对照真实物体"]
    TASKS --> VAR["任务变体 Table I: 行为变体 对视觉不明显 / 视觉 + 行为变体"]
    ASSETS --> VAR
    ASSETS --> SIM["§IV 仿真套件: Newton 物理引擎 + MuJoCo Warp 求解器<br/>与实物配对的资产, 弹簧阻尼驱动, 刚度 K 与预载 τ0 可调"]
    SIM --> EXTRA["可选 GelSight Mini 触觉模拟 / 键盘, 手柄, SpaceMouse 遥操作 / 脚本策略"]
    TASKS --> EXP["§V 实验: 选 6 个短时域任务, UR5e + WSG50, UMI 手持夹爪采集示范<br/>策略 DP, RDP, π0.5, FTP-1; 每方法每任务 30 回合"]
    VAR --> EXP
    EXP --> MET["指标: 成功回合数 SR / 任务进度 Prog / 峰值力超过 40 N 的比例 / 相对人类的完成时间"]
    MET --> Q["§V-B 纯视觉策略的主要挑战 → §V-C 视觉 - 触觉策略能否解决 → §V-D 行为泛化案例"]
```

图依据：基准构建与评测协议图，论文 Fig. 1-3、§III-A 至 §III-D、Table I、§IV-A 至 §IV-D、§V-A、Table III（指标含义）、§V 开头的三个研究问题与 §V-B 至 §V-D 小节标题。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.24512](https://arxiv.org/abs/2610.24512) |
| **开源** | **待发布** |
| **要点** | ContactBench 1.0用可打印物体与可调阻力仿真覆盖日常接触技能；六项实测任务中，加入触觉未带来明显提升，提示传感器与策略利用能力需分别评估。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（待发布）** — 项目页 HTTP 200，仿真代码入口标 Simulation code (coming soon)。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**把日常接触技能做成可复现任务：ContactBench 1.0用可打印物体与可调阻力仿真覆盖日常接触技能；六项实测任务中，加入触觉未带来明显提升，提示传感器与策略利用能力需分别评估。**

1. 开源边界：**待发布** — 项目页 HTTP 200，仿真代码入口标 Simulation code (coming soon)。
2. 核心机制：ContactBench 1.0用可打印物体与可调阻力仿真覆盖日常接触技能；六项实测任务中，加入触觉未带来明显提升，提示传感器与策略利用能力需分别评估。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 评测基准在选型闭环中的定位与取舍

## 参考来源

- [contactbench_1_0_arxiv_2610_24512.md](../../sources/papers/contactbench_1_0_arxiv_2610_24512.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.24512](https://arxiv.org/abs/2610.24512)
- [arXiv PDF](https://arxiv.org/pdf/2610.24512)：已通读摘要、§I-§IV、§V-A 实验设置与 Table III 的指标定义；§V-B 至 §V-D 的失败分析与泛化案例、§VI 讨论与附录只读小节标题与总述，未逐段读

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.24512)
- [项目页](https://jingyi-xiang.github.io/contactbench/)
