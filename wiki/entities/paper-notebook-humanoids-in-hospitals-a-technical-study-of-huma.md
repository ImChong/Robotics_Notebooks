---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2503.12725"
related:
  - ./paper-humanoid-surgeon-in-vivo-laparoscopy.md
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_humanoids-in-hospitals.md
summary: "本文探索用人形机器人经遥操作执行医疗任务，以缓解医护人力短缺。研究为 Unitree G1 搭建一套双臂系统，集成高保真位姿跟踪、定制抓取配置、阻抗控制器（用于工具操作）。跨七类医疗流程评测——体检、急救干预、通气（ventilation）、超声引导（ultrasound-guided）、精密穿针等。结果显示：人形能复现关键医疗评估，在通气与超声引导任务上有可观的定量表现，但仍面临力限与传感灵敏度带来的挑战，影响临床精度。"
---

# Humanoids in Hospitals

**Humanoids in Hospitals: A Technical Study of Humanoid Robot Surrogates for Dexterous Medical Interventions** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

本文探索用人形机器人经遥操作执行医疗任务，以缓解医护人力短缺。研究为 Unitree G1 搭建一套双臂系统，集成高保真位姿跟踪、定制抓取配置、阻抗控制器（用于工具操作）。跨七类医疗流程评测——体检、急救干预、通气（ventilation）、超声引导（ultrasound-guided）、精密穿针等。结果显示：人形能复现关键医疗评估，在通气与超声引导任务上有可观的定量表现，但仍面临力限与传感灵敏度带来的挑战，影响临床精度。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Surrogate | 替身，远程代替医护操作 |
| Impedance Control | 阻抗控制，柔顺工具操作 |
| Pose Tracking | 位姿跟踪 |
| Ventilation | 通气（医疗操作） |
| Ultrasound-guided | 超声引导操作 |
| Needle Task | 穿针/进针任务 |

## 为什么重要

- **医疗是高价值但高要求的人形落地场景**：需灵巧 + 柔顺 + 精密；
- **阻抗控制**对接触/工具医疗操作不可或缺；
- **力限与传感灵敏度**是当前硬件瓶颈，提示本体改进方向；
- 系统性技术研究为后续自主化/学习化医疗操作奠基。

## 解决什么问题

医护**人力短缺**，能否用**人形替身**远程做医疗操作？ - 医疗任务**灵巧、精密、需柔顺**； - 不清楚现有人形（G1）能做到什么程度、卡在哪。

论文要：搭建医疗双臂遥操作系统并**系统评测**人形做医疗干预的能力与局限。

## 核心机制

1. **医疗人形遥操作系统**：G1 双臂 + 位姿跟踪 + 定制抓取 + 阻抗控制；
2. **七类医疗流程系统评测**：体检/急救/通气/超声/穿针；
3. **能力与局限并陈**：通气/超声可观，力限/传感制约精度；
4. **医疗替身可行性研究**：面向人力短缺场景。

方法拆解（深读笔记小节）：Unitree G1 双臂医疗遥操作系统；七类医疗流程评测；发现；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Humanoids_in_Hospitals__Humanoid_Surrogates_for_Dexterous_Medical_Interventions/Humanoids_in_Hospitals__Humanoid_Surrogates_for_Dexterous_Medical_Interventions.html> |
| arXiv | <https://arxiv.org/abs/2503.12725> |
| 作者 | Soofiyan Atar、Xiao Liang、Calvin Joyce、Florian Richter、Michael Yip 等（UC San Diego） |
| 发表 | 2025 年 3 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- **姊妹活体路线：** 同 UCSD 团队的 [Humanoid Surgeon（Nature 2026）](./paper-humanoid-surgeon-in-vivo-laparoscopy.md) 将人形医院/手术路线推进至 **in vivo 腹腔镜** 系统评估

## 参考来源

- [humanoid_pnb_humanoids-in-hospitals.md](../../sources/papers/humanoid_pnb_humanoids-in-hospitals.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Humanoids_in_Hospitals__Humanoid_Surrogates_for_Dexterous_Medical_Interventions/Humanoids_in_Hospitals__Humanoid_Surrogates_for_Dexterous_Medical_Interventions.html>
- 论文：<https://arxiv.org/abs/2503.12725>

## 推荐继续阅读

- [机器人论文阅读笔记：Humanoids in Hospitals](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Humanoids_in_Hospitals__Humanoid_Surrogates_for_Dexterous_Medical_Interventions/Humanoids_in_Hospitals__Humanoid_Surrogates_for_Dexterous_Medical_Interventions.html)
