---
type: entity
tags: [paper, humanoid-paper-notebooks, humanoid, manipulation, teleoperation, medical-robotics, impedance-control, ucsd, unitree]
status: complete
updated: 2026-09-28
arxiv: "2503.12725"
related:
  - ./paper-humanoid-surgeon-in-vivo-laparoscopy.md
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../tasks/teleoperation.md
  - ../concepts/impedance-control.md
  - ../methods/admittance-control.md
  - ./paper-notebook-humanoidvlm-vision-language-guided-impedance-con.md
sources:
  - ../../sources/papers/humanoid_pnb_humanoids-in-hospitals.md
summary: "本文探索用人形机器人经遥操作执行医疗任务，以缓解医护人力短缺。研究为 Unitree G1 搭建一套双臂系统，集成高保真位姿跟踪、定制抓取配置、阻抗控制器（用于工具操作）。跨七类医疗流程评测——体检、急救干预、通气（ventilation）、超声引导（ultrasound-guided）、精密穿针等。结果显示：人形能复现关键医疗评估，在通气与超声引导任务上有可观的定量表现，但仍面临力限与传感灵敏度带来的挑战，影响临床精度。"
---

# Humanoids in Hospitals

**Humanoids in Hospitals: A Technical Study of Humanoid Robot Surrogates for Dexterous Medical Interventions** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

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
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Humanoids_in_Hospitals__Humanoid_Surrogates_for_Dexterous_Medical_Interventions/Humanoids_in_Hospitals__Humanoid_Surrogates_for_Dexterous_Medical_Interventions.html> |
| arXiv | <https://arxiv.org/abs/2503.12725> |
| 源码 | **未开源**：项目页 <https://surgie-humanoid.github.io> 截至 2026-09-28 未列代码仓库 |
| 作者 | Soofiyan Atar、Xiao Liang、Calvin Joyce、Florian Richter、Michael Yip 等（UC San Diego） |
| 发表 | 2025 年 3 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**设置**：Unitree G1 双臂遥操作；操作端为 HTC Vive 追踪器 + 多相机手部姿态追踪 + 双踏板（离合与模式切换），末端按离合后的相对手部运动映射；踏板可切入阻抗模式，由估计的末端力调整位置以保证柔顺接触。**所有实验由非临床人员操作**，7 项任务覆盖体检、急救与精密穿针。

| 任务 | 结果 |
|------|------|
| 心脏听诊 | 每次重新摆放听诊器约 6.35 s；机器人自身振动混入听诊信号 |
| Leopold 四步触诊（孕妇模拟人） | 能完成手法序列，但手部几何使拇指与其他手指难以同时获得压力读数 |
| 球囊面罩通气（BVM） | 单手：间隔 6 s、送气 0.99 s、平均潮气量 471 mL、86.7% 在 400–600 mL 内；双手：6 s、1.06 s、533 mL、93.3%。三名人类对照：间隔 4.8–4.9 s、潮气量达标率 53–93% |
| 气管插管 | 打开并保持口腔需约 44 N，超过 G1 最大力，需人工协助拉喉镜；导管可由机器人自行置入 |
| 气管切开（10 次） | 切口成功 30%、部分成功 40%、失败 30%；成功 / 部分成功平均 19 s，导管尖端置入平均 18.4 s |
| 超声引导注射（20 次） | 命中 45%、调整后命中 25%（合计 70%）、未见针尖 30%；经验医生约 90%，未受训医学生约 36.4% |
| 缝合（16 次） | 与专用手术系统差距大，插针阶段失败最多（视野受限时针在组织中意外转向） |

## 与其他工作对比

| 系统 | 形态 | 与本文的差异 |
|------|------|------|
| 专用手术机器人（如 da Vinci 类） | 专用器械 + 专用持针 | 缝合等精细任务明显更好；人形优势在于直接使用医院现有工具 |
| [Humanoid Surgeon（Nature 2026）](./paper-humanoid-surgeon-in-vivo-laparoscopy.md) | 同 UCSD 团队 | 推进到在体腹腔镜的系统评估 |
| 人类操作者（论文对照） | — | BVM 节律上人形更稳定；超声引导注射上仍低于经验医生 |

## 结论

**这是一份把「人形医疗替身」能力边界量出来的技术研究，结论是能复现关键医疗评估、但还够不到临床精度，而卡点主要在本体硬件而非遥操作方案本身。**

- 系统侧真正吃劲的是 **阻抗控制**：医疗工具操作需要柔顺接触，位姿跟踪与定制抓取只解决够不够得着，不解决压得对不对。
- 结果分层清晰：通气与超声引导任务有可观的定量表现，而 **力限与传感灵敏度** 制约了精密穿针一类的临床精度——这是硬件瓶颈，也直接指向本体改进方向。
- 适用边界：全程为遥操作而非自主，平台是 Unitree G1 双臂系统；本文定位是可行性与局限的系统评测，不是可临床部署的方案，但为后续自主化/学习化医疗操作奠基。
- 路线延伸：同 UCSD 团队的 [Humanoid Surgeon（Nature 2026）](./paper-humanoid-surgeon-in-vivo-laparoscopy.md) 把这条线推进到 in vivo 腹腔镜的系统评估。
- 量化上最有说服力的是节律类任务：双手 BVM 潮气量达标率 93.3%、间隔稳定在 6 s；最弱的是精密操作——超声引导注射合计 70%（经验医生约 90%），气管切开切口完全成功仅 30%。

## 局限与风险

- **力量不足**：气管插管需约 44 N 开口力，超出 G1 能力，必须人工协助；气管导管也难以完全插入高摩擦模拟气管。
- **振动干扰**：机器人振动会混入听诊器等敏感测量。
- **缺少触觉 / 力反馈**：组织张紧与切割、穿针时的针姿控制都受限，作者认为触觉反馈对精密针类任务必不可少。
- **灵巧度有限**：Leopold 触诊与缝合等复杂手法难以有效完成。
- **全程遥操作、探索性研究**：没有自主能力，操作者非临床人员，样本量小（10–20 次），未做临床试验。
- **开源边界**：未见代码；源码运行时序图 **不适用**。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- **姊妹活体路线：** 同 UCSD 团队的 [Humanoid Surgeon（Nature 2026）](./paper-humanoid-surgeon-in-vivo-laparoscopy.md) 将人形医院/手术路线推进至 **in vivo 腹腔镜** 系统评估
- 任务语境：遥操作：[teleoperation](../tasks/teleoperation.md)
- 柔顺接触所用阻抗控制：[impedance-control](../concepts/impedance-control.md)
- 导纳控制：[admittance-control](../methods/admittance-control.md)
- 同为 G1 上的阻抗调节：[paper-notebook-humanoidvlm-vision-language-guided-impedance-con](./paper-notebook-humanoidvlm-vision-language-guided-impedance-con.md)

## 参考来源

- [humanoid_pnb_humanoids-in-hospitals.md](../../sources/papers/humanoid_pnb_humanoids-in-hospitals.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Humanoids_in_Hospitals__Humanoid_Surrogates_for_Dexterous_Medical_Interventions/Humanoids_in_Hospitals__Humanoid_Surrogates_for_Dexterous_Medical_Interventions.html>
- 论文：<https://arxiv.org/abs/2503.12725>
- 论文正文（实验结果与讨论节）：<https://arxiv.org/html/2503.12725>

## 推荐继续阅读

- [机器人论文阅读笔记：Humanoids in Hospitals](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Humanoids_in_Hospitals__Humanoid_Surrogates_for_Dexterous_Medical_Interventions/Humanoids_in_Hospitals__Humanoid_Surrogates_for_Dexterous_Medical_Interventions.html)
- 项目页：<https://surgie-humanoid.github.io>
