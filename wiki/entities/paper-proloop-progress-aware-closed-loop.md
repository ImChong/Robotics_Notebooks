---
type: entity
tags:
  - paper
  - long-horizon
  - manipulation
status: complete
updated: 2026-11-02
arxiv: "2610.39319"
project_id: proloop-progress-aware-closed-loop
related:
  - ../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/proloop-progress-aware-closed-loop_arxiv_2610_39319.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md
summary: "ProLoop：基于结构化状态关系的进度感知长程机器人闭环规划（arXiv:2610.39319）：提出 State-Relation Monitor，根据目标状态和实际执行状态之间的结构化关系，判断任务处于执行中、偏离目标还是已经完成。在 AgentBoard ALFWorld 和真实机器人实验中"
---

# ProLoop：基于结构化状态关系的进度感知长程机器人闭环规划

**ProLoop**（*ProLoop: Progress-Aware Closed-Loop Framework via Structured State Relations for Long-Horizon Robotic Task Planning*，[arXiv:2610.39319](https://arxiv.org/abs/2610.39319)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（Manipulation）。本页依据**周报条目 + arXiv 摘要页**整理，流程图依据论文方法部分（见参考来源），未逐页核对实验数字；指标与结论以论文为准。

## 一句话定义

**ProLoop：基于结构化状态关系的进度感知长程机器人闭环规划——提出 State-Relation Monitor，根据目标状态和实际执行状态之间的结构化关系，判断任务处于执行中、偏离目标还是已经完成。在 AgentBoard ALFWorld 和真实机器人实验中，整体成功率提升 **12.94%**，环境交互次数减少 **32.3%**。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| IL | Imitation Learning | 模仿学习 |

## 为什么重要

- 周报所述待解决的问题：长程操作通常分解为多个子目标，但 LLM 规划器对任务进度的判断主要依赖历史文本，容易误判某个子任务是否真正完成。

## 流程总览

```mermaid
flowchart TD
    PLAN["LLM 子目标规划器 P_LLM(G, M_k)<br/>每次生成一个活动子目标 g_k 和保留线索 r_k"] --> DES["期望状态构造：C_LLM(g_k, 任务状态 schema Ω)<br/>稀疏期望条件集 {χ_k,j}，每条 = ⟨字段类型, 实体绑定, 期望值⟩"]
    EXEC["ReAct 式执行器：思考 + 动作"] --> ENV["环境 / 机器人<br/>观测与反馈 o_k,t"]
    ENV --> GRD["执行状态落地 Φ_m + 状态更新 U<br/>文本环境：确定性解析；实体机器人：VLM 落地"]
    GRD --> ES["执行状态 s^e_k,t<br/>agent 位置 / 目标可见 / 物体属性 / 物体关系 / 持有物"]
    DES --> MON["状态-关系监测器（学习得到）<br/>冻结语义编码器 E_ψ + 成对特征 → MLP 关系头 f_θ"]
    ES --> MON
    MON --> REL["逐条件关系：supported / unresolved / contradicted"]
    REL --> AGG["关系聚合（保守规则）"]
    ENV -->|"无效动作反馈 Inv(o)"| AGG
    AGG -->|"存在 contradicted 或无效动作"| DEV["deviated 偏离"]
    AGG -->|"全部 supported 且无无效动作"| COM["completed 完成"]
    AGG -->|"其他"| ONG["ongoing 进行中"]
    ONG -->|"Continue：不加干预提示"| EXEC
    DEV -->|"Correct：冲突信息构造纠正提示，生成 Corrective Reflection → 纠正动作，不换子目标"| EXEC
    COM -->|"Transition"| MEM["完成门控记忆更新：LLM 按 r_k 总结局部轨迹为 m_k<br/>M_k+1 = M_k ⊕ (g_k, m_k)"]
    MEM -->|"总目标未满足"| PLAN
    MEM -->|"总目标满足"| END["结束"]
    ANN["离线：LLM 标注三类关系标签，加权交叉熵只训练关系头"] -.-> MON
```

图依据：论文 §IV-A 至 §IV-F、Fig. 2/3、式 (11)–(14)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.39319](https://arxiv.org/abs/2610.39319) · [PDF](https://arxiv.org/pdf/2610.39319) |
| **作者（arXiv）** | Zou, Runxin, Liang, Qingmiao, Zhang, Jiatao, Lv, Xinyan, Qie, Liangyu, Bu, Meiyi, Song, Wei, Zhu, Shiqiang |
| **机构（周报所述，未逐条核实）** | 待核实（作者：Runxin Zou、Shiqiang Zhu 等）。 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 提出 State-Relation Monitor，根据目标状态和实际执行状态之间的结构化关系，判断任务处于执行中、偏离目标还是已经完成。在 AgentBoard ALFWorld 和真实机器人实验中，整体成功率提升 **12.94%**，环境交互次数减少 **32.3%**。 |

### arXiv 摘要节选（英文原文）

> Long-horizon robotic tasks are commonly decomposed into semantic subgoals to simplify planning. However, subgoal progress in LLM-based planning is typically inferred implicitly from interaction histories.

## 源码运行时序图

**不适用**：arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布），无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [Manipulation周报技术地图](../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布） |

## 结论

**ProLoop：基于结构化状态关系的进度感知长程机器人闭环规划。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-manipulation-technology-map](../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [proloop-progress-aware-closed-loop_arxiv_2610_39319.md](../../sources/papers/proloop-progress-aware-closed-loop_arxiv_2610_39319.md)
- [wechat_senlanke_weekly_manipulation_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
- [arXiv:2610.39319](https://arxiv.org/abs/2610.39319)
- [arXiv PDF](https://arxiv.org/pdf/2610.39319)：已通读方法部分（实验与附录以略读为准）：§III 预备知识、§IV 方法（子目标规划器 / 结构化状态构造 / 状态关系监测器 / 进度引导适配 / 完成门控记忆更新 / 离线关系学习）、§V-A 实验设定

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.39319)
