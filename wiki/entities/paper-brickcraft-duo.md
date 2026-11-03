---
type: entity
tags: ['paper', 'manipulation', 'bimanual', 'long-horizon', 'assembly']
status: complete
updated: 2026-09-24
arxiv: "2609.28281"
related:
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
  - ../concepts/contact-dynamics.md
  - ../overview/embodied-13-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md
  - ../../sources/papers/brickcraft-duo_arxiv_2609_28281.md
  - ../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md
summary: "BrickCraft-Duo（arXiv:2609.28281）：互锁积木双臂装配：可复用单/双臂技能 + 稳定性感知组合 + 人机定向修正，最长 **9 步** 长时任务。"
---

# BrickCraft-Duo（arXiv:2609.28281）

**BrickCraft-Duo: Efficient Dual-Arm Skill Learning and Refinement for Compositional Long-Horizon Assembly**（[项目页](https://jichuan-yu.github.io/BrickCraft-Duo/)，[arXiv:2609.28281](https://arxiv.org/abs/2609.28281)）来自 [具身智能小站 · 13 篇盘点](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)（2026-09-24）。

## 一句话定义

**互锁积木双臂装配：可复用单/双臂技能 + 稳定性感知组合 + 人机定向修正，最长 **9 步** 长时任务。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RL | Reinforcement Learning | 强化学习 |
| TO | Trajectory Optimization | 轨迹优化 |
| HITL | Human-in-the-Loop | 人在回路 |

## 为什么重要

- 互锁结构同时考双臂协作、支撑关系、插入容差与长时依赖，是接触丰富组装的浓缩测试床。
- 纳入 [13 篇技术地图](../overview/embodied-13-papers-technology-map.md) 阅读坐标。
- 开源结论（步骤 2.5，2026-09-24）：**待发布**。

## 流程总览

```mermaid
flowchart TD
    Target["目标积木结构 B = {b1 … bN}"] --> Plan["稳定性感知装配规划<br/>反向 assembly-by-disassembly 启发式 DFS<br/>可行性 C = 连通 ∧ 可观测 ∧ 可操作 ∧ 稳定<br/>+ 连续同臂惩罚 J_alt"]
    Plan --> Role{"插入载荷下结构是否稳定？<br/>（StableLEGO 受力分析）"}
    Role -->|"稳定"| Single["R_sup = ∅：单臂步"]
    Role -->|"不稳定"| Dual["另一臂作 R_sup：双臂协作步"]
    Single --> P["装配计划 P = (η1 … ηN)<br/>每步含 b_att、b_ref、任务编码 τ、臂角色"]
    Dual --> P
    P --> Comp["图编译器 Γ(P, S)<br/>单臂步展开 Transit–Pick–Transit–UniAssembly<br/>双臂步展开 Transit–Pick–Transit + 支撑臂 Transit → CoAssembly"]
    Lib["技能库 S：Pick / Transit / UniAssembly / CoAssembly<br/>σ = (R, Ξ, π, c_pre, c_post)"] --> Comp
    Comp --> Graph["技能执行图 G（DAG）<br/>前驱完成且 c_pre 满足才激活节点"]
    Graph --> Exec["执行：Transit 用运动规划，Pick 用预定义抓取<br/>UniAssembly / CoAssembly 用学习的闭环 diffusion policy"]
    Demo["人类示范<br/>双边对称对齐：右臂镜像到左臂约定；<br/>CoAssembly 经角色交换 + 反射对齐"] --> Pol["共享策略<br/>输入：视觉（SAM 2 高亮 b_att / b_ref）、本体状态、τ"]
    Pol --> Exec
    Exec -->|"表现不佳的装配步"| Corr["人工 SpaceMouse 遥操作局部纠正<br/>成功后从更新状态继续自主执行"]
    Corr -->|"纠正轨迹 + τ 与原示范合并，微调对应技能"| Pol
```

图依据：论文 §III-A–D 与 Fig. 1–4（规划、技能执行图、对称对齐学习、人类纠正微调）。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.28281](https://arxiv.org/abs/2609.28281) |
| **开源** | **待发布** |
| **要点** | 对称共享与角色分配学 reusable skills；稳定性推理组合 skill graph；HITL 修正薄弱步骤。 |
| **文内指标** | 五任务长时成功率 **≥60%**、逐步完成率 **≥95%**；Scale/Eagle 9 步经 refine 后 completion **~97%** 量级。 |


## 源码运行时序图

**不适用**（截至 2026-09-24 项目页/论文未提供可运行官方代码；开源状态：**待发布**）。


## 实验与评测

- 五任务长时成功率 **≥60%**、逐步完成率 **≥95%**；Scale/Eagle 9 步经 refine 后 completion **~97%** 量级。
- **读法：** 公众号归纳；逐项 baseline 与协议以 arXiv PDF 为准。

## 与其他工作对比

- 横向索引见 [13 篇技术地图](../overview/embodied-13-papers-technology-map.md)。

## 结论

**组合式长时任务先拆 skill 再补 HITL** — 读 initial vs refined 成功率区分数据与修正贡献。

1. 开源边界：**待发布** — 以项目页/仓库实际链接为准（入库日 2026-09-24）。
2. 核心机制：对称共享与角色分配学 reusable skills；稳定性推理组合 skill graph；HITL 修正薄弱步骤。…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

- [Manipulation](../tasks/manipulation.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [Contact Dynamics](../concepts/contact-dynamics.md)
- [Embodied 13 Papers Technology Map](../overview/embodied-13-papers-technology-map.md)

## 参考来源

- [13 篇盘点（公众号）](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)
- [BrickCraft-Duo: Efficient Dual-Arm Skill Learning and Refinement for Compositional Long-Horizon Assembly](../../sources/papers/brickcraft-duo_arxiv_2609_28281.md)
- [arXiv PDF](https://arxiv.org/pdf/2609.28281)：已通读方法部分（实验与附录以略读为准）：摘要、§I 引言、§III 方法（A–D）、§IV-A 实现细节（略读其余实验）

## 推荐继续阅读

- [arXiv:2609.28281](https://arxiv.org/abs/2609.28281) — 原文
