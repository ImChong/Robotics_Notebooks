---

type: entity
tags: [paper, bfm, behavior-foundation-model, awesome-bfm-papers, meta]
status: complete
updated: 2026-06-25
arxiv: "2312.02682"
venue: "2024 · ICLR"
code: https://github.com/facebookresearch/hgap
summary: "generalist planner 管理低层人形控制；BFM 进入系统后的分层问题。"
related:
  - ../concepts/behavior-foundation-model.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-02-goal-conditioned-learning.md
sources:
  - ../../sources/papers/bfm_awesome_hgap_arxiv_2312_02682.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
---

# H-GAP

**H-GAP** 收录于 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) **第 18/41** 篇，归类为 **02 Goal-conditioned 学习**（2024 · ICLR）。

## 一句话定义

generalist planner 管理低层人形控制；BFM 进入系统后的分层问题。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- generalist planner 管理低层人形控制；BFM 进入系统后的分层问题。
- 在 [BFM 41 篇技术地图](../overview/bfm-41-papers-technology-map.md) 中属于 **02 Goal-conditioned 学习**（#18/41）。

## 流程总览

```mermaid
flowchart TD
    D["MoCapAct 大规模数据集<br/>状态 s(本体感知) + 动作 a(目标关节角)"] --> E["编码器：拼接 (s,a) 为 transition 向量<br/>→ 1D 因果卷积 + 因果 Transformer 解码块"]
    E --> C["按 L 步分块 → 块内 max pooling<br/>→ 拆成 M 个 code embedding"]
    C --> Q["VQ：在 codebook 中取最近邻 → 离散隐码 z"]
    Q --> DEC["解码器 fdec(s1, z) → 重建轨迹<br/>(损失：重建 + VQ + commitment)"]
    Q --> PR["Prior Transformer<br/>自回归建模 p(zi | z&lt;i, s1)"]
    S1["初始状态 s1"] --> PR
    S1 --> DEC
    subgraph MPC["零样本下游任务：MPC 规划"]
        CS["当前状态 s1"] --> SM["Prior 采样 N 条隐码序列<br/>(温度 &gt; 1 + top-p 过滤)"]
        SM --> DC["fdec 解码为轨迹"]
        DC --> SC["按目标函数 R(τ)=Σ r(s) 打分"]
        SC --> BEST["返回得分最高的轨迹"]
    end
    PR --> SM
    DEC --> DC
```

图依据：论文 §2.1–2.3 / Fig.1、Algorithm 1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 18/41 |
| 分组 | 02 Goal-conditioned 学习 |
| 出处 | 2024 · ICLR |
| 论文 | <https://arxiv.org/abs/2312.02682> |
- **代码/项目：** <https://github.com/facebookresearch/hgap>

## 核心机制（归纳）

### 1）策展导读要点

以 **goal / reference / command** 为条件训练全身跟踪或交互策略，扩展人形可执行动作库。

### 2）策展导读要点

数据侧常融合 MoCap、视频、遥操作与 HOI；控制侧强调 **抗扰、恢复与跨参考泛化**。

### 3）策展导读要点

在 BFM taxonomy 中回答「身体能覆盖多少目标条件技能」。

## 结论

**H-GAP 关心的不是身体能做什么，而是「基座进入系统之后谁来指挥」：用 generalist planner 管理低层人形控制的分层问题。**

- 归位在 **02 Goal-conditioned 学习**（#18/41）：以 goal / reference / command 条件化全身策略，扩展人形可执行动作库。
- 分层视角下瓶颈仍在底层——抗扰、恢复与跨参考泛化决定上层规划器能不能稳定使唤这具身体。
- 代码公开（<https://github.com/facebookresearch/hgap>），但本页只保留 survey 坐标与机制归纳，实验与消融请回原文。

## 常见误区

1. Goal-conditioned 跟踪不等于 unlimited skills：仍受数据分布、接触建模与实机 Sim2Real 约束。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)
- BFM 概念：[behavior-foundation-model.md](../concepts/behavior-foundation-model.md)
- 原始 source：[bfm_awesome_hgap_arxiv_2312_02682.md](../../sources/papers/bfm_awesome_hgap_arxiv_2312_02682.md)

## 参考来源

- [bfm_awesome_hgap_arxiv_2312_02682.md](../../sources/papers/bfm_awesome_hgap_arxiv_2312_02682.md) — awesome-bfm 策展摘录
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 41+10 总表
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — 微信公众号编译导读
- 论文：<https://arxiv.org/abs/2312.02682>
- [arXiv PDF](https://arxiv.org/pdf/2312.02682)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 方法（2.1 离散化、2.2 隐码先验、2.3 MPC 规划与 Algorithm 1）

## 推荐继续阅读

- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 完整列表与数据集表
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487) — TPAMI 2025 综述
