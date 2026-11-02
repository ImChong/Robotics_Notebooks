---

type: entity
tags: [paper, bfm, behavior-foundation-model, awesome-bfm-papers, meta]
status: complete
updated: 2026-06-25
arxiv: "2102.11271"
venue: "2021 · ICML"
code: https://github.com/denisyarats/proto
summary: "原型表示组织行为空间，便于上层组合与选择技能。"
related:
  - ../concepts/behavior-foundation-model.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-03-intrinsic-reward-pretraining.md
sources:
  - ../../sources/papers/bfm_awesome_proto_rl_icml_2021.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
---

# Reinforcement Learning with Prototypical Representations

**Reinforcement Learning with Prototypical Representations** 收录于 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) **第 27/41** 篇，归类为 **03 Intrinsic reward 预训练**（2021 · ICML）。

## 一句话定义

原型表示组织行为空间，便于上层组合与选择技能。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 原型表示组织行为空间，便于上层组合与选择技能。
- 在 [BFM 41 篇技术地图](../overview/bfm-41-papers-technology-map.md) 中属于 **03 Intrinsic reward 预训练**（#27/41）。

## 流程总览

```mermaid
flowchart TD
    subgraph P["任务无关预训练阶段（500k 环境交互）"]
        OBS["增强后的图像帧 x_t, x_t+1<br/>（随机平移增强，3 帧堆叠）"] --> ON["在线网络：卷积编码器 f_θ → y_t<br/>投影器 g_θ → z_t；预测器 v_θ → u_t"]
        ON --> PRO["与 M=512 个原型 c_i 做 softmax → p_t"]
        OBS --> TG["目标网络 f_ξ, g_ξ（在线网络的 EMA）<br/>编码 x_t+1 → z_t+1"]
        TG --> SK["Sinkhorn-Knopp 聚类 → 软分配目标 q_t+1"]
        PRO --> LOSS["交叉熵损失 L_SSL(p_t, q_t+1)<br/>只更新在线网络与原型"]
        SK --> LOSS
        TG --> Q["原型聚类重平衡：候选 z 按原型分簇后均匀采样，存入队列 Q"]
        Q --> RI["熵内在奖励 r̂_t = ‖z_t+1 − 第 k 近邻（Q 中）‖"]
        RI --> SAC["探索智能体 SAC：在 (y_t, a_t, r̂_t, y_t+1) 上训练<br/>RL 损失梯度不回传编码器"]
        SAC -- "新交互数据" --> OBS
    end
    subgraph D["下游任务阶段（500k 交互）"]
        FZ["冻结编码器 f_θ 与原型"] --> RL["基于状态的标准 RL（SAC）<br/>奖励 r_t + α·r̂_t"]
    end
    LOSS --> FZ
    RI -. "下游探索奖励" .-> RL
```

图依据：论文 §4.1 原型表示学习、§4.2 最大熵探索、§4.3 任务无关预训练、§4.4 下游任务应用与 Fig.2 / Fig.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 27/41 |
| 分组 | 03 Intrinsic reward 预训练 |
| 出处 | 2021 · ICML |
| 论文 | <https://arxiv.org/abs/2102.11271> |
- **代码/项目：** <https://github.com/denisyarats/proto>

## 核心机制（归纳）

### 1）策展导读要点

无明确外部任务时，用 **intrinsic reward**（探索、多样性、后继态等）预训练身体策略。

### 2）策展导读要点

为后续 goal-conditioned 或 imitation 提供 **可迁移的探索先验**，降低冷启动样本需求。

## 结论

**Proto-RL 属于 BFM 栈的「冷启动」层：它不负责完成任务，而是用原型表示把无外部奖励阶段探索到的行为组织成可检索的空间，供下游 goal-conditioned 或模仿阶段复用。**

- 起作用的是 **intrinsic reward**（探索、多样性、后继态等）驱动的无任务预训练，产出可迁移的探索先验，从而降低下游冷启动的样本需求。
- 边界很硬：intrinsic 预训练不替代任务奖励，它提供的是 **探索覆盖**，下游仍需任务信号或示范对齐才能定向到具体行为。
- 本页为清单坐标（awesome-bfm #27/41，**03 Intrinsic reward 预训练**，2021 · ICML，代码 <https://github.com/denisyarats/proto>）；量化 benchmark、消融与实机指标以原文 PDF 为准。

## 常见误区

1. Intrinsic 预训练不替代任务奖励；它提供 **探索覆盖**，下游仍需任务或示范对齐。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)
- BFM 概念：[behavior-foundation-model.md](../concepts/behavior-foundation-model.md)
- 原始 source：[bfm_awesome_proto_rl_icml_2021.md](../../sources/papers/bfm_awesome_proto_rl_icml_2021.md)

## 参考来源

- [bfm_awesome_proto_rl_icml_2021.md](../../sources/papers/bfm_awesome_proto_rl_icml_2021.md) — awesome-bfm 策展摘录
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 41+10 总表
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — 微信公众号编译导读
- 论文：<https://arxiv.org/abs/2102.11271>
- [arXiv PDF](https://arxiv.org/pdf/2102.11271)：已通读方法部分（实验与附录以略读为准）：§4 Proto-RL Algorithm（§4.1 至 §4.4）、§5.1 实验设定

## 推荐继续阅读

- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 完整列表与数据集表
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487) — TPAMI 2025 综述
