---
type: overview
tags: [hub, embodied-foundation-model, vlm, vln, vla, vlx, world-model, taxonomy]
status: complete
updated: 2026-11-14
original:
  - ../../sources/papers/paligemma_arxiv_2407_07726.md
  - ../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md
  - ../../sources/papers/gr00t_n1_arxiv_2503_14734.md
  - ../../sources/papers/wam_survey_reading_arxiv_2605_12090.md
  - ../../sources/papers/bfm_survey_reading_arxiv_2506_20487_btm41.md
related:
  - ../concepts/embodied-foundation-model-hardware-codesign.md
  - ../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md
  - ../concepts/3d-spatial-vqa.md
  - ../concepts/foundation-policy.md
  - ../concepts/world-action-models.md
  - ../concepts/latent-imagination.md
  - ../concepts/behavior-tree-vla-orchestration.md
  - ../concepts/humanoid-policy-network-architecture.md
  - ../concepts/visual-representation-for-policy.md
  - ../concepts/hierarchical-quadruped-navigation-stack.md
  - ../comparisons/humannet-table1-human-video-corpora.md
  - ./hub-vla.md
sources:
  - ../../sources/blogs/wechat_shenlan_five_embodied_model_taxonomy.md
summary: "具身大模型分类学选型闭环知识链枢纽：把 VLM 感知理解 → VLN 空间导航 → VLA 动作执行 → VLX 一体化扩展 → 世界模型时序推演五大家族沉淀为一条贯通的选型链，统一各家族的 I/O 边界、数据需求、泛化能力与实时性取舍入口。"
---

# 具身大模型分类学选型闭环（知识链汇总）

> **知识链定位**：本页是「感知理解 → 空间导航 → 动作执行 → 一体化扩展 → 世界模型推演」五层具身大模型家族的统一入口，把近周密集 ingest 的 VLM / VLN / VLA / VLX / World-Model 五大家族从分散的实体/方法/对比页收拢为一条可导航的选型链。

## 一句话定义

**具身大模型分类学选型闭环** 指按 **跨模态感知（VLM）→ 空间导航（VLN）→ 动作执行（VLA）→ 一体化多任务扩展（VLX）→ 世界模型时序推演（WM）** 逐层分工的家族谱系，各层共享 Transformer 与多模态编码底座，但在 I/O 边界、数据需求、泛化能力与实时性上各有取舍，需按任务组合选型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 跨模态感知理解骨干（图文对齐、空间 VQA） |
| VLN | Vision-Language Navigation | 语言指令驱动的空间导航 |
| VLA | Vision-Language-Action | 视觉-语言-动作统一策略，直接对接硬件 |
| VLX | Vision-Language-X | 一体化多任务扩展（X 为可插拔模态/技能头） |
| WM | World Model | 时序虚拟预演，与 VLA 形成决策–预演闭环 |
| WAM | World-Action Model | 把世界模型预测与动作生成合一的家族 |

## 为什么重要

- **补一条贯通的选型视角**：仓库已有各家族的实体/方法/对比页，但缺「从感知到推演逐层如何分工、各家族边界与取舍」的统一决策入口。
- **暴露家族间取舍矛盾**：端到端 VLA vs 分层 VLN、显式世界模型预测 vs 无模型反应式、大模型泛化 vs 实时控制带宽、统一 VLX vs 专精分立——这些矛盾只有并置在一条链上才看得清。
- **与人形产业叙事同向**：「大模型高层 + 实时低层控制」的分层部署，正是五层闭环在硬件上的落地形态；为何大模型公司仍自研本体见 [具身大模型与本体协同设计](../concepts/embodied-foundation-model-hardware-codesign.md)。

## 五层选型闭环

| 层次 | 家族 | 典型职责 | 站内入口 |
|------|------|----------|----------|
| 感知理解 | VLM | 图文对齐、3D 空间 VQA | [3D 空间 VQA](../concepts/3d-spatial-vqa.md)、[策略视觉表征](../concepts/visual-representation-for-policy.md) |
| 空间导航 | VLN | 语言指令驱动导航 | [分层四足导航栈](../concepts/hierarchical-quadruped-navigation-stack.md) |
| 动作执行 | VLA | 视觉-语言-动作统一策略 | [VLA 知识链](./hub-vla.md)、[基础策略](../concepts/foundation-policy.md)、[人形策略网络架构](../concepts/humanoid-policy-network-architecture.md) |
| 一体化扩展 | VLX | 多任务/多模态一体化 | [行为树 × VLA 编排](../concepts/behavior-tree-vla-orchestration.md) |
| 世界模型推演 | WM / WAM | 时序预演与决策闭环 | [World Action Models](../concepts/world-action-models.md)、[潜空间想象](../concepts/latent-imagination.md) |
| 数据底座 | 跨层 | 人类视频语料等训练数据 | [HumanNet Table1 人类视频语料](../comparisons/humannet-table1-human-video-corpora.md) |

## 家族选型的关键取舍

- **端到端 vs 分层**：VLA 端到端泛化强但闭环稳定性与可解释性弱；VLN 分层栈稳定可控但泛化受限于模块接口。
- **显式世界模型 vs 无模型反应式**：WM/WAM 用预演换取长程规划能力，代价是推理延迟；VLA 反应式实时性好但缺长程预测。
- **大模型泛化 vs 实时控制带宽**：家族越靠上游（VLM/VLX）泛化越强、频率越低，越靠执行端（VLA + 低层控制器）越要压延迟。
- **统一 VLX vs 专精分立**：一体化模型省接口成本但单点能力上限受限；专精分立各自最优但需编排（如行为树）粘合。

## 与其他知识链的关系

- **[VLA 与基础策略知识链](./hub-vla.md)**：执行层的开源谱系与 BFM 身体接口。
- **[五大具身模型分类对比](../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md)**：本闭环的家族边界与递进关系原始对比。
- **[具身大模型与本体协同设计](../concepts/embodied-foundation-model-hardware-codesign.md)**：模型族选型之外的硬件侧约束（数据闭环、分层频率、安全边界）。

## 核心原理

五层具身大模型家族并非彼此独立的并行流派，而是沿**语义理解 → 空间接地 → 连续控制**逐层压缩与解耦的同一条信息链路：

1. **输入与表征共享，输出与带宽分化**：上游 VLM 将图像与语言指令投影到共享语义空间；VLN 把语义对齐到拓扑/几何路点；VLA 在语言–视觉表征之上接入动作头（离散 action token 或连续 flow/diffusion 专家）；VLX 扩展到力、触觉、音频或工具调用等异构输出头。
2. **显式世界模型作为前瞻层**：WM/WAM 在潜空间或像素空间预演动作后果，可与 VLA 级联排序候选轨迹，也可通过联合建模蒸馏给反应式执行头，把长程因果推理与高频闭环分离。
3. **跨层协同设计**：越靠近硬件执行，对延迟、动作维度和实体几何的约束越强，因此高泛化语义模型通常与专用低层控制器或行动基础模型（BFM）分层部署。

## 工程实践

- **按控制频率划分边界**：先测量任务所需的闭环频率（例如高频接触力控与全身平衡 vs 1–5 Hz 语义规划），将重参数 VLM/VLA 放在低频决策层，高频执行下沉给动作专家或 WBC。
- **优先复用统一评测与数据基础设施**：在进入全量微调前，对照 [HumanNet Table1 人类视频语料](../comparisons/humannet-table1-human-video-corpora.md) 与 [VLM/VLN/VLA/VLX/WM 分类对比](../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) 明确任务缺少的是语义先验、空间感知还是物理接触数据。
- **增量演进栈**：工程落地通常按「预训练 VLM/策略骨架 → 领域示教微调 → 行为树/安全壳编排 → 必要时引入 WM 校验」渐进叠加，避免一步到位引入过高推理开销。

## 局限与风险

- **分类边界在最新工作上持续模糊**：部分模型（如统一 WAM 或多任务 VLX）同时输出未来预测与连续动作，表格分层仅作职责划分而非绝对排他分类。
- **跨家族接口标准尚不统一**：不同家族在动作空间（关节位控 vs 末端增量）、历史窗口长度与异步 chunk 协议上差异显著，直接拼接易因坐标系与延迟失配失效。
- **上层泛化不自动等价于真机鲁棒**：互联网图文预训练无法替补真实物理交互中的接触不确定性与长尾故障恢复能力。

## 关联页面

- [宅码 Common Know-How（二手知识点汇编）](../entities/zhaima-common-know-how.md) — VLA 与机器人学习知识点的二手索引，结论需回一手论文核对
- [Spotter](../entities/paper-spotter.md) — 具身模型主导、VLM 只为确认错误介入的失败修复层

- [ERR（Replan for Error Recovery）](../entities/paper-err-replan-error-recovery.md) — 具身 VLM 在线错误恢复与 turn 级 RL
- [EVO-WAM](../entities/paper-evo-wam.md) — 世界动作模型通过视频-动作验证自我演化
- [VLM/VLN/VLA/VLX/世界模型分类对比](../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md)
- [3D 空间 VQA](../concepts/3d-spatial-vqa.md)
- [基础策略（Foundation Policy）](../concepts/foundation-policy.md)
- [World Action Models](../concepts/world-action-models.md)
- [潜空间想象（Latent Imagination）](../concepts/latent-imagination.md)
- [行为树 × VLA 编排](../concepts/behavior-tree-vla-orchestration.md)
- [人形策略网络架构](../concepts/humanoid-policy-network-architecture.md)
- [策略视觉表征](../concepts/visual-representation-for-policy.md)
- [分层四足导航栈](../concepts/hierarchical-quadruped-navigation-stack.md)
- [HumanNet Table1 人类视频语料](../comparisons/humannet-table1-human-video-corpora.md)
- [具身大模型与本体协同设计](../concepts/embodied-foundation-model-hardware-codesign.md)

## 参考来源

- [wechat_shenlan_five_embodied_model_taxonomy.md](../../sources/blogs/wechat_shenlan_five_embodied_model_taxonomy.md) — 深蓝具身智能《五大具身模型详解：VLM、VLA、VLN、VLX、世界模型》
- [PaliGemma 原文归档（arXiv:2407.07726）](../../sources/papers/paligemma_arxiv_2407_07726.md) — 开放 VLM 视觉–语言骨干基础
- [RT-2 原文归档（arXiv:2307.15818）](../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md) — VLA 概念确立与 co-fine-tuning 配方
- [GR00T N1 原文归档（arXiv:2503.14734）](../../sources/papers/gr00t_n1_arxiv_2503_14734.md) — 双系统推理–动作人形基础模型
- [WAM 综述原文归档（arXiv:2605.12090）](../../sources/papers/wam_survey_reading_arxiv_2605_12090.md) — 世界动作模型推演与策略联合建模
- [BFM 综述原文归档（arXiv:2506.20487）](../../sources/papers/bfm_survey_reading_arxiv_2506_20487_btm41.md) — 行为基础模型与身体级控制边界
- 本页归纳自 [五大具身模型分类对比](../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) 及各家族概念/方法页

## 推荐继续阅读

- [具身大模型分类学选型闭环 Query](../queries/embodied-fm-taxonomy-loop.md) — 按闭环选型问题逐层定位对应家族页
- [VLA 与基础策略知识链](./hub-vla.md) — 深入操作与开源 VLA 谱系
- [具身模型实时性 ↔ 泛化能力取舍](../concepts/embodied-fm-latency-generalization-tradeoff.md) — 带宽与推理时延约束分析
