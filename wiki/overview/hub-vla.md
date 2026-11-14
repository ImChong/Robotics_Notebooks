---
type: overview
tags: [hub, hub-vla, vision-language-action, foundation, manipulation]
status: complete
updated: 2026-11-14
original:
  - ../../sources/papers/rt_1_arxiv_2212_06817.md
  - ../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md
  - ../../sources/papers/openvla_reading_arxiv_2406_09246.md
  - ../../sources/papers/pi0_arxiv_2410_24164.md
  - ../../sources/papers/vla_action_tokenization_survey_arxiv_2507_01925.md
summary: "VLA 与基础策略知识链汇总：视觉-语言-动作统一建模、OpenVLA/π0/GR00T 等开源谱系，以及 BFM 身体接口与 loco-manip 任务接口；含社区多基准榜入口与驾驶 VLA（S²-VLA）锚点。"
---

# VLA 与基础策略（知识链汇总）

> **知识链汇总**：本页是相关概念/方法的统一入口；对应策展纵深见图谱 [路线视图](../../docs/graph.html?depth=vla) 与 [路线页](../../roadmap/depth-vla.md)。

## 一句话定义

**VLA（Vision-Language-Action）** 把 **视觉观测、自然语言指令与机器人动作** 统一到同一策略或基础模型中，面向多任务操作与 loco-manip 的「一个模型多种技能」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| BFM | Behavior Foundation Model | 可复用/可调用的身体行为基座 |
| VLM | Vision-Language Model | 视觉-语言预训练骨干 |
| RT | Robotics Transformer | 早期 transformer 机器人策略代表 |
| OXE | Open X-Embodiment | 跨具身开源数据集倡议 |

## 为什么重要

- **降低任务专用策略成本**：语言与视觉提供泛化接口。
- **与人形产业叙事同向**：「运控基座 + 任务头」分层部署。
- **与 WBT / 抓取 / Sim2Real 交叉**：VLA 常作为高层，WBC/低层控制负责执行。

## 本知识链覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 对比 | 五大模型分类 | [VLM/VLN/VLA/VLX/WM 分类](../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) |
| 方法 | VLA 定义与路线 | [VLA](../methods/vla.md) |
| 总览 | 决策骨干在六支架构中的位置 | [AI 架构地图](./ai-architecture-map.md) |
| 概念 | 行为基础模型 | [Behavior Foundation Model](../concepts/behavior-foundation-model.md) |
| 概念 | Foundation Policy | [Foundation Policy](../concepts/foundation-policy.md) |
| 地图 | BFM 41 篇技术地图 | [BFM 技术地图](./bfm-41-papers-technology-map.md) |
| 地图 | VLA/WM 14 篇阅读路线 | [CLIP→RT→OpenVLA→π₀ + 四篇 WM](./vla-wm-reading-roadmap-14-papers-technology-map.md) |
| 概念 | BT 编排 VLA 部署 | [行为树 × VLA 编排](../concepts/behavior-tree-vla-orchestration.md) |
| 实体 | ROBOTIS Physical AI 栈 | [Cyclo Intelligence](../entities/cyclo-intelligence.md) |
| 榜站 | VLA / 灵巧手多基准相对位次 | [VLA SOTA Leaderboard](../entities/vla-sota-leaderboard.md) |
| 实体 | Perceptron 开源通才（非 NVIDIA） | [Isaac 0.5](../entities/perceptron-isaac-05.md) |
| 实体 | 相机系锚点几何通才 | [UCAG-P](../entities/paper-ucag-p.md) — 人手与机器人共享 \(p_0/p_1\)；代码待发布 |
| 实体 | 相机可配置的多视角 VLA | [VersaCamVLA](../entities/paper-versacamvla.md) — 可变相机集合编码为固定大小场景 token；代码待发布 |
| 驾驶 VLA | 语义∥空间双流规划（NAVSIM） | [S²-VLA](../entities/paper-s-squared-vla.md) |
| 数据 | 跨具身数据倡议 | [Open X-Embodiment](../concepts/open-x-embodiment.md) |

## 与其他知识链的关系

- **[IL/RL](./hub-learning.md)**：VLA 训练常混合 IL 与 RLHF/微调。
- **[视觉骨干](./hub-vision-backbone.md)**：感知表征质量影响 VLA 上限。
- **[WBT](./hub-wbt.md)**：全身技能与 VLA 任务接口的分层。

## 核心原理

VLA 知识链贯穿的核心共性是**把预训练视觉–语言表征接入连续或离散动作空间**：

1. **离散动作分词路线（RT-1 → RT-2 → OpenVLA → FAST）**：把连续末端或关节轨迹量离散为 token 或频域系数，复用 VLM 自回归解码与图文–动作联合微调。
2. **连续生成动作专家路线（π₀ / Flow Matching / Diffusion）**：保留预训练 VLM 的语义骨干，将高频多峰连续控制交由单独或 MoT 耦合的扩散/流匹配动作专家，在 20–50 Hz chunk 输出中兼顾灵巧性与多步一致性。
3. **跨层解耦路线（VLA + BFM / 行为树）**：在人形或移动操作场景中，由 VLA 输出任务语义、关键点或子目标，由行为基础模型（BFM）或行为树调度底层全身控制闭环。

## 工程实践

- **入门与复现优先级**：新手推荐按 `RT-1 / RT-2（建概念）→ OpenVLA / SmolVLA（跑通 7B/轻量推理与 LoRA 微调）→ π₀ / OpenPI（理解流匹配动作专家）` 递进，结合 [VLA Open-Source Landscape 2025](./vla-open-source-repro-landscape-2025.md) 选择适配单卡或真机的仓。
- **部署必须核对的三项接口**：动作坐标系与归一化统计（`norm_stats`）、视觉观测历史与动作 chunk 异步节拍、以及与底层伺服/WBC 的安全限幅边界。
- **微调经验**：在小样本场景下优先冻结或低秩适配 VLM 骨干并重点校对相机视角分布；对语言改写敏感的任务需关注语义–动作解耦与指令增强。

## 局限与风险

- **跨本体迁移不自动成立**：即便在 Open X-Embodiment 上混训，末端空间、相机外参与动力学差异仍使零样本迁移严重受限于硬件一致性。
- **高频接触任务对推理延迟极其敏感**：大型 7B+ 自回归 VLA 难以直接进入高频力控内环，盲目追求更大骨干反而恶化动态抓握与干扰恢复。
- **基准分数不可横向直比**：SIMPLER、LIBERO、CALVIN 与自建真机协议在重置、随机化与视角上差异极大，跨工作对照需回看原始实验设置。

## 关联页面

- [VLA / 世界模型 14 篇阅读路线](./vla-wm-reading-roadmap-14-papers-technology-map.md)
- [统一机器人学习综述](../entities/paper-unified-robot-learning-survey.md) — 表征–VLA–WM 耦合诊断（TMLR 2026）
- [VLA Open-Source Landscape 2025](./vla-open-source-repro-landscape-2025.md)
- [Perceptron Isaac 0.5](../entities/perceptron-isaac-05.md)
- [Whole-Body VLA 相关实体](../entities/paper-hrl-stack-30-wholebodyvla.md)
- [World Action Models](../concepts/world-action-models.md)
- [UCAG-P](../entities/paper-ucag-p.md) — 相机系腕/抓取锚点几何通才操作
- [AI 架构地图](./ai-architecture-map.md)

## 参考来源

- [RT-1 原文归档（arXiv:2212.06817）](../../sources/papers/rt_1_arxiv_2212_06817.md) — 规模化真机 Transformer 策略与离散动作 bin
- [RT-2 原文归档（arXiv:2307.15818）](../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md) — VLA 命名确立与图文–动作联合微调
- [OpenVLA 原文归档（arXiv:2406.09246）](../../sources/papers/openvla_reading_arxiv_2406_09246.md) — 7B 开源 VLA 与 OXE 跨本体训练配方
- [π0 原文归档（arXiv:2410.24164）](../../sources/papers/pi0_arxiv_2410_24164.md) — VLM + 流匹配动作专家通才架构
- [VLA 动作分词综述归档（arXiv:2507.01925）](../../sources/papers/vla_action_tokenization_survey_arxiv_2507_01925.md) — 离散 bin、FAST 与连续生成动作头统一分析
- 本库归纳自 [VLA](../methods/vla.md)、[Behavior Foundation Model](../concepts/behavior-foundation-model.md)、[BFM 技术地图](./bfm-41-papers-technology-map.md)
- [UCAG-P 论文摘录](../../sources/papers/ucag_p_arxiv_2608_26058.md) — 相机系锚点几何通才操作
- 知识链定义：[docs/depth-filters.js](../../docs/depth-filters.js)（`vla` 命中规则）
- 上游原始资料（本链概念页共同的 ingest 来源）：[BFM 综述（arXiv:2506.20487）](../../sources/papers/bfm_survey_arxiv_2506_20487.md)、[BFM：人形机器人行为基础模型（arXiv:2509.13780）](../../sources/papers/bfm_humanoid_arxiv_2509_13780.md)、[awesome-bfm-papers](../../sources/repos/awesome_bfm_papers.md)

## 推荐继续阅读

- [VLA 纵深路线](../../roadmap/depth-vla.md) — Stage 0–5 系统学习路径
- [VLA Open-Source Landscape 2025](./vla-open-source-repro-landscape-2025.md) — 按复现目标挑选开源仓与权重
- [操作 VLA 与视频-动作架构选型指南](../queries/manipulation-vla-architecture-selection.md) — 不同数据条件与控制任务下的选型决策树
