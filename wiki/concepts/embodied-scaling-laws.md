---
type: concept
tags: [scaling-laws, data-engine, foundation-policy, machine-learning]
status: complete
updated: 2026-08-27
related:
  - ./bitter-lesson.md
  - ./open-x-embodiment.md
  - ../entities/paper-from-agi-to-asi.md
  - ../entities/paper-rynnbrain-1-1.md
  - ../entities/generalist-gen15-one-shot.md
  - ../entities/skild-s1.md
  - ../entities/generalist-gen1-thousand-hands.md
  - ../entities/dyna-2.md
  - ../entities/perceptron-isaac-05.md
  - ../methods/vla.md
  - ../methods/egoscale.md
  - ../concepts/world-action-models.md
  - ../methods/auto-labeling-pipelines.md
  - ../formalizations/foundation-policy-alignment.md
sources:
  - ../../sources/blogs/sutton_bitter_lesson.md
  - ../../sources/blogs/generalist_gen15_one_shot.md
  - ../../sources/blogs/generalist_thousand_hands.md
  - ../../sources/blogs/dyna_2_million_hour_wam.md
  - ../../sources/papers/rl_foundation_models.md
  - ../../sources/papers/egoscale_arxiv_2602_16710.md
  - ../../sources/papers/rynnbrain_1_1_arxiv_2607_17977.md
  - ../../sources/papers/data_pyramid_embodied_manipulation_arxiv_2607_24744.md
  - ../../sources/blogs/skild_s1_in_context_learning.md
  - ../../sources/blogs/perceptron_isaac_05.md
summary: "具身规模法则（Embodied Scaling Laws）探讨了具身智能模型中数据规模、模型参数量与下游任务泛化能力之间的幂律关系；含 EgoScale（~20k h）、Dyna-2（1M h 人→机跨具身）与 Perceptron Isaac 0.5（1M h 视频置换 teleop）等案例。"
---

# Embodied Scaling Laws (具身规模法则)

**具身规模法则**：在机器人学习中，随着训练数据（演示轨迹、仿真经验）、模型参数量和计算资源的增加，模型在未见任务、未见物体和未见环境上的表现呈现出可预测的性能提升趋势（通常遵循幂律分布）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| RT-2 | Robotics Transformer 2 | 将 web 规模 VLM 能力迁移到机器人控制的代表工作 |

## 核心观察

在 NLP 和 CV 领域，Scaling Laws 已经得到了充分验证（如 GPT-4, Llama）。在机器人领域，**Open X-Embodiment** 等项目的研究表明，类似的规律同样存在：

1. **跨形态泛化**：在大规模混合数据集（来自不同机器人形态）上训练的模型，其表现优于仅在单一形态数据上训练的模型。
2. **数据多样性 vs 质量**：对于基础策略模型（Foundation Policies），数据的**多样性**（多样化的环境、光照、物体）往往比单一任务的高精度演示更重要。
3. **涌现能力**：当数据规模跨越某个临界点时，模型开始展现出零样本（Zero-shot）逻辑推理能力（如“拿起与水果颜色相同的方块”）。
4. **人侧视频监督的可预测缩放（案例）**：EgoScale 在 **1k–20k 小时** egocentric 人操纵轨迹上报告 **验证损失与数据规模近 log-linear**，并与 **真机灵巧后训练表现** 强相关（见 [EgoScale](../methods/egoscale.md)）；可与机器人日志 scaling 对照阅读，但 **指标域与任务族** 并不自动等价。
5. **人→机跨具身缩放（产业案例）**：[Dyna-2](../entities/dyna-2.md) 将人视频梯子推到 **1k–1M 小时**，在 **预训练零机器人数据、零对齐适配** 设定下报告 **零样本机器人离线指标** 与后训练真机归一化均值随小时数单调上升，并主张 **视频共训** 是跨具身幂律出现的必要条件；**闭源自报**，协议定义与 EgoScale（含对齐 mid-training）不同，宜对照读而非直接合并曲线。
6. **非均匀具身 scaling（案例）**：[RynnBrain 1.1](../entities/paper-rynnbrain-1-1.md) 在统一配方下对比 matched **Qwen3.5（2B→122B）**：一般认知双方随规模上升；**推理密集型认知** 上 RynnBrain 上升而 Qwen3.5 **负缩放**；**定位** 上最大 Qwen 仍低于最小 RynnBrain——说明 **显式空间/具身监督** 与 **参数缩放** 互补而非可替代。
7. **末端接口多样性（产业案例）**：[GEN-1 千手](../entities/generalist-gen1-thousand-hands.md) 主张在 **>50 万小时** 交互与 **~9k 末端变体** 上预训练同一基座，用 task-vector 权重更新度量「新手」新颖度；属 **闭源自报**，作多样性轴对照而非可复现定律拟合。
8. **预训练时长与适应成本（产业案例）**：[GEN-1.5](../entities/generalist-gen15-one-shot.md) 在 **8+ 月** 持续预训练后报告 **无显式 ICL 训练** 的 one-shot physical prompting 与 **1–10 梯度步** 适应；作者主张更多预训练使新任务适应趋近「可忽略」——**闭源自报**，与 EgoScale / Dyna-2 的指标域不同。
9. **ICL vs 语言 prompt（产业案例）**：[S1](../entities/skild-s1.md) 在同一数据/架构/算力下把预训练从 1k 推到 **100k 小时**：已见任务小数据语言 VLA 更好，未见任务上 ICL **66%** vs 语言 **9%**（约 7×）。读法是 **任务指定通道** 改变 scaling 斜率，不是另一条参数幂律；**闭源自报**。

## 宏观算力背景（与具身 scaling 的层级差）

DeepMind 技术报告 [*From AGI to ASI*](../entities/paper-from-agi-to-asi.md)（arXiv:2606.12683）在 **后 AGI** 情景下讨论 **有效算力 ~10×/年** 的历史复合增长，并强调 **并行实例、测试时算力与多智能体集体** 可能在单模型平台期后仍推高系统级能力。该叙事针对 **通用认知栈**，不替代本页的 **轨迹/参数/任务** 幂律；二者可对照阅读：宏观算力决定 **仿真集群、数据工厂与多机 fleet** 的上限，微观 embodied scaling 决定 **单位算力能换多少泛化**。

## 机器人领域的特殊挑战

不同于互联网文本，具身数据的 Scaling 面临物理瓶颈：

- **数据稀缺性**：真实机器人轨迹采集成本极高。
- **维度灾难**：机器人动作空间复杂，且存在时序强耦合。
- **Sim2Real 效率**：仿真数据虽然易于 Scaling，但其多样性受限于物理引擎的建模能力。

## 解决路径：数据引擎 (Data Engine)

为了满足 Scaling Laws 的需求，行业正在从“人工采集”转向“**自动数据工厂**”：
- **生成式增强**：利用 [Generative Data Augmentation](../methods/generative-data-augmentation.md) 扩充长尾数据。
- **自动标注**：利用 VLM 自动为原始轨迹添加语义标签。
- **基础模型引导**：利用已有的 VLA 模型作为专家，在仿真中自动收集海量负样本。
- **视频置换遥操作（开源对照）：** [Perceptron Isaac 0.5](../entities/perceptron-isaac-05.md) 在固定 80:30:30 通用:ego:UMI 混合物上报告：通用视频从 1k h 升到 1M h，达到 held-out 动作损失 2.50 所需 teleop 从约 **5.9k h → 28 h**（**210×**）。与 [Dyna-2](../entities/dyna-2.md) 的闭源 1M h ego 缩放对照：Isaac 给的是 **开源代码 + teleop 置换等高线**，不是人→机零样本 WAM 幂律。

## 关联页面
- [The Bitter Lesson](./bitter-lesson.md) — 宏观方法论：可扩展 search/learning vs 人类先验
- [具身数据金字塔综述](../entities/paper-data-pyramid-embodied-manipulation.md) — 五层数据生态 × 六维属性的类目级坐标系；数据配方的「该扩哪一层」决策入口
- [VLA (Vision-Language-Action Models)](../methods/vla.md)
- [RynnBrain 1.1](../entities/paper-rynnbrain-1-1.md) — 统一配方下相对 Qwen3.5 的非均匀具身 scaling
- [EgoScale（人视频规模预训练 VLA）](../methods/egoscale.md)
- [Dyna-2（百万小时 WAM 跨具身缩放）](../entities/dyna-2.md) — 闭源 1M h 人→机缩放主张
- [Perceptron Isaac 0.5](../entities/perceptron-isaac-05.md) — 开源 1M h 通用视频置换 teleop（210×；权重入库日未齐）
- [World Action Models](./world-action-models.md) — Dyna-2 所属 Joint WAM 族谱
- [GEN-1.5 一次示范学习](../entities/generalist-gen15-one-shot.md) — 预训练规模与 one-shot / 极少步适应
- [S1（Skild）](../entities/skild-s1.md) — ICL vs 语言条件 VLA 的未见任务 scaling 分叉
- [GEN-1 千手（跨末端多样性）](../entities/generalist-gen1-thousand-hands.md) — 闭源产业多样性轴对照
- [自动化标注流水线](../methods/auto-labeling-pipelines.md)
- [基础策略对齐](../formalizations/foundation-policy-alignment.md)

## 参考来源
- [The Bitter Lesson 原始资料](../../sources/blogs/sutton_bitter_lesson.md) — Sutton 2019 scaling 方法论（宏观层对照）
- [Dyna-2 研究长文摘录](../../sources/blogs/dyna_2_million_hour_wam.md)
- Padalkar, A., et al. (2023). *Open X-Embodiment: Robotic Learning at Scale*.
- Brohan, A., et al. (2023). *RT-2: Vision-Language-Action Models Transfer Knowledge from Web to Robots*.
- [EgoScale 论文摘录（arXiv:2602.16710）](../../sources/papers/egoscale_arxiv_2602_16710.md) — 人视频小时数与验证损失 / 真机表现联动的案例材料
- [From AGI to ASI 论文摘录（arXiv:2606.12683）](../../sources/papers/agi_to_asi_arxiv_2606_12683.md) — 有效算力与集体 scaling 的宏观框架
- [RynnBrain 1.1 论文摘录（arXiv:2607.17977）](../../sources/papers/rynnbrain_1_1_arxiv_2607_17977.md) — 具身 vs 通用 VLM 的非均匀 scaling 案例
- [GEN-1.5 博客归档](../../sources/blogs/generalist_gen15_one_shot.md) — one-shot physical prompting 产业样本
- [S1 博客归档](../../sources/blogs/skild_s1_in_context_learning.md) — ICL vs 语言 prompt 的内部 scaling 对照
- [GEN-1 千手博客归档](../../sources/blogs/generalist_thousand_hands.md) — 多末端多样性 scaling 产业样本
- [Isaac 0.5 博客归档](../../sources/blogs/perceptron_isaac_05.md) — 无动作视频小时置换 teleop 的开源对照
