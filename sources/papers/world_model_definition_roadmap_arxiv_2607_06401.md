# A Definition and Roadmap for World Models（上海人工智能实验室视角文）

> 来源归档（ingest）

- **标题：** A Definition and Roadmap for World Models
- **类型：** paper（perspective）
- **机构：** Physical Intelligence Team，上海人工智能实验室（Shanghai AI Laboratory）
- **arXiv：** <https://arxiv.org/abs/2607.06401v1>（v1）
- **PDF：** <https://arxiv.org/pdf/2607.06401v1.pdf>
- **HTML：** <https://arxiv.org/html/2607.06401v1>
- **入库日期：** 2026-09-05
- **最后更新：** 2026-09-05
- **一句话说明：** 把世界模型定义为有限算力下对物理状态转移的压缩建模，并在 Fei-Fei 功能分类之上加表征轴，给出倒金字塔数据流与三阶段路线图。

## 为什么值得保留

- 明确批评功能分类 **只分类解码/投影、不定义内部模型**：同一压缩表征可按查询接口解成像素、状态或动作。
- 给出可操作的 **二维 taxonomy**（功能 × 架构）以及「WAM 不是第四列」的定位，和本库 [WAM](../../wiki/concepts/world-action-models.md) 综述对齐。
- 数据侧把「天花板由物理经验多样性决定」写成 **倒金字塔**（互联网视频 → 过滤/合成 → 少量真机任务数据），可直接对照具身数据金字塔讨论。
- 路线图把「统一多模态 → 统一物理表征 → 基础规模交互仿真器」写成阶段，而不是产品口号。

## 核心摘录（面向 wiki 编译）

### 1) 定义与三条性质

**Definition 2.1：** 世界模型是在 **有限计算资源** 约束下，对物理世界 **状态转移过程** 的压缩建模。文内与「物理世界模型」互换；**不等于** 交互/3D 一致视频生成器。

三条性质：

| 性质 | 含义 |
|------|------|
| **Omnimodal workscope** | 全模态统一潜表征，不只文本/视觉 |
| **Multidimensional Asynchronicity** | 多通道、不同采样频率的异步序列 |
| **Locality** | 局部观测 + 外部干预 → POMDP |

认识论问题从「接下来会发生什么」扩成「正在发生什么、为什么、接下来会怎样」。物理系统非平稳，静态语料可能永远不见安全关键故障。

### 2) 理解优先于预测

- **Understanding-oriented：** 压缩出实体/关系/机制；预测主要是训练信号。
- **Prediction-oriented：** 以可滚未来、可规划为准（Sora 式视频、LeCun 预测世界模型）。
- 文内立场：物理世界模型应以 **理解为先、预测服务理解**。好看 rollout 若丢隐状态/因果/干预语义，仍不是控制用仿真器。

### 3) 二维 taxonomy（功能 × 架构）

功能轴沿用 Fei-Fei（2026）：Renderer / Simulator / Planner，并映射到 POMDP 的 \(O\) / \(P^\star\) / 决策。

架构轴：

| 列 | 预测底物 | 代表 |
|----|----------|------|
| Observation-level | 像素 / 视频 token | Sora、Seedance、Genie 3 |
| Latent-space | 紧凑状态转移 | JEPA、Dreamer、VLA-JEPA、V-JEPA 2-AC |
| 3D / structured | 几何、物体、占用、splat | Marble、OccWorld、GWM |

- **WAM 不是第四实现类**：跨架构的功能范式，承诺「预测状态 ↔ 生成动作」耦合；功能轴上偏 Planner，但通常横跨 Simulator。
- Cosmos 3 被写成 **同一骨干的多种 I/O 配置**，不是单点类别。

### 4) 倒金字塔数据流

前提：固定架构与算力时，**数据多样性** 决定物理泛化天花板；架构/算力只决定逼近速度。

1. 顶层：开放互联网视频（隐式编码物体恒存、刚/软体、运动学、遮挡因果、人类动作先验）。
2. 中层：自动过滤 + 合成，抽出可标准化的动作/交互。
3. 底层：少量任务对齐真机数据，用于具身微调。

核心任务被写成 **信息论压缩**：保留决策相关因果/物理，丢掉光度 nuisance。生成与仿真是好表征的下游能力，不是目标本身。

### 5) 训练与失败模式（MBRL 遗产）

- 背景规划 vs 决策时规划；三种复发失败：**compounding error**、**objective mismatch**、**optimism / 模型剥削**。
- WAM vs 解耦 MBRL：WAM 吃互联网视频先验、难隔离动力学误差与策略误差；MBRL 可换模型、可标定不确定性。
- Chain-of-Imagination：推理在动作条件动力学空间里展开，不是自然语言 CoT。
- 反事实要求 **same-world constraint**（只改决策、不改质量/身份/光照）。

### 6) 机器人三角色 + 身体先验

机器人侧把世界模型收成：**Data Engine / Environment Simulator / Action Planner**；并单独提出 **Embodiment World Model / BFM**（身体能做什么）与外部世界先验互补。

### 7) 三阶段路线图

1. 统一多模态（外观、3D、状态、动作、长程推理一起相关）。
2. **一个内部物理状态、多种解码**（渲染 / 仿真 / 规划都是 query）。
3. 基础规模交互仿真器：可闭环验证，不只视觉可信。

Outlook 的 Trinity：**Agent / Evaluator / World Model**（世界模型兼课程设计师，提出刚超出当前能力的任务）。

### 8) 开放瓶颈

数据不对称（Renderer 有视频、Simulator/Planner 缺可执行 3D 与动作轨迹）、保真 vs 精度、长程误差累积、Sim2Real、评测碎片化、安全/透明/可持续（含联邦世界模型与自动化偏见）。

科学域与有界社会系统只作范围延拓；大规模社会预测被明确排除出「世界模型」称号。

## 开源边界（步骤 2.5，2026-09-05）

- arXiv abs / HTML **无** 项目页、GitHub、权重或数据链接。
- 视角文，无实验代码承诺。
- 结论：**确认未开源**。源码运行时序图不适用。

## 对 wiki 的映射

- 升格：[paper-sa-2607-06401-a-definition-and-roadmap-for-world-models](../../wiki/entities/paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md)
- 功能轴概念：[世界模型功能分类](../../wiki/concepts/functional-taxonomy-world-models.md)
- 交叉：[生成式世界模型](../../wiki/methods/generative-world-models.md)、[WAM](../../wiki/concepts/world-action-models.md)、[训练闭环三线](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)、[具身数据金字塔](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md)、[Cosmos 3](../../wiki/entities/cosmos-3.md)、[Marble](../../wiki/entities/marble-world-model.md)
