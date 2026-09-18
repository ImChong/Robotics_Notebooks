# ICI-VLA: In-Context Imitation with Spatiotemporally Aligned Demonstrations for Vision-Language-Action Models

> 来源归档（ingest）

- **标题：** ICI-VLA: In-Context Imitation with Spatiotemporally Aligned Demonstrations for Vision-Language-Action Models
- **短名：** ICI-VLA
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.07581>
- **PDF：** <https://arxiv.org/pdf/2609.07581>
- **作者：** Songhua Yang, Ziyu Liu, Xuetao Li, Ruqi Xiao, Kangxin Zhu, Miao Li（武汉大学 / 武汉大学工业科学研究院）
- **入库日期：** 2026-09-18
- **一句话说明：** 固定 text-action VLA（Qwen3-VL-4B）+ DTW 监督的 RD-Encoder 检索子任务级 micro-demonstration；Target Action Masking 抑制轨迹抄录；LIBERO 97.7%、RoboTwin 2.0 60.4%、真机四任务 83.2%，推理零梯度。

## 开源状态（步骤 2.5）

- **确认未开源（ICI-VLA 本体）**：截至入库日 arXiv **无独立项目页、无官方 GitHub / 权重链接**；GitHub 检索 `ICI-VLA` 无作者团队仓库；第一作者 GitHub（SupritYoung）未列出本文实现。
- **相关开源资源**：方法对照 **VLA-0**（text-action VLM 范式，[arXiv:2508.02062](https://arxiv.org/abs/2508.02062)）；骨干 **Qwen3-VL** / **Qwen3-VL-Embedding** 为公开模型族；基准 **LIBERO**、**RoboTwin 2.0** 为社区开源。

## 核心摘录（面向 wiki 编译）

### 摘录 1：固定 text-action VLA + 检索式 in-context 适应

- **问题：** 主流 VLA 新任务依赖额外梯度更新；朴素 ICL 把长轨迹塞进 prompt 易 **时序错位**（temporal aliasing）与 **语义–动力学错配**（semantic dynamic mismatch），示范变成误导性 action prefix。
- **范式：** 沿用 **VLA-0**「零结构修改」——连续动作与本体状态 **文本化**，Qwen3-VL-4B **无 action head**；离线训练 RD-Encoder 与策略，**推理时参数全冻结**，仅靠检索 micro-demonstration 作上下文。
- **Query：** \(I_q=\langle T, T_{\mathrm{sub}}, O_t, Z_t\rangle\)；视觉规划器把全局指令分解为有序子任务并选当前 \(T_{\mathrm{sub}}\)。

**对 wiki 的映射：** [paper-ici-vla-spatiotemporal-icl](../../wiki/entities/paper-ici-vla-spatiotemporal-icl.md)、[机器人 In-Context Learning](../../wiki/concepts/robot-in-context-learning.md)、[VLA](../../wiki/methods/vla.md)

### 摘录 2：micro-demonstration 库 + RD-Encoder（DTW 对比学习）

- **库构建：** ~11,200 条长轨迹（LIBERO、RoboTwin 2.0、~1,000 条双臂 Aloha 真机）→ Qwen3-VL 启发式分段标注 → **~139,659** 条子任务级 micro-demo \(E=\langle T, T_k, O, Z, A_{i:i+k}\rangle\)。
- **RD-Encoder：** 微调 **Qwen3-VL-Embedding-2B**；迭代 **语义 hard-filter**（top-64 候选）+ **DTW** 在标注轨迹上挖 positive / phase-misaligned hard negative；最多 5 轮直至 positive 变化 <5%。
- **推理：** 冻结 encoder 仅用可观测量 \(I_q\) 排序，**不** 对 query 动作算 DTW；默认检索 **3** 条示范。

**对 wiki 的映射：** [StellaVLA](../../wiki/entities/paper-stellavla-structured-icl-vla.md)（同为检索 ICL，表征为结构化语言 vs 本文 micro-demo + 几何对齐）

### 摘录 3：Target Action Masking 与主结果

- **Target Action Masking：** 训练时对 context 内 action token 随机 [MASK]，只监督 query 段未 mask 位置 → 降低 **直接抄录** 检索轨迹，强化当前观测 grounding。
- **LIBERO（Table 1）：** 平均 **97.7%**（Spatial 98.5 / Object 98.7 / Goal 98.0 / Long 96.8）；相对 VLA-0 **94.5**、朴素 ICL **71.5**。
- **RoboTwin 2.0：** Easy **72.4** / Hard **46.3** / Avg **60.4**；较报告最强基线均值（OpenVLA-OFT **41.1**）**+19.3 pp**；VLA-0 仅 **34.6**。
- **真机四任务（Aloha 双臂）：** 平均 **83.2%** [80.8, 85.4]；π₀ **66.4**、VLA-0 **63.0**。
- **消融（Table 2）：** 去 DTW → RoboTwin **31.4**；去语义过滤 → **38.1**；去 Masking（朴素 ICL）→ **10.7**。
- **检索质量：** Recall@1 **27.8→70.8**、Recall@5 **52.4→90.1**（五轮 RD-Encoder 训练）。

**对 wiki 的映射：** [manipulation](../../wiki/tasks/manipulation.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-ici-vla-spatiotemporal-icl.md`](../../wiki/entities/paper-ici-vla-spatiotemporal-icl.md)
- 与 [StellaVLA](../../wiki/entities/paper-stellavla-structured-icl-vla.md)、[RoboTTT](../../wiki/entities/paper-robottt-test-time-training-vla-context.md)、[BPP](../../wiki/entities/paper-behavior-prompting-policy.md) 构成「VLA 部署期适应」对照轴（DTW 对齐 micro-demo ICL vs 结构化语言 ICL vs TTT vs 原始示范 prompt）

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与 ICL 概念回链
