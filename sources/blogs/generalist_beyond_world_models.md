# Going Beyond World Models & VLAs（Generalist AI）

> 来源归档（blog / Generalist AI 官方，Idea 栏目）

- **标题：** Going Beyond World Models & VLAs
- **类型：** blog（观点 / 研究哲学，非技术报告）
- **作者 / 组织：** Pete Florence and the Generalist Team / Generalist AI
- **原始链接：** <https://generalistai.com/blog/beyond-world-models>
- **发表日期：** 2026-04-07（页面署名 April 7, 2026）
- **入库日期：** 2026-10-09
- **抓取方式：** `curl` 抓取官方页静态 HTML，抽取正文与脚注
- **一句话说明：** GEN-1 发布五天后的配套观点文：解释 **GEN-1 约 99% 参数从零训练**、既不是「VLM + 动作头」的 VLA 也不只是世界模型，而是 **物理交互原生基础模型**；论证三点——目标重于方法标签、把「A 或 B」改问「能走多远」、约束（机器人数据稀缺）本身会变化，视觉-语言预训练只是数据不足时的「拐杖」。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-10-09） |
|----|-------------------------|
| 代码 / 权重 / 数据 | **不适用 / 未开源**：观点文，无代码、权重、数据或架构细节；所指模型 GEN-1 闭源（见 [generalist_gen1.md](./generalist_gen1.md)） |
| 可信度边界 | 作者观点与公司路线说明；「从零训练总会赢」等为立场性论断，仅引一篇蒸馏论文作旁证 |

## 核心摘录（归纳，非全文）

### 从零训练的事实陈述

- **GEN-1 中约 99% 的参数从零训练**（未说明剩余约 1% 来自何处）。
- 这是公司 **两年来** 的刻意选择：数据足够时，完全掌控底层模型能更快推前沿。
- GEN-1「不是在视觉-语言模型上外挂机器人动作的微调模型，也不只是世界模型」，而是 **面向物理交互的一等公民原生基础模型**。
- 「有足够数据和算力时，从零训练总会赢」——引 Beyer & Zhai et al. 2022（*Knowledge Distillation: A Good Teacher is Patient and Consistent*）作「越来越多证据」之一（作者立场）。
- 架构、训练、推理的每个方面「都不受他人为别的目的所做决策的约束」。

### 为什么不贴 VLA / 世界模型标签

- 时间线判断：VLA 的热潮在 **2023–2025**，世界模型在 **2026 年初**；作者称学术界有「随大流」倾向。
- 团队成员 **共同发明了 VLA**（引 RT-2，Brohan et al. 2023），**2023 年起** 发表机器人世界模型相关工作（引 Video Language Planning，Du et al. 2023），且更早开始研究。
- 不贴标签的三个理由：

| 论点 | 要点 |
|------|------|
| **1. 目标比工具标签重要** | 引 Schulman「idea-driven vs goal-driven」研究：当前世界模型讨论是 idea-driven；应先问目标是什么 |
| **2. 不要问「A 或 B」，问「能走多远」** | 把「或」转为「与」→「各多少」→ 更深的目标与约束问题；举例 Chinchilla（compute-optimal）；团队 **一年多** 来在组合 VLA、世界模型及其他思路，组合越多越难归类 |
| **3. 供给侧会变化** | 「机器人数据少」不是长期约束；已有 **>50 万小时** 物理交互数据；视觉-语言训练（乃至互联网视频）是数据不足时的 **拐杖**——「拐杖之后呢？还需要拐杖吗？」 |

### 目标驱动路线图

- 长期目标示例：**完全零样本机器人**——从未见过的整类任务，高成功率、高速度、零任务数据；任务足够多样复杂时相当于 **完整 physical AGI**。
- 中间里程碑：允许每任务少量机器人数据 **X**，高性能执行；路线 = **持续减小 X、同时提高性能**。
- 具体可测里程碑：**约 1 小时机器人数据 → 广泛达 99%+ 成功率**，即具备广泛商业可行性（与 GEN-1 博文的结果口径一致）。
- 旁证：PaLM-E（早期多模态语言模型之一）出自机器人目标，却也被用于医疗基准（Med-PaLM M）——目标驱动反而能外溢。

### 收尾

- 三条原则：目标强于方法；在约束下优化而非站队分类；约束本身会改变。
- 回顾已展示能力：机器人 scaling law、数小时内泛化到新环境与新具身、大规模预训练涌现即兴智能；「More soon.」

## 对 wiki 的映射

- [generalist-gen1](../../wiki/entities/generalist-gen1.md) — 作为「从零训练与超越 VLA / 世界模型」一节并入 GEN-1 实体页
- [vla](../../wiki/methods/vla.md) — VLA 主流路线（VLM 初始化 + 动作头）的反向立场
- [world-action-models](../../wiki/concepts/world-action-models.md) — 世界模型与动作生成耦合路线对照
- [paper-rt-2](../../wiki/entities/paper-rt-2.md)、[paper-palm-e-embodied-language-model](../../wiki/entities/paper-palm-e-embodied-language-model.md) — 文中自引的 VLA / 多模态起源工作

## 可信度与使用边界

- 观点文，无实验；「从零训练总会赢」是外推论断，引用的蒸馏论文为视觉分类场景，并非机器人直接证据。
- 「99% 参数从零训练」无架构拆解，外部无法核对剩余参数的来源与作用。
- 「co-invented VLAs」指团队成员参与 RT-2 等早期工作（作者自述），不代表公司实体在当时存在。

## Citation

> 原文未提供 Citation 区块；以下为本库按页面署名整理。

```bibtex
@misc{florence2026beyondwm,
  author = {Pete Florence and Generalist Team},
  title = {Going Beyond World Models \& VLAs},
  howpublished = {Generalist AI Blog},
  year = {2026},
  note = {https://generalistai.com/blog/beyond-world-models}
}
```
