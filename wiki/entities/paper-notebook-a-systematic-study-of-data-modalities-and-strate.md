---
type: entity
tags: [paper, humanoid-paper-notebooks, manipulation, vla, co-training, large-behavior-model, vision-language, bimanual, toyota-research]
status: complete
updated: 2026-09-28
arxiv: "2602.01067"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../methods/vla.md
  - ../concepts/open-x-embodiment.md
  - ../concepts/flow-matching-embodied-policy.md
  - ./paper-latent-actions-matter.md
  - ./paper-notebook-sim-and-real-co-training-a-simple-recipe-for-vis.md
sources:
  - ../../sources/papers/humanoid_pnb_a-systematic-study-of-data-modalities-and-strate.md
summary: "大行为模型（Large Behavior Models）把模仿学习扩展到多任务机器人数据的大规模训练，展现强灵巧操作能力，但泛化仍受限于机器人数据覆盖不足。为在不昂贵额外采集的前提下扩覆盖，近期工作依赖协同训练（co-training）：联合学习目标机器人数据与异构数据模态。但不同协同训练数据模态与策略如何影响策略性能仍理解不足。本文做系统研究：在 4000 小时机器人/人类操作数据 + 5000 万视觉-语言样本上，跨 89 个策略、5.8 万次仿真 + 2835 次真机 rollout，比较五类模态（视觉-语言数据、稠密语言标注、跨本体机器人数据、人类视频、离散动作 token）与单/多阶段训练策略。主要发现：视觉-语言与跨本体机器人数据显著提升对分布偏移、新任务与语言理解的泛化；离散动作 token 收益甚微；组合有效模态可累加增益；纯机器人训练会损害视觉-语言能力，而协同训练能恢复；思维链（CoT）条件在其基准上无性能增益。"
---

# A Systematic Study of Data Modalities and Strategies for Co-training Large Behavior Models for Robot Manipulation

**A Systematic Study of Data Modalities and Strategies for Co-training Large Behavior Models for Robot Manipulation** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

大行为模型（Large Behavior Models）把模仿学习扩展到多任务机器人数据的大规模训练，展现强灵巧操作能力，但泛化仍受限于机器人数据覆盖不足。为在不昂贵额外采集的前提下扩覆盖，近期工作依赖协同训练（co-training）：联合学习目标机器人数据与异构数据模态。但不同协同训练数据模态与策略如何影响策略性能仍理解不足。本文做系统研究：在 4000 小时机器人/人类操作数据 + 5000 万视觉-语言样本上，跨 89 个策略、5.8 万次仿真 + 2835 次真机 rollout，比较五类模态（视觉-语言数据、稠密语言标注、跨本体机器人数据、人类视频、离散动作 token）与单/多阶段训练策略。主要发现：视觉-语言与跨本体机器人数据显著提升对分布偏移、新任务与语言理解的泛化；离散动作 token 收益甚微；组合有效模态可累加增益；纯机器人训练会损害视觉-语言能力，而协同训练能恢复；思维链（CoT）条件在其基准上无性能增益。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| LBM | Large Behavior Model，大行为模型 |
| Co-training | 协同训练，目标数据 + 异构模态联合学 |
| Cross-Embodiment | 跨本体机器人数据 |
| Action Token | 离散动作 token |
| VLA | Vision-Language-Action |
| CoT | Chain-of-Thought，思维链条件 |

## 为什么重要

- **"加什么数据"比"加更多数据"更重要**：视觉-语言与跨本体最划算；
- **纯机器人训练会损害通用视觉-语言能力**——协同训练是必要的"保养"；
- **离散动作 token / CoT 未必有用**，提醒不要盲目堆技巧；
- 对人形 VLA/大行为模型的数据配方有直接指导（TRI 大规模实证）。

## 解决什么问题

大行为模型泛化受**机器人数据覆盖不足**所限： - 协同训练（加异构数据）能扩覆盖； - 但**哪些模态、哪种策略**有效**理解不足**，缺系统证据。

论文要：用**大规模、系统化**实验，厘清**协同训练**的数据模态与策略选择。

## 核心机制

1. **协同训练的系统性研究**：五模态 × 单/多阶段策略，大规模评测；
2. **关键结论**：视觉-语言 + 跨本体数据最有效，离散动作 token 收益小；
3. **组合累加 + 协同恢复**：有效模态可叠加，协同训练修复纯机器人训练的视觉-语言退化；
4. **CoT 无增益**：在其基准上的反直觉发现。

方法拆解（深读笔记小节）：五类协同训练数据模态；单/多阶段训练策略 + VLA 架构；大规模评测；主要发现；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models.html> |
| arXiv | <https://arxiv.org/abs/2602.01067> |
| 源码 | **未开源**：项目页 <https://co-training-lbm.github.io/> 截至 2026-09-28 未列代码或数据链接 |
| 作者 | Fanqi Lin、Kushal Arora、Jean Mercat、Haruki Nishimura、Paarth Shah、Jose Barreiros 等（Toyota Research Institute, TRI） |
| 发表 | 2026 年 2 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**设置**：以 TRI-Ramen 目标机器人连续动作 + 流匹配为无协同基线，逐一加入 8 类协同数据（标准视觉–语言数据、脚本 / VLM 生成的机器人轨迹语言标注、跨本体机器人数据 OXE-Ramen、人类视频的潜动作与 VLM 标注、FAST / VQ-VAE 离散动作 token），并比较单阶段、两阶段仅第一阶段、两阶段全程三种协同策略。总规模：4000 小时机器人与人类操作数据 + 5000 万视觉–语言样本，评估 89 个策略、5.8 万次仿真 rollout、2835 次真机 rollout。

- **仿真**（Drake 基准）：13 个已见 + 8 个未见任务，每任务 50 次，分名义条件与分布偏移（光照、背景、相机参数、纹理）。
- **真机**：双臂 Franka；语言跟随分已见物体 / 指令改写 / 未见物体三种设置，各 15 种布局 × 3 条指令 = 45 次，共 49 种已见与 52 种未见物体，按评分细则计任务完成度。
- **有效模态**：标准 VL 数据、轨迹语言标注、跨本体数据、人类视频 VLM 标注都能提升分布偏移鲁棒性、未见任务泛化与语言跟随，**对分布内表现无统计显著影响**。VLM 标注优于脚本标注；轨迹标注与跨本体数据放在第一阶段最有效，VL 数据与人类视频标注在第二阶段继续混入还能提升未见物体的语言跟随。
- **无效或有限的模态**：潜动作只在目标机器人数据很少时有帮助，数据 / 算力充足时无增益；FAST token 协同不提升，反而损害未见任务泛化。
- **组合可累加**：最终模型仿真未见任务 **72.6%**（比基线 +36.4%），真机语言跟随平均完成度 **69.4%**（+45.3%）。
- **微调新长时程灵巧任务**（装袋、盛汤、收纳洗净餐具，平均 13 步、93 秒，每任务 200 条演示）：最终模型微调后平均完成度 **90.2%**，比微调基线高 22.8%、比单任务从零训练高 42.9%。
- **CoT 条件**：让动作生成显式条件于协同数据学到的思维链，对这些目标明确的操作任务没有收益。

## 与其他工作对比

| 工作 | 研究对象 | 与本文的差异 |
|------|------|------|
| [Sim-and-Real Co-Training](./paper-notebook-sim-and-real-co-training-a-simple-recipe-for-vis.md) | 仿真 + 真实数据的混合配方 | 单任务、Diffusion Policy；本文是 VLA 规模的多模态消融 |
| [What Matters for Latent Actions](./paper-latent-actions-matter.md) | 潜动作设计 | 本文结论是潜动作的收益只在低数据区间明显 |
| [Humanoid Policy ~ Human Policy](./paper-notebook-humanoid-policy-human-policy.md) | 人类数据以统一动作空间协同 | 使用人手动作；本文只用了潜动作与语言标注这类粗粒度人类视频表示 |
| [Open X-Embodiment](../concepts/open-x-embodiment.md) | 跨本体数据集 | 本文确认其在第一阶段最有效、第二阶段增益有限 |

## 结论

**这篇工作给出的结论不是「协同训练有用」，而是「哪几类数据协同训练才有用」——视觉-语言与跨本体是划算的两类，离散动作 token 与思维链在其基准上并不划算。**

- 增益主要来自 **视觉-语言数据** 与 **跨本体机器人数据**：二者显著改善对分布偏移、新任务与语言理解的泛化，且有效模态组合起来可累加。
- 一个易被忽略的副作用被摆到台面上：**纯机器人数据训练会损害模型原有的视觉-语言能力**，协同训练在这里更像是对通用能力的「保养」，而不只是数据扩增。
- 负结果同样是产出：离散动作 token 收益甚微、CoT 条件在其基准上无性能增益——提醒不要盲目堆技巧。
- 说服力来自规模与对照密度而非新方法：4000 小时机器人/人类操作数据 + 5000 万视觉-语言样本，跨 89 个策略、5.8 万次仿真与 2835 次真机 rollout（TRI）。
- 适用边界：结论是 **数据配方层面** 的实证，绑定其所用基准、模态划分与 VLA 架构；汇总收益为仿真未见任务 72.6%（+36.4%）、真机语言跟随 69.4%（+45.3%），逐模态数值只在论文图中。

## 局限与风险

- **结果以图表示**：各模态的逐项成功率在图 4–11 中，正文只给出组合与微调的汇总数字。
- **VL 数据未按任务类型拆分**：VQA、描述、检测、空间推理等子类各自贡献未分析。
- **人类视频表示粗粒度**：只用了潜动作与语言标注，没有提取细粒度手部动作。
- **CoT 形式有限**：只考察了协同数据里自然出现的低层动作抽象，未涉及高层规划。
- **只研究模仿学习**：世界模型、强化学习范式下的协同训练未涉及。
- **开源边界**：未见代码与数据；源码运行时序图 **不适用**。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- VLA 方法总览：[vla](../methods/vla.md)
- 跨本体数据 OXE：[open-x-embodiment](../concepts/open-x-embodiment.md)
- 连续动作的流匹配目标：[flow-matching-embodied-policy](../concepts/flow-matching-embodied-policy.md)
- 潜动作设计的对照研究：[paper-latent-actions-matter](./paper-latent-actions-matter.md)
- 仿真 + 真实协同训练配方：[paper-notebook-sim-and-real-co-training-a-simple-recipe-for-vis](./paper-notebook-sim-and-real-co-training-a-simple-recipe-for-vis.md)

## 参考来源

- [humanoid_pnb_a-systematic-study-of-data-modalities-and-strate.md](../../sources/papers/humanoid_pnb_a-systematic-study-of-data-modalities-and-strate.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models.html>
- 论文：<https://arxiv.org/abs/2602.01067>
- 论文正文（III-B–III-F 结果与讨论节）：<https://arxiv.org/html/2602.01067>

## 推荐继续阅读

- [机器人论文阅读笔记：A Systematic Study of Data Modalities and Strategies for Co-training Large Behavior Models for Robot Manipulation](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models.html)
- 项目页：<https://co-training-lbm.github.io/>
