# What comes after AGI?（AGI 之后是什么？）

> DeepMind Institute 原始文章归档（ingest；仅整理观点，不转载全文）。

- **类型：** blog（研究人员署名观点文章，不是新论文）
- **英文标题：** What comes after AGI?
- **中文标题：** AGI 之后是什么？从通用智能到超级智能
- **作者：** Tim Genewein（Google DeepMind 研究科学家）
- **机构 / 平台：** DeepMind Institute
- **发布日期：** 2026-11-19
- **入库日期：** 2026-11-20
- **官方原文：** <https://institute.deepmind.com/essays/what-comes-after-agi/>
- **依据的技术报告：** *From AGI to ASI*，arXiv:2606.12683v3，2026-11-10，<https://arxiv.org/html/2606.12683>
- **核对范围：** 已直接读取官方文章全文，逐节核对「The continuum of general intelligence」「Cognitive industrialization」「Four pathways from AGI to ASI」「Scaling models and data」「Technological paradigm shifts」「Improving agent coordination」「Recursive self-improvement」「The many roads to ASI」以及免责声明；与 arXiv v3 正文的对应论点、Table 3/4 交叉核对。
- **通读：** 已完成
- **署名和立场边界：** 网页免责声明明确：DMI 文章供研究讨论，反映作者观点，不能视为 Google 的官方立场。
- **源码 / 复现：** 属于议论文及技术报告解读，未提供可运行模型/训练/评测工程入口；源码运行时序图不适用。
- **一句话：** 用「有效算力推动认知工业化」和四条可叠加的 AGI→ASI 技术路径，解释为什么单模型进展放缓并不必然意味着集体智能进展停止。

## 核心主张与证据口径

### 1. 通用智能连续谱

- 文中 AGI 约指广泛认知任务达到人类水平；ASI 指在几乎所有人类关心的领域长期稳定地超越大型人类专家集体，而不只是某个游戏或专业测验超人类。
- Universal AI / AIXI 描述理论上界，但不可据此推断实际系统必然达到该上界。
- 这些是研究用的操作性刻画，并非全球统一测量标准。

### 2. 认知工业化（Cognitive industrialization）

作者将三个历史增长因素相乘：单位投资的硬件算力约 1.5×/年、AI 硬件投入约 2.5×/年、算法效率约 3×/年，推演全球有效算力存量至少约 10×/年。文章还分别讨论推理成本下降与大量同能力 Agent 副本的规模扩张。

**注意：** 这些是基于过去趋势的外推场景，不是未来持续增长的观测事实，也不是 AGI/ASI 到达时间预测。受能源、芯片、电网、经济投入与互联能力等约束；认知工作变廉价，不代表所有物理任务也同比例变廉价。

### 3. 四条路径可并行

| 路径 | 机制 | 文中提醒的主要不确定性 |
|---|---|---|
| Scaling models and data | 训练模型/数据/有效算力及测试时推理扩展 | 数据墙、收益递减、物理与经济资源约束 |
| Technological paradigm shifts | 持续学习、世界模型、交互决策、更新架构与优化算法 | 现有范式渐进增强是否足够、何时需要质变 |
| Improving agent coordination | 同质或异质 Agent 的任务分解、群体组织与共享经验 | 协调成本、组织设计、多智能体 scaling law 尚待验证 |
| Recursive self-improvement | 代码/设计改进、数据再利用、群体分工三类循环 | 评测反馈不可靠时会退化或停滞 |

文中在 RSI 节将代码/设计、数据/文化、群体协作明确分为三种机制；**不意味着今天已实现无监督、无限制的递归智能爆炸**。

### 4. 跨路径耦合与评测

- 四条路径可以相互加强：算力用于产生更好数据和更多实验，Agent 协作帮助研发，而研发改进又提高计算效率。
- 单一训练算力约束未必同时约束其他能力通道，但这只是作者对发展轨迹与治理的判断。
- 作者主张「model, measure, forecast, repeat」，需要从仅用人类水平静态基准，转向非饱和、能评估高级群体系统与递归改进的测量方法。
- 原技术报告 **v3 Table 4 列七项**：Data wall、Economic and natural resource demand、Neural paradigm、Research gets harder、Abstraction barrier、Evaluation Problem、Deliberate slowdown；不能沿用旧版「六类」摘要漏掉评测问题。

## 对 Robotics_Notebooks 的可操作映射

1. [From AGI to ASI](../../wiki/entities/paper-from-agi-to-asi.md)：既有独立论文详情节点，本次只做增量修订，不再新建重复论文。
2. [Embodied Scaling Laws](../../wiki/concepts/embodied-scaling-laws.md)：区分宏观认知算力和具身策略成功率 scaling；不要把前者系数直接用于预测真机性能。
3. [Data Flywheel](../../wiki/concepts/data-flywheel.md)：仿真/交互数据再利用的机器人类比，必须用独立真机评测检验。
4. [Recursive Self-Improvement](../../wiki/concepts/recursive-self-improvement.md)：区分研发自动化、有限闭环改善、真正递归自改进。
5. [MARL](../../wiki/methods/marl.md)：Agent 数量和任务可分解性共同影响协作收益。

## 局限

- **不是机器人研究实验**：未给出实体机器人成功率、真实部署数据或 Sim2Real 比较。
- **不是 AGI/ASI 预测模型的实测验证**：10×/年和未来 Agent 人口属于条件情景，不应当作确定结论。
- **不是 Google 公司正式政策声明**：依官方页面免责声明，是研究人员讨论意见。
- **未发现可运行的官方源码**：该文章没有模型仓库或入口，不适用源码运行时序图。

## 原始链接

- 官方文章：<https://institute.deepmind.com/essays/what-comes-after-agi/>
- 论文 HTML（v3）：<https://arxiv.org/html/2606.12683>
- 论文摘要与版本：<https://arxiv.org/abs/2606.12683>
- 本仓论文原归档：[agi_to_asi_arxiv_2606_12683.md](../papers/agi_to_asi_arxiv_2606_12683.md)
