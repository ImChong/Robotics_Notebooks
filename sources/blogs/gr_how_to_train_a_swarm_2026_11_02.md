# How to Train a Swarm｜如何训练智能体群：集体推理时扩展

> 来源归档（blog；官方一手研究解读，不转载全文）

- **英文标题：** How to Train a Swarm
- **中文标题：** 如何训练智能体群：让并行工作与协作在强化学习中变得有利
- **类型：** blog
- **作者：** General Reasoning team（General Reasoning）
- **原始链接：** https://www.gr.inc/blog/how-to-train-a-swarm
- **发表日期：** 2026-11-02（原文页面标注）
- **入库日期：** 2026-11-16
- **核对范围：** 通读文章 Introduction、Designing incentives for RL、Making parallel work pay、Communication primitives、Signs of life、Outstanding problems、Conclusion 与 Citation/References；核对公式、图表文字说明及作者声明的实验口径。未将链接的 Avalon-M1 技术报告当作已通读，也未运行训练。
- **通读：** 已完成
- **材料边界：** 该页是集体式 AI 软件 Agent 的方法讨论，**不是实体机器人编队的实验报告**；数字来自作者的 Avalon-M1 研究展示，不应外推到真机群控。
- **开源核对：** 原文引用 Avalon-M1 技术报告（https://gr.inc/avalonm1.pdf），但本文没有提供可运行的训练源码入口；不据此创建源码归档、声称权重已公开或补写源码运行时序图。

## 一句话摘要

General Reasoning 讨论 **collective inference-time scaling（集体推理时扩展）**：不只是让单个推理模型想得更久，而是让共享策略的多个 Agent 在共享奖励和延迟压力下，自主选择并行劳动、消息沟通与任务交接。文中把这一训练思路称为 **RLCR（reinforcement learning from collective rewards）**，以 Avalon-M1 的实验作例子。

## 原文机制与重要区分

1. **训练对象：** 多个 Agent 共享策略参数；同一集体回报生成较粗的 group/collective advantage，再反馈给各 Agent 的策略更新。不是各 Agent 独立得到精确的个体贡献分数。
2. **为什么需要时间压力：** 若不同组织方式最终任务分数相同，只看任务分数的目标不会鼓励更快完成。并行并非天然更好；对可完全独立尝试的任务，沟通反而可能徒增开销。
3. **两种延迟激励：**
   - 显式 latency penalty：文章给出一种可能的奖励塑形形式，把 wall-clock 用时按任务平均通过率和上限归一化扣分；该式是**候选方式**，不要误写成 Avalon-M1 采用的公式。
   - **Avalon-M1 实际选择 wall-clock time limit**：到截止时间按已完成的进度给回报，训练中逐渐延长上限，使更长时域协作的收益有机会覆盖起步协调成本。**截止前同分但更快不额外加分**，与显式时间惩罚不同。
4. **前提：anytime-return。** 任务在任意截断点都应可评估中间进度；文中举软件工程测试通过数和金融环境累计回报为例，并非声称所有任务都能这样测。
5. **通信原语：** 直接消息、公共频道或消息板、可选知识归档/issue board。Avalon-M1 包含 CEO 招聘/解雇专家的先验结构，同时允许任意 Agent 相互通信；相比之下 Kimi Agent Swarm 强调训练 orchestrator、冻结子 Agent。
6. **观察学习是否发生：** 不能只看 reward。应并看对环境产生作用的计算量、单位时长通信强度以及**达到相同分数的 wall-clock 延迟**；消息多本身不代表有效合作。

## 作者给出的实验观察（保留限制条件）

| 原文观察 | 读法与边界 |
|---|---|
| 5–8 名受雇专家的集体到训练 step 200，环境“有用工作”代理指标接近平均 rollout 的 **2 倍** | “有用工作”按对环境执行动作的 turn 中输出/推理 token 统计，**不等同于真实有效产出翻倍** |
| 匹配总体输出 token 预算时，扩展推理的单 Agent 达到类似能力需要约 **4 倍** wall-clock 时间 | 是文章报告的任务/基线比较，非一般规律 |
| EdgeBench 长时任务中，至多 8 名受雇专家的组织约在 **2.4 小时**的 derived latency 后追上并超过单 Agent | 初期组织受沟通开销拖累，曲线有任务和时长前提 |

## 作者列出的未解问题

- **Village problem：** 训练时最多 9 个 Agent（CEO + 最多 8 hires），部署时可能扩到几百、几千个；规模泛化尚不清楚。
- **Uneasy triangle：** 集体共享且平坦的 advantage 难以针对长轨迹中各 Agent 的具体贡献做信用分配，环境跨域也降低了每类任务的学习样本密度。
- **跨 rollout 公共资产：** 单轮收益很难奖励“写代码库或论文供未来 Agent 使用”这类延迟兑现的公共品贡献。
- **可迁移性：** 文章未提供实体机器人多机安全、物理通信时延、真实运动控制等验证。

## 对 wiki 的映射

- **主沉淀：** [RLCR 与集体推理时扩展](../../wiki/methods/rlcr-collective-inference-time-scaling.md)，梳理奖励激励、流程 Mermaid、实测指标及边界。
- **交叉链接：** [MARL 方法](../../wiki/methods/marl.md)；共同研究多智能体协作，但前者是软件 Agent 的共享回报与 wall-clock 机制，后者也包含实体机器人分布式控制问题。
- **纵深路线核查：** 已核对 roadmap/README.md 的已有纵深路线；本文并非机器人群控的直接训练与部署路线，**不强行挂入**人形群控展演或 RL 运控路线。

## 进一步阅读（未纳入本次全文通读）

- [Avalon-M1: Reinforcement Learning from Collective Rewards（技术报告，文章引用）](https://gr.inc/avalonm1.pdf)
- [General Reasoning 原始博客](https://www.gr.inc/blog/how-to-train-a-swarm)
