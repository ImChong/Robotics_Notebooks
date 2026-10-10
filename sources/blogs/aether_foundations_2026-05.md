# Aether AI 奠基四篇博文（2026-05-17）

> 来源归档（blog / Aether AI 官方，四篇同日发布，官网编号 01–04）

- **类型：** blog（官网 Blog 栏目；无单独作者署名，按公司发布处理）
- **组织：** Aether AI（创始人 Biwei Huang / 黄碧薇，UCSD）
- **发表日期：** 2026-05-17（四篇均为此日期）
- **入库日期：** 2026-10-10
- **抓取方式：** `curl -sSL -A "Mozilla/5.0"` 读取官网静态 HTML 后去标签核对全文
- **一句话说明：** 公司开站时的四篇纲领文：①为什么因果是下一代 AI 范式；②Causal Copilot 因果分析智能体（可运行原型，MIT 开源）；③World Agent 的「因果大脑」= 记忆 + 世界模型 + 模块化能力 + 因果；④因果世界模型的闭环配方（因果引导探索 → 统一潜动作 → 因果校验器 → 控制）。

| 编号 | 标题 | 原始链接 | 阅读时长（官网标注） |
|------|------|----------|----------------------|
| 01 | Causality and the Next AI Paradigm | <https://aetherlabs.ai/articles/causality-and-the-next-ai-paradigm.html> | ~18 min |
| 02 | Causal Copilot: Toward AI That Discovers Before It Acts | <https://aetherlabs.ai/articles/causal-copilot-toward-ai-that-discovers-before-it-acts.html> | ~9 min |
| 03 | Building the Causal Brain of World Agent | <https://aetherlabs.ai/articles/building-the-causal-brain-of-world-agent.html> | ~6 min |
| 04 | Learning Causal World Models: A Closed-Loop Recipe for Exploration, Representation, and Decision Making | <https://aetherlabs.ai/articles/learning-causal-world-models.html> | ~12 min |

## 开源 / 项目页核查

| 项 | 结论（截至 2026-10-10） |
|----|-------------------------|
| Causal Copilot 代码 | <https://github.com/Lancelot39/Causal-Copilot>（博文 02 文末链接；`git ls-remote` 可达；LICENSE 为 **MIT**；最新提交 2026-05-15；含 `causal_discovery/`、`causal_inference/`、`report/`、`web_demo/`、CPU/GPU Dockerfile） |
| Causal Copilot Demo | <https://causalcopilot.com>（HTTP 200） |
| Causal Copilot 论文 | Wang, Zhou, Wu, …, Huang. *Causal-Copilot: An Autonomous Causal Analysis Agent*. [arXiv:2504.13263](https://arxiv.org/abs/2504.13263)（2025-04-17 首发） |
| 博文 03 / 04 | 无代码；03 引用的是团队既有论文（记忆、C-World、PAN、能力模块化、长 CoT 激活控制），04 只给图表截图，未给数据表、论文编号或代码 |

## 核心摘录（归纳，非全文）

### 01 Causality and the Next AI Paradigm（观点 / 综述）

- **主张：** 预测结构 ≠ 因果结构。模型可以在观测分布上表现很好，但在干预、分布漂移或部署时失败；AI 从被动预测走向推荐、规划、实验和行动，因果问题就躲不开。
- **六条脉络：** 因果推断（潜在结果框架 vs Pearl 结构因果模型、`do(X=x)` 与条件化的区别）；因果发现（PC / GES / LiNGAM / ANM / NOTEARS）；面向鲁棒与迁移的因果机器学习（独立因果机制、不变预测）；因果表示学习（因果变量通常未给定，需从像素、轨迹等中恢复）；决策与控制中的因果（动作即干预，关联 action-sufficient 表征、AdaRL、非平稳 RL、可辨识分解世界模型）；**因果世界模型即因果基础模型**。
- **与邻近范式的区分（作者观点）：** VLA 把动作当输出而非干预；视频生成的视觉合理不等于因果正确；几何 / 3D 重建不编码力、接触与动力学。
- **结论：** 「Causality does not replace scaling」——规模提供容量，因果提供结构（变量、机制、干预、不变性、反事实）。
- **自我定位：** 多段把 Biwei Huang 的工作（异质 / 非平稳因果发现、潜变量层级结构、可辨识潜模型、因果 RL）与 Pearl、Glymour、Schölkopf、Kun Zhang 的谱系并列；附 38 条参考文献，文末引 Causal-Copilot 论文。属公司自我叙事。

### 02 Causal Copilot: Toward AI That Discovers Before It Acts（系统介绍）

- **定义：** 自主因果分析智能体，把一个因果问题从问题形式化、假设管理，依次带过因果发现、可识别性分析、效应估计、反事实推理、稳健性检验和解释。公司称其为「early, runnable expression」。
- **推理循环：** ①把用户意图翻译成处理 / 结局 / 分析单位 / 时间窗 / 协变量 / estimand；②先做数据画像（表格：类型、缺失、分布、线性、异质性；时序：平稳性、滞后结构）再选方法；③多族因果发现（约束型、打分型、连续优化、LiNGAM 类、Markov blanket、Granger 类、时序方法），bootstrap 估边置信度，LLM 只对中等置信边做结构化合理性检查，可接人类反馈重跑；④识别与估计（调整集、前门 / 后门、IV、匹配、双稳健等）；⑤反事实问题；⑥敏感性分析、安慰剂检验、替代设定等稳健性检验。
- **覆盖面（自报）：** 集成 20 多种因果分析技术；输出是可检查的分析报告（问题、诊断、选法理由、不确定边、结论依赖的假设）。
- **评测（自报）：** 在表格与时序合成场景下变维度、密度、样本量、噪声、离散性、测量误差、缺失率等，对照单个因果发现方法与「不带上下文的 GPT-4o 基线」；博文只给 F1 / ATE RMSE 柱状图截图，正文无具体数值。
- **与机器人关系：** 博文明确说机器人里要把「动作当作干预」，在执行前问动作会改变什么；但 Causal Copilot 本身面向表格 / 时序数据分析，**不是机器人控制器**。

### 03 Building the Causal Brain of World Agent（愿景）

- **World Agent：** 通过内部因果模型行动的智能体；需理解「action → state change → observation → next action」链条，并能把失败追溯到更早的某个动作（例：GUI 中改日期影响价格与支付选项）。
- **四个组成：** **记忆**（团队工作：视频扩散的即插即用记忆、GUI agent 结构化 / 连续记忆）；**世界模型**（C-World 计算机使用环境生成器、PAN 通用世界模型；强调「action-valid futures」）；**模块化**（能力定位于少量参数以迁移 / 恢复 / 合并；激活控制按需触发长 CoT）；**因果**（把上述三者串起来）。
- **因果大脑回答四问：** 发生了什么（记忆）、为什么（因果推理）、接下来会怎样（世界模型）、现在该做什么（模块化控制）。循环：`observe → retrieve memory → simulate futures → reason about causes → act → update memory`。
- **边界：** 无新实验；引用的 7 篇均为既有 arXiv 论文（2505.17697、2510.09038、2511.09057、2511.19229、2601.06328、2601.09398、2603.10291）。

### 04 Learning Causal World Models（方法配方）

- **问题：** 标准世界模型学 `p(z_{t+1} | z_t, a_t)`，预测好不等于知道哪些因素可控、哪些相关是伪相关、哪些机制可复用、想象的长程计划是否物理可达。
- **闭环：** `Explore → abstract states and actions → learn causal structure → plan, verify, and generalize → explore again`；智能体主动探测（推、抬、转、扰动）暴露任务相关因素，模型只保留控制所需因素。
- **因果引导探索：** 数据即干预——选对任务相关因素信息量最大的轨迹；同一状态下两种干预的结果对比可识别动作效应，同一干预在两种情境下的对比可识别情境因素。
- **统一潜动作：** `u_t = h(o_t, o_{t+1})` 或 `h(z_t, a_t, z_{t+1})`，逆动力学推潜动作、前向模型 `z_{t+1} = G(z_t, u_t)` 检验其充分性；跨本体时 `a_t^e → u_t → z_{t+1}`，即不同机器人共享「因果动作语言」而非命令空间（与后续 SCAR 博文同一路线）。
- **因果校验器：** 「生成模型提议未来，因果校验器检查动力学」；动力学一致性能量 `E_dyn = Σ_t ||f(h_t, a_t) − h_{t+1}||²`，推理时可用来拒绝或引导采样，无需重训视频大模型。
- **实验（仅图示，自报）：** World Arena 接触丰富 rollout 对比；Aloha → Franka 零样本迁移的长程任务成功率；Meta-World 组合泛化；Procgen 与 RoboTwin 跨本体泛化。正文无数值表。

## 对 wiki 的映射

- [aether-ai](../../wiki/entities/aether-ai.md) — 公司页「奠基四篇博文」「Causal Copilot」「核心原理」小节

## 可信度与使用边界

- 01、03 是观点 / 愿景文，04 是配方加图示，三者都没给可复现数值；只有 02 有可运行代码。
- 01 中对创始人学术谱系的定位是公司自述。
