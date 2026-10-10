---
type: entity
tags: [company, aether-ai, causal-ai, causal-world-model, world-model, embodied-foundation-model, causal-discovery, world-agent]
title: Aether AI（因果世界模型公司）
status: complete
summary: "Aether AI（2026 年春成立，San Diego）由 UCSD 助理教授黄碧薇（Biwei Huang）创办，主张用因果世界模型替代纯相关性 scaling，首个落地方向是 Physical AI。2026-06 完成 2000 万美元种子轮（MPCi/经纬创投领投）。官网 11 篇博文覆盖因果范式、Causal Copilot、因果大脑、闭环因果世界模型、TC-WM、接触几何、CD-LAM、SCAR、RSIAgent、CausalWM 与 CRIS-0。"
updated: 2026-10-10
institutions: [aether-ai]
related:
  - ./paper-task-centric-world-models.md
  - ./paper-geometry-of-contact.md
  - ./paper-cd-lam.md
  - ./paper-scar-continuous-action.md
  - ./aether-cris-0.md
  - ./paper-causalwm.md
  - ./aether-rsiagent.md
  - ./paper-causalvae-world-models.md
  - ./paper-sa-2511-09057-pan-a-world-model-for-general-interactable-and-l.md
  - ./paper-latent-actions-matter.md
  - ./generalist-ai-robotics.md
  - ./cosmos-3.md
  - ../concepts/world-action-models.md
  - ../concepts/functional-taxonomy-world-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ../overview/overseas-embodied-ai-labs-landscape-2026.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/sites/aetherlabs-blog-index.md
  - ../../sources/blogs/aether_foundations_2026-05.md
  - ../../sources/blogs/aether_seed_round_2026-06.md
  - ../../sources/blogs/aether_rsiagent.md
---

# Aether AI（因果世界模型）

## 一句话定义

**Aether AI**：2026 年春在 San Diego 成立的 AI 公司，创始人是 UCSD 助理教授 **黄碧薇（Biwei Huang）**，出自 CMU 因果发现学派。公司主张 **因果世界模型（Causal World Model）**：模型要识别因果变量、学习因果结构、推演干预后果，而不只是拟合统计相关。首个落地方向是 **Physical AI**，即给机器人做一个位于感知与控制之间的「因果大脑」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CWM | Causal World Model | 因果世界模型；公司核心概念，建模状态、动作、机制与结局 |
| WM | World Model | 世界模型；给定状态与动作预测未来 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略；公司认为它把动作当输出而非干预 |
| SCM | Structural Causal Model | 结构因果模型；Pearl 框架，用 `do(·)` 表示干预 |
| ATE | Average Treatment Effect | 平均处理效应；Causal Copilot 的效应估计目标之一 |
| LAM | Latent Action Model | 潜动作模型；CD-LAM / SCAR 博文的对象 |
| CoT | Chain-of-Thought | 思维链；CausalWM 提出「因果思维链」 |
| GUI | Graphical User Interface | 图形界面；World Agent 与 RSIAgent 的评测环境 |
| UCSD | University of California San Diego | 加州大学圣地亚哥分校；创始人任职单位 |

## 与同名「Aether」区分

arXiv 2503.18945 *Aether: Geometric-Aware Unified World Modeling*（署名 Aether Team，Haoyi Zhu 等，2025-03）是另一组作者的几何世界模型项目，与本公司无关，见 [该页](./paper-sa-2503-18945-aether-geometric-aware-unified-world-modeling.md)。

## 为什么重要

- **「结构 vs 规模」的产业样本：** 多数具身基础模型公司押注更多数据与更大模型（对照 [Generalist AI](./generalist-ai-robotics.md)）；Aether 押注 **因果结构**，声称能以更少数据换来更好泛化，是这一路线少见的商业化样本。
- **学派背书清楚：** 创始人在异质 / 非平稳因果发现、潜变量因果结构、因果 RL 上有长期发表，并是开源 Causal-Learn、Causal-Copilot 的作者。这让公司叙事有可追溯的论文谱系。
- **世界模型评测的新维度：** 公司强调「视觉合理 ≠ 因果正确」，并提出 **因果校验器**、**统一潜动作**、**因果思维链** 等可以单独借鉴的组件（见 [CausalWM](./paper-causalwm.md)、[CRIS-0](./aether-cris-0.md)）。
- **引用纪律：** 成立不到一年，数据效率 20–30%、榜单第一等数字都是 **自报**；RSIAgent、Causal Copilot 有代码，机器人侧系统多数未开源。

## 公司概况

| 项 | 内容 | 来源 |
|----|------|------|
| 成立 | **2026 年春**（官网未写成立日期） | San Diego Business Journal 2026-07-06："Huang launched Aether this spring" |
| 总部 | San Diego，CA | 官网融资公告电头 |
| 创始人 | Biwei Huang（黄碧薇）：UCSD HDSI 助理教授（2022 起）；CMU 博士（导师 Kun Zhang、Clark Glymour），曾在马普智能系统所（MPI-IS）；100 余篇论文；Causal-Learn、Causal-Copilot 作者 | 官网公告；SDBJ；机器之心 |
| 使命 | 让因果推理成为下一代 AI 的基础能力；从「识别模式」走向「理解机制」（自报） | 官网主页 / 公告 |
| 首个场景 | Physical AI 与机器人，不造本体，做感知与控制之间的「decision brain」；长期延伸到科学发现（生物、医学、长寿） | 官网主页；机器之心 |
| 融资 | 2026-06 种子轮 **$20M**；**MPCi（经纬创投）领投**，Inno Angel Fund（英诺）、SWC Global、Unity Ventures（九合创投）等参投；估值未披露 | 官网（金额）+ 公司通稿（投资方） |
| 学界关联 | 官网称受 Judea Pearl、Bernhard Schölkopf、Clark Glymour、Peter Spirtes、Kun Zhang 等 "strongly supported and affected"；中文通稿称「学术顾问网络」 | 官网公告；中文通稿 |
| 渠道 | 官网 <https://aetherlabs.ai/>；X [@AetherLab_AI](https://x.com/AetherLab_AI)；GitHub 组织 AetherLabsAI（见 [RSIAgent](./aether-rsiagent.md)） | 官网页脚 |

> 官网主页 HTML 里还有被注释隐藏的「创始团队」和「Scientific Advisors」名单，页面不展示。本页不把这些名单当作公开事实。

## 官方动态时间线（2026-10-10 共 11 篇博文 + 3 条新闻）

完整核查见 [官网 Blog / News 列表归档](../../sources/sites/aetherlabs-blog-index.md)。日期取官网列表；News 中的 CausalWM、CRIS-0 两条与对应博文是同一 URL。

| 日期 | 类型 | 标题 | 要点（自报） | 本库入口 |
|------|------|------|--------------|----------|
| 2026-05-17 | Blog 01 | [Causality and the Next AI Paradigm](https://aetherlabs.ai/articles/causality-and-the-next-ai-paradigm.html) | 预测结构 ≠ 因果结构；因果世界模型即因果基础模型 | 本页「奠基四篇博文」 |
| 2026-05-17 | Blog 02 | [Causal Copilot](https://aetherlabs.ai/articles/causal-copilot-toward-ai-that-discovers-before-it-acts.html) | 可运行的因果分析智能体；20+ 种因果技术；MIT 开源 | 本页「Causal Copilot」 |
| 2026-05-17 | Blog 03 | [Building the Causal Brain of World Agent](https://aetherlabs.ai/articles/building-the-causal-brain-of-world-agent.html) | 记忆 + 世界模型 + 模块化 + 因果 = World Agent 的因果大脑 | 本页「奠基四篇博文」 |
| 2026-05-17 | Blog 04 | [Learning Causal World Models](https://aetherlabs.ai/articles/learning-causal-world-models.html) | 闭环配方：因果引导探索 → 统一潜动作 → 因果校验器 → 控制 | 本页「奠基四篇博文」 |
| 2026-06-17 | News | [$20M Seed Round](https://aetherlabs.ai/news/aether-ai-raises-20m-seed-round.html) | 种子轮 2000 万美元；首攻 Physical AI | 本页「2026-06 种子轮」 |
| 2026-07-09 | Blog 05 | [Task-Centric World Models from Visual Foundations](https://aetherlabs.ai/articles/task-centric-world-models.html) | 单个线性投影从冻结视觉基础模型中取紧凑任务状态 | [TC-WM](./paper-task-centric-world-models.md) |
| 2026-07-16 | Blog 06 | [The Geometry of Contact](https://aetherlabs.ai/articles/the-geometry-of-contact.html) | Interaction-Weighted Resampling；真机空气曲棍球 5/20 → 12/20 | [Geometry of Contact（IWR）](./paper-geometry-of-contact.md) |
| 2026-07-27 | Blog 07 | [CD-LAM](https://aetherlabs.ai/articles/cd-lam-causal-debiasing-for-embodied-world-models.html) | 潜动作空间因果去偏；动作跟随误差降 30% 以上，后训练少 10 倍 | [CD-LAM](./paper-cd-lam.md) |
| 2026-08-09 | Blog 08 | [SCAR](https://aetherlabs.ai/articles/scar-self-supervised-continuous-action-representation-learning.html) | 从视觉转移学统一潜动作接口，跨本体迁移 | [SCAR](./paper-scar-continuous-action.md) |
| 2026-09-15 | Blog 09 | [RSIAgent](https://aetherlabs.ai/articles/rsiagent-autonomous-exploration-for-recursive-self-improvement.html) | 不更新参数的探索式自我改进；OSWorld 2.0 78.98% | [RSIAgent](./aether-rsiagent.md) |
| 2026-09-19 | Blog 10 + News | [CausalWM](https://aetherlabs.ai/articles/causalwm-causal-chain-of-thought-reasoning-for-embodied-world-model.html) | 先预测运动与几何再生成视频；TriWorldBench 66.04 第一 | [CausalWM](./paper-causalwm.md) |
| 2026-10-08 | Blog 11 + News | [CRIS-0](https://aetherlabs.ai/articles/real-world-autonomous-robotic-system-with-causality-driven-agent-and-world-model.html) | 因果引导机器人智能体 + 因果世界模型；扰动恢复、长程无干预 | [CRIS-0](./aether-cris-0.md) |

## 核心原理

### 因果大脑四层架构

公司在 CVPR 2026 演讲通稿（2026-06-06）、中文融资通稿（2026-06-18）和机器之心深度稿（2026-06-24）中描述了同一套「Four-Layer Causal Brain Architecture」。官网博文没有把它写成单独一篇文章。

```mermaid
flowchart BT
  L1["① Causation Transformer 层<br/>token 级因果依赖，保持可扩展性"]
  L2["② 模块化神经架构层<br/>按机制拆分：接触 / 支撑 / 摩擦 / 动作影响"]
  L3["③ 因果世界模型层（核心）<br/>像素 → 因果变量 → 干预下的动力学"]
  L4["④ 因果驱动智能体系统层<br/>规划 · 归因 · 记忆 · 恢复"]
  L1 --> L2 --> L3 --> L4
  OBS[视频 / 文本 / 传感器] --> L3
  L4 --> ACT[机器人动作 = 干预]
  ACT -. 新证据 .-> L3
  subgraph caps["三类基础能力"]
    C1[因果特征表示]
    C2[因果结构发现]
    C3[因果动力学建模]
  end
  caps -.-> L3
```

| 层 | 公司表述 | 与对外成果的对应（推测） |
|----|----------|--------------------------|
| ④ 智能体系统 | 因果驱动的规划、归因、记忆；失败时定位根因再恢复 | Blog 03 World Agent、[RSIAgent](./aether-rsiagent.md)、[CRIS-0](./aether-cris-0.md) 中的 Causality-guided Robot Agent |
| ③ 因果世界模型 | 从像素到物理层面的因果变量识别与动力学建模；架构核心 | Blog 04 配方、TC-WM、CD-LAM、SCAR、[CausalWM](./paper-causalwm.md) |
| ② 模块化架构 | 受大脑功能分区启发，把因果机制做成可复用、可组合模块 | Blog 03 引用的能力参数定位（arXiv 2601.09398） |
| ① Causation Transformer | 在可扩展 Transformer 上引入 token 级因果性，「改这里，结果是否随之改变」 | **截至 2026-10-10 未见公开论文或代码** |

「对应」列是本库根据主题做的匹配，公司没有逐项对应说明。

### 三类基础能力

公司在 CVPR 演讲和中文通稿中都把因果世界模型拆成三条标准：

1. **因果特征表示**：从原始观测中恢复可解释的潜在因子（「概念」），而非黑箱 embedding。创始人的说法是「structured compression is intelligence」。
2. **因果结构发现**：找出因子之间谁影响谁，可能跨层级。
3. **因果动力学建模**：学习系统在不同动作 / 干预下随时间的演化，支持反事实推理。

### 闭环因果世界模型配方（Blog 04）

`探索 → 抽象状态与动作 → 学因果结构 → 规划、校验、泛化 → 再探索`：

- **数据即干预：** 主动推、抬、转、扰动物体，用「同状态不同干预」和「同干预不同情境」的对比识别动作效应与情境因素。
- **统一潜动作：** `u_t = h(o_t, o_{t+1})`，前向 `z_{t+1} = G(z_t, u_t)`；不同本体各自实现同一潜动作（`a_t^e → u_t`），共享的是「因果动作语言」，不是命令空间。
- **因果校验器：** 生成模型提议未来，用动力学一致性能量 `E_dyn = Σ_t ||f(h_t, a_t) − h_{t+1}||²` 打分或引导采样，不用重训视频大模型。

## 奠基四篇博文（2026-05-17）

详细摘录见 [归档](../../sources/blogs/aether_foundations_2026-05.md)。

- **Causality and the Next AI Paradigm：** 梳理因果推断（潜在结果 vs SCM、`do(X=x)` 与条件化）、因果发现（PC / GES / LiNGAM / NOTEARS）、因果 ML、因果表示学习、因果决策控制五条脉络，结论是「因果世界模型即因果基础模型」。作者认为 VLA 把动作当输出，视频生成的视觉合理不等于因果正确，几何重建不编码力与接触。立场是「因果不取代 scaling，而是给 scaling 更对的目标」。
- **Building the Causal Brain of World Agent：** World Agent 的因果大脑由 **记忆**（过去）、**世界模型**（未来，强调 action-valid futures）、**模块化能力**（按需激活）和 **因果**（把三者串起来）组成，循环为 `observe → retrieve memory → simulate futures → reason about causes → act → update memory`。这是愿景文，引用的 7 篇都是团队已有论文（含 [PAN](./paper-sa-2511-09057-pan-a-world-model-for-general-interactable-and-l.md)、C-World），没有新实验。
- **Learning Causal World Models：** 见上文「闭环配方」。只给 World Arena、Aloha → Franka 零样本、Meta-World、Procgen / RoboTwin 的图示，正文没有数值表。
- **Causal Copilot：** 见下节。

### Causal Copilot

- **是什么：** 自主因果分析智能体，把一个因果问题依次带过问题形式化、数据画像、因果发现、可识别性分析、效应估计、反事实推理、稳健性检验和可检查报告。公司称它是因果路线「早期、可运行的表达」。
- **关键设计：** 先画像后选法（表格看缺失 / 线性 / 异质性，时序看平稳性 / 滞后）；多族因果发现方法加 bootstrap 边置信度；LLM 只对 **中等置信** 的边做结构化合理性检查；遇到不可识别的问题允许 **拒答**。
- **评测（自报）：** 在表格与时序合成场景下对照单一因果发现算法和不带上下文的 GPT-4o 基线；博文只放 F1 / ATE RMSE 图，没有给数值。
- **开源：** 代码 <https://github.com/Lancelot39/Causal-Copilot>（**MIT**，含 CPU/GPU Dockerfile 与 `web_demo/`）；Demo <https://causalcopilot.com>；论文 [arXiv:2504.13263](https://arxiv.org/abs/2504.13263)（2025-04，早于公司成立）。
- **定位边界：** 它面向表格 / 时序数据分析，**不是机器人控制器**。与机器人的联系只在理念上：把动作当作干预，执行前先问会改变什么。

## 2026-06 种子轮

详见 [归档](../../sources/blogs/aether_seed_round_2026-06.md)。

- **金额（官网）：** **$20M** 种子轮，已完成交割；资金用于因果世界模型研发、工程基础设施与科学团队扩张、Physical AI / 机器人首批商业部署。
- **投资方（公司通稿，官网正文未列）：** GlobeNewswire 英文稿（2026-06-18）写 **MPCi 领投**，Inno Angel Fund、SWC Global、Unity Ventures 等参投；中文稿写 **经纬创投领投**，英诺基金、SWC Global、九合创投等参投。两稿都由同一位合伙人（Ti Tong / 童倜）发言，所以 MPCi 就是经纬创投；其余名称按两份名单对应推断。
- **未披露：** 估值。
- **早期结果（自报）：** 选定操作任务上数据效率提升 20–30%；部分案例约 50 条高质量因果标注就让持续失败的任务达到可靠成功率。没有给任务、基线或数据。
- **日期口径：** 官网列表 06-17，正文电头 06-16，通稿 06-18。

## 数据与开源就绪度

| 成果 | 代码 / 权重 | 备注 |
|------|-------------|------|
| Causal Copilot | ✅ GitHub（MIT）+ 在线 Demo | 表格 / 时序因果分析，不含机器人 |
| RSIAgent | ✅ 见 [RSIAgent](./aether-rsiagent.md) | GUI / 软件环境智能体 |
| TC-WM / Geometry of Contact / SCAR | 有 arXiv 论文（2605.25620 / 2606.11525 / 2605.16412，媒体给出、本库已核对标题） | 代码状态由各详情页核查 |
| CausalWM / CRIS-0 | 见 [CausalWM](./paper-causalwm.md)、[CRIS-0](./aether-cris-0.md) | 以详情页核查为准 |
| Causation Transformer、模块化架构层 | ❌ 未见 | 只出现在通稿与媒体稿 |

## 工程实践

| 场景 | 建议 |
|------|------|
| 做表格 / 时序因果分析 | 直接用 Causal Copilot 开源代码；先读它的数据画像和稳健性检验流程，不要只看最终图 |
| 设计世界模型评测 | 借鉴「因果校验器」思路：除了视频质量，还要测动作跟随误差、接触前后物体是否违规移动（对照 [动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)） |
| 跨本体动作接口 | 对照 Blog 04 / SCAR 的统一潜动作与 [Latent Actions Matter](./paper-latent-actions-matter.md)；注意潜动作是否编码了本体外观这类捷径 |
| 写综述 / 产业对照 | 可把 Aether 当作「结构优先」路线，与 [Generalist AI](./generalist-ai-robotics.md) 的「规模优先」对照；引用数字时标注自报 |
| 产线选型 | 不要假设能下载机器人侧的因果大脑；四层架构多数层没有公开实现 |

## 局限与风险

- **成立时间短：** 2026 年春成立，到 2026-10 只有 5 个月的公开记录，机器人系统（CRIS-0）是第 11 篇博文才出现。
- **自报为主：** 20–30% 数据效率、「50 条标注」、榜单第一、OSWorld 分数都来自公司。融资公告没有给任务或基线细节。
- **架构叙事领先于证据：** 四层架构里 Causation Transformer 与模块化层只出现在通稿，没有论文或代码；底层「token 级因果性」怎么做、怎么评测都未公开。
- **「因果」的边界：** 多数机器人侧工作（TC-WM、接触重采样、潜动作）本质是表示学习与数据重加权，它们与严格意义上可识别因果结构的关系需看各论文的假设（推测）。
- **学界背书措辞：** 官网写 "supported and affected by" Pearl、Schölkopf 等，没有写正式顾问；中文通稿的「顾问网络」说法更强，引用时以官网措辞为准。
- **同名混淆：** 与 arXiv 2503.18945 的 Aether 几何世界模型无关。

## 关联页面

- [CRIS-0：因果驱动的真实世界机器人系统](./aether-cris-0.md)
- [CausalWM：因果思维链具身世界模型](./paper-causalwm.md)
- [RSIAgent：自主探索式递归自我改进](./aether-rsiagent.md)
- [CausalVAE World Models](./paper-causalvae-world-models.md) — 另一条「世界模型 + 因果层」学术路线
- [PAN 世界模型](./paper-sa-2511-09057-pan-a-world-model-for-general-interactable-and-l.md) — Blog 03 引用的通用世界模型
- [Latent Actions Matter](./paper-latent-actions-matter.md) — 潜动作对照
- [World Action Models](../concepts/world-action-models.md)
- [世界模型功能分类](../concepts/functional-taxonomy-world-models.md)
- [生成式世界模型](../methods/generative-world-models.md)
- [VLA](../methods/vla.md)
- [Cosmos 3](./cosmos-3.md) — CausalWM 博文的对比对象
- [Generalist AI](./generalist-ai-robotics.md) — 规模优先路线的商业对照
- [机器人世界模型：动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- [海外具身智能实验室版图 2026](../overview/overseas-embodied-ai-labs-landscape-2026.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [Aether AI 官网 Blog / News 列表核查（2026-10-10，11 篇 + 3 条）](../../sources/sites/aetherlabs-blog-index.md)
- [奠基四篇博文（2026-05-17，来源归档）](../../sources/blogs/aether_foundations_2026-05.md)
- [$20M 种子轮公告与通稿（来源归档）](../../sources/blogs/aether_seed_round_2026-06.md)
- [RSIAgent（来源归档）](../../sources/blogs/aether_rsiagent.md)
- 官网主页：<https://aetherlabs.ai/>
- 融资公告：<https://aetherlabs.ai/news/aether-ai-raises-20m-seed-round.html>
- GlobeNewswire 通稿（转载页，投资方）：<https://kdhnews.com/online_features/press_releases/aether-ai-raises-20-million-seed-round-to-build-causal-world-models-for-the-next/article_17d56198-cba6-5c6d-ae05-7b34fa40b7aa.html>
- 中文通稿（凤凰网，四层架构）：<https://tech.ifeng.com/c/8u3GHGpPynC>
- CVPR 2026 演讲通稿（四层架构英文表述）：<https://kdhnews.com/online_features/press_releases/beyond-correlation-aether-ais-prof-biwei-huang-introduces-causal-world-models-at-cvpr-2026/article_929091dc-674b-58a9-bdd8-5863aefa341a.html>
- 机器之心深度稿（36 氪）：<https://www.36kr.com/p/3866596553561095>
- 成立时间：San Diego Business Journal，*Aether Taking AI to Its Next Logical Step*（2026-07-06）<https://sdbj.com/technology/aether-taking-ai-to-its-next-logical-step/>
- Causal-Copilot 论文：<https://arxiv.org/abs/2504.13263>

## 推荐继续阅读

- [Causality and the Next AI Paradigm](https://aetherlabs.ai/articles/causality-and-the-next-ai-paradigm.html) — 公司纲领与 38 条因果文献
- [Learning Causal World Models](https://aetherlabs.ai/articles/learning-causal-world-models.html) — 闭环配方原文
- [Causal-Copilot GitHub](https://github.com/Lancelot39/Causal-Copilot) — 唯一可直接运行的奠基期成果
