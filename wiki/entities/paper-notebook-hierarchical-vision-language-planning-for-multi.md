---
type: entity
tags: [paper, humanoid-paper-notebooks, humanoid, manipulation, vlm, hierarchical-control, imitation-learning, whole-body-control, mit, unitree]
status: complete
updated: 2026-09-28
arxiv: "2506.22827"
venue: "RSS 2025 Workshop (Robot Planning in the Era of Foundation Models)"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ./paper-exbody-expressive-humanoid.md
  - ../methods/action-chunking.md
  - ../methods/hierarchical-reinforcement-learning.md
  - ../methods/saycan.md
  - ./paper-notebook-towards-proprioception-aware-embodied-planning-f.md
sources:
  - ../../sources/papers/humanoid_pnb_hierarchical-vision-language-planning-for-multi.md
summary: "让人形可靠执行复杂多步操作对工业/家庭部署很关键。本文提出一个分层规划与控制框架，含三层：① 底层——基于 RL 的控制器，负责跟踪全身动作目标；② 中层——一组用模仿学习训练的技能策略，为任务各步产生动作目标；③ 高层——一个视觉-语言规划模块，用预训练 VLM 决定执行哪个技能并实时监控其完成。在 Unitree G1 人形上做非抓握式（non-prehensile）取放任务、40+ 次真机试验，完整序列成功率 73%。"
---

# Hierarchical Vision-Language Planning for Multi-Step Humanoid Manipulation

**Hierarchical Vision-Language Planning for Multi-Step Humanoid Manipulation** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

让人形可靠执行复杂多步操作对工业/家庭部署很关键。本文提出一个分层规划与控制框架，含三层：① 底层——基于 RL 的控制器，负责跟踪全身动作目标；② 中层——一组用模仿学习训练的技能策略，为任务各步产生动作目标；③ 高层——一个视觉-语言规划模块，用预训练 VLM 决定执行哪个技能并实时监控其完成。在 Unitree G1 人形上做非抓握式（non-prehensile）取放任务、40+ 次真机试验，完整序列成功率 73%。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Hierarchical | 分层（高/中/低三层） |
| VLM | Vision-Language Model |
| Skill Policy | 技能策略（中层，IL 训练） |
| Whole-Body Tracking | 全身动作目标跟踪（底层 RL） |
| Non-prehensile | 非抓握式（推/拨等） |
| Real-time Monitoring | 实时监控技能完成 |

## 为什么重要

- **分层是多步长序列任务的可靠之道**：高层决策、中层技能、底层控制各司其职；
- **VLM 实时监控"某步是否完成"**是闭环关键，避免盲目推进；
- **非抓握操作**（推/拨）拓展了人形操作类型；
- 与 Proprio-MLLM、BiBo 等 VLM 规划工作互补。

## 解决什么问题

人形**多步操作**要可靠： - 单层端到端难覆盖**长序列**； - 需要**高层决策 + 中层技能 + 底层控制**协同； - 还要**实时知道某步是否完成**。

论文要：一个**分层、可监控**的多步人形操作框架。

## 核心机制

1. **三层规划-控制框架**：VLM 规划 + IL 技能 + RL 全身控制；
2. **VLM 高层选技能 + 实时监控完成**；
3. **多步可靠执行**：面向工业/家庭长序列任务；
4. **真机验证**：G1 非抓握取放，序列成功率 73%。

方法拆解（深读笔记小节）：三层架构；VLM 高层：选技能 + 监控；评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Hierarchical_Vision-Language_Planning_for_Multi-Step_Humanoid_Manipulation/Hierarchical_Vision-Language_Planning_for_Multi-Step_Humanoid_Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2506.22827> |
| 源码 | **未开源**：项目页 <https://vlp-humanoid.github.io/> 只提供论文 PDF，截至 2026-09-28 未列代码仓库 |
| 作者 | André Schakkal、Ben Zandonati、Zhutian Yang、Navid Azizan（MIT） |
| 发表 | 2025 年 6 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**平台**：29-DoF Unitree G1，外接两路 ELP RGB 相机提供双目第一视角；两张桌子的室内场景。

**三层结构**：
1. **底层**：在 ExBody 基础上扩展到 G1 全 29 DoF 的 RL 全身跟踪控制器（PPO，Isaac Gym 4096 并行环境，大量域随机化），50 Hz 输出关节目标，200 Hz PD 执行。
2. **中层**：用 RGB 人体姿态估计（HybrIK）+ 重定向做遥操作采集演示，训练类 ACT 的 HIT 模仿学习技能（每次输出 50 步动作块）。
3. **高层**：GPT-4o 规划器生成技能序列，VLM 执行监视器约 1 Hz（平均验证延迟约 1 s）判定每个技能是否完成并触发切换。

**任务**：从第一张桌子提起袋子，放到第二张桌子。

| | 取 | 放 | 完整取放 |
|------|---:|---:|---:|
| 试验次数 | 30 | 30 | 40 |
| 成功次数 | 27 | 25 | 29 |
| 成功率 | 90% | 83% | **73%** |

- **失败来源**（按频率）：中层技能失败最多（多在抓取阶段，物体位置超出训练分布）；其次是监视器过早判定完成（如袋子搭在桌边被误判为成功）；偶有规划器错误定位导致技能序列错误。
- 连续执行多个 50 步动作块时块间有轻微位置跳变，作者建议加入块间平滑。

## 与其他工作对比

| 路线 | 特点 | 与本文的差异 |
|------|------|------|
| 开环预定义技能序列 | 按脚本依次执行 | 本文用 VLM 监视器闭环判定技能完成 |
| 端到端 VLA | 依赖大规模配对演示，可解释性弱 | 本文规划与监视可直接检查，便于定位失败 |
| 符号式 TAMP | 需要领域专家写 PDDL | 本文用自然语言技能描述，扩展新技能更容易 |
| [Proprio-MLLM](./paper-notebook-towards-proprioception-aware-embodied-planning-f.md) | 改造 MLLM 注入本体感受，仿真评测 | 本文用现成 GPT-4o，在真机上验证执行闭环 |

## 结论

**这篇工作的主张不是某一层更强，而是把多步人形操作切成 VLM 规划 / IL 技能 / RL 全身控制三层，并让 VLM 额外承担「这一步做完了没有」的实时监控。**

- 真正起作用的是闭环监控：预训练 VLM 既选技能又实时判定完成，长序列里才不会盲目推进到下一步。
- 关键指标是整条序列而非单步——Unitree G1 真机 40+ 次试验、完整序列成功率 73%。
- 任务覆盖很窄：只有一个「提起袋子 → 放到另一张桌子」的桌面任务族（论文讨论节称之为非抓握式操作），尚未验证更多样的家居操作。
- 能力上限受中层技能库约束：技能策略由模仿学习训练，超出技能集合的步骤高层无从调度。
- 与 Proprio-MLLM、BiBo 等 VLM 规划工作互补，本页偏「分层 + 可监控」的执行侧闭环，而非规划表达能力本身。
- 单技能成功率（取 90%、放 83%）与整条序列（73%）的差距说明误差会沿序列累积；主要失败仍来自中层技能对分布外物体位置的泛化，而不是规划器。

## 局限与风险

- **评测范围很窄**：只有一个桌面「提袋取放」任务族，40 次试验，难以代表家居操作的组合多样性（论文自述）。
- **监视频率低**：1 Hz 验证适合慢速动作，不足以支撑快速恢复等技能，这也限定了技能抽象的时间尺度。
- **中层技能泛化弱**：遮挡或分布外物体位姿会让模仿技能停在抓取 / 放置前。
- **规划器幻觉**：GPT-4o 偶尔虚构物体、加入多余步骤或冗余的切换判断。
- **依赖商用 API**：高层调用 GPT-4o。
- **开源边界**：未见代码；源码运行时序图 **不适用**。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 底层跟踪控制器所基于的 ExBody：[paper-exbody-expressive-humanoid](./paper-exbody-expressive-humanoid.md)
- 中层技能 HIT 所借鉴的 ACT：[action-chunking](../methods/action-chunking.md)
- 分层控制：[hierarchical-reinforcement-learning](../methods/hierarchical-reinforcement-learning.md)
- LLM 选技能的早期代表 SayCan：[saycan](../methods/saycan.md)
- MLLM 双臂人形规划对照：[paper-notebook-towards-proprioception-aware-embodied-planning-f](./paper-notebook-towards-proprioception-aware-embodied-planning-f.md)

## 参考来源

- [humanoid_pnb_hierarchical-vision-language-planning-for-multi.md](../../sources/papers/humanoid_pnb_hierarchical-vision-language-planning-for-multi.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Hierarchical_Vision-Language_Planning_for_Multi-Step_Humanoid_Manipulation/Hierarchical_Vision-Language_Planning_for_Multi-Step_Humanoid_Manipulation.html>
- 论文：<https://arxiv.org/abs/2506.22827>
- 论文正文（方法、Table I、失败与局限）：<https://arxiv.org/html/2506.22827>

## 推荐继续阅读

- [机器人论文阅读笔记：Hierarchical Vision-Language Planning for Multi-Step Humanoid Manipulation](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Hierarchical_Vision-Language_Planning_for_Multi-Step_Humanoid_Manipulation/Hierarchical_Vision-Language_Planning_for_Multi-Step_Humanoid_Manipulation.html)
- 项目页：<https://vlp-humanoid.github.io/>
