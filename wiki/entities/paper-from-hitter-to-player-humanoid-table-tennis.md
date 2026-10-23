---
type: entity
project_id: from-hitter-to-player-humanoid-table-tennis
tags: [paper, humanoid, table-tennis, motor-control, self-play, strategy]
status: complete
updated: 2026-10-23
arxiv: "2610.14067"
venue: "2026 · arXiv preprint"
related:
  - ../methods/table-tennis-strategy-skill-learning.md
  - ./paper-notebook-towards-versatile-humanoid-table-tennis.md
  - ./paper-sony-ai-ace-table-tennis.md
sources:
  - ../../sources/papers/from_hitter_to_player_arxiv_2610_14067.md
summary: "From Hitter to Player（arXiv:2610.14067）：先通过闭环自博弈提升人形击球执行，再冻结动作栈、单独训练按对手状态选落点的竞争策略。"
---

# From Hitter to Player：人形乒乓球中的动作执行与策略分离

**From Hitter to Player**（Dong et al., 2026，[arXiv:2610.14067](https://arxiv.org/abs/2610.14067)）把人形机器人竞技乒乓球分成**回球动作执行**和**落点策略选择**两层，以期望落点作为两者接口；其关键实验是先在闭环回合中自博弈改进动作策略，再冻结整个动作栈训练逐拍选点策略。

## 一句话定义

**先让人形机器人学会把球稳定打到指定区域，再固定击球能力，单独学习如何根据对手位置选择落点。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过环境交互学习运动与决策策略 |
| MDP | Markov Decision Process | 将逐拍对抗中的状态、动作与回报形式化 |
| Sim2Real | Simulation to Reality | 从仿真训练迁移到真实硬件执行的过程 |

## 为什么重要

- **明确分离执行与战术。** 落点作为模块接口，使得“打得到指定位置”和“选择有利位置”可以分别训练、评估。
- **动作层在真实来球分布中改进。** 闭环自博弈让机器人处理对手产生的来球，不只针对预先指定的来球训练。
- **战术改进可单独归因。** 冻结完整运动栈后只改变选点策略，报告的得分胜率差异更能反映策略层效果。

## 核心流程

1. **目标条件击球策略：** 模型式击球规划器给出可执行击球参考，学习控制器执行全身动作；目标变量是预期回球落点。
2. **闭环动作自博弈：** 运动策略在连续回合中接收由对手生成的来球，落点目标仍按任务采样，重点是改善击球执行。
3. **冻结动作栈并训练战术：** 固定规划器和运动控制栈，训练逐拍选择回球落点的策略，通过竞争性自博弈优化比赛得分。

**数据流：** 期望落点 → 模型式击球规划器 → 学习到的全身控制器 → 冻结的动作执行栈；逐拍策略根据对手状态更新期望落点。

## 关键结果

| 评测 | 论文报告 |
|------|----------|
| 对中路落点基线 | 学习策略赢得 89% 的分数 |
| 对随机落点基线 | 学习策略赢得 82% 的分数 |
| 对远离对手落点基线 | 学习策略赢得 73% 的分数 |
| 真实硬件对打 | 与休闲人类对手完成 11 分制比赛，包含多拍正手与反手回球 |

上述百分比是论文摘要所述的**分数胜率**，不要误读为整场比赛胜率。

## 与相关乒乓球工作的关系

| 工作 | 任务与贡献 | 与本文的区别 |
|------|------------|--------------|
| [PhysicsPingPong](../methods/table-tennis-strategy-skill-learning.md) | 仿真角色中的多击球技能与落点策略 | 本文聚焦人形机器人全身击球及真实硬件对打 |
| [Towards Versatile Humanoid Table Tennis](./paper-notebook-towards-versatile-humanoid-table-tennis.md) | 人形乒乓球统一强化学习论文条目 | 独立论文；不要把两者的方法或结果合并 |
| [Sony AI Ace](./paper-sony-ai-ace-table-tennis.md) | 定制 8-DOF 机器人在规则对局中的竞技乒乓球 | 平台、感知与系统架构不同；本文使用人形机器人并显式冻结动作栈训练策略 |

## 工程实践

- **先定接口，再分层优化：** 用落点目标作为可观测、可评估的执行接口，减少战术策略与低层击球控制相互干扰。
- **评估运动层和策略层：** 对动作层报告落点准确度与合法回球率；冻结动作栈后再比较不同选点策略的对局结果。
- **把自博弈阶段说清楚：** 运动自博弈使用不考虑对手的目标采样；只有后续冻结运动栈后的竞争策略阶段明确学习对手相关选点。
- **开源边界待核实：** arXiv 关联项目页当前无法访问，本页不据此声称代码或权重已开源。

## 结论

**这项工作把人形乒乓球中的“击球控制”与“比赛决策”拆成可独立训练的两层：先用闭环自博弈强化动作，再在冻结动作栈上优化逐拍落点选择。** 论文报告的策略胜率是得分级而非比赛级指标；项目页及代码可用性仍需后续核实。

## 关联页面

- [Table Tennis Strategy & Skill Learning（PhysicsPingPong）](../methods/table-tennis-strategy-skill-learning.md)
- [Towards Versatile Humanoid Table Tennis](./paper-notebook-towards-versatile-humanoid-table-tennis.md)
- [Sony AI Ace](./paper-sony-ai-ace-table-tennis.md)

## 参考来源

- [来源归档：From Hitter to Player](../../sources/papers/from_hitter_to_player_arxiv_2610_14067.md)
- [arXiv:2610.14067](https://arxiv.org/abs/2610.14067) — 摘要、作者、日期、项目页链接与正文。

## 推荐继续阅读

- [论文 HTML](https://arxiv.org/html/2610.14067v1)
- [arXiv 摘要与 PDF](https://arxiv.org/abs/2610.14067)
