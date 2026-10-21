# ZitongBo/awesome-embodied-self-improvement

- **URL：** <https://github.com/ZitongBo/awesome-embodied-self-improvement>
- **类型：** 具身自我改进论文与基准策展清单
- **许可证：** MIT（仓库元数据）
- **范围：** 机器人 / embodied intelligence / Physical AI 的部署后自我改进；纯软件 agent 仅作软件域参照
- **主要入口：** [README](https://github.com/ZitongBo/awesome-embodied-self-improvement)、[中文 README](https://github.com/ZitongBo/awesome-embodied-self-improvement/blob/main/README.zh-CN.md)、[分级方法](https://github.com/ZitongBo/awesome-embodied-self-improvement/blob/main/GRADING.md)、[逐篇表格](https://github.com/ZitongBo/awesome-embodied-self-improvement/blob/main/PAPERS.md)、[分类复核](https://github.com/ZitongBo/awesome-embodied-self-improvement/blob/main/CLASSIFICATION.md)
- **入库日期：** 2026-10-17
- **开源状态：** 清单、论文分级记录与分类材料公开（MIT）；这是文献导航仓库，不是可运行的机器人自改进框架。条目所指代码与数据需逐篇核验。
- **当前规模（README，2026-10-17 检查）：** 45 项具身自我改进工作、3 个 benchmark、4 篇综述，另有 2 个软件域参照；README 报告 L1 8 项、L1–L2 边界项 1 项、L2 13 项、L3 17 项、L4 下界 1 项及新生工作 5 项。
- **Tags：** #embodied-self-improvement #physical-ai #robot-learning #recursive-self-improvement #curated-list

## 一句话

按「改进后是否新增此前无法执行的动作 / 技能」对部署后机器人自我改进工作分级的开放文献索引，覆盖从 episode 内反馈到改进机制自身演化的 L1–L4。

## 核心判据

清单使用一个主判据区分持久优化与能力扩张：自我改进后，系统的可执行动作 / 技能集合是否增加了一个之前无法完成的动作？

| 级别 | 判定 | 阅读时的含义 |
|------|------|--------------|
| L1 | 改进在 episode 重置后归零 | 当前任务内的计划、提示或约束调整，不代表持续学习 |
| L2 | 改进可保留，但动作集合不变 | 记忆、规则、调度、参数或已有技能使用方式更好 |
| L3 | 持久增加此前不可行的动作 / 技能 | 能力集合扩张，可指出新增的可执行单元 |
| L4 | 改进「产生改进」的机制本身 | 诊断、验证或晋级逻辑也被系统改进 |

该标准把自我描述与机制证据分开：某工作称为 self-evolving 或 RSI，不足以决定等级；混合 L2/L3 的系统按是否出现真实新增可执行单元来判定。README 说明目前前沿集中在 L3，PhysEvo 触及 L4 的较弱下界，完整的 Physical AI L4 仍是开放问题。

## 使用边界

- 这是活跃的社区策展清单；数量与分级会随维护者审查和贡献更新，应以仓库当前版本为准。
- README 中的等级不替代逐篇论文核读。检查实验是否真机或仿真、改进状态是否跨 reset 保留，以及评测是否有 held-out / promotion gate。
- 该仓库本身不提供统一训练 / 推理运行时。其 MIT 许可证指向该清单仓库，不能据此推断所列论文代码或数据也采用相同许可。
- 可与通用 agent 层 [Awesome RSI](https://github.com/Prism-Shadow/awesome-rsi) 对照：后者按被改 artifact 与 online/offline protocol 整理，本文清单则聚焦物理具身任务中的能力增量。

## 对 wiki 的映射

- 概念页：[递归自改进（RSI）](../../wiki/concepts/recursive-self-improvement.md)
- 通用 RSI 策展页：[Awesome RSI](../../wiki/entities/awesome-rsi.md)

## 参考来源（原始）

- [README（统计与范围）](https://github.com/ZitongBo/awesome-embodied-self-improvement/blob/main/README.md)
- [GRADING.md（等级定义）](https://github.com/ZitongBo/awesome-embodied-self-improvement/blob/main/GRADING.md)
- [CLASSIFICATION.md（重分类检查）](https://github.com/ZitongBo/awesome-embodied-self-improvement/blob/main/CLASSIFICATION.md)
- [PAPERS.md（逐篇证据表）](https://github.com/ZitongBo/awesome-embodied-self-improvement/blob/main/PAPERS.md)
- [CONTRIBUTING.md（范围与投稿要求）](https://github.com/ZitongBo/awesome-embodied-self-improvement/blob/main/CONTRIBUTING.md)
