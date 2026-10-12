# Discovering state-of-the-art reinforcement learning algorithms

> 来源归档；以下为归纳，不是论文全文转载。

- **类型：** paper / meta-learning / RL-algorithm-discovery
- **论文：** <https://www.nature.com/articles/s41586-025-09761-x>
- **DOI：** 10.1038/s41586-025-09761-x
- **作者：** Junhyuk Oh、Gregory Farquhar、Iurii Kemaev、Dan A. Calian、Matteo Hessel、Luisa Zintgraf、Satinder Singh、Hado van Hasselt、David Silver
- **机构：** 谷歌 DeepMind（Google DeepMind）
- **发表：** 2025-10-22；Nature 648, 312–319 (2025)
- **项目：** <https://google-deepmind.github.io/disco_rl/>
- **代码：** <https://github.com/google-deepmind/disco_rl>
- **入库 / 核查：** 2026-10-12
- **配套归档：** [项目页](../sites/discorl.md)、[官方实现](../repos/disco_rl.md)
- **一句话说明：** 通过双层优化学习策略与预测的更新目标，把 RL 算法设计本身变成可扩展的元学习任务。

## 核心摘录

1. **学习规则而非动作策略。** 内层智能体参数 θ 随经验更新，外层元网络参数 η 根据更新后的收益优化。未来奖励、终止标记和预测参与目标构造；y、z 的含义不由人预先指定。
2. **保留归纳偏置。** 策略、观测条件预测 y、动作条件预测 z 以目标匹配学习；还保留动作价值 q 和辅助策略预测 p。它不是没有人类设计约束的代码生成系统。
3. **跨任务的是规则。** Disco57 在 Atari57 上发现，Disco103 在 Atari57 + ProcGen16 + DMLab30 上发现；迁移后仍训练新智能体，不是零样本直接执行任务。
4. **结果与预算分开读。** Disco57 的 Atari 人类归一化 IQM 为 13.86；Disco103 在 Crafter 达到人类水平，在 Sokoban 接近 MuZero。论文的发现预算分别为 1,024 TPUv3 核 × 64 小时和 2,048 核 × 60 小时，不是部署算力。
5. **机制分析。** 学到的预测含未来显著事件信息；移除预测或阻止目标依赖未来预测会降低表现。外层使用跨 20 次智能体更新的截断元梯度；该窗口不是控制频率。

## 对 wiki 的映射

- [DiscoRL 统一论文实体](../../wiki/entities/paper-discorl.md)：机制、评测、运行时序与机器人适用边界。
- [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)：补充自动发现学习规则的方向入口。

## 开放与核验边界

项目页直接链接官方仓库；仓库含 Catch 示例、元训练 / 评估 notebook 和 Disco103 元参数。公开的是最小 JAX harness，不等于完整大规模实验集群。论文材料为 CC-BY，源码 Apache-2.0；本归档注明原作者与官方来源。此轮检查源码与文档，未运行官方训练。
