# Iterative Gumbel Planning（arXiv:2611.02236）

> 来源归档（paper）

- **标题：** Iterative Gumbel Planning for Continuous Control
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.02236>
- **PDF：** <https://arxiv.org/pdf/2611.02236>
- **项目页：** <https://liushaohuai5.github.io/ez_igp/>
- **代码：** <https://github.com/liushaohuai5/EZ-IGP>
- **入库日期：** 2026-11-04
- **线索来源：** [机器人做错了，奖励模型为什么还在加分？｜附代码与评测入口](../blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)（具身智能小站，2026-11-04）
- **一句话说明：** 用分维离散策略对齐搜索动作，减少把胜出候选压回高斯分布的误差；结合Gumbel采样与多轮搜索，在连续控制基准中比较性能和样本效率。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-04）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 让连续控制策略与搜索候选对齐。
- 用分维离散策略对齐搜索动作，减少把胜出候选压回高斯分布的误差；结合Gumbel采样与多轮搜索，在连续控制基准中比较性能和样本效率。
- 指标：公众号未给具体数字，本页不引用。

## 对 wiki 的映射

- [paper-iterative-gumbel-planning](../../wiki/entities/paper-iterative-gumbel-planning.md)
- [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-04-57-papers-technology-map.md)
