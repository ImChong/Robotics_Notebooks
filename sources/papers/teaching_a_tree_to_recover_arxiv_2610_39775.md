# Teaching a Tree to Recover（arXiv:2610.39775）

> 来源归档（paper）

- **标题：** Teaching a Tree to Recover: Online Distillation of an LLM Supervisor with Counterfactual Replay
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.39775>
- **PDF：** <https://arxiv.org/pdf/2610.39775>
- **代码：** <https://github.com/kristianceder/Counterfactual-Tree-Teaching>
- **入库日期：** 2026-10-30
- **线索来源：** [机器人做错了，奖励为什么还在增加？｜附代码与项目入口](../blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)（具身智能小站，2026-10-30）
- **一句话说明：** 利用反事实回放判断恢复动作是否必要或有害，在线更新决策树。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-30）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 把语言模型的脱困经验教给决策树。测试时无需调用语言模型即可处理熟悉类型的异常；当前主要证据来自预设导航布局的仿真。
- 利用反事实回放判断恢复动作是否必要或有害，在线更新决策树。
- 指标：公众号转述：测试时无需调用语言模型即可处理熟悉类型的异常；当前主要证据来自预设导航布局的仿真。

## 对 wiki 的映射

- [paper-teaching-a-tree-to-recover](../../wiki/entities/paper-teaching-a-tree-to-recover.md)
- [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-30-28-papers-technology-map.md)
