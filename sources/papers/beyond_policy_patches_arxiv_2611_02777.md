# Beyond Policy Patches（arXiv:2611.02777）

> 来源归档（paper）

- **标题：** Beyond Policy Patches: Distilling RL Specialists into Vision-Language-Action Policies
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.02777>
- **PDF：** <https://arxiv.org/pdf/2611.02777>
- **项目页：** <https://repair-vla.github.io/>
- **入库日期：** 2026-11-04
- **线索来源：** [机器人做错了，奖励模型为什么还在加分？｜附代码与评测入口](../blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)（具身智能小站，2026-11-04）
- **一句话说明：** 先用专家数据初始化，再采集自身执行状态进行价值加权蒸馏；让专家只提供训练监督，部署回到单一VLA，在精细操作阶段比较成功率与动作平滑性。

## 开源状态

- **项目页未列 GitHub**（步骤 2.5 核查，2026-11-04）：项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。

## 核心摘录（公众号转述，未核验原文表格）

- 把强化学习专家收进同一个VLA。
- 先用专家数据初始化，再采集自身执行状态进行价值加权蒸馏；让专家只提供训练监督，部署回到单一VLA，在精细操作阶段比较成功率与动作平滑性。
- 指标：公众号未给具体数字，本页不引用。

## 对 wiki 的映射

- [paper-beyond-policy-patches](../../wiki/entities/paper-beyond-policy-patches.md)
- [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-04-57-papers-technology-map.md)
