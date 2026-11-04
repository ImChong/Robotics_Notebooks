# RAFT（arXiv:2611.04368）

> 来源归档（paper）

- **标题：** RAFT: Reasoning-Aligned Fine-Tuning Improves Robustness of VLA Post-Training
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.04368>
- **PDF：** <https://arxiv.org/pdf/2611.04368>
- **项目页：** <https://princeton-prism.github.io/RAFT/>
- **代码：** <https://github.com/princeton-prism/RAFT>
- **入库日期：** 2026-11-04
- **线索来源：** [机器人做错了，奖励模型为什么还在加分？｜附代码与评测入口](../blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)（具身智能小站，2026-11-04）
- **一句话说明：** 先用任务推理标注训练适配器，再冻结适配器学习动作，缓解微调后鲁棒性下降；作者在DROID、双臂YAM及仿真中检查视觉与指令变化。

## 开源状态

- **部分开源**（步骤 2.5 核查，2026-11-04）：公众号代码 princeton-prism/RAFT 经 git ls-remote 失败，项目页 HTTP 404，不能当作已开源。

## 核心摘录（公众号转述，未核验原文表格）

- 微调任务成功，也保住分布外能力。
- 先用任务推理标注训练适配器，再冻结适配器学习动作，缓解微调后鲁棒性下降；作者在DROID、双臂YAM及仿真中检查视觉与指令变化。
- 指标：公众号未给具体数字，本页不引用。

## 对 wiki 的映射

- [paper-raft-reasoning-aligned-finetuning](../../wiki/entities/paper-raft-reasoning-aligned-finetuning.md)
- [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-04-57-papers-technology-map.md)
