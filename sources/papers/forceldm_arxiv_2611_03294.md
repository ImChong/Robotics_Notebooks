# ForceLDM（arXiv:2611.03294）

> 来源归档（paper）

- **标题：** ForceLDM: A Force-aware Latent Dynamics Model for Contact-Rich Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.03294>
- **PDF：** <https://arxiv.org/pdf/2611.03294>
- **项目页：** <https://tpy001.github.io/ForceLDM/>
- **代码：** <https://github.com/tpy001/ForceLDM-code>
- **入库日期：** 2026-11-04
- **线索来源：** [机器人做错了，奖励模型为什么还在加分？｜附代码与评测入口](../blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)（具身智能小站，2026-11-04）
- **一句话说明：** 训练时利用未来力矩和光流构建教师，再将接触动力学表征蒸馏给当前观测策略；在五类真机接触任务中检查新物体、扰动及噪声课程的作用。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-04）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 把未来接触经验蒸馏给当前策略。
- 训练时利用未来力矩和光流构建教师，再将接触动力学表征蒸馏给当前观测策略；在五类真机接触任务中检查新物体、扰动及噪声课程的作用。
- 指标：公众号未给具体数字，本页不引用。

## 对 wiki 的映射

- [paper-forceldm](../../wiki/entities/paper-forceldm.md)
- [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-04-57-papers-technology-map.md)
