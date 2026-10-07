# Freeform Preference Learning 项目页

> 来源归档（site）

- **标题：** Freeform Preference Learning for Robotic Manipulation
- **项目页：** <https://freeform-pl.github.io/fpl.website/>
- **论文：** <https://arxiv.org/abs/2606.32027>
- **代码：** 仿真 <https://github.com/freeform-pl/fpl>；真实机器人 <https://github.com/freeform-pl/fpl_real>
- **机构：** Stanford University
- **入库日期：** 2026-10-07
- **一句话说明：** 将自由文本定义的多轴人类偏好变成稠密奖励和可测试时调节的机器人策略。

## 项目页核查

- **代码状态：** 页面分别链接仿真和真实机器人代码仓库，均已公开。
- **任务：** 摆方块入碗、叠短裤、摆盘吐司、布置餐桌等四项真实操作；另有两个仿真任务。
- **结果：** 页面报告四项真实任务平均 progress 为 75，最佳基线为 37，提升 38 个百分点。

## 方法摘要

比较两条轨迹时可沿若干自然语言轴独立给偏好，奖励模型为每个轴生成分数；策略学习时可组合轴，在测试时重新设定各轴目标而无需重新训练。
