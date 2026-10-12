# VaDar: Implicit Goal Conditioning via Value Disaggregation

> 来源归档（ingest）

- **类型：** paper
- **论文 PDF：** <https://shashwatsaxena13.github.io/vadar-site/assets/vadar-paper.pdf>
- **项目页：** <https://shashwatsaxena13.github.io/vadar-site/>
- **作者：** Shashwat Saxena、Mehul Goel、Sreyas Venkataraman、Sarvesh Patil、Max Simchowitz
- **机构：** Carnegie Mellon University；Amazon Scholar / Amazon FAR（Max Simchowitz）
- **入库日期：** 2026-10-12
- **一句话说明：** 分别监督有序子目标的价值分量并共享 critic 表征，使用未来子目标价值训练部署时 goal-agnostic 的策略。
- **沉淀到 wiki：** [VaDar 实体页](../../wiki/entities/paper-vadar.md)

## 方法与问题摘要

长时程任务被拆为有序子目标。只围绕当前子目标训练可能诱发短视行为：阶段内进度更快，不代表整条任务链更容易完成。VaDar 对各子目标分别预测 action value，以对应阶段的时序差分目标监督共享 critic；然后聚合未来子目标价值指导策略更新，使测试时策略不必将显式子目标标签作为输入。

本归档不抄录未核对的精确结果数值；数字、基线和统计口径以论文 PDF 为准。

## 代码开放核查

项目页链接研究展示和论文。对应 GitHub Pages 源码是网站资产，不是算法实现；未发现官方训练 / 评估代码或公开数据入口。论文 PDF 可获取不等于实现已开源。

## 对 wiki 的映射

- [VaDar 实体页](../../wiki/entities/paper-vadar.md)
- [分层强化学习](../../wiki/methods/hierarchical-reinforcement-learning.md)
- [强化学习](../../wiki/methods/reinforcement-learning.md)
