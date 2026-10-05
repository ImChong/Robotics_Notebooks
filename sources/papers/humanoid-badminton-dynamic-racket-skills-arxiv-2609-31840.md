# Humanoid Badminton: Learning Dynamic Racket Skills from Limited Human Motion Data

> 来源归档（论文；关键内容摘录与方法/实验归纳）

- **类型：** paper
- **作者：** Jingzhi Cui, Zhexiong Wang, Bangjie Xu, Pengyu Zhao, Youyuan Li, Zhi Su, Peng Ren, Mengdi Xu, Chao Yu, Yi Wu, Luyang Wang, Zhongyu Li
- **机构：** Tsinghua University; Hong Kong Embodied AI Lab; The Chinese University of Hong Kong; Beijing University of Civil Engineering and Architecture; Zhejiang University; DeepCybo
- **会议：** CoRL 2026（项目页标注 Accepted at CoRL 2026）
- **arXiv：** <https://arxiv.org/abs/2609.31840>
- **HTML：** <https://arxiv.org/html/2609.31840v1>
- **PDF：** <https://arxiv.org/pdf/2609.31840v1>
- **项目页：** <https://sunlight02.github.io/humanoid-badminton/>
- **提交日期：** 2026-09-25
- **入库/核查日期：** 2026-10-05
- **一句话说明：** 三阶段分层强化学习将有限的人类羽毛球击球动作扩为目标条件化潜技能，并在 Unitree G1 上完成多技能回球和人机连续对打。

## 核心摘录（策展，非全文）

- **问题与方法：** 稀疏且不完美的人类击球动作难以覆盖连续变化的羽毛球来球；直接任务优化则可能让动作显得不自然。论文提出任务随机化动作扩增、潜技能高层规划、上下文条件对抗正则三个训练阶段。
- **动作扩增：** 从重建/重定向动作中标注击球事件，围绕球拍接触位置、球拍速度和拍面朝向采样随机目标，以强化学习训练技能编码器和低层控制器。
- **高层规划：** 低层控制器冻结后，规划器根据来球状态和机器人状态输出连续潜技能码，在线组合技能。
- **仿真：** MJLab 中的 29 自由度 Unitree G1；每个策略对 1,000 个随机来球评测。本文方法 SR 88.3%、AC 5.98，Direct PPO 为 79.0% / 4.14，AMP 为 84.6% / 5.06。
- **真机：** 20 轮人机连续对打；报告 SR 89.4%、平均连续成功回球 8.42 次、最长 23 次。评测依赖动捕提供羽毛球与基座位置，人工标注回球种类；人类侧失误不计入机器人失败。
- **消融：** 去除任务随机化扩增后 SR 为 57.0%、无反手成功；过早加入规划器正则后 SR 为 55.9%，支持先学会回球、再约束技能使用的阶段顺序。

## 局限与风险

- 策略以成功回到对方有效区域为目标，尚未显式控制落点以实现战术性击球。
- 真机状态依赖动捕，不代表机载视觉已完成球位和基座估计闭环。
- 项目页未提供代码、模型权重或公开数据集链接；论文结果目前难以从公开资源独立复现。

## 对 wiki 的映射

- [Humanoid Badminton：从有限人类动作学习动态球拍技能](../../wiki/entities/paper-humanoid-badminton-dynamic-racket-skills.md)
- 对照：[Humanoid Whole-Body Badminton（Annealed RL）](../../wiki/entities/paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md)
- 对照：[LHBS：人形拟人羽毛球技能学习](../../wiki/entities/paper-notebook-learning-human-like-badminton-skills-for-humanoi.md)

## 参考来源（原始）

- [arXiv 摘要页](https://arxiv.org/abs/2609.31840)
- [arXiv HTML 全文 v1](https://arxiv.org/html/2609.31840v1)
- [作者项目页](https://sunlight02.github.io/humanoid-badminton/)
