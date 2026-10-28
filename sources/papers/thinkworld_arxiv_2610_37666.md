# ThinkWorld: World Imagination Reinforces Chain-of-Dynamics Reasoning

- **类型：** 论文
- **作者：** Chaehyun Kim、Chi-Pin Huang、Jisoo Kim、Min-Hung Chen、Yu-Chiang Frank Wang、Seungryong Kim、Fu-En Yang
- **机构：** 英伟达（NVIDIA）、韩国科学技术院（KAIST）
- **论文：** <https://arxiv.org/abs/2610.37666> · [PDF](https://arxiv.org/pdf/2610.37666)
- **项目页：** <https://thinkworld-project.github.io/>
- **日期：** 2026-10-27（v1）；2026-10-28（入库）
- **关联 wiki：** [ThinkWorld](../../wiki/entities/paper-thinkworld.md)

## 核心摘录与 wiki 映射

1. **动作推理：** Chain-of-Dynamics（CoD）以离散运动 token 表示连接当前和未来状态的中间动力学；首先用有动作标签的示范初始化。映射到 [核心机制](../../wiki/entities/paper-thinkworld.md)。
2. **世界想象：** 预测 CoD 对应的未来 DINO 特征，与视频中观测到的未来特征比较，以强化学习提升一致性更高的 CoD；无需为视频标注动作。映射到 [核心机制](../../wiki/entities/paper-thinkworld.md)。
3. **技能组合：** 烤箱实验使用 100 条机器人原子技能轨迹及 90 条人类完整任务视频；论文展示开门、放左侧物体、关门的成功轨迹，属于定性结果。映射到 [实验与评测](../../wiki/entities/paper-thinkworld.md)。
4. **评测边界：** 其他三个真机任务的分布外测试为 52/60 次成功；训练结果来自单个随机种子。映射到 [实验与评测](../../wiki/entities/paper-thinkworld.md)。

## 开放状态

截至 2026-10-28，项目页访问失败，代码、权重和数据是否公开均待核实；不能据此判断是否开源。参见 [项目页记录](../sites/thinkworld.md)。
