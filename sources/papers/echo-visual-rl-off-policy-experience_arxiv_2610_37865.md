# Echo（离策略视觉 RL）（arXiv:2610.37865）

> 来源归档（paper）

- **标题：** Echo: Scaling Visual Reinforcement Learning with Off-Policy Experience
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.37865>
- **PDF：** <https://arxiv.org/pdf/2610.37865>
- **作者：** Khandate, Gagan
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** Echo：利用离策略经验扩展视觉强化学习。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】人形／四足33篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
2. **周报所述问题：** 直接视觉 RL 需要持续渲染图像并执行视觉网络，计算成本高；Teacher–Student 两阶段蒸馏又存在模仿误差。
3. **周报所述创新点：** 让状态输入辅助策略反向模仿持续更新的视觉策略，并采集无需渲染的廉价经验；这些经验通过共享特权 Critic 参与离策略价值学习，直接更新视觉策略。该工作目前为初步仿真实验。
4. **arXiv 摘要（英文节选）：** Scaling visual reinforcement learning (RL) with GPU-accelerated simulation is computationally expensive due to rendering and visual-policy inference costs. Consequently, two-stage teacher--student distillation is commonly used to train visual policies, but introduces an imitation gap. In this preliminary study, we introduce \echomethod{}, \textbf{a single-stage off-policy RL framework that scales experience collection without proportionally increasing visual computation}.

**对 wiki 的映射**

- [paper-echo-visual-rl-off-policy-experience](../../wiki/entities/paper-echo-visual-rl-off-policy-experience.md)
