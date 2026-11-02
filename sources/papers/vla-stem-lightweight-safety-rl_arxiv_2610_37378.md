# VLA-Stem（arXiv:2610.37378）

> 来源归档（paper）

- **标题：** VLA-Stem: A Lightweight Safety Mechanism for Vision-Language-Action Models via Reinforcement Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.37378>
- **PDF：** <https://arxiv.org/pdf/2610.37378>
- **作者：** Zou, Bojun, Yu, Wenhao, Li, Jiaohao, Lin, Jiaxuan, Ye, Guanting, Geng, Haohan, Yu, Weihao, Zhang, Yanyong et al.
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** VLA-Stem：基于强化学习的轻量化视觉语言动作模型安全机制。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** VLA 主要学习完成操作任务，但面对动态障碍物和空间约束时，原始动作块可能发生碰撞。直接重新微调 VLA 成本较高。
3. **周报所述创新点：** 在冻结 VLA 下方增加强化学习安全策略，将 VLA 动作块转换为参考轨迹。安全策略根据本体状态和局部障碍物信息调整动作，同时尽量保留原始任务意图。训练采用 Teacher–Student 蒸馏和课程学习，在六种受约束仿真环境中，平均无碰撞任务成功率达到 **52.8%**，最强安全基线为 **23.7%**。
4. **arXiv 摘要（英文节选）：** Vision-Language-Action (VLA) models integrate high-level reasoning with low-level action execution for end-to-end robotic control. However, most VLA models target static manipulation in unconstrained environments, overlooking spatial constraints and dynamic obstacles. Fine-tuning VLAs on obstacle-avoidance data is a straightforward solution, but collecting such data is costly, especially for dynamic scenarios, and may degrade their original task capabilities.

**对 wiki 的映射**

- [paper-vla-stem-lightweight-safety-rl](../../wiki/entities/paper-vla-stem-lightweight-safety-rl.md)
