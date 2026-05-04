# kung_fu_athlete_bot

> 来源归档（ingest）

- **标题：** KungFuAthleteBot
- **类型：** repo / paper
- **来源：** GitHub / arXiv
- **入库日期：** 2026-05-04
- **最后更新：** 2026-05-04
- **一句话说明：** A Kung Fu Athlete Bot That Can Do It All Day: Highly Dynamic, Balance-Challenging Motion Dataset and Autonomous Fall-Resilient Tracking

## 核心论文摘录（MVP）

### 1) A Kung Fu Athlete Bot That Can Do It All Day: Highly Dynamic, Balance-Challenging Motion Dataset and Autonomous Fall-Resilient Tracking（Zhongxiang Lei et al., 2026）
- **链接：** <https://github.com/NPCLEI/KungFuAthleteBot> / <https://arxiv.org/abs/2602.13656>
- **核心贡献：**
  - 提供了一个高度动态、具有平衡挑战的武术动作数据集（包含长拳、南拳、太极等）。
  - 提出了一种针对 GVHMR（基于视频的人体网格恢复）提取数据中的根节点高度漂移问题的极端点校正方法。
  - 使用强化学习（分为三阶段：粗跟踪+基础跌倒恢复、精确运动跟踪、增强鲁棒性减少跌倒）在仿真和真机上实现了复杂动作序列（如5分钟连续太极动作）的抗跌倒跟踪。
  - 提供了一套完整的数据处理流水线：包含动作可视化（GVHMR）、动作重定向到机器人（GMR）以及用于训练的高效工作流（Unitree RL Mjlab）。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面的参考来源段落已添加 ingest 链接
