# FIBER 论文来源归档

## 元数据

- **题目：** Action-Aligned Latent Planning for Versatile Humanoid Loco-Manipulation
- **作者：** Yutang Lin, Guanya Shi, C. Karen Liu
- **机构：** Amazon Frontier AI & Robotics（Amazon FAR）、Carnegie Mellon University（CMU）、Stanford University
- **预印本：** https://arxiv.org/abs/2610.24675
- **项目页：** https://fiberaa.github.io/（见 [项目页归档](../sites/fiber.md)）
- **官方代码仓库：** https://github.com/Yutang-Lin/FIBER（见 [仓库归档](../repos/fiber.md)）
- **版本：** v1 于 2026-10-17 提交；arXiv 页面显示 v2 于 2026-10-20 修订
- **一句话说明：** FIBER 用冻结行为基础模型的动作对齐潜变量连接感知规划与高频全身控制，在 G1 上统一箱子拾取、开门和抓取。

## 论文要解决的问题

人形移动操作需要在高层任务决策与全身闭环控制之间建立合适接口。运动学目标或接触目标只对预先指定变量反馈；直接预测低层动作则要求规划器自己建模高频闭环响应。

## 方法摘录

1. **独立训练技能专家：** 箱子拾取、开门和物体抓取分别训练，各自使用对应观测、奖励与任务设置。
2. **Latent Policy Projection（LPP）：** 沿专家轨迹寻找冻结 BFM 的潜变量序列，使 BFM 在访问到的状态上诱发与专家局部控制响应相匹配的动作。
3. **汇总闭环潜轨迹：** 用投影后的技能运行收集 latent rollouts；专家策略不再作为规划器输入。
4. **单一离线感知规划器：** 条件输入包含深度、本体状态和任务指令，以 flow matching 学习多个任务共享的潜变量规划策略。
5. **Adaptive-Horizon BFM（AH-BFM）：** 通过 multi-gamma Forward-Backward 学习支持精细控制和时间跨度更长的全身行为；部署时 BFM 继续根据新状态闭环执行。

## 论文报告的结果

- **仿真任务：** 箱子拾取、开门、物体抓取；每项任务覆盖 128 种变化。
- **平均成功率：** FIBER 42.0%；Projected Latent Replay 11.6%；Kinematic Planning 3.5%。
- **真机：** 官方项目页展示 Unitree G1 从机载深度观测执行箱子拾取、开门与抓取，并报告技能串联、waypoint 条件引导和 sim-to-real 演示；这些演示不等同于摘要中的仿真成功率。

## 开源与复现边界

官方项目页链接到 [Yutang-Lin/FIBER](https://github.com/Yutang-Lin/FIBER)。截至 2026-10-21，仓库 README 表示代码 coming soon，仓库当前主要包含预览素材、论文/项目链接、演示入口与引用；尚不能据此复现训练、投影或部署流程。详见 [官方仓库归档](../repos/fiber.md)。

## Robotics_Notebooks 映射

- 详情节点：[wiki/entities/paper-fiber-action-aligned-latent-planning.md](../../wiki/entities/paper-fiber-action-aligned-latent-planning.md)
- 任务入口：[wiki/tasks/loco-manipulation.md](../../wiki/tasks/loco-manipulation.md)

## 一手来源

- arXiv 摘要与版本记录：<https://arxiv.org/abs/2610.24675>
- 官方项目页：<https://fiberaa.github.io/>
- 官方代码仓库：<https://github.com/Yutang-Lin/FIBER>
