# SWAP：测试时可控的人形机器人全身避障（arXiv:2610.25711）

> 来源归档（ingest）

- **标题：** SWAP: Test-Time Steerable Humanoid Whole-Body Collision Avoidance
- **作者：** Zeji Yi, Guanqi He, Xiaofeng Guo, Guannan Qu, Pieter Abbeel, Yan Duan, Karen Liu, Guanya Shi
- **机构：** Carnegie Mellon University、Amazon FAR
- **类型：** paper / humanoid / whole-body-control / collision-avoidance / diffusion-policy
- **arXiv abs：** <https://arxiv.org/abs/2610.25711>
- **arXiv HTML：** <https://arxiv.org/html/2610.25711>
- **PDF：** <https://arxiv.org/pdf/2610.25711>
- **项目页：** <https://iscoyizj.github.io/SWAP-steerable-humanoid-oa-website/>
- **代码：** 截至 2026-10-21，待开放；本条按项目提交信息记录，后续需复核项目页
- **平台：** Unitree G1；论文报告仿真与真机实验
- **一句话说明：** 将感知条件化的扩散式全身避障提案与低层 RL 稳定控制分层，再用测试时 Monte Carlo 采样和简单代价信号引导动作候选。

## 摘要级要点

- **问题：** 人形机器人需在狭窄、杂乱空间中协调全身避障并保持动态平衡；单一确定性动作难表达同一障碍的多种安全通过方式。
- **方法：** 高层感知条件化扩散策略生成多模态、避障动作提案；低层全身 RL 控制器跟踪提案并维持平衡。
- **训练监督：** 低层策略在随机指令下滚动执行，再仅在动作扫掠体积之外放置障碍并重新渲染机器人视角，构造无碰撞监督标签。
- **测试时引导：** Monte Carlo 多样采样结合速度、朝向或姿态等简单代价，按任务需求引导扩散采样，不依赖繁复 reward tuning。
- **实验口径：** 仿真及 Unitree G1 真机在静态与动态障碍中评估；测试时 steerability 对目标奖励平均提升 34%，同时保持较高避障率。该数值是目标奖励提升，不应改读为成功率提升。

## 相关资料

| 类型 | 链接 | 说明 |
|------|------|------|
| 论文 | <https://arxiv.org/abs/2610.25711> | 方法与仿真、G1 真机评测 |
| 项目页 | <https://iscoyizj.github.io/SWAP-steerable-humanoid-oa-website/> | 项目介绍与演示；代码待开放 |

## 对 wiki 的映射

- [SWAP 全身避障实体页](../../wiki/entities/paper-swap-humanoid-collision-avoidance.md)
- [Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md) — 动态平衡与全身运动上下文
- [Whole-Body Control](../../wiki/concepts/whole-body-control.md) — 高层动作提案与低层稳定控制分工
