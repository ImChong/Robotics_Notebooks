---
type: entity
tags: [paper, humanoid, loco-manipulation, sparse-pose, whole-body-control, VLA]
status: draft
updated: 2026-10-23
project_id: sp-hlm-sparse-pose-humanoid-control
arxiv: "2610.30044"
project: https://sp-hlm.github.io/
related:
  - ../tasks/loco-manipulation.md
  - ../methods/vla.md
  - ./paper-smpc2rl-loco-manipulation.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md
  - ../../sources/papers/sp_hlm_arxiv_2610_30044.md
  - ../../sources/sites/sp-hlm.md
summary: "SP-HLM combines five explicit task-critical body poses with a learned motion prior, linking a VLA planner to real-time humanoid whole-body control for loco-manipulation."
---

# SP-HLM：稀疏姿态驱动的人形全身控制

**一句话定义：** SP-HLM 用五个任务关键身体姿态表达空间意图，再用学习到的运动先验补全全身协调，连接高层 VLA 与低层全身控制。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SP-HLM | Sparse-Pose-Driven Motion Generation and Whole-Body Control for Humanoid Loco-Manipulation | 论文提出的稀疏姿态驱动全身控制框架 |
| VLA | Vision-Language-Action | 从多模态观察与指令预测任务目标的高层模型 |
| SP-MGM | Sparse-Pose Motion Generation Model | 根据稀疏目标实时生成协调全身动作 |
| SP-GMT | Sparse-Pose General Motion Tracker | 联合跟踪显式目标与隐式运动先验的控制器 |

## 为什么重要

高层策略若直接预测完整关节配置，动作维度高且要同时表达任务意图和全身协调；只给稀疏末端目标又不足以确定躯干、腿部如何配合。SP-HLM 将两种信息分开表示，减少高层动作接口负担。

## 核心原理

框架显式提供骨盆、左右手腕、左右脚踝五个任务关键姿态，同时由预训练运动编码器提供全身协调 latent。SP-VLA 从多模态观察和指令预测稀疏姿态与手指关节；SP-MGM 经两阶段蒸馏从稀疏目标生成全身动作；SP-GMT 将姿态约束与运动 latent 一起用于跟踪控制。

## 方法栈

`多模态观察 + 指令 → SP-VLA 稀疏目标 → SP-MGM 协调动作 / latent → SP-GMT 全身跟踪`

异步 action-chunk 推理使高层预测可以和低层动作执行并行，保持闭环响应。

## 实验与评测

论文摘要称，仿真和真实人形实验中，方法提升了运动跟踪、全身协调与 loco-manipulation 成功表现。摘要还报告 SP-MGM 相对生成模型基线达到 1.45× 生成速率、SP-HLM 在八个仿真任务总体成功率 52.75%；这些数字需结合论文定义的任务与比较基线解读。

## 与其他工作对比

| 接口 | 显式空间目标 | 全身协调表示 |
|------|--------------|--------------|
| SP-HLM | 骨盆、双腕、双踝五点姿态 | 学习到的 motion latent |
| [FIBER](./paper-smpc2rl-loco-manipulation.md) | 高层计划与动作基础模型接口 | 冻结行为基础模型生成交互动作 |
| 关节空间动作 | 完整关节目标 | 由高层直接承担动作协调 |

## 工程实践

- 高层 VLA 预测稀疏姿态和手指目标，低层运动模型负责补全全身动作。
- 异步 action-chunk 推理可减少高层推理阻塞对低层闭环的影响。
- **代码运行时序图不适用：** 本次核查未能访问项目页并确认可运行代码入口；待官方发布代码后补全复现时序。

## 结论

**SP-HLM 的核心取舍是显式保留任务空间目标，同时把全身协调交给学习到的运动先验。**

1. 五点接口比完整关节目标更紧凑，但明确携带骨盆、双腕和双踝目标。
2. 稀疏目标本身并不能决定全身动作，motion latent 是该接口的重要补充。
3. 52.75% 总体成功率来自论文定义的八个仿真任务，不应脱离任务集推广。
4. 项目页开放状态待核实，当前按 arXiv 来源记录，不声明代码已开源。

## 局限与风险

- 稀疏关键点选择会限制可直接表达的任务约束；关键点不足时依赖运动先验补全。
- VLA、动作生成和跟踪控制是多模块系统，误差可能跨层传递。
- 本次无法核验项目页中的代码和数据发布信息。

## 关联页面

- [Loco-Manipulation（移动操作）](../tasks/loco-manipulation.md)
- [VLA](../methods/vla.md)
- [FIBER](./paper-smpc2rl-loco-manipulation.md)

## 参考来源

- [SP-HLM 论文归档](../../sources/papers/sp_hlm_arxiv_2610_30044.md)
- [SP-HLM 项目页归档](../../sources/sites/sp-hlm.md)
- [arXiv:2610.30044](https://arxiv.org/abs/2610.30044)

## 推荐继续阅读

- [SP-HLM 项目页](https://sp-hlm.github.io/) — 论文关联的补充结果与视频
- [FIBER](https://fiberaa.github.io/) — 另一种高层规划与全身行为基础模型接口
