# MimicX: Policy-in-the-Loop Supervision Refinement for Video-Driven Humanoid Motion Tracking（arXiv:2610.09055）

> 来源归档（ingest）

- **标题：** MimicX: Policy-in-the-Loop Supervision Refinement for Video-Driven Humanoid Motion Tracking
- **类型：** paper / humanoid / motion-tracking / reinforcement-learning
- **arXiv abs：** <https://arxiv.org/abs/2610.09055>
- **PDF：** <https://arxiv.org/pdf/2610.09055>
- **项目页：** <https://nebulis-lab.com/MimicX> — [归档](../sites/mimicx-nebulis-lab.md)
- **官方代码：** **已开源** — <https://github.com/NEBULIS-Lab/MimicX>（Apache-2.0）；训练、失败诊断、监督精炼和异构执行器均有实现，见[仓库归档](../repos/mimicx.md)
- **策略与资产：** [MimicX-Policies](https://huggingface.co/Shuaijun/MimicX-Policies)、[MimicX-Assets](https://huggingface.co/datasets/Shuaijun/MimicX-Assets)
- **作者：** Shuaijun Liu、Chenglong Zhang、Xuhao Liu、Feiyang You、Yifan Liao、Shuyang Hao、Chaozhe Zhang、Chengyu Wu、Zhen Sun、Ningxin Su
- **机构：** 香港科技大学（广州）（The Hong Kong University of Science and Technology (Guangzhou)）
- **发表日期：** 2026-10-06（arXiv 首次提交）
- **入库日期：** 2026-10-12
- **一句话说明：** 把人形策略的执行失败变成可操作的监督信号：定位难时间段和出错身体部位，调整跟踪目标与 reset curriculum，并以重复 rollout 和 tracking guard 决定是否接受新策略。

## 核心摘录（面向 wiki 编译）

### 方法定位

论文聚焦视频驱动动作跟踪中「参考看起来合理、物理执行却失败」的问题。它不是单独提出新的单目动作重建器，而是在 GVHMR 重建、GMR 重定向和 PPO 跟踪策略之后加入闭环：用 rollout 错误诊断修改下一轮监督。

### 执行反馈如何改变训练

- **定位失败：** 从 rollout 找出困难时间窗口及误差突出的身体区域 / 通道。
- **提出候选：** 有界地调整 tracking objective 与 reset curriculum，再从当前策略继续训练。
- **门控保留：** 重复执行候选策略，以执行进度为优先，同时用 tracking guard 防止跟踪质量退化；不满足条件时保护当前策略。
- **异构加速：** MimicX-HLoop 将 GPU rollout、CPU 诊断和 I/O 组织成依赖感知任务，并核验并行执行结果。

### 评测坐标

| 项目 | 论文报告 |
|------|----------|
| 人形平台 | Unitree G1，29 DoF |
| 仿真与控制 | MuJoCo / MjLab；仿真 200 Hz，PPO policy 50 Hz |
| 核心任务 | Tennis Swing、Football Juggling、Dance Sequence、Kung Fu Sequence |
| 固定参考基线对照 | 四个任务中的 12 个配对比较均改善 body-tracking error 与 Robust Execution Horizon |
| 任务平均结果 | body-tracking error 降低 25.7%；execution horizon 提升 255.6% |

这些数字是四任务任务平均结果，不应外推为任意视频、机器人形态或真机上的普遍增益。

### 开源状态与复现边界

- 官方仓库含核心包、训练和 refinement 入口、文档、测试与论文配置；策略和资产在 Hugging Face 分别托管。
- 官方说明不再分发原始人类视频、SMPL 系列模型 / 参数、GVHMR 和 SONIC 预训练权重及第三方动作数据；部分授权参考只提供身份 / 哈希记录。
- 因而算法代码已开源，但复现实验仍需从获授权来源准备上游模型、视频和动作参考，并核对输入版本与校验和。

## 对 wiki 的映射

- 新建统一项目实体：[MimicX](../../wiki/entities/paper-mimicx.md)（论文、项目页、源码和 HF 资产共用一个项目节点）。
- 交叉：[SONIC 运动跟踪](../../wiki/methods/sonic-motion-tracking.md)、[模仿学习](../../wiki/methods/imitation-learning.md)、[GVHMR](../../wiki/entities/gvhmr.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)。
