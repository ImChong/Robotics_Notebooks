# TRACE: Efficient Transfer Learning for Neural Whole-Body Control Across Humanoid Embodiments

> 来源归档（ingest · arXiv v1）

- **中文题目：** TRACE：面向不同人形机器人形态的神经全身控制高效迁移学习
- **类型：** paper / humanoid / behavior-foundation-model / cross-embodiment / whole-body-tracking
- **arXiv：** https://arxiv.org/abs/2610.17999
- **HTML：** https://arxiv.org/html/2610.17999v1
- **项目页：** https://bfm-trace.github.io/
- **作者：** Qingyu Lan、Zhihao Tian、Zifan Wang、Junwei Liang
- **机构：** 香港科技大学（广州）、香港科技大学
- **提交日期：** 2026-10-13
- **入库日期：** 2026-10-17
- **一句话说明：** 复用 SONIC 的 G1 预训练控制器，通过关节运动学/动力学预对齐、筛选后的紧凑动作集、编码器蒸馏与低秩强化学习，将动作跟踪迁移到 Unitree H2；论文报告约 8 小时训练和 96.08% 成功率。

## 摘录 1：跨具身问题与预对齐

单一人形机体上的行为基础模型虽能跟踪多样动作，换机后仍会因关节数量与排列、连杆几何、质量分布、执行器及接触动力学不同而失配。TRACE 先把运动学先验注入输入/输出投影层，再对齐动作尺度与关节名义动力学响应，降低微调开始时的形态差异。

## 摘录 2：动作集与参数高效适配

论文从经过质量筛选的参考动作中构造紧凑迁移集，兼顾动作类别平衡与命令特征空间覆盖。训练结合参考编码器蒸馏和低秩强化学习：前者把目标机表征锚定到预训练运动潜空间，后者适配目标机反馈控制；两阶段课程先建立稳定跟踪，再集中改进困难动作。

## 摘录 3：实验与可复用能力

TRACE 将 Unitree G1 上的 SONIC 迁移至 Unitree H2，使用 1000 万动作帧，在单张 RTX PRO 6000 上训练约 8 小时。论文报告在 67,953 段评测动作上的高度缩放跟踪成功率为 96.08%。作者将适配开销与 SONIC 从头预训练的 21,000 GPU 小时比较；该比值仅是硬件未归一化的 GPU 小时比较，且不包含 SONIC 源模型预训练成本。

论文还展示了 H2 真机动态动作跟踪和全身遥操作，并以采集的演示数据支持后续 VLA 策略后训练。

## 对 wiki 的映射

- 主实体：[TRACE 跨具身人形控制迁移](../../wiki/entities/paper-trace-humanoid-transfer.md)
- 对照：[SONIC-Transfer](../../wiki/entities/paper-sonic-transfer.md)、[Any2Any](../../wiki/entities/paper-any2any-cross-embodiment-wbt.md)
- 方法上下文：[SONIC motion tracking](../../wiki/methods/sonic-motion-tracking.md)、[Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md)、[跨具身策略迁移选型指南](../../wiki/queries/cross-embodiment-transfer-strategy.md)

## 来源

- arXiv 摘要与论文 HTML：<https://arxiv.org/html/2610.17999v1>
- arXiv 摘要页：<https://arxiv.org/abs/2610.17999>
