# Dream What I Can：长程机器人操作的稀疏视觉规划（arXiv:2610.24981）

> 来源归档（paper）

- **标题：** Dream What I Can (DreamCan): Sparse Visual Planning for Long-Horizon Robot Manipulation
- **类型：** paper / robot manipulation / visual planning / video model
- **arXiv：** <https://arxiv.org/abs/2610.24981>
- **项目页：** <https://dreamcan.github.io>（由 arXiv 摘要页链接）
- **提交日期：** 2026-10-18
- **作者：** Chongkai Gao、Wenlong Huang、Haonan Chen、Lin Shao、Yilun Du、Jiajun Wu、Ruohan Zhang、Li Fei-Fei
- **一句话说明：** DreamCan 将预训练视频模型改造成高层规划器，生成稀疏视觉子目标，再由低层 visuomotor 策略执行，以支持长程操作。

## 摘要级要点

- 高层规划器不预测稠密动作轨迹，而是选择可达性与任务效用兼顾的稀疏视觉子目标。
- 论文从无动作标注的机器人视频估计子目标效用与可达性，并用动态规划选择子目标序列。
- 报告在 LIBERO-Long 上达到 91.2% 成功率，低层策略规模为 0.1B；作者还报告 Language-Table 与真实双臂任务结果。
- arXiv 页面链接项目网站，但未列 GitHub 代码仓库；项目页当前未能通过检索工具核实，代码开放状态待复核。

## 对 wiki 的映射

- [DreamCan 论文实体页](../../wiki/entities/paper-dreamcan-sparse-visual-planning.md)
- [HINT 长程操作](../../wiki/entities/paper-hint-robot-manipulation.md)
- [世界动作模型](../../wiki/entities/unifolm-world-model-action.md)
