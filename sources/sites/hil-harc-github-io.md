# HIL-HARC 项目页

> 来源归档

- **标题：** HIL-HARC — Efficient Real-World Online RL via CTDE and Critic Decomposition
- **类型：** site / project-page
- **URL：** <https://hil-harc.github.io/>
- **论文：** <https://arxiv.org/abs/2608.09762> — [`sources/papers/hil_harc_arxiv_2608_09762.md`](../papers/hil_harc_arxiv_2608_09762.md)
- **代码：** 截至 **2026-08-12** 项目页 **未列** 训练/推理 GitHub；仅存在静态仓 `https://github.com/HIL-HARC/HIL-HARC.github.io`
- **机构：** IIT HHCM / HRI²；University of Genova；TU Delft
- **入库日期：** 2026-08-12
- **一句话说明：** 官方项目站：CTDE + HRA 方法叙事、真机/仿真视频、相对 HIL-SERL 的成功率与样本效率表；投稿后增补 bottle stowing。

## 开源核查（步骤 2.5，截至 2026-08-12）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到代码 | **否** — Resources 仅 Citation 占位 |
| GitHub `HIL-HARC/HIL-HARC.github.io` | **项目页静态仓**，非训练栈 |
| 权重 / 数据 | **未列** |
| 综合判定 | **确认未开源** |

## 页面要点

- 双 actor：连续 Cartesian 臂 + 离散夹爪；集中式多头 critic（task / grasp）。
- 真机平均成功率 **75%**（+35 pp vs HIL-SERL）；最大绝对提升 **+70 pp**；收敛干预率 **0%**。
- 任务：网球 / 香蕉 P&P、锅复位、仿真 G1 搬块；增补 bottle stowing **85%**（17/20）。
- 训练：RLPD + 人干预；异步远端 learner。

## 关联资料

- 论文：[`sources/papers/hil_harc_arxiv_2608_09762.md`](../papers/hil_harc_arxiv_2608_09762.md)
- Wiki：[`wiki/entities/paper-hil-harc.md`](../../wiki/entities/paper-hil-harc.md)
