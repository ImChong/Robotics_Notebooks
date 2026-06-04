# extreme-parkour.github.io（Extreme Parkour 项目页）

- **标题：** Extreme Parkour with Legged Robots — 官方项目页
- **类型：** site / project-page
- **URL：** <https://extreme-parkour.github.io/>
- **PDF：** <https://extreme-parkour.github.io/resources/parkour.pdf>
- **入库日期：** 2026-06-04
- **配套论文：** [Extreme Parkour（arXiv:2309.14341）](https://arxiv.org/abs/2309.14341) — 归档见 [`sources/papers/extreme_parkour_arxiv_2309_14341.md`](../papers/extreme_parkour_arxiv_2309_14341.md)
- **代码：** <https://github.com/chengxuxin/extreme-parkour>

## 一句话摘要

CMU Pathak Lab 的 **Extreme Parkour（ICRA 2024）** 官方站点：展示 **Go1 级四足** 在 **单目前向深度** 下完成高跳、远跳、手倒立、斜 ramp 变向与 **新障碍组合泛化** 的实机视频；含 **Vision locomotion 失败对照**、**clearance reward** 与 **direction distillation** 消融，以及 CoRL 2023 **2 分钟不间断** demo。

## 公开信息要点（截至入库日）

- **会议标签：** ICRA 2024；CoRL 2023 Generalist / Roboletics / Deployable Workshop（Oral）。
- **摘要主张：** 低成本、执行不精确、深度相机低频抖动条件下，**单 NN 端到端 RL** 仍可输出高精度跑酷控制；人类式「练习习得」而非分模块精密工程。
- **Extreme Cases 演示：** High jump（~2× 身高）、Long jump（~2× 体长）、Handstand、Tilted ramp（**全自主、无遥操作**）。
- **Parkour Course：** Step / Gap / Hurdle 组合与 **Robustness** 片段。
- **Baselines / Ablations（页面级）：**
  - Vision locomotion 项目在楼梯场景 **跌落**（对照端到端跑酷必要性）；
  - **无 clearance reward**：大 gap 贴边触边失败；
  - **无 direction distillation**：ramp 摇杆控制困难。
- **Media：** 机器之心、量子位、Hacker News 报道链接。
- **BibTeX：** `@article{cheng2023parkour, ... arXiv:2309.14341}`

## 为何值得保留

- **非 PDF 证据：** 动态技能与 ablation 视频比论文静态图更直观。
- **与 GitHub README 三角互证：** 训练阶段划分、延迟与相机 flag 与代码文档一致。
- **跑酷技术谱系锚点：** 与 Robot Parkour Learning、DreamWaQ++、PHP 等形成四足→人形、2023–2026 时间线参照。

## 关联资料

- 论文归档：[`sources/papers/extreme_parkour_arxiv_2309_14341.md`](../papers/extreme_parkour_arxiv_2309_14341.md)
- 代码归档：[`sources/repos/extreme-parkour.md`](../repos/extreme-parkour.md)
- Wiki 实体：[`wiki/entities/extreme-parkour.md`](../../wiki/entities/extreme-parkour.md)
