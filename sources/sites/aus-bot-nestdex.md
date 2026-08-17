# NestDex 项目页（aus.bot）

> 来源归档（site / project page）

- **标题：** NestDex · Nested Dexterous Policies
- **类型：** site / project-page
- **URL：** <https://aus.bot/research/nestdex/>
- **论文：** <https://arxiv.org/abs/2608.13362> — 归档见 [`sources/papers/nestdex_arxiv_2608_13362.md`](../papers/nestdex_arxiv_2608_13362.md)
- **代码：** 截至 **2026-08-17** 项目页 SPA **未列** GitHub / Hugging Face / 数据集；JS bundle 仅链 arXiv 与 PAIR Lab 研究站
- **机构：** PAIR Lab / The University of Sydney（School of Computer Science + Australian Centre for Robotics）；Vanderbilt University
- **入库日期：** 2026-08-17
- **核查日期：** 2026-08-17
- **一句话说明：** NestDex 官方研究站：内层本体感觉手技能 + 单自由度 clutch copilot 采数，再训独立外层 visuomotor；六任务演示与接触闭环消融视频。

## 开源核查（步骤 2.5，截至 2026-08-17）

| 核查项 | 结论 |
|--------|------|
| 项目页 Code / Resources | **无** — SPA 入口为方法叙事、视频、论文链接 |
| JS bundle 外链 | 仅 [arXiv:2608.13362](https://arxiv.org/abs/2608.13362) 与 [PAIR Lab 研究站](https://aus.bot/research/)；**0** 次 `github` / `huggingface` |
| GitHub 检索 `nestdex` | 无 PAIR Lab / 作者官方训练仓；命中仓库与本文无关 |
| 论文 PDF Code availability | **未承诺仓库 URL**，亦无 “code will be released” 链接 |
| 综合判定 | **确认未开源**（无可运行训练 / 部署实现） |

## 页面要点

- Hero：把学到的手技能嵌进遥操作，让操作员只管任务级臂运动；这些技能只服务示范，外层策略部署时独立运行。
- 三步管线：多视手跟踪重定向 → clutch 调节内层技能进度 → 完整任务示范训外层 visuomotor。
- H-VAE：20 维手关节指令压到 10 维 latent，臂指令仍走关节空间。
- 接触叙事：同一 grasp 内层策略只吃关节位置与力矩，无物体图像/身份，四物体接触构型不同。
- 技能切换：Toast Preparation 腕相机选 Button Press，手回到技能起始姿态，clutch 执行。
- 双臂长程演示：Toast Preparation（四技能阶段）、Binder Filing（可复用手技能）。

## 关联资料

- 论文摘录：[`sources/papers/nestdex_arxiv_2608_13362.md`](../papers/nestdex_arxiv_2608_13362.md)
- Wiki 实体：[`wiki/entities/paper-nestdex.md`](../../wiki/entities/paper-nestdex.md)
- 同实验室对照：[`sources/sites/aus-bot-autointervene.md`](./aus-bot-autointervene.md)
