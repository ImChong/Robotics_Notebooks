# shooting-for-contact.github.io（Shooting for Contact 项目页）

> 来源归档（ingest）

- **标题：** Shooting for Contact — Contact-Implicit Multiple Shooting for Dynamic Motion Retargeting
- **类型：** site / project-page
- **URL：** <https://shooting-for-contact.github.io/>
- **入库日期：** 2026-08-07
- **配套论文：** [Shooting for Contact（arXiv:2608.03116）](https://arxiv.org/abs/2608.03116) — 归档见 [`sources/papers/shooting_for_contact_arxiv_2608_03116.md`](../papers/shooting_for_contact_arxiv_2608_03116.md)
- **代码：** <https://github.com/sesteban951/shooting-for-contact> — 归档见 [`sources/repos/shooting-for-contact.md`](../repos/shooting-for-contact.md)
- **机构：** 加州理工学院（Caltech）；德保罗大学（DePaul University）

## 一句话摘要

Caltech / DePaul 团队的 **DSMS** 官方站点：展示接触隐式多重打靶如何把运动学参考转为动力学可行全身轨迹，并链接 arXiv、PDF 与 **已开源** GitHub 实现；真机证据包括 G1 命令条件化爬行与 180° 跳转。

## 公开信息要点（截至入库日）

- **站点源码：** <https://github.com/shooting-for-contact/shooting-for-contact.github.io>（Nerfies 模板改编）
- **Paper / arXiv / Code** 三按钮齐全；PDF 镜像路径 `static/pdfs/Contact_Rich_Locomotion.pdf`
- **方法卖点（页内）：** Whole-body feasible · Contact-implicit · Arbitrary constraints · Morphology-agnostic
- **硬件分区：** 180° Jump-Turn；限高爬行；前后爬行；草地爬坡；实验室长程 twist 驾驶
- **仿真能力展示：** Humanoid backflip / Super-hero backflip / Side-rolling / Quadruped jump-turn（蓝 ghost = 动力学不可行参考）

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 代码链接 | **有** — `https://github.com/sesteban951/shooting-for-contact` |
| 开放程度 | **已开源**（trajopt / MPC / MuJoCo 模型与示例轨迹） |
| 数据集 | 未单独发 HF；参考 clips 随仓 `trajectories/` |
| RL / 真机栈 | 项目页展示结果；**训练与部署代码不在本仓**（论文 mjlab） |

## 为何值得保留

- **非 PDF 证据：** 接触丰富爬行与高动态跳转的真机/仿真视频，便于与 OmniRetarget / DynaRetarget 选型对照。
- **三角溯源：** 项目页 ↔ arXiv ↔ DSMS 代码仓固定入口。

## 关联资料

- 论文归档：[`sources/papers/shooting_for_contact_arxiv_2608_03116.md`](../papers/shooting_for_contact_arxiv_2608_03116.md)
- 代码：[`sources/repos/shooting-for-contact.md`](../repos/shooting-for-contact.md)

## 对 wiki 的映射

- [`wiki/entities/paper-shooting-for-contact.md`](../../wiki/entities/paper-shooting-for-contact.md)
- [`wiki/methods/dsms-contact-implicit-multiple-shooting.md`](../../wiki/methods/dsms-contact-implicit-multiple-shooting.md)
