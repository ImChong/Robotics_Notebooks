# pei-xu.github.io/basketball（Learning to Ball 项目页）

- **标题：** Learning to Ball: Composing Policies for Long-Horizon Basketball Moves
- **类型：** site / project-page
- **URL：** <https://pei-xu.github.io/basketball>
- **配套论文：** [Learning to Ball（arXiv:2509.22442）](https://arxiv.org/abs/2509.22442) — 归档见 [`sources/papers/learning_to_ball_arxiv_2509_22442.md`](../papers/learning_to_ball_arxiv_2509_22442.md)
- **代码：** <https://github.com/xupei0610/basketball> — 归档见 [`sources/repos/learning-to-ball.md`](../repos/learning-to-ball.md)
- **入库日期：** 2026-07-28

## 一句话摘要

SIGGRAPH Asia 2025 / ACM TOG 官方项目页：展示物理仿真角色通过 **策略组合 + soft router** 完成长程篮球连招，以及子技能、过渡与多人交互演示。

## 公开信息要点（截至入库日）

- **作者 / 机构：** Pei Xu、Zhen Wu、Ruocheng Wang、Vishnu Sarukkai、Kayvon Fatahalian（Stanford）；Ioannis Karamouzas（UC Riverside）；Victor Zordan（Roblox & Clemson）；C. Karen Liu（Stanford）。
- **方法三块：** (1) 从非结构化异构数据学原始子技能（无球轨迹参考）；(2) 面向 ill-defined 中间态的策略过渡学习；(3) soft router 策略组合。
- **Primitive Policies 演示：** Dribble、Shoot、Rebound、Catch & Pass、Locomotion + Defend。
- **Policy Transitions 演示：** Shoot off Dribble、Pass & Catch、Shoot off Catch、Pass off Dribble、Catch to Dribble、Rebound to Dribble。
- **Multi-Agent：** 实时多人交互控制演示。
- **资源链接：** 页内指向 arXiv / 视频；**代码入口为官方 GitHub `xupei0610/basketball`**（MIT）。

## 为何值得保留

- **非 PDF 证据：** 过渡类型与多人 2v2 演示比表格更直观。
- **开源三角互证：** 项目页技能名与仓库 `cfg/` / `pretrained/` 子技能目录对齐。
- **与 Paper Notebooks 分类 13** 篮球物理动画线（SkillMimic 等）对照入口。

## 关联资料

- 论文归档：[`sources/papers/learning_to_ball_arxiv_2509_22442.md`](../papers/learning_to_ball_arxiv_2509_22442.md)
- 代码仓库：[`sources/repos/learning-to-ball.md`](../repos/learning-to-ball.md)
- Paper Notebooks 锚点：[`sources/papers/humanoid_pnb_learning-to-ball.md`](../papers/humanoid_pnb_learning-to-ball.md)
