# DAVIS: A Depth-Only End-to-End Active-Vision Framework for Humanoid Soccer Skills

> 来源归档（ingest）

- **标题：** DAVIS: A Depth-Only End-to-End Active-Vision Framework for Humanoid Soccer Skills
- **类型：** paper / humanoid / active-vision / depth / reinforcement-learning / soccer
- **arXiv：** [2609.28175](https://arxiv.org/abs/2609.28175)
- **PDF：** https://arxiv.org/pdf/2609.28175
- **项目页：** <https://thusi-lab.github.io/DAVIS/>
- **作者：** Jiakang Jin、Yixiao Huo、Pengyuan Wang 等；Xiaoyu Tian、Yiming Li（通讯）
- **机构：** Noetix Robotics（松延动力）；清华大学
- **代码：** **待发布** — [项目页核查](../sites/davis-thusi-lab.md) 无 GitHub（2026-09-25）
- **入库日期：** 2026-09-24（初稿）；**2026-09-25** 项目页深读升格
- **一句话说明：** **仅头部深度 + 本体历史 + 低维指令** 端到端输出 **25-DoF** PD 目标，学习 **主动视觉** 人形足球射门/带球；训练期可见性门控几何 + GT→prediction annealing + AMP；**无**运行时检测/规划模块。

## 相关资料

| 类型 | 链接 |
|------|------|
| 项目页 | [thusi-lab.github.io/DAVIS](https://thusi-lab.github.io/DAVIS/) |
| 站点归档 | [`davis-thusi-lab.md`](../sites/davis-thusi-lab.md) |
| 公众号初摘 | [`wechat_embodied_13_papers_forgetmimic_2026-09-24.md`](../blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md) |

## 核心摘录（项目页 + 摘要）

- **问题：** 人形足球需闭环感知—接近—对齐—击球—恢复；头动改变视角，球常出画。
- **部署严格性：** 无外部检测、定位、状态估计或足球规划器；actor 仅 depth + proprio + command。
- **LightDepthEncoder + HIM**；每物体 **3D center + visibility** 辅助头；confidence gate [0.05,1]。
- **技能：** 射门（球+门）与带球（12 向指令）**独立 checkpoint**。
- **指标：** 仿真点球/任意球 total SR **~0.85**；真机点球 **0.55–0.68** 分档；Repeated-S dribbling mean **0.65**。

## 对 wiki 的映射

- [`paper-davis-humanoid-soccer.md`](../../wiki/entities/paper-davis-humanoid-soccer.md)
