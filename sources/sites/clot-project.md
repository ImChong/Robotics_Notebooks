# CLOT 项目页（zhutengjie.github.io/CLOT.github.io）

> 来源归档

- **标题：** CLOT: Closed-Loop Global Motion Tracking for Whole-Body Humanoid Teleoperation
- **类型：** site（项目页 + 实验视频）
- **URL：** <https://zhutengjie.github.io/CLOT.github.io/>
- **说明：** 论文摘要写作 `CLOT.github.io`；裸域 <https://clot.github.io/> 非官方页（勿混用）
- **论文：** <https://arxiv.org/abs/2602.15060>
- **代码：** <https://github.com/zhutengjie/CLOT>
- **机构：** 上海交通大学 · 上海人工智能实验室
- **入库日期：** 2026-06-12
- **一句话说明：** 官方页：闭环全局位姿反馈的长时程全身遥操作；OptiTrack 120 Hz + 在线 Pinocchio IK 重定向 + Transformer 策略（50 Hz）+ 400 Hz PD；Adam Pro 31 DoF 真机；含全身跟踪、鲁棒性、长时程 loco-manipulation 与长期稳定性演示视频。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Abstract | 全局漂移问题；闭环同步；Observation Pre-shift；AMP 正则；20 h 人体数据；1300+ GPU h 训练 |
| Video | 系统总览 |
| Whole-Body Tracking Demos | 高动态全身跟踪 |
| Robustness Demos | 扰动拒绝 |
| Long-Horizon Loco-Manipulation | 长时程移动操作 |
| Long-Term Stability Demos | 扩展运行无全局漂移 |
| BibTeX | `@article{zhu2026clot,...}` arXiv:2602.15060 |

## 对 wiki 的映射

- 主实体：[CLOT（论文实体）](../../wiki/entities/paper-amp-survey-16-clot.md)
- 论文摘录：[clot_arxiv_2602_15060.md](../papers/clot_arxiv_2602_15060.md)
- 代码仓库：[clot.md](../repos/clot.md)
