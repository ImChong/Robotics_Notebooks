# FlashRFCL（CoRL 2026）

> 来源归档（paper）

- **标题：** FlashRFCL: Demonstration-Efficient Sparse Reward RL via Reverse-Forward Curriculum Learning in High-Throughput GPU Simulation
- **类型：** paper
- **作者：** Yong Woo Choi, Stone Tao, Henrik I. Christensen
- **会场：** 10th Annual Conference on Robot Learning（CoRL 2026）
- **arXiv：** 未检索到（arXiv API 按标题与关键词查询 0 结果，2026-11-12）
- **论文页（仅摘要）：** <https://hichristensen.com/publication/2026-corl-flashrfcl/>
- **OpenReview：** <https://openreview.net/forum?id=DBLXUxxslN>（本环境被验证挑战挡住）
- **代码：** 未找到公开仓库
- **入库日期：** 2026-11-12
- **一句话说明：** 把「反向 + 正向课程」（RFCL）与可扩展 RL、高吞吐 GPU 仿真结合，仅用每任务一条演示求解稀疏奖励的高维机器人任务。

## 核对范围与通读状态

- **核对范围：** 仅读到作者主页论文页与 Papers with Code 镜像页给出的摘要（约 100 词）；未取得论文全文（arXiv 无、OpenReview 被挡、无项目页 / 代码仓库）。方法细节、实验任务、数值、消融、局限均未读到。
- **通读：** 未完成

## 核心摘录（均出自摘要）

1. 稀疏奖励 + 演示的 RL 可避免奖励塑形，但已有方法常需要大量课程工程、长训练时间和 / 或不可行数量的演示。
2. 作者结合可扩展 RL、高吞吐 GPU 仿真与基于演示的自动课程，称 FlashRFCL 能更快求解多种高维机器人任务，每任务仅需 1 条演示。
3. 摘要称做了消融，展示各设计选择对训练速度与性能的影响（相对已有工作）。

## 相关一手资料（前作，未在本归档通读）

- RFCL：*Reverse Forward Curriculum Learning for Extreme Sample and Demonstration Efficiency in Reinforcement Learning*，ICLR 2024，<https://arxiv.org/abs/2405.03379>。

## 对 wiki 的映射

- [paper-flashrfcl](../../wiki/entities/paper-flashrfcl.md)
