# RFPO 官方项目页归档

> 来源归档

- **标题：** RFPO: Rectified Flow Policy Optimization for Embodied Control
- **类型：** site（论文项目页 + 方法图示 + 机器人演示）
- **URL：** <https://aigeeksgroup.github.io/RFPO/>
- **论文：** <https://arxiv.org/abs/2610.10453>
- **代码：** <https://github.com/AIGeeksGroup/RFPO>
- **作者：** Ting Huang, Lisiyu Pan, Haoyu Wang, Zeyu Zhang, Siyuan Qian, Yanjun Li, Yandong Guo, Boxin Shi, Hao Tang
- **机构：** School of Computer Science, Peking University；AI² Robotics
- **核对日期：** 2026-10-11
- **一句话说明：** 项目页展示 RFPO 的 reward-aware online reflow、多预算训练框架与 Go2/G1 sim-to-real 结果，并链接论文与官方代码。

## 页面要点

| 区块 | 内容要点 |
|------|----------|
| 方法概览 | 以 reward-aware online reflow 适配少步 flow policy，并以冻结 Gaussian PPO 控制器提供动作空间监督 |
| 推理预算 | 训练学生策略适配多个 Euler 积分预算，最终目标为单步推理 |
| 机器人评测 | 展示 Unitree Go2、Spot、H1、G1 的论文评测结果；项目页演示包含 Go2/G1 sim-to-real 场景 |
| 论文与代码 | 项目页链接 arXiv 论文与 AIGeeksGroup/RFPO 官方实验代码 |

## 对 wiki 的映射

- 主实体：[RFPO](../../wiki/entities/paper-rfpo-rectified-flow-policy-optimization.md)
- 论文摘录：[rfpo_arxiv_2610_10453.md](../papers/rfpo_arxiv_2610_10453.md)
- 代码仓库：[rfpo.md](../repos/rfpo.md)
