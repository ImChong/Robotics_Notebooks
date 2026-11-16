# DKWBC 官方项目页

> 来源归档（site）

- **标题：** DKWBC: Decoupling Motion Tracking and Balancing for Humanoid Whole-Body Control
- **类型：** site
- **链接：** <https://robbyant-research.github.io/DKWBC/>
- **论文：** <https://arxiv.org/abs/2611.09605>
- **代码链接：** <https://github.com/robbyant-research/DKWBC>（实际为项目网页仓库）
- **核查日期：** 2026-11-10
- **核对范围：** 官方页的人员机构、论文 / Code 两个链接、动作对比视频、2–128 小时数据缩放曲线与动力学解耦公式；检查官方仓库 README 与默认分支。
- **通读：** 已完成
- **开放情况：** 项目展示已公开，**训练/推理/部署代码及权重尚未见公开**；不要将展示网站视为算法开源。
- **沉淀到 wiki：** [paper-dkwbc](../../wiki/entities/paper-dkwbc.md)

## 摘录

官方将关节扭矩拆为 `τ_cmd = τ_trk + τ_bal`：显式动力学 + PD 跟踪负责参考运动，学习残差补偿支持、外力与平衡。项目页展示在 Unitree G1 上的动作与外载荷实验，但没有列出可运行方法的使用命令。
