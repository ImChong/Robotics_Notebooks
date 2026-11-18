# World Engine: Towards the Era of Post-Training for Autonomous Driving

- **类型：** 原始论文摘录（不转存论文）
- **论文：** https://arxiv.org/html/2606.19836
- **arXiv ID：** 2606.19836
- **机构：** OpenDriveLab等
- **代码：** https://github.com/OpenDriveLab/WorldEngine
- **核对日期：** 2026-11-18
- **核对范围：** 论文§2–§4仿真、行为世界模型、强化后训练及实验附录；官方README、scripts/closed_loop_test.sh和docs/algengine_usage.md
- **通读：** 已完成

## 方法

真实日志经MTGS三维高斯重建形成SimEngine交互环境，AlgEngine中的VADv2闭环策略产生反馈，失败样本筛选后用于安全约束强化后训练；论文提出Behavior World Model。

## 原文实验及边界

仓库Quick Test对288个稀有场景进行闭环测试，读取VADv2 50%数据训练的模型并算NAVSIM v1 PDMS。

官方README把Behavior World Model integration仍列为待办；simengine和algengine需分开装依赖；3DGS重建画面不保证物理交互真实。

## 对 wiki 的映射

- [唯一详情节点](../../wiki/entities/paper-worldengine-2606-19836.md)
- [2026 自动驾驶十项目](../../wiki/overview/autonomous-driving-open-source-top10-2026.md)
