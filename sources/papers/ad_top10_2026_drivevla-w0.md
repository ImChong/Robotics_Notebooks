# DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving

- **类型：** 原始论文摘录（不转存论文）
- **论文：** https://arxiv.org/html/2510.12796
- **arXiv ID：** 2510.12796
- **机构：** 中国科学院自动化研究所等
- **代码：** https://github.com/BraveGroup/DriveVLA-W0
- **核对日期：** 2026-11-18
- **核对范围：** 论文§3 VLA、世界建模、专家，§4实验及附录；官方README推理与评测脚本
- **通读：** 已完成

## 方法

历史图像、驾驶指令和动作进入VLA；训练期通过VQ或扩散世界模型预测未来图像形成稠密监督；推理时动作专家生成waypoints。

## 原文实验及边界

公开NAVSIM实验配置及模型权重；脚本名PDMS_87.2是具体配置名称。

世界模型在训练期辅助学习，不能把视觉生成指标直接当成驾驶规划安全指标。

## 对 wiki 的映射

- [唯一详情节点](../../wiki/entities/paper-sa-2510-12796-drivevla-w0-world-models-amplify-data-scaling-la.md)
- [2026 自动驾驶十项目](../../wiki/overview/autonomous-driving-open-source-top10-2026.md)
