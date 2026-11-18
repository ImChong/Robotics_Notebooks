# ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving

- **类型：** 原始论文摘录（不转存论文）
- **论文：** https://arxiv.org/html/2506.08052
- **arXiv ID：** 2506.08052
- **机构：** 华中科技大学、小米汽车
- **代码：** https://github.com/xiaomi-research/recogdrive
- **核对日期：** 2026-11-18
- **核对范围：** 论文§4分层认知、条件扩散、DiffGRPO，§5与附录局限；README与docs/Train_Eval.md源码入口
- **通读：** 已完成

## 方法

三阶段驾驶问答知识生成、提炼、质检微调InternVL3；将VLM隐状态送入DiT连续轨迹规划；DiffGRPO用规划奖励对扩散去噪策略进行后训练。

## 原文实验及边界

官方NAVSIM 2B VLM/IL/RL 依次为84.1/86.5/90.8 PDMS；还提供Bench2Drive闭环结果。

部分第三方数据受许可限制；VLM隐状态缓存约占1–2TB磁盘；不同数据集分数不可混合。

## 对 wiki 的映射

- [唯一详情节点](../../wiki/entities/paper-recogdrive-2506-08052.md)
- [2026 自动驾驶十项目](../../wiki/overview/autonomous-driving-open-source-top10-2026.md)
