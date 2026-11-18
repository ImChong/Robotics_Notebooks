# SGDrive: Scene-to-Goal Hierarchical World Cognition for Autonomous Driving

- **类型：** 原始论文摘录（不转存论文）
- **论文：** https://arxiv.org/html/2601.05640
- **arXiv ID：** 2601.05640
- **机构：** 复旦大学、理想汽车、上海创智学院、同济大学、萨里大学
- **代码：** https://github.com/LogosIntelligenceLab/SGDrive
- **核对日期：** 2026-11-18
- **核对范围：** 论文§3.1–3.5分层查询、注意力、DiT训练，§4实验及附录；官方README和docs/Train_Eval.md
- **通读：** 已完成

## 方法

在VLM中引入场景/agent/goal分层世界知识查询，通过block-wise attention避免信息泄漏，世界表征作为条件送入DiT扩散规划器，并可进行强化微调。

## 原文实验及边界

官方NAVSIM v1 navtest报告VLM 85.5、IL 87.4、RL 91.1 PDMS；v2另用EPDMS指标。

部分权重仍有coming soon标记，需核对HF可得资产；训练缓存需求可达1–2TB；无法由PDMS推导真实道路安全。

## 对 wiki 的映射

- [唯一详情节点](../../wiki/entities/paper-sgdrive-2601-05640.md)
- [2026 自动驾驶十项目](../../wiki/overview/autonomous-driving-open-source-top10-2026.md)
