# FlowDPG 官方项目页来源归档

- **官网：** https://flowdpg.github.io/
- **论文：** https://arxiv.org/html/2606.22303v2
- **论文 PDF：** https://flowdpg.github.io/static/images/main.pdf
- **机构：** Skild AI、CMU、UPenn
- **会议：** CoRL 2026
- **核对日期：** 2026-11-13
- **核对范围：** 页面作者与机构、Paper/arXiv/Video/BibTeX 入口、方法概览、Franka AirPods 和 YAM 炒蛋演示、奖励消融、扰动恢复和逐阶段评分标准。
- **通读：** 已完成

## 已核验事实
- Franka 任务是两台 Franka 机械臂完成 AirPods 8 阶段装配，官方报告离线 88%、在线 92% 整任务成功率。
- 炒蛋使用两台 YAM 机械臂，86% 左右指 14 阶段 rubric 而非端到端成功率。
- 核心训练方案是把 Critic 梯度蒸馏进 Flow Matching 速度场，部署保持多步 ODE 推理。
- **开放程度：** 当前官网未列明确官方 GitHub、数据下载或模型权重。故未创建虚假的源码仓库归档和源码调用时序图。

## 映射
- [FlowDPG 原文阅读](../papers/flowdpg_arxiv_2606_22303.md)
- [FlowDPG 知识实体](../../wiki/entities/paper-flowdpg.md)
