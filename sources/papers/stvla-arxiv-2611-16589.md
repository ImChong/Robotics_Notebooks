# STVLA: Spatial-aware and Temporal-memory for Vision-Language-Action Models

- **类型：** paper / robotics
- **中文标题：** STVLA：融合空间几何与时序记忆的视觉语言动作模型
- **论文：** [arXiv:2611.16589v1](https://arxiv.org/abs/2611.16589)（2026 年 11 月）
- **HTML 全文：** https://arxiv.org/html/2611.16589v1
- **PDF：** https://arxiv.org/pdf/2611.16589
- **作者：** Zhuo Han、Yeqiang Wang、Xinran Wang 等
- **机构：** 智元机器人（AgiBot）
- **项目：** https://stvla.github.io/
- **源码：** 未查到独立可运行的官方发布入口
- **入库日期：** 2026-11-16
- **核对范围：** 已核对原始论文 HTML v1 的引言、方法、实验、对比、结论与局限，并参照作者项目页核对核心机制及口径；参考文献不逐篇复读。
- **通读：** 已完成
- **一句话说明：** 将预训练视觉几何基础模型的空间特征与视觉 token 一起写入有界时序记忆，按相关性和互补性选取历史证据来条件化 VLA 动作。

## 原文核对摘要与 wiki 映射

### 方法与控制接口

当前图像同时经过视觉编码和预训练几何 backbone，空间信息经压缩后与图像 token 组成记忆条目；有界 memory bank 保留历史。读取时以当前 query 与历史条目的余弦相似度评估 relevance，并利用差异项奖励非冗余信息，标准化后给 token attention 加偏置，形成条件上下文；用 flow-matching 动作目标并配合空间重建、KL 损失训练。

- **对 wiki 的映射：** [STVLA：融合空间几何与时序记忆的视觉语言动作模型](../../wiki/entities/paper-stvla-spatial-temporal-memory.md) — 核心原理及 Mermaid 流程。

### 实验与数值边界

论文报告 LIBERO、LIBERO-Mem、MIKASA-Robo、SimplerEnv Bridge 仿真与 AGIBOT G2 真机。真机堆块、物体归置、手机包装三项任务，较 π0.5 成功率分别高 58.33、30.00、35.00 **百分点**（不是相对百分比）。SimplerEnv Bridge 平均 74.0% 对 MemoryVLA 的 71.9%。

- **对 wiki 的映射：** 主实体 — 实验与评测、结论及误读警示。

### 开源、局限与工程取舍

- **开源状态（2026-11-16）：** 项目站点未提供可运行源码仓库链接。
- **论文边界：** 几何模型及记忆检索会引入附加计算与存储，结果主要在论文指定基准和 G2 双臂上验证；项目页未发现官方可运行训练/评测代码。
- **对 wiki 的映射：** 主实体 — 工程实践、源码运行时序与局限。

## 原始链接

- [arXiv HTML v1](https://arxiv.org/html/2611.16589v1)
- [作者项目页](https://stvla.github.io/)
