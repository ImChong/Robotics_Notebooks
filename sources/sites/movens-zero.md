# Movens-Zero 项目主页（一手来源）

> 来源归档（site；对应同一论文实体）

- **标题：** Movens-Zero: Scaling Humanoid Motion Foundation Models with Large-Scale Human Videos
- **类型：** site
- **链接：** https://movens-zero.github.io/
- **论文：** [arXiv:2610.41877v1](../papers/movens_zero_arxiv_2610_41877.md)
- **代码：** [wendell0218/Movens-Zero](../repos/movens-zero.md)
- **数据集：** [Movens-5M](../datasets/movens-5m.md)
- **Movens-Intent：** https://huggingface.co/datasets/wendell0218/Movens-Intent
- **核对日期：** 2026-11-23
- **核对范围：** 项目页头部 Resources、Dataset、Method、Scaling、Capabilities，以及代码/数据/基准/模型四类公开入口。
- **通读：** 已完成
- **沉淀到 wiki：** [Movens-Zero / Movens-5M](../../wiki/entities/paper-movens-zero.md)

## 官方一手信息

- 机构页列出浙江大学、宇树科技（Joint Laboratory）、Bit Oasis、HiThink Research 等机构；主本体为 Unitree G1。
- 项目数据引擎五阶段：切片、质量筛选、语义标注、人体动作恢复与重定向、物理修复与闭环跟踪验证。
- 报告 15M 人体动作、5M 可执行动作、10K+ 小时、1.08B+ 帧、297 类动作、16 类场景、6 类物理检查和 4 套跟踪策略。
- 官方项目页链接 **GitHub 代码**、**HF 数据集**、**Movens-Intent 基准**和**S/M/L/XL 模型权重**，因此为**已发布部分可运行代码和权重，并列出全量数据分片入口**；不意味着本地已跑通全部训练或完整下载。

## 开放边界

- HF 主数据页实际提供 Parquet **预览**，其官方数据卡把全量数据分发至四个 tar shards 仓库，见 [数据集归档](../datasets/movens-5m.md)。
- 中央数据页 license 为 `other`，需单独阅读并确认视频版权和用途许可。
- 项目主页和论文的实验是作者报告，复现能力需结合代码配置、模型权重及目标硬件独立验证。
