# DexTacWAM — 作者项目页

- **项目页：** https://dextacwam.github.io/
- **论文：** https://arxiv.org/abs/2609.24976v1
- **代码：** https://github.com/dextacwam/DexTacWAM
- **权重及数据：** https://huggingface.co/collections/JensenYuan/dextacwam-6aba3aa25d3362d882cdefbc
- **核对范围：** 2026-11-13 阅读作者项目页的 Introduction、Method、Encoder、Results、Q&A、Code/Weights/Data 与发布说明，结合源码 README 核查开放边界。
- **通读：** 已完成

## 资料定位和开放结论

本项目提出将双手十指的接触演化作为预测世界状态，不只是动作策略当前触觉条件。主页提供真实机器人任务演示、六任务分数、触觉编码与压缩效率、一手论文以及代码/权重/数据链接。

**部分开放：** 完整三阶段训练及离线推理代码、488 条基础触觉记录、六任务示范和 Stage 1 编码器权重公开；但作者**未提供 Stage 2 任务世界模型和 Stage 3 动作专家检查点**。真机部署脚本在代码仓的 `deploy` 分支，复现需要重新按任务训练和对应硬件。

## 对 wiki 的映射

- [DexTacWAM 主实体](../../wiki/entities/paper-dextacwam.md)
- [论文归档](../papers/dextacwam_arxiv_2609_24976.md)
- [代码归档](../repos/dextacwam-official.md)