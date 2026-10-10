# H-JEPA 官方代码仓库

> 来源归档（ingest · GitHub）

- **类型：** repository
- **仓库：** <https://github.com/kevinghst/H-JEPA>
- **作者 / 维护者：** Wancong Zhang（kevinghst）
- **关联论文：** [H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning](../papers/hjepa_arxiv_2610_06805.md)
- **项目页：** <https://h-jepa.com/>
- **许可证：** MIT（仓库）
- **核查日期：** 2026-10-10
- **沉淀到 wiki：** [H-JEPA](../../wiki/entities/paper-h-jepa-visual-planning.md)

## 仓库内容

官方仓库提供 H-JEPA 的训练与规划代码，面向 Visual AntMaze、FourRoom Distractors、OGBench Cube、Push-T，并支持在 DROID 上进行离线规划。README 提供不同层数（2/3/4 层）的配置和论文结果复现入口；依赖 LeWM 与 stable-worldmodel 相关代码。

## 使用边界

这是研究代码仓库，不代表论文已验证物理机器人闭环部署。复现实验需按 README 准备相应数据集、评测任务及模型检查点，并核对仓库当前环境要求与资源需求。
