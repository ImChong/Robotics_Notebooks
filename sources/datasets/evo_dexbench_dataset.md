# Evo-DexBench 数据集与资产

> 来源归档

- **类型：** dataset / simulation assets
- **论文：** https://arxiv.org/abs/2610.21180
- **官方 Hugging Face collection：** https://huggingface.co/collections/MINT-SJTU/evo-dexbench
- **数据集：** https://huggingface.co/datasets/MINT-SJTU/Evo-DexBench-dataset
- **资产：** https://huggingface.co/datasets/MINT-SJTU/Evo-DexBench_Assets
- **代码与下载工具：** https://github.com/MINT-SJTU/Evo-DexBench
- **入库日期：** 2026-10-19

## 已核实信息

- arXiv 摘要报告 **1,248 条多模态示范**，由模拟 Shadow Hands 采集；每只手配置 **123 个 taxel**。
- 官方仓库将数据集与资产分开发布。README 中给出的任务规模为 **30 项**，每项包括三个场景布局；具体条目见官方 [task catalog](https://github.com/MINT-SJTU/Evo-DexBench/blob/main/docs/task-catalog.md)。
- 下载仿真资产需先申请官方 README 所指的 [PartNet-Mobility 数据访问](https://huggingface.co/datasets/sapien-sim/PartNetMobility)，再通过仓库的 `dexbench assets download` / `verify` 命令管理本地资产。
- 本归档不推断 Hugging Face collection 中各数据集的文件大小、拆分或许可细节；这些信息以对应卡片和文件元数据为准。

## Robotics_Notebooks 映射

- 项目实体：[Evo-DexBench](../../wiki/entities/paper-evo-dexbench.md)
- 论文来源：[evo_dexbench_arxiv_2610_21180](../papers/evo_dexbench_arxiv_2610_21180.md)
- 仓库来源：[evo_dexbench](../repos/evo_dexbench.md)
