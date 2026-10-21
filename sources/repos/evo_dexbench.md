# Evo-DexBench 官方仓库来源

> 来源归档

- **标题：** Evo-DexBench
- **类型：** repo
- **链接：** https://github.com/MINT-SJTU/Evo-DexBench
- **论文：** https://arxiv.org/abs/2610.21180
- **数据与资产：** [Hugging Face collection](https://huggingface.co/collections/MINT-SJTU/evo-dexbench)；README 列出的 [Evo-DexBench-dataset](https://huggingface.co/datasets/MINT-SJTU/Evo-DexBench-dataset) 与 [Evo-DexBench_Assets](https://huggingface.co/datasets/MINT-SJTU/Evo-DexBench_Assets)
- **项目页：** https://mint-sjtu.github.io/Evo-DexBench.io/（README 标注 coming soon）
- **许可证：** Apache-2.0（仓库代码与文档；第三方资产依各自许可）
- **入库日期：** 2026-10-19
- **一句话说明：** 基于 ManiSkill 3 的灵巧操作 benchmark 与模拟环境，支持单手 / 双手任务、VR 遥操作、示范采集、LeRobot 导出及可插拔策略评测。

## 开源核查（2026-10-19）

| 资源 | 状态 | 说明 |
|------|------|------|
| 仿真、基准与评测代码 | **已开源** | GitHub 提供 Conda 环境安装、`dexbench run`、`dexbench evaluate` 与模型适配接口 |
| 遥操作及数据采集 | **已开源** | README 记录键盘和 Quest 3 遥操作、episode 采集与质量验证流程 |
| 数据与资产 | **已发布** | 官方 README 链接 Hugging Face 数据集和资产仓库；PartNet-Mobility 另需单独申请访问 |
| 项目页 | **待上线** | 官方 README 的项目页徽章仍标注 coming soon |

**判定：代码与基准已开源，数据/资产已有公开分发入口。** 获取带门控的 PartNet-Mobility 资产须遵循其 Hugging Face 访问条件。

## 仓库 README 公开要点

- 基于 **ManiSkill 3**，覆盖 **30 项任务**：15 项单手、15 项双手任务；论文摘要将其归为 6 个功能家族。
- 标准评测覆盖空间、相机、光照和物理泛化，报告 **Success Rate (SR)** 与 **StageScore**。
- 仿真通常以 **30 Hz 控制 / 120 Hz 仿真**运行；具体任务与评测参数可通过 `dexbench tasks show` 和官方文档核对。
- 支持键盘和 Quest 3 遥操作、示范采集、HDF5 数据验证与 LeRobot 导出。
- 策略通过 `dex_benchmark.policy_api` 的 `PolicySession` / `SemanticAction` 接口接入；策略模型与模拟器运行时可分离。README 展示 KineDex 示例，不表示 benchmark 只能使用 KineDex。
- 资产下载工具要求用户先取得 PartNet-Mobility 仓库访问权；机器人描述随软件包提供。

## 关联来源

- 论文：[evo_dexbench_arxiv_2610_21180](../papers/evo_dexbench_arxiv_2610_21180.md)
- 项目页状态：[evo-dexbench-project-page](../sites/evo-dexbench-project-page.md)
- 数据集与资产：[evo_dexbench_dataset](../datasets/evo_dexbench_dataset.md)
- wiki 实体：[paper-evo-dexbench](../../wiki/entities/paper-evo-dexbench.md)
