# HandsOn 官方代码仓库来源

> 来源归档

- **标题：** HandsOn: A Large-Scale Simulation Benchmark for Generalizable Dexterous Manipulation
- **类型：** repo
- **链接：** https://github.com/TsinghuaC3I/HandsOn
- **论文：** https://arxiv.org/abs/2610.26765
- **项目页：** https://llltttwww.github.io/HandsOn-website/
- **数据集：** https://huggingface.co/datasets/Cyan666/HandsOn-Datasets
- **许可证：** MIT（仓库 LICENSE；第三方任务资产遵循 NOTICE 中各自许可）
- **入库日期：** 2026-10-25
- **一句话说明：** 基于 NVIDIA Isaac Sim / Isaac Lab 的跨具身灵巧操作仿真平台，统一 66 项任务、策略适配与闭环评测，并提供数据读取和 OpenPI 训练/服务栈。

## 开源核查（2026-10-25）

| 资源 | 状态 | 说明 |
|------|------|------|
| 仿真任务与评测 | **已开源** | 仓库含 66 个任务环境、统一策略接口、评测入口、任务验收材料及论文评测协议数据 |
| 训练 / 策略适配 | **部分开源** | 包含适配的 OpenPI π0.5 训练/服务代码及多模型评测适配器；论文 checkpoint 不随仓库分发 |
| 示范数据 | **已发布（PhysX）** | Hugging Face 发布 18,000 条轨迹；Newton 后端数据标注为待发布 |
| 项目页 | **已上线** | 项目页同时列出论文、代码、数据与任务结果 |

## 仓库 README 核查要点

- 任务分为 Core-50（有示范并评测执行）与 Transfer-16（目标任务无示范，测迁移、组合和可变形操作）。
- 默认 PhysX 环境要求 Python 3.11、Isaac Sim 5.1.0.0、Isaac Lab v2.3.2（Python 包版本 0.54.2）及 NVIDIA GPU；可选 Newton evaluator 使用独立 Python 3.12 / Isaac Sim 6 环境，当前任务范围较小。
- 默认评测入口为 `python -m handson.evaluation`；支持注册任务、选择策略适配器或连接策略服务，并记录视频、图像与 runtime receipt。
- 操作状态与命令采用 EEF52 约定；动作以 30 Hz 执行，具体 horizon、历史输入与评分例外依模型配置和论文协议。
- 任务源码、环境与成功谓词可公开检查；策略 checkpoint 未再分发，不能只凭仓库克隆即复现全部论文分数。

## 关联来源

- 论文：[handson_arxiv_2610_26765](../papers/handson_arxiv_2610_26765.md)
- 项目页：[handson-project-page](../sites/handson-project-page.md)
- 数据集：[handson_datasets](../datasets/handson_datasets.md)
- wiki 实体：[paper-handson](../../wiki/entities/paper-handson.md)
