# physicalvision / artisan

> 来源归档

- **标题：** Artisan（官方实现与 benchmark）
- **类型：** repo
- **维护者（GitHub 显示）：** physicalvision
- **链接：** https://github.com/physicalvision/artisan
- **项目页：** https://physicalvision.github.io/artisan/
- **论文：** https://arxiv.org/abs/2610.13947
- **数据集：** https://huggingface.co/datasets/mastercai/Artisan-Benchmark
- **许可证：** Apache-2.0（代码仓库）
- **入库日期：** 2026-10-25
- **一句话说明：** Artisan 的官方代码与评测仓库；README 提供 Linux 安装、文本/图像条件资产生成、交互 viewer 与多项 benchmark 评估入口。
- **沉淀到 wiki：** 是 → [`wiki/entities/artisan.md`](../../wiki/entities/artisan.md)

## README 入口

- **安装：** 在 Ubuntu/Debian x86_64 上运行 `./install.sh`，再配置 API key 并运行 `./run.sh doctor`。
- **资产生成：** `./run.sh run "<描述>" --image <参考图>`；README 提供的默认设置包含 GPT-6 Astra 与每个案例的费用预算。具体模型、预算和图像工具可在项目配置中调整。
- **交互检查：** `./run.sh viewer` 启动本地查看器，可检查生成资产并操作关节。
- **Benchmark：** README 描述 ArtVIP、LightWheel、Stability 与 ABO-Mass 数据集的生成和评估命令。

## 备注

- 官方 README 当前记录的默认 API 模型与费用预算可能随仓库更新变化，使用前应检查配置文件。
- 论文性能数字以论文及其 benchmark 定义为准；代码运行需遵循上游依赖与各数据集许可。
