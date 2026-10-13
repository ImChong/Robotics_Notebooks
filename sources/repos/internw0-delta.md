# InternW0-Δ — 官方代码仓库归档

- **仓库：** <https://github.com/InternRobotics/InternW0-Delta>
- **项目页：** <https://internrobotics.github.io/InternW0-Delta/>
- **默认分支：** main
- **核验日期：** 2026-10-13
- **许可：** 主代码 MIT；仓库 NOTICE 中的第三方依赖和模型/数据资产应各按其自身许可处理。
- **策展节点：** [InternW0-Δ](../../wiki/entities/paper-internw0-delta.md)

## 仓库当前开放范围

根目录已有 `src/wam`、`configs`、`deploy`、`eval`、`scripts`、`tools`、`tests`、`docs`、`LICENSE`、`NOTICE`、`README.md` 和中文 README。README 给出了安装、预训练、4D 蒸馏、目标任务 post-training、评测、真实机器人部署和可视化入口；不是只有 project-page 的演示仓。

安装说明要求 Linux、Python 3.11、CUDA GPU、FFmpeg；示例环境安装 PyTorch 2.10.0 / CUDA 12.8。预训练入口示例包括 `python tools/path_index.py`、`python tools/pretrain_text_cache.py`、`NPROC_PER_NODE=8 bash run.sh pretrain`；目标任务 post-training 有相应 task config。

## 当前发布状态需分开描述

- **训练/评测/部署实现：已公开**，MIT 代码许可。
- **预训练和任务模型权重：** README/model guide 截至核验日标注为 forthcoming / 按指南下载；需检查具体 checkpoint 是否已发布后再声称可直接复现实验。
- **数据处理代码与数据：** 仓库开放相关处理入口，但原始和处理后数据能否下载/再分发取决于各自许可。
- **第三方模型：** Wan2.2、RynnBrain、Track4World 等组件不因本仓 MIT 声明而自动变为 MIT。

## 代码导航（README）

| 主题 | 官方入口 |
|---|---|
| 数据准备 | README 的 Data Preparation guide |
| 模型预训练与 4D 蒸馏 | README 的 Training、4D Distillation |
| 下游 post-training | LIBERO、RoboTwin、RoboDojo、EBench、RTC |
| 真机部署 | RTC 和同步/异步部署 guide |
| 推理加速 | LiteGen、VAE/VLM caching、MoT compile |
| 状态/动作接口 | `robot.yaml` 中统一 80D 带 mask 语义布局 |
