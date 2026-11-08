# GazeVLA — 官方训练代码仓库

- **类型：** repo / research-code / vla / gaze
- **仓库：** https://github.com/lichy2004/GazeVLA
- **项目页：** https://gazevla.github.io/
- **论文：** https://arxiv.org/abs/2604.22615
- **数据处理：** https://github.com/lichy2004/GazeVLA-Data
- **核对范围：** 官方 README；`examples/process_data_1.py`、`examples/process_data_2.py`、`scripts/compute_norm_stats.py`、`scripts/train/train.py`、`src/openpi/training/config.py`、`src/openpi/training/data_loader.py`、`src/openpi/models/VLIA_config.py`、`src/openpi/models_pytorch/VLIA_pytorch.py`；按 README 的训练入口与实际 import/构造调用核对。
- **核对日期：** 2026-11-08

## README 所给复现入口

1. 获取 AV-ALOHA 数据；`examples/process_data_1.py` 将 Parquet/视频/眼动观测转换成 HDF5，`examples/process_data_2.py` 转换为 RLDS（包括 resized gaze 和有效性字段）。
2. `python scripts/compute_norm_stats.py --config-name lfa-cam1-chunk25`：从相应数据 loader 计算动作归一化统计。
3. `torchrun --nnodes=1 --nproc_per_node=1 scripts/train/train.py lfa-cam1-chunk25 --exp_name=lfa-cam1-chunk25`：`TrainConfig` → `data_loader.create_data_loader` → Accelerate 训练循环 → gaze CE/MSE 与 action loss → optimizer/checkpoint。
4. 模型实现 `VLIA_pytorch.py` 定义了 gaze tokenizer/head、视觉语言 prefix、状态/动作 suffix 与 flow-matching 动作专家。配置定义于 `VLIA_config.py`。

## 源码级复现风险（已检查具体路径，未实际训练）

- 当前 `train.py` 虽导入 `openpi.models.VLIA_config`、`openpi.models_pytorch.VLIA_pytorch`，但模型构造处仍访问 `openpi.models.IntentionVLA_config.IntentionVLAConfig` 与 `openpi.models_pytorch.IntentionVLA_pytorch.IntentionVLA_pytorch`；这两个指定路径在当前默认分支未找到。**README 命令不能直接视为已验证可运行**。
- README 中 clone 例子指向 `lichy2004/GazeVLA-Code.git`，而项目主页链接的是 `lichy2004/GazeVLA`；应以官网给出的真实仓库为准。
- `examples/process_data_1.py` 写 `data/av_aloha`，README 示例为 `data/av-aloha`；`process_data_2.py` 默认包含作者本机绝对路径，运行前需逐项修改和确认。
- 官方 TODO：训练代码已提交；**预训练 checkpoint 未发布、evaluation scripts 未发布**，且评价部分要求手改 LFA 官方项目。
- **许可证需复核**：GitHub 顶部显示仓库 LICENSE 为 Apache-2.0；README 顶部徽章写 MIT；底部写项目及数据 CC BY-NC 4.0。不能仅按一种表述推定商用许可。

## 与论文实现的边界

论文 §3 / 附录 B 解释了意图→动作架构及训练，仓库给出若干模型/训练组成，但未经环境安装和端到端训练验证，不能宣称它已完成论文中所有预训练、评测及实机部署。

## 映射

- [论文原文](../papers/gazevla_arxiv_2604_22615.md)
- [项目主页](../sites/gazevla_project.md)
- [数据处理仓库](./gazevla_data.md)
- [唯一实体节点](../../wiki/entities/paper-gazevla.md)
