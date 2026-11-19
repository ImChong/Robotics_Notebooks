# AWARE 官方源码：可复现入口与开放状态

> 来源归档（repo），供 [AWARE 独立论文节点](../../wiki/entities/paper-aware.md) 溯源。

- **官方仓库：** <https://github.com/adaptive-intelligent-robotics/aware>
- **论文：** <https://arxiv.org/html/2610.32742v1>
- **入库：** 2026-10-29；**源码核验：** 2026-11-19
- **核对范围：** 仓库 README，`scripts/generate_data.py`、`scripts/train.py`、`scripts/evaluate.py` 和各自的 `configs/*.yaml`；核验到真实运行入口，未运行 GPU 环境。
- **通读：** 已完成

## 开源状态

**已开源并有可运行入口。** README 明确给出仿真生成、模型训练和数据集评测命令；提供高/低噪声 240 条真实轨迹和模型 checkpoint，代码 MIT，数据 CC BY-NC-SA 4.0；CUDA GPU 为使用要求。

## 实际入口与调用

1. `python scripts/generate_data.py num_episodes=10`：Hydra 实例化模拟环境与 operator，调用 `aware.data_generation.generate_dataset`，反复 `agent.get_action` 和 `env.step`，数据写入 `data/sim`。
2. `python scripts/train.py model=rssm_aware dataset_path=...`：实例化 cfg.model / cfg.trainer 并调用 `trainer.train()`；`model=transformer_aware` 是另一训练配置。
3. `python scripts/evaluate.py checkpoint=checkpoints/rssm_aware_seed1`：构建 `Agent_Double_Discriminator`、`anomaly.load_dataset`、`anomaly.evaluate`，输出 AP、AUROC、F1 等指标到 `outputs/evaluation/`。

最短复现应先读 README 对 Python/CUDA/Docker、ckpt 目录、数据授权的要求；无需重新生成论文量级 1.3 TB 模拟数据就可对公开轨迹执行模型评测。具体目录与参数均以官方仓库当前版 README 为准。

## 参考来源

- [官方 README](https://github.com/adaptive-intelligent-robotics/aware/blob/main/README.md)
- [生成入口](https://github.com/adaptive-intelligent-robotics/aware/blob/main/scripts/generate_data.py)
- [训练入口](https://github.com/adaptive-intelligent-robotics/aware/blob/main/scripts/train.py)
- [评测入口](https://github.com/adaptive-intelligent-robotics/aware/blob/main/scripts/evaluate.py)
