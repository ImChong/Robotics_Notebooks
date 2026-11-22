# AR-WAM 官方 GitHub 源码与训练入口

> 一手 repo 归档；复用 [AR-WAM 论文实体](../../wiki/entities/paper-ar-wam.md)。

- **类型：** repo
- **官方仓库：** https://github.com/AR-WAM/AR-WAM
- **论文：** https://arxiv.org/abs/2609.23578
- **主页：** https://ar-wam.github.io/
- **数据集：** https://modelscope.cn/datasets/Jasonggg15/AR-WAM-RoboTwin-Dataset
- **核对日期：** 2026-11-22
- **核对范围：** README Installation/Data/Pretraining/Policy Training、train_rla.py、train.py、models/model_runner.py、dataloader/rla_hdf5_dataloader.py、configs/stage1_robotwin_adaln.yaml；检查实际公开代码所支持的训练路径与缺失的独立推理/Agent 部署入口。
- **通读：** 已完成

## 官方源码入口与执行关系

| 步骤 | 已核实代码/配置 | 作用 |
|---|---|---|
| 获取数据 | README / ModelScope | RoboTwin clean/random HDF5、schema、norm_stats；README 校验 100 份压缩包 SHA256 |
| 前瞻 gist 预训练 | train_rla.py + configs/rla_robotwin.yaml | RLAHDF5PairDataset 取时序两端点，冻结 DINO 特征、RlaAutoencoder 预测未来特征，保存 encoder/decoder/checkpoint |
| 策略 Stage 1 | train.py + configs/stage1_robotwin_adaln.yaml | HDF5 数据→冻结 DINOv3→VLAWrapper/ModelFactory/VLAModel；构造 RLA teacher、动作/前瞻/Mask/条件损失，优化与保存 checkpoint |
| 策略 Stage 2 | train.py + configs/stage2_robotwin_adaln.yaml | --warm-start 读取 Stage1 权重，启用新 optimizer/scheduler，继续训练 |
| 恢复训练 | train_rla.py 或 train.py 的 --resume | 恢复检查点；对世界规模、训练配置及（策略阶段）归一化统计进行核验 |

**官方运行顺序（从仓库根目录执行，以下均为训练）：**

```bash
conda env create -f environment.yml
conda activate ar-wam
python -m torch.distributed.run --standalone --nproc_per_node=8 train_rla.py --config configs/rla_robotwin.yaml --output-dir runs/rla
python -m torch.distributed.run --standalone --nproc_per_node=8 train.py --config configs/stage1_robotwin_adaln.yaml --norm-stats data/arwam_release/norm_stats.json --output-dir runs/stage1
python -m torch.distributed.run --standalone --nproc_per_node=8 train.py --config configs/stage2_robotwin_adaln.yaml --norm-stats data/arwam_release/norm_stats.json --output-dir runs/stage2 --warm-start runs/stage1/checkpoints/step_0550000.pt
```

- RLA 共 150k 更新，Stage1 550k 更新，Stage2 200k 更新；配置中 action_dim=14、state_dim=16、action_chunk_size=32，global_batch_size=128，八卡仅为官方示例。
- 前提：Python 3.10、PyTorch 2.7.1、CUDA 12.8、DINOv3 ViT-L/16 预训练权重；官方声明数据压缩包 455 GiB、解压 616 GiB（总空间至少 1.1 TiB）。
- **开源边界：** 仓库当前提供训练配方、模型/数据加载器；未在 README 给出对外 Agent detect/execute/query 真机完整部署的运行命令，也未见可直接下载的 AR-WAM 预训动作策略权重。因此源码时序图只能画**已公开可运行的训练链**，不把论文 Algorithm 1 当作仓库中已实现可执行 Agent harness。
- **执行验证：** 已按文件阅读入口和参数，未在本环境下载 TB 级数据，也未执行 GPU 训练/真机复现。

## 关联归档

- [论文原文](../papers/ar-wam-arxiv-2609-23578.md)
- [官网](../sites/ar-wam.md)
- [论文详情](../../wiki/entities/paper-ar-wam.md)
