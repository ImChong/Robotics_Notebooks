# eth-siplab/EgoPoser 官方代码仓库核对归档

> 来源归档（GitHub 源码实体核对）

- **仓库：** <https://github.com/eth-siplab/EgoPoser>
- **类型：** repo
- **许可证：** MIT（基于 AvatarPoser 代码库扩展）
- **核对范围：** 核对 `README.md`、`main_train.py`、`main_test.py`、`model_egoposer.py`、`networks/egoposer.py`、`data/dataset_amass.py`、`options/train_egoposer.yaml`、`options/train_egoposer_handtracking.yaml` 与 `options/test_egoposer_hps.yaml`。
- **通读：** 已完成
- **入库日期：** 2026-11-14

## 源码架构与论文差异核对

1. **核心模块与继承结构**：`main_train.py` 与 `main_test.py` 由 YAML 配置驱动，实例化 `AMASS_Dataset`（`data/dataset_amass.py`）与 `ModelEgoPoser`（`model_egoposer.py`），内部骨干网络在 `networks/egoposer.py` 中沿用 `AvatarNet` 类名。
2. **SlowFast 实现细节（`networks/egoposer.py`）**：前向传播将 `x_fast` 与 `x_slow` 均送入同一个 `self.linear_embedding` 后逐元素相加，代码中虽声明了 `self.linear_embedding_fast` 属性但未在前向路径中调用。
3. **YAML 与论文超参差异**：默认训练配置 `train_egoposer.yaml` 与 `train_egoposer_handtracking.yaml` 使用 `input_dim: 60`、`batch_size: 64` 及损失权重 `(0.02, 1, 1 或 20, 0.01)`，与论文正文所写的 59 维特征、batch size 256 与权重 `(0.05, 1, 1, 0.01)` 存在差异；复现时需以目标配置为准。
4. **HPS 大场景评测配置（`test_egoposer_hps.yaml`）**：注释明确在具备大高差变化的户外大场景（HPS）评测时设置 `spatial_normalization: false`。

## 对 wiki 的映射

- [paper-notebook-egoposer-robust-real-time-egocentric-pose-estima](../../wiki/entities/paper-notebook-egoposer-robust-real-time-egocentric-pose-estima.md)
