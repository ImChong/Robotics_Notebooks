# FARM（HaoranPei-casia/FARM）

> 来源归档

- **标题：** FARM
- **类型：** repo
- **链接：** <https://github.com/HaoranPei-casia/FARM>
- **论文：** <https://arxiv.org/abs/2609.11445>
- **入库日期：** 2026-09-12
- **再核日期：** 2026-09-12
- **一句话说明：** 冻结 VLA-JEPA 预测态上的轻量失败 readout（33,985 参数）；`farm train` / `evaluate` / `adapt` CLI；不含私有数据、backbone 权重与第三方模型源码。

## 仓库入口（README，2026-09-12）

| 命令 | 说明 |
|------|------|
| `pip install -e ".[test]"` | Python ≥3.10 安装 |
| `python examples/make_toy_data.py` | 合成 smoke 数据 |
| `farm train --config configs/train.example.json` | 源域 readout 训练 |
| `farm evaluate --checkpoint ... --manifest ... --split test` | 评测 |
| `farm adapt --config configs/adapt.example.json` | few-shot readout 适配 |
| `pytest` | 单元测试 |

## 数据与特征约定

- 输入：每轨迹 `.npy` 形状 `[T, N, D]`（`B1` token；主配置 `D=1024`）
- Manifest CSV 五列：`trajectory_id, task_id, label, b1_path, split`
- 特征须由用户自有冻结 backbone 抽取；[`DATA.md`](https://github.com/HaoranPei-casia/FARM/blob/main/DATA.md) 公布轨迹 ID 与固定划分，**不含**轨迹内容

## 对 wiki 的映射

- 论文：[`sources/papers/farm-failure-readout_arxiv_2609_11445.md`](../papers/farm-failure-readout_arxiv_2609_11445.md)
- 沉淀 **[`wiki/entities/paper-farm-failure-readout.md`](../../wiki/entities/paper-farm-failure-readout.md)**
