# EgoDex（apple/ml-egodex）

> 来源归档

- **标题：** EgoDex
- **类型：** repo / dataset
- **来源：** Apple
- **链接：** <https://github.com/apple/ml-egodex>
- **论文：** <https://arxiv.org/abs/2505.11709>
- **数据：** README 内 Apple CDN 下载链接（约 2.0 TB）
- **许可：** 示例代码使用 Apple 源码许可；数据为 CC BY-NC-ND
- **入库日期：** 2026-07-28
- **一句话说明：** EgoDex 官方数据与轻量工具入口，公开 829 小时第一视角操作数据，并提供加载、2D/3D 可视化和 best-of-K 指标脚本。
- **开源状态：** **部分开源** — 数据与样例工具公开；论文使用的 X-IL 大规模训练实现、配置和权重不在本仓库。
- **沉淀到 wiki：** [`wiki/entities/paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md`](../../wiki/entities/paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md)

## 仓库概况（2026-07-28）

| 字段 | 值 |
|------|-----|
| 安装 | Python 3.11、FFmpeg 7.1.1、`pip install -r requirements.txt` |
| 数据入口 | 同编号 `.mp4` + `.hdf5`；训练 5×300 GB、测试 16 GB、追加 200 GB |
| 可运行入口 | `simple_dataset.py`、`visualize_2d.py`、`visualize_3d.py`、`compute_metrics.py` |
| 未提供 | 论文 14 个 X-IL 模型的训练入口、checkpoint 与完整训练配置 |

## 对 wiki 的映射

- 论文来源：[`humanoid_pnb_egodex.md`](../papers/humanoid_pnb_egodex.md)
- 论文实体：[`paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md`](../../wiki/entities/paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md)
