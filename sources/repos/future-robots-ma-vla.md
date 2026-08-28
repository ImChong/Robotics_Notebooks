# MA-VLA / Future（zhangzaibin/future-robots）

- **URL：** <https://github.com/zhangzaibin/future-robots>
- **许可：** Apache-2.0（LICENSE 文件；README 徽章写 MIT）
- **配套论文：** [arXiv:2608.25864](https://arxiv.org/abs/2608.25864)

## 状态（2026-08-28）

| 项 | 状态 |
|----|------|
| 训练入口 | `uv run scripts/train.py …_mavla` |
| 数据转换 | `scripts/tasks/convert_h5_lerobot_stackcubes.py` |
| 部署 | README Deployment 节 |
| MACG 基准 | 数据生成管线已列入 Features |

可运行路径对齐：`scripts/train.py`、`scripts/compute_norm_stats.py`、`src/openpi/training/config.py`。

## wiki

- [`wiki/entities/paper-ma-vla.md`](../../wiki/entities/paper-ma-vla.md)
