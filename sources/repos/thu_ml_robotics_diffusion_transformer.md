# RoboticsDiffusionTransformer（thu-ml/RoboticsDiffusionTransformer）

- **URL：** <https://github.com/thu-ml/RoboticsDiffusionTransformer>
- **项目页：** <https://rdt-robotics.github.io/rdt-robotics/>
- **论文：** [rdt_1b_arxiv_2410_07864.md](../papers/rdt_1b_arxiv_2410_07864.md)
- **入库日期：** 2026-09-28

## 一句话说明

RDT-1B 官方 PyTorch：`models/rdt_runner.py`、DeepSpeed `train/train.py`、HF **rdt-1b** / **rdt-170m** 权重、ALOHA `scripts/agilex_inference.py`。

## 关键入口

| 路径 | 用途 |
|------|------|
| `train/train.py` | 微调 / 预训练 |
| `models/rdt_runner.py` | RDT 主体 |
| `scripts/agilex_inference.py` | 真机部署示例 |
| `docs/pretrain.md` | 预训练数据列表 |

## 交叉链接

- [paper-rdt-1b](../../wiki/entities/paper-rdt-1b.md)
