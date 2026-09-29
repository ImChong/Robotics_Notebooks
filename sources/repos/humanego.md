# HumanEgo（TX-Leo/HumanEgo）

> 来源归档（GitHub README，2026-09-29）

- **标题：** HumanEgo
- **类型：** repo
- **代码：** <https://github.com/TX-Leo/HumanEgo>
- **论文：** <https://arxiv.org/abs/2605.24934>
- **项目页：** <https://humanego-ai.github.io/>
- **入库日期：** 2026-09-29
- **一句话说明：** 官方实现：**预处理 → FlowMatching 训练 → 双臂推理模板**；含 `scripts/download_data.py`、示例任务 `serve_bread` / `water_flowers`。

## 三条使用路径（README）

| 路径 | 入口 |
|------|------|
| 5 分钟 smoke test | 下载 2 条 `--input-only` → `preprocess.Preprocess` → `FlowMatchingTrainer --job HumanEgo` |
| 官方 HumanEgo 数据 | `download_data.py --task all`（含预计算 preprocess）→ 直接训练 |
| 自采数据 | Project Aria + MPS → 自定义 `cfg/preprocess/tasks/*.yaml` → 训练 → `inference/` |

## 关键模块

| 目录 | 作用 |
|------|------|
| `preprocess/` | MPS → 训练就绪标签 |
| `training/` | `FlowMatchingTrainer` |
| `inference/` | `run_inference.py` + 硬件抽象 |
| `datacollection/` | Aria 录制与 MPS 指南 |
| `cfg/` | preprocess / training / inference YAML |

## 对 wiki 的映射

- 论文实体：[`paper-sa-2605-24934-humanego-...`](../../wiki/entities/paper-sa-2605-24934-humanego-zero-shot-robot-learning-from-minutes-o.md)
