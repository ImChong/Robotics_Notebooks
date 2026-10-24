# Anyverse-Dynamics/Phoenix-MWA（机器人世界动作模型）

> 来源归档

- **标题：** Phoenix-MWA — Learning Generalizable World–Action Models with Structural and Semantic Representations
- **类型：** repo / robot-manipulation / world-action-model
- **来源：** Anyverse Dynamics
- **链接：** <https://github.com/Anyverse-Dynamics/Phoenix-MWA>
- **项目页：** <https://openphoenix-team.github.io/Phoenix-pages/index_zh.html>
- **模型入口：** <https://huggingface.co/AnyverseDynamics>
- **入库日期：** 2026-10-24
- **一句话说明：** Phoenix-MWA 官方预训练代码；以多视角观测、语言、机器人状态和本体提示为条件，通过视频专家与动作专家联合预测未来视觉状态及动作。
- **沉淀到 wiki：** [Phoenix-MWA](../../wiki/entities/phoenix-mwa.md)

## 代码范围

当前 main 分支 README 将本仓库定位为 Phoenix-MWA 的**预训练实现**，主要包括模型、视觉 codec、数据集和训练 runtime。README 指向 post-train 分支作为后训练与基准评测入口；encoder 预训练、数据处理与真实机器人 RTC 异步部署分支仍标注为 coming soon。

训练需要另外准备 `resources/`：Wan 初始化权重、Phoenix Encoder、CS-VAE、数据 manifest 和文本缓存不随 Git 克隆提供；原始视频与数值轨迹存放于外部共享存储。因此当前仓库并非完整数据和部署包。

## 关键入口

| 路径 | 用途 |
|------|------|
| `src/wam/` | 模型、冻结视觉 codec、数据集和训练 runtime |
| `configs/phoenix_pretrain.yaml` | 预训练配方 |
| `scripts/pretrain.sh` | 预训练启动脚本 |
| `tools/` | 数据集与多视角数据准备工具 |
| `docs/` | 资源准备和训练说明 |

## 对 wiki 的映射

- 实体页：[Phoenix-MWA](../../wiki/entities/phoenix-mwa.md)
- 方法背景：[World Action Models](../../wiki/concepts/world-action-models.md)
- 项目页：[Phoenix 项目页](../sites/phoenix-mwa.md)
