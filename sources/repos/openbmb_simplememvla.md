# OpenBMB/SimpleMemVLA

> 来源归档

- **标题：** SimpleMemVLA（官方实现）
- **类型：** repo
- **组织 / 作者：** OpenBMB（面壁智能 ModelBest）；Yin Cheng 等
- **代码：** <https://github.com/OpenBMB/SimpleMemVLA>
- **论文：** <https://arxiv.org/abs/2609.05533>
- **模型集合：** <https://huggingface.co/collections/yinchenghust/simplememvla>
- **许可：** MIT
- **入库日期：** 2026-09-20
- **一句话说明：** SimpleMemVLA 官方仓：Qwen3.5-4B 原生视频历史 + sub-task 条件 DiT 动作头；统一 `train.py` 与五基准闭环 `eval_*.sh`；HF 释出每套件 checkpoint 与 LeRobot v3 数据集。

## 开源核查（2026-09-20）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是（公开，MIT；组织 `OpenBMB`） |
| 已发布 | 五基准 SFT 训练、闭环评测、开环 proxy 评测；每套件 HF checkpoint（含 `config.json` / processor / `stats.json`）；LeRobot v3 训练集；真机部署结果与视频 |
| 镜像 | ModelScope checkpoint + RMBench 仿真资产 |
| 结论 | **已开源** — 训练/评测/权重/数据均可公开获取；SAPIEN 类基准需按 README **分环境** 安装 |

## 仓库入口（README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `train.py` | 统一 SFT 入口（`--benchmark rmbench\|robomme\|mikasa\|robomemarena\|libero`） |
| `scripts/train.sh <benchmark>` | 各基准发布配方启动脚本 |
| `scripts/eval_<benchmark>.sh` | 官方协议闭环成功率评测 |
| `scripts/eval_<benchmark>_openloop.sh` | 数据集开环 action L1 + sub-task 准确率 |
| `simplememvla/benchmarks/` | 每基准相机/窗口/fps/动作维等规格 |
| `simplememvla/model/` | `SimpleMemVLAConfig`、`SimpleMemVLAForActionPrediction`、DiT 动作头 |
| `simplememvla/data/` | LeRobot 数据集、collator、prompt、`subtask_span_mask` |
| `simplememvla/training/` | Trainer（split LR、cosine、NaN guard） |
| `*_sim/` | 各基准 vendored 仿真 + `policy.py` + `eval_success.py` |
| `configs/sft_params.py` | 模型/数据/训练参数 |
| `configs/zero2.json` 等 | DeepSpeed ZeRO 配置 |

## Hugging Face 资源

| 类型 | 资源 |
|------|------|
| Backbone | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Checkpoints | `yinchenghust/simplememvla_{rmbench,robomme,mikasa,robomemarena,libero}` |
| Datasets | `yinchenghust/{rmbench,robomme,mikasa,robomemarena,libero}_lerobot`（LeRobot v3） |
| Collection | [`yinchenghust/simplememvla`](https://huggingface.co/collections/yinchenghust/simplememvla) |

## 最短复现路径（以 RMBench 为例）

1. `conda create -n simplememvla-rmbench python=3.10 -y` → `pip install -r requirements.txt`
2. `bash scripts/install/install_fast_path.sh`（flash-attn 等）
3. `bash scripts/install/install_rmbench_sim.sh`（SAPIEN + 资产）
4. 下载 `yinchenghust/rmbench_lerobot` 与 `simplememvla_rmbench` checkpoint
5. `bash scripts/train.sh rmbench` 或 `CHECKPOINT=... bash scripts/eval_rmbench.sh`

> **环境注意：** RMBench/MIKASA 需 SAPIEN beta；RoboMME 需 stable SAPIEN；RoboMemArena/LIBERO 为 MuJoCo/robosuite — **不可混装于同一 conda env**。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [SimpleMemVLA](../../wiki/entities/paper-simplememvla.md) | 论文实体：原生视频上下文记忆 VLA |
| [VLA](../../wiki/methods/vla.md) | 长程记忆增强谱系：native context vs 检索/压缩/循环 |
| [KEMO](../../wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md) / [Chronos](../../wiki/entities/paper-chronos.md) | 稀疏关键帧 vs SSM 全历史 vs 本文原生视频 |
| [BridgeVLA++](../../wiki/entities/paper-bridgevla-plusplus.md) | 3D heatmap + 时空记忆对照 |

## 参考来源

- [SimpleMemVLA 论文归档](../papers/simplememvla_arxiv_2609_05533.md)
- [GitHub: OpenBMB/SimpleMemVLA](https://github.com/OpenBMB/SimpleMemVLA)
