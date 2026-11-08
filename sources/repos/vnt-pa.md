# VNT-PA 官方源码（beimingli0626/VNT-PA）

- **类型：** repo / code / visual-navigation / imitation-learning
- **官方仓库：** <https://github.com/beimingli0626/VNT-PA>
- **论文：** <https://arxiv.org/abs/2609.21212>，见 [论文归档](../papers/vnt-pa_arxiv_2609_21212.md)
- **官方作者主页：** <https://beimingli0626.github.io/>，见 [主页归档](../sites/beimingli-vnt-pa.md)
- **开源状态：** 已开源（MIT；third_party 子模块适用各自许可）
- **入库日期：** 2026-11-08
- **核对范围：** README、`src/pan/collection/explore.py`、`tokens.py`、`src/pan/planner/model.py`、`dataset.py`、`train.py`、`eval.py` 的实际入口和主要逻辑。
- **通读：** 已完成

## 运行入口（README + 源码）

```bash
# 采集 exploration tours（需要 HM3D v0.2）
python -m pan.collection.explore --scene-root "$HM3D/train" --dataset-config "$HM3D/hm3d_basis.scene_dataset_config.json" --num-scenes 800 --seeds-per-scene 5 --seed-offset 100 --out-root "$DATA/traj/train"
# 冻结 DeFM 提取 patch tokens + 覆盖率关键帧标签
python -m pan.collection.tokens --corpus "$DATA/traj/train" --out "$DATA/tokens/train" --gpu 0
# 验证集 expert episode bank（验证探索和 tokens 须事先生成）
python -m pan.collection.bank --token-root "$DATA/tokens/val" --out "$DATA/banks/bank_val.npz" --per-tour 5 --queries 16 --chunk 8 --seed 1
# imitation training
torchrun --nproc_per_node=1 -m pan.planner.train --run-name pan --accum 4 --wandb-project '' --token-root "$DATA/tokens/train" --val-token-root "$DATA/tokens/val" --val-bank "$DATA/banks/bank_val.npz"
# closed-loop evaluation
python -m pan.planner.eval --ckpt logs/pan/checkpoints/final.pt --token-root "$DATA/tokens/val" --bank "$DATA/banks/bank_val.npz" --gpu 0 --out logs/pan/eval/keyframes.json
```

## 代码结构与证据

| 源码 | 核查事实 |
|---|---|
| [collection/explore.py](https://github.com/beimingli0626/VNT-PA/blob/main/src/pan/collection/explore.py) | 采集仿真探索轨迹并写出深度、位姿等数据 |
| [collection/tokens.py](https://github.com/beimingli0626/VNT-PA/blob/main/src/pan/collection/tokens.py) | `DeFMEncoder.encode` 离线提取 patch tokens；`gate_labels` 生成 keyframe keep 位，存 HDF5 |
| [planner/dataset.py](https://github.com/beimingli0626/VNT-PA/blob/main/src/pan/planner/dataset.py) | `TourStore` 读取关键帧 token / 位姿；`TourDataset` 给出 query poses / goals / expert chunk |
| [planner/model.py](https://github.com/beimingli0626/VNT-PA/blob/main/src/pan/planner/model.py) | `Planner` 依次 `summarize` → `encode_map` → `read` → `decode`；`apply_3d_rope` 实现平面 x/z/yaw RoPE |
| [planner/train.py](https://github.com/beimingli0626/VNT-PA/blob/main/src/pan/planner/train.py) | `TrainConfig`、`build_model`、`TourDataset`、交叉熵训练/存检查点 |
| [planner/eval.py](https://github.com/beimingli0626/VNT-PA/blob/main/src/pan/planner/eval.py) | `load_planner` → `summarize_tour` / `encode_for_goal` → `closed_loop_eval` |
| [planner/env.py](https://github.com/beimingli0626/VNT-PA/blob/main/src/pan/planner/env.py) | `Stepper` 与 `policy_rollout` 执行导航动作并回收评价指标 |

## 依赖、可用性与限制

- README 指定 Linux/NVIDIA GPU、Python 3.10、PyTorch 2.5.1/CUDA 12.1、habitat-sim 0.3.3 (headless + Bullet)。
- DeFM ViT-S/14 冻结权重首次运行从 `leggedrobotics/defm` 下载；HM3D v0.2 全 split 约 240GB。
- 仓内提供训练/评估源码，**不等于发布预训练 VNT-PA checkpoint**；未执行端到端复现。本归档为静态核验结果。

## 对 wiki 的映射

- [VNT-PA 实体节点](../../wiki/entities/paper-vnt-pa.md)
