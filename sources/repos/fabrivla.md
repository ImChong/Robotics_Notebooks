# FabriVLA（Youi-FabriX/FabriVLA）

> 来源归档

- **标题：** FabriVLA — Lightweight VLA for Precise Multi-Task Manipulation
- **类型：** repo
- **组织：** Youi-FabriX（优艾智合 FabriX 团队）× 澳门大学 Mohismlab 合作实现
- **代码：** <https://github.com/Youi-FabriX/FabriVLA>
- **权重：** Hugging Face [`Youi-FabriX/FabriVLA`](https://huggingface.co/Youi-FabriX/FabriVLA)（`checkpoint_step_93000.pt`，Apache-2.0）
- **论文：** <https://arxiv.org/abs/2607.08575>
- **许可：** Apache-2.0
- **入库日期：** 2026-07-23
- **一句话说明：** FabriVLA 官方训练/评测代码：InternVL3.5-1B + gated self-attention flow-matching 动作头 + shallow layer fusion；LeRobot 格式 Meta-World 数据；DeepSpeed ZeRO-2 + FP32 master weights 单阶段联合微调；`evaluations/metaworld/evaluate_mt50.py` 复现 MT50。
- **沉淀到 wiki：** [FabriVLA（论文实体）](../../wiki/entities/paper-fabrivla.md)

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | **0.89B** 轻量 flow-VLA；Meta-World tier-avg **90.0%** |
| [Evo-1](../../wiki/entities/paper-evo1-lightweight-vla.md) | 架构启发来源；训练数据用公开 **Evo-1 Meta-World** 演示集 |
| [VLA SOTA Leaderboard](../../wiki/entities/vla-sota-leaderboard.md) | 榜站 Meta-World 开源区收录（入库日 rank 1） |
| [Action Chunking](../../wiki/methods/action-chunking.md) | 50 步 × 24 维动作块 + receding-horizon 执行 |
| [LeRobot](../../wiki/entities/lerobot.md) | 数据路径按 LeRobot parquet/video 布局配置 |

## 工程要点（README 摘要）

- **布局：** `metawold-vla/`（训练：`src/train.py`、`model/action_head.py`、`internvl_embedder.py`）+ `evaluations/metaworld/`。
- **训练：** `accelerate launch --num_processes 5 --multi_gpu ... train.py --config ...1-exp_shallow_concat_proj_scratch_100k.yaml`；缺 DeepSpeed FP32 master 时 BF16 VLM 更新会被量化噪声淹没（`train.py` 会报错）。
- **评测：** `evaluate_mt50.py --checkpoint ... --episodes 10 --episode-horizon 400 --exec-horizon 5 --num-inference-timesteps 50`。
- **相机：** Meta-World 用 `corner2`（论文叙述亦称 corner RGB）作策略输入。

## 为何值得保留

- **可复现轻量 VLA：** 代码 + 93k 步权重齐全，可对照 Evo-1 验证「单阶段 + gated SA + shallow fusion」配方。
