# Movens-Zero 官方源码与运行入口

> 来源归档（repo；对应 arXiv:2610.41877，不另建同名论文节点）

- **仓库：** https://github.com/wendell0218/Movens-Zero
- **论文：** [Movens-Zero arXiv:2610.41877v1](../papers/movens_zero_arxiv_2610_41877.md)
- **项目页：** [Movens-Zero](../sites/movens-zero.md)
- **数据集：** [Movens-5M](../datasets/movens-5m.md)
- **基准：** https://huggingface.co/datasets/wendell0218/Movens-Intent
- **核对日期：** 2026-11-23
- **核对范围：** README、pyproject.toml、movens_zero/inference.py、rollout.py、rollout_root40.py、model.py、export.py、training/train.py、training/data.py 及其调用关系；未在本环境启动 GPU 推理/训练或真机。
- **通读：** 已完成
- **对应 wiki：** [Movens-Zero / Movens-5M](../../wiki/entities/paper-movens-zero.md)

## 可运行代码与模型

- `pyproject.toml` 将 CLI `movens-zero` 绑定到 `movens_zero.inference:main`；支持 `text` / `music` / `speech` / `video`。`Python >=3.10,<3.12`；有 CUDA 模型与依赖要求。
- 推理：`inference.py` 解析 mode/config/checkpoint/初始位姿，加载 `MovensZeroModel.from_pretrained`、`MotionNormalizer`，调用 `rollout.generate_motion`；当配置 `action_dim==40` 时委派 `rollout_root40.generate_root40_motion`。
- 运动分块：`rollout_root40.py` 在首块调用 `predict_action`，后续根据上块重叠区构造 prefix 并调用 `predict_action_rtc`；每块反归一化为 G1 姿态，按 `action_exec_horizon` 推进。`rollout.py` 为另一套绝对运动表示实现；**36D 与 40D 不可混用**。
- 导出：`export_motion` 输出 `joint_pos.csv`、`joint_vel.csv`、`body_pos.csv`、`body_quat.csv`、`body_lin_vel.csv`、`body_ang_vel.csv`、`motion.npy` 和 `metadata.json`；G1 姿态包含 29 关节与根的位置/四元数。
- 训练：`training/train.py` 接受 `--manifest`、`--config`、`--checkpoint` / `--resume` / `--fresh`，`training/data.py` 按模态和 50 Hz motion 读取、采样 batch；loss 及模型另见 `training/objective.py` / `movens_zero/model.py`。多 GPU 可用 DDP，未经本次实测。
- README 列出 S/M/L/XL checkpoints、Movens-Intent evaluation、Sim2Sim / Sim2Real 入口与运行指南；**发布代码及权重≠公开训练数据自动与仓库本地路径就绪**。

## 最小复现阅读路径

1. [README / Installation](https://github.com/wendell0218/Movens-Zero#readme) → `pyproject.toml` 与 checkpoint 依赖。
2. [`inference.py`](https://github.com/wendell0218/Movens-Zero/blob/main/movens_zero/inference.py) → [`rollout_root40.py`](https://github.com/wendell0218/Movens-Zero/blob/main/movens_zero/rollout_root40.py) → [`export.py`](https://github.com/wendell0218/Movens-Zero/blob/main/movens_zero/export.py)。
3. [`training/train.py`](https://github.com/wendell0218/Movens-Zero/blob/main/training/train.py) → [`training/data.py`](https://github.com/wendell0218/Movens-Zero/blob/main/training/data.py) → `training/objective.py`。
4. [数据发布说明](../datasets/movens-5m.md) 分别检查中央 Parquet 预览与四个 tar shard 仓库的许可、格式和文件完整性。

**许可**：代码仓库和数据发布页面的条款须分别核对；中央 HF 数据卡标记 `other`，不推断通用可商用许可。
