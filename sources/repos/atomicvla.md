# AtomicVLA GitHub 仓库

> 来源归档（ingest）

- **项目名称：** AtomicVLA
- **GitHub 地址：** <https://github.com/zhanglk9/AtomicVLA>
- **许可证：** MIT
- **基座：** [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)（子模块）
- **核心功能：** 统一规划–执行 VLA；**SG-MoE** 原子技能库；LIBERO 训练/推理与 openpi 式 policy server 部署。
- **入库日期：** 2026-09-18

## 仓库结构（README 对齐）

| 路径 | 作用 |
|------|------|
| `scripts/compute_norm_stats.py` | 按 config 名（如 `Atomic_libero`）计算归一化统计 |
| `scripts/train.py` | 训练入口：`Atomic_libero --exp-name=...` |
| `scripts/serve_policy.py` | Policy server（openpi 范式，websocket + 硬件 client） |
| `INSTALL.md` | conda 可选安装说明 |
| `assets/images/` | 方法管线示意图 |

## 关键复现路径

1. `git clone --recurse-submodules` → `GIT_LFS_SKIP_SMUDGE=1 uv sync && uv pip install -e .`
2. 数据放 `$HF_HOME/`；准备 **reasoning annotation JSON**（episode → segments：帧范围、skill verb、CoT）
3. 训练：`uv run scripts/compute_norm_stats.py --config-name Atomic_libero` → `uv run scripts/train.py Atomic_libero --exp-name=my_experiment --overwrite`
4. 推理：起 `serve_policy.py`（`--policy.config=Atomic_libero`、`--policy.dir=<ckpt>`）+ 硬件 client
5. 或直接用 HF：`likui/AtomicVLA-libero`

## 关联 Wiki 页面

- [AtomicVLA 论文实体](../../wiki/entities/paper-atomicvla.md)
- [π₀ / openpi](../../wiki/entities/paper-pi0.md)
- [VLA](../../wiki/methods/vla.md)
