# Mondo-Robotics / PMT（Perceptive Motion Tracking）

> 来源归档（ingest · Perceptive BFM 官方代码）

- **标题：** PMT — Perceptive Motion Tracking
- **类型：** repo
- **官方入口：** <https://github.com/Mondo-Robotics/PMT>
- **项目页：** <https://acodedog.github.io/perceptive-bfm/>
- **论文：** <https://arxiv.org/abs/2606.08059>
- **资产镜像：** Hugging Face dataset `aCodeDog/PMT-assets`；Google Drive 备份见 README
- **机构：** 摩多机器人（Mondo Robotics）；香港科技大学（广州）；香港科技大学；妙动科技等（以论文作者为准）
- **入库日期：** 2026-07-22
- **一句话说明：** Perceptive BFM 官方实现：Isaac Lab 上 Unitree G1 的 config-driven 运动跟踪 RL（`scripts/train.py` / `play.py`），含 **TCRS** 地形运动优化器、teacher→distill→finetune 管线、预训练 checkpoint 与样例 `raw`/`optimized` 地形片段。

## 开源状态（2026-07-22 项目页 + README 核查）

| 产物 | 状态 |
|------|------|
| 代码 | **已开源** · [Mondo-Robotics/PMT](https://github.com/Mondo-Robotics/PMT) |
| 预训练策略 | `checkpoints/pretrained/`（HF `aCodeDog/PMT-assets` / Git LFS） |
| 样例运动 + 地形 | `assets/` 下 99-pair `raw`/`optimized` + meshes |
| TCRS | 仓内 [`TCRS/`](https://github.com/Mondo-Robotics/PMT/tree/main/TCRS) MPPI 优化器 |
| 浏览器 demo | 项目页 MuJoCo WASM + ONNX（无需装仓） |

> **范围：** 研究/训练代码，**依赖已有 Isaac Lab 环境**（不 vendor Isaac Sim）；平坦任务族亦可走 mjlab（MuJoCo-Warp），见 `docs/MJLAB_USAGE.md`。

## 仓库结构（README 摘要）

| 路径 | 职责 |
|------|------|
| `scripts/train.py` / `scripts/play.py` | 训练与回放入口（~25 个 G1 task id） |
| `pmt_tasks/` | 任务层：组合 `robot` / `terrain` / `motion` / `obs` / `reward` / `network` / `algorithm` / `stage` |
| `configs/` | Hydra/YAML 配置与 `paths.yaml`（`PMT_*` 环境变量覆盖） |
| `motion_tracking_rl/` | RL 核心（editable `pip install -e .`） |
| `TCRS/` | 平地 `.npz` + 地形 `.xml` → `raw/` + `optimized/` + `ghost/` |
| `checkpoints/` | 预训练策略（含 distill 等） |

## 关键复现路径

1. 激活含 Isaac Lab 的 conda/venv；`export OMNI_KIT_ACCEPT_EULA=YES`
2. `pip install -e .`（仓库根）
3. `hf download aCodeDog/PMT-assets --repo-type dataset` → rsync `assets` / `checkpoints` / `TCRS` 进仓根
4. 平坦多运动：`python scripts/train.py --task PMT-G1-MultiMotionV2-Flat-v0 …`；回放：`python scripts/play.py --task … --resume_path …`
5. 地形 / 视觉 / distill 任务走 Isaac Lab；TCRS 单独生成 `optimized` 监督片段

## 对 wiki 的映射

- [Perceptive BFM 论文实体](../../wiki/entities/paper-perceptive-bfm.md)
- [项目页归档](../sites/perceptive-bfm-github-io.md)
- [论文摘录](../papers/perceptive_bfm_corl_2026.md)
