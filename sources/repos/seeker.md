# zheyu-zhuang/seeker

> 来源归档

- **标题：** Seeker（动作监督视觉瓶颈）
- **类型：** repo
- **组织 / 作者：** Zheyu Zhuang 等（KTH / 弗莱堡大学 / 汉堡大学）
- **代码：** <https://github.com/zheyu-zhuang/seeker>
- **默认分支：** `open_source`（**不是** `main`；`main` 上 README 404）
- **论文：** arXiv:2608.13422 — [`sources/papers/seeker_arxiv_2608_13422.md`](../papers/seeker_arxiv_2608_13422.md)
- **入库日期：** 2026-08-15
- **一句话说明：** 冻结 DINOv3 + 动作监督 ROI；CLI 覆盖 MimicGen 重渲染、Seeker 预训练、冻结 ROI 训 Diffusion Policy。**已开源、可运行**。

## 开源核查（2026-08-15）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是（公开；默认分支 `open_source`；约 1★） |
| License | **MIT**（`LICENSE`）。捆绑 DINOv3 / Diffusion Policy / 钉死的 robosuite·robomimic·mimicgen 走各自上游许可 |
| 可运行入口 | **有** — `seeker=seeker.scripts.cli:main`；`seeker setup` / `rerender-dataset` / `train` / `merge-datasets` / `playback-dataset`；`notebooks/inspect_seeker_weights.ipynb` |
| 权重 | `seeker setup` 拉到 `.weights/`，含 `seeker.mimicgen.pth` 与 `dinov3.vits16plus.pth`；出处与校验见 `seeker/model/WEIGHTS.md` |
| 数据 | 公开 MimicGen HDF5：<https://huggingface.co/datasets/amandlek/mimicgen_datasets/>；仓内不托管全量轨迹 |
| 结论 | **已开源**（训练、重渲染、可视化、发布权重）。真机 xArm 协议在论文附录，仓主线是 MimicGen |

## 入口速查

| 路径 / 命令 | 作用 |
|-------------|------|
| `mamba env create -f conda_environment.yaml` | 建 `seeker` 环境 |
| `seeker setup` | 钉死 robosuite/robomimic/mimicgen（`.dep/`）、下权重、建 `task_emb_cache.npz` |
| `seeker rerender-dataset` | 把 MimicGen HDF5 重渲到 240×240 LMDB |
| `seeker train --config-name=train_visual_focus_seeker` | 单任务 / 多任务训 Seeker |
| `seeker train --config-name=train_focus_policy` | 冻结 ROI 训下游 Diffusion Policy |
| `seeker/config/method/{seeker,mirroraug,rvt2,oracle}.yaml` | 输入级对照与 overlay 协议 |
| `seeker/model/dinov3_core/` | 捆绑 DINOv3 |
| `seeker/policy/diffusion_policy.py` | 改编自 real-stanford/diffusion_policy |

**最短路径：** `seeker setup` → 下一份 MimicGen HDF5 → `rerender-dataset` → 开 `inspect_seeker_weights.ipynb` 看发布权重的 mask/crop。

识别的任务族（`seeker/util/task_meta.py`）：`coffee_preparation`、`mug_cleanup`、`square`、`nut_assembly`、`stack_three`、`three_piece_assembly`、`pick_place`、`threading`。发布权重覆盖其中 6 个（不含 `mug_cleanup` / `nut_assembly`）。

## 对 wiki 的映射

- 论文：[`sources/papers/seeker_arxiv_2608_13422.md`](../papers/seeker_arxiv_2608_13422.md)
- 沉淀 **[`wiki/entities/paper-seeker.md`](../../wiki/entities/paper-seeker.md)**
