# zheyu-zhuang/seeker

> 来源归档

- **标题：** Seeker（动作监督视觉瓶颈）
- **类型：** repo
- **组织 / 作者：** Zheyu Zhuang 等（KTH / 弗莱堡大学 / 汉堡大学）
- **代码：** <https://github.com/zheyu-zhuang/seeker>
- **默认分支：** `open_source`（**不是** `main`；`main` 上 README 404）
- **论文：** arXiv:2608.13422v2 — [`sources/papers/seeker_arxiv_2608_13422.md`](../papers/seeker_arxiv_2608_13422.md)
- **入库日期：** 2026-08-15（源码通读与核对 2026-11-19）
- **通读：** 已完成
- **核对范围：** 通读 `README.md`、`conda_environment.yaml`、`setup.py`、`seeker/model/WEIGHTS.md`、`seeker/scripts/cli.py`、`setup.py`、`train.py`、`rerender_dataset.py`、`merge_lmdb_caches.py`、Workspace/Policy/Model/Encoder 源码及 YAML 配置，验证 `num_refinement_iters: 3` 与演示入口 `seeker_demo.ipynb`
- **一句话说明：** 冻结 DINOv3 + 动作监督 ROI；CLI 覆盖 MimicGen 重渲染、Seeker 预训练、冻结 ROI 训 Diffusion Policy。**已开源、可运行**。

## 开源核查（复核 2026-11-19）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是（公开；默认分支 `open_source`） |
| License | **MIT**（`LICENSE`）。捆绑 DINOv3 / Diffusion Policy / 钉死的 robosuite·robomimic·mimicgen 走各自上游许可 |
| 可运行入口 | **有** — `seeker=seeker.scripts.cli:main`；`seeker setup` / `rerender-dataset` / `train` / `merge-datasets` / `playback-dataset`；演示笔记本 `seeker_demo.ipynb` |
| 权重 | `seeker setup` 拉到 `.weights/`，含 `seeker.mimicgen.pth`、`rvt2_heatmap.mimicgen.pth`、`dinov3.vits16plus.pth` 与 `task_emb_cache.npz`；出处与校验见 `seeker/model/WEIGHTS.md` |
| 数据 | 公开 MimicGen HDF5：<https://huggingface.co/datasets/amandlek/mimicgen_datasets/>；仓内不托管全量轨迹 |
| 结论 | **已开源**（训练、重渲染、可视化、发布权重）。真机 xArm 协议在论文附录，仓主线是 MimicGen |

## 入口速查

| 路径 / 命令 | 作用 |
|-------------|------|
| `mamba env create -f conda_environment.yaml` | 建 `seeker` 环境 |
| `seeker setup` | 调 `setup_suite_deps.sh` + `setup_assets.py`：钉死 robosuite/robomimic/mimicgen（`.dep/`）、下权重、建 `task_emb_cache.npz` |
| `seeker rerender-dataset` | `DatasetRerenderToCache` 把 MimicGen HDF5 重渲到 240×240 `images.lmdb` + `obs_arrays.npz` |
| `seeker merge-datasets` | `merge_lmdb_caches` 合并多个任务的 LMDB 缓存 |
| `seeker train --config-name=train_visual_focus_seeker` | `TrainSeekerWorkspace` + `SeekerTrainingPolicy` + `SeekerTrainingEncoder` 训 Seeker（`T=3`） |
| `seeker train --config-name=train_focus_policy` | `TrainFocusPolicyWorkspace` + `DiffusionPolicy` + `FocusConditionedObsEncoder`（经 `FocusViewTransform` 加载冻结 `seeker.mimicgen.pth`）训下游策略 |
| `seeker/config/method/{seeker,mirroraug,rvt2,oracle}.yaml` | 输入级对照与 overlay 协议 |
| `seeker/model/dinov3_core/` | 捆绑 DINOv3 |
| `seeker/policy/diffusion_policy.py` | 改编自 real-stanford/diffusion_policy |

**最短路径：** `seeker setup` → 下一份 MimicGen HDF5 → `rerender-dataset` → 开 `seeker_demo.ipynb` 看发布权重的 mask/crop。

识别的任务族（`seeker/util/task_meta.py`）：`coffee_preparation`、`mug_cleanup`、`square`、`nut_assembly`、`stack_three`、`three_piece_assembly`、`pick_place`、`threading`。发布权重覆盖其中 6 个（不含 `mug_cleanup` / `nut_assembly`）。

## 对 wiki 的映射

- 论文：[`sources/papers/seeker_arxiv_2608_13422.md`](../papers/seeker_arxiv_2608_13422.md)
- 沉淀 **[`wiki/entities/paper-seeker.md`](../../wiki/entities/paper-seeker.md)**
