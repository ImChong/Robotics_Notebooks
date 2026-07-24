# EgoVerse（GaTech-RL2/EgoVerse）

> 来源归档（ingest）

- **标题：** EgoVerse: Egocentric Data for Robot Learning from Around the World
- **类型：** repo / dataset tooling / training code
- **代码：** <https://github.com/GaTech-RL2/EgoVerse>
- **项目页：** <https://egoverse.ai/>
- **论文：** <https://arxiv.org/abs/2604.07607>
- **数据浏览器：** <https://partners.mecka.ai/egoverse>
- **许可证：** MIT
- **默认分支：** `main`
- **入库日期：** 2026-07-24
- **一句话说明：** EgoVerse 官方仓库：统一 zarr 数据处理、EgoDB/SQL 元数据访问、人–机协同训练（HPT + flow matching BC）与可视化工具；训练入口 `egomimic/trainHydra.py`。

## 开源边界（入库日核实）

| 项 | 状态 |
|----|------|
| **代码** | **已开源**（MIT）：处理、训练、评测、可视化 |
| **算法实现** | `egomimic/algo`：ACT、EgoMimic（HPT）、Pi 等 |
| **数据** | 经 SQL 过滤 + `sync_s3.py` / 训练管线自动从 Cloudflare R2 拉取；**需官方云访问配置**，非公开匿名全量下载 |
| **权重** | README 以训练脚本为主；具体公开 checkpoint 以仓库/项目页后续说明为准 |

## 仓库结构（README 对齐）

| 路径 | 作用 |
|------|------|
| `egomimic/trainHydra.py` | 主训练入口（PyTorch Lightning + Hydra，支持 DDP） |
| `egomimic/hydra_configs/` | 算法与数据配置（如 `train_zarr_cartesian`、`data/eva_human_cotrain`） |
| `egomimic/algo/` | ACT / EgoMimic(HPT) / Pi |
| `egomimic/scripts/aria_process/` | Aria VRS → zarr / LeRobot |
| `egomimic/scripts/aloha_process/` | ALOHA hdf5 → zarr / LeRobot |
| `egomimic/scripts/data_download/sync_s3.py` | 按 filter 同步子集到本地 |
| `egomimic/scripts/data_visualization/latent_inspector.py` | 本地 episode zarr 浏览器 |
| `training.md` | BC / flow-matching 训练速查 |
| `CONTRIBUTING_DATA.md` | 数据贡献与 zarr 约定（含 intrinsics 强制字段） |

## 运行时注意（changelog 摘要）

- **Camera intrinsics 强制**：每个 episode `zarr.json` 必须含 `{camera_key: 3×4 K}`；缺失则写入失败。
- **Human embodiment 折叠**：人示教统一为 `human_*`（ids 1–3），机器人 Eva 为 `eva_*`（ids 4–6）；旧 `aria_*` / `mecka_*` / `scale_*` 标签已移除，本地缓存需 **重新下载**。
- **Aria EE/腕朝向修正**：左右手共享与 Eva 工具系对齐的 `T_ROT_CAM`；需重处理 Aria 数据。

## 对 wiki 的映射

- [EgoVerse 论文实体](../../wiki/entities/paper-egoverse.md) — 源码运行时序图对齐本仓库入口
- [EgoWAM](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md)
- [Imitation Learning](../../wiki/methods/imitation-learning.md)

## 交叉链接（sources 互指）

- 项目页：[egoverse-ai.md](../sites/egoverse-ai.md)
- 论文：[egoverse_arxiv_2604_07607.md](../papers/egoverse_arxiv_2604_07607.md)
