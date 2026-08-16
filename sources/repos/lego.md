# WHU-USI3DV/LEGO

> 来源归档

- **标题：** LEGO: Leveled Language Gaussian Splatting
- **类型：** repo
- **组织 / 作者：** WHU-USI3DV（Yuning Peng / Haiping Wang / Yuan Liu / Yipeng Lu / Zhen Dong / Bisheng Yang）
- **代码：** <https://github.com/WHU-USI3DV/LEGO>
- **项目页：** <https://pz0826.github.io/LEGO-Webpage/>
- **论文：** arXiv:2608.10057 — [`sources/papers/lego_leveled_language_gs_arxiv_2608_10057.md`](../papers/lego_leveled_language_gs_arxiv_2608_10057.md)
- **许可：** CC BY-NC-SA 4.0（非商业研究）；MASt3R / SAM / gsplat / OpenCLIP 保留各自许可
- **入库日期：** 2026-08-16
- **一句话说明：** 官方可运行实现：`lego` CLI 覆盖 SAM 掩码 → MASt3R/COLMAP 重建 → 层级定级 → RGB + 层级特征蒸馏 → HDBSCAN 树 / 关系图 → CLIP / 评测 / Viser 查看器。
- **沉淀到 wiki：** [`wiki/entities/paper-lego-leveled-language-gaussian-splatting.md`](../../wiki/entities/paper-lego-leveled-language-gaussian-splatting.md)

## 开源核查（2026-08-16）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是（公开；约 6★；2026-08-12 创建） |
| 语言 / 体积 | Python；约 1.9k（含 vendored `third_party/gsplat`） |
| 可运行入口 | **有** — `lego run` / `eval` / `viewer` / `validate` / `doctor` |
| 权重 | 无场景 checkpoint（`checkpoints/` 仅 `.gitkeep`）；需 `scripts/download_models.sh` 拉 MASt3R 等，OpenCLIP 首次用时缓存 |
| License | CC BY-NC-SA 4.0 |
| 结论 | **已开源、可运行训练 / 评测 / 可视化** |

## 入口速查（对齐 README / `src/lego`）

| 路径 / 命令 | 作用 |
|-------------|------|
| `scripts/setup_env.sh` | Conda 环境 `lego`（Python 3.11 / PyTorch 2.5.1 / CUDA 12.1）并编译 CUDA 扩展 |
| `scripts/download_models.sh --accept-mast3r-license` | 本地下载 MASt3R 等权重 |
| `lego doctor` | 检查环境、扩展与 checkpoint |
| `lego config <dataset/scene>` | 解析并写出 resolved 配置 |
| `lego run <dataset/scene>` | 全管线；可用 `--from-stage` / `--to-stage` / `--set` |
| `lego validate <dataset/scene>` | 检查 `clustering/` 下 label / tree / CLIP / relation_graph |
| `lego eval <dataset/scene>` | 调 `benchmarks/` 协议（LERF / Mip-NeRF 360 / 3D-OVS / NVOS / SPIn-NeRF / CoR） |
| `lego viewer <dataset/scene>` | Viser：RGB / 层级 / 簇 / 开放词汇；`--scene-graph` 开 LLM CoR |
| `src/lego/pipeline/runner.py` | `STAGE_NAMES`：`generate-masks` → `reconstruct` → `project` → `map-masks` → `assign-levels` → `export-levels` → `train-rgb` → `train-features` → `build-tree` → `build-relation-graph` → `select-views` → `match-sam` → `extract-clip` |
| `src/lego/gaussian/train_rgb.py` / `train_hierarchy.py` | RGB 场与层级对比蒸馏 |
| `src/lego/leveling/` | 掩码提升、定级、导出 |
| `src/lego/scene_graph/` | HDBSCAN 树、邻接图、SAM 匹配 |
| `src/lego/semantics/clip.py` | 最优视角 CLIP |
| `benchmarks/` | 各数据集评测脚本 |
| `configs/scenes/` | 论文场景 YAML（`3d_ovs/room`、`lerf/teatime`、`cor/teatime` 等） |

环境变量：`LEGO_DATA_ROOT`、`LEGO_OUTPUT_ROOT`。CoR 标注在 [Google Drive](https://drive.google.com/drive/folders/1hpuLBeH6CTDWLMg_F7teCfcN8gjPblfP?usp=sharing)，放到 `<data_root>/cor`。LLM 查询需 `LLM_API_KEY`（默认 OpenRouter）。

README 声明验证环境：Ubuntu 20.04、24GB GPU。论文：两阶段 GS 约 20–60 min/场景（RTX 4090），树 + CLIP 再 5–10 min。

## 对 wiki 的映射

- 论文：[`sources/papers/lego_leveled_language_gs_arxiv_2608_10057.md`](../papers/lego_leveled_language_gs_arxiv_2608_10057.md)
- 项目页：[`sources/sites/pz0826-lego-webpage.md`](../sites/pz0826-lego-webpage.md)
- 沉淀 **[`wiki/entities/paper-lego-leveled-language-gaussian-splatting.md`](../../wiki/entities/paper-lego-leveled-language-gaussian-splatting.md)**
