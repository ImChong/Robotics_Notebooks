# egosteer/egosmith

> 来源归档

- **标题：** EgoSmith（egocentric 视频 → 全标注 VLA 预训练数据管线）
- **类型：** repo
- **组织 / 作者：** egosteer（PKU / PKU–PsiBot）
- **代码：** <https://github.com/egosteer/egosmith>
- **依赖上游：** HaWoR（submodule）、DPVO、Any4D、MANO 资产（需自 MANO 站点下载）
- **论文：** <https://arxiv.org/abs/2607.09701>
- **项目页：** <https://egosteer.github.io/>
- **许可：** Apache-2.0
- **入库日期：** 2026-07-23
- **一句话说明：** 预过滤 → 度量 4D 手/相机重建 → 多粒度语言标注 → 后过滤；相对 HaWoR 约 **9×** 吞吐，产出可训 WebDataset。

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `bash scripts/setup/setup_env.sh` | conda `egosmith`（CUDA 12.8 + Torch + DPVO） |
| `git submodule update --init thirdparty/hawor_upstream` + `fetch_hawor_base.sh` | 拉取 HaWoR |
| `bash scripts/setup/download_weights.sh` | 下载管线权重（不含 MANO） |
| `python scripts/run_dataset_pipeline.py --config configs/my_video.yaml` | 单视频端到端策展 |
| `python demo.py --video_path …` | 手部重建可视化 |
| `docs/dataset_format.md` | WebDataset / 116-d `lowdim` 字段说明 |

## 与本仓库知识的关系

- 论文归档：[`sources/papers/egosteer_arxiv_2607_09701.md`](../papers/egosteer_arxiv_2607_09701.md)
- 下游训练仓：[`egosteer`](./egosteer.md)
- wiki：[`wiki/entities/paper-egosteer.md`](../../wiki/entities/paper-egosteer.md)
