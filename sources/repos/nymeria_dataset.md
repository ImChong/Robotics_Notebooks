# nymeria_dataset — Nymeria / NymeriaPlus 官方工具仓

> 来源归档（ingest · 步骤 2.5）

- **仓库：** <https://github.com/facebookresearch/nymeria_dataset>
- **类型：** repo / dataset-tools
- **许可：** 以仓库 LICENSE 为准（数据本身 CC BY-NC 4.0）
- **入库日期：** 2026-09-21
- **一句话说明：** Nymeria 与 NymeriaPlus 的 **下载、布局与可视化 API**；`main` 支持 NymeriaPlus，原版可切 `nymeria_dataset_legacy` 分支。

## 使用入口

```bash
git clone https://github.com/facebookresearch/nymeria_dataset.git
cd nymeria_dataset && uv sync
# 从 Dataset Explorer 获取 *_download_urls.json 后：
aria_dataset_downloader --cdn_file Nymeria_download_urls.json --output_folder ./nymeria --data_types 0
```

## 对 wiki 的映射

- [nymeria-dataset.md](../../wiki/entities/nymeria-dataset.md)
- [paper-nymeria.md](../../wiki/entities/paper-nymeria.md)
