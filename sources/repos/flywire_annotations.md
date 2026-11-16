---
type: source
source_type: repo
title: "flywire_annotations 官方注释仓库与数据集发布核查"
urls:
  - "https://github.com/flyconnectome/flywire_annotations"
  - "https://codex.flywire.ai/"
  - "https://doi.org/10.5281/zenodo.10676866"
  - "https://doi.org/10.5281/zenodo.10877326"
created: 2026-11-14
updated: 2026-11-14
tags: [connectomics, dataset, drosophila, flywire, cambridge, princeton]
---

# `flyconnectome/flywire_annotations` 仓库核查

- **官方仓库：** <https://github.com/flyconnectome/flywire_annotations>
- **核对范围：** 核查 `README.md`、`supplemental_files/README.md`（至 `v3.2.0` 归档说明）、`supplemental_files/Supplemental_file1`–`Supplemental_file5`、`cocoglancer/README.md` 与 `code/annotation_counts.ipynb`（单元格 1–100 及统计汇总单元 770–869），并对照主论文 Zenodo 静态归档与 Codex 下载机制
- **通读：** 已完成

## 一、开源开放状态核查（步骤 2.5）

| 资源层级 | 开放状态 | 说明与限制 |
|---|---|---|
| 细胞注释与分型表（`flywire_annotations`） | **已开源** | GitHub 仓库 `supplemental_files/` 公开发布版本化 TSV/CSV 注释表（含 `Supplemental_file1`–`Supplemental_file5`，至 `v3.2.0`）；主论文与分型数据均以 **CC-BY 4.0** 开放 |
| 静态连接组与骨架归档（Zenodo） | **已开源** | 突触表、网格/骨架与 NBLAST 比对结果静态归档于 Zenodo（`10.5281/zenodo.10676866`、`10.5281/zenodo.12588557`、`10.5281/zenodo.10877326`），可免登录直接复现 Nature 2024 快照 |
| 在线交互门户与实时库（Codex / CAVE） | **部分开放（需账号认证）** | `https://codex.flywire.ai/` 与 CAVE/PyChunkedGraph 后端对学界开放免费访问，但交互检索与服务端 API 查询要求 **Google 账号登录 / CAVE Token** |
| 平台完整 Web/分布式校对服务代码 | **分散开源（非单一 monorepo）** | `flywire_annotations` 本身仅含数据表与统计脚本；可视化与查询工具分散于 `Neuroglancer`、`CAVEclient`、`fafbseg-py`、`navis` / `coconatfly` 与 `codex` 等独立仓库 |

## 二、仓库结构与可运行代码核查

- `README.md`：说明仓库作为 FlyWire FAFB 全脑连接组（v783）细胞类型与层级注释的发布和跟踪载体。
- `supplemental_files/README.md` 与 `Supplemental_file1`–`Supplemental_file5`：存放 Nature 2024 细胞分型论文对应的补充数据表（含神经元元数据、发育谱系/半谱系、Hemibrain 共识映射与跨连接组注释，迭代至 `v3.2.0`）。
- `cocoglancer/README.md`：记录与 Hemibrain 共注册的 Neuroglancer 预加载图层入口（`tinyurl.com/flywirehb`）。
- `code/annotation_counts.ipynb`：唯一脚本/笔记本文件，使用 `pandas` 读取补充 CSV/TSV 表格并复算论文表中的神经元分类计数（如 `super_class`、`cell_type` 分布）。**仓库本身不包含可执行训练、推理或机器人控制入口**；面向 FAFB/FlyWire 的程序化连接组分析由外部 Python/R 生态（`fafbseg-py`、`CAVEclient`、`navis`、`coconatfly`）承担。

## 参考链接

- [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
- [FlyWire Codex](https://codex.flywire.ai/)
- [Whole-brain annotation and multi-connectome cell typing of Drosophila (Nature 2024)](https://doi.org/10.1038/s41586-024-07686-5)
