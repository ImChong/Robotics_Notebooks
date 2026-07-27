# DiffGI Project Page

> 来源归档

- **标题：** DiffGI — Differentiable Geometry Images for High-Fidelity Thin-Shell 3D Generation
- **类型：** site / project page
- **URL：** <https://ejshim.github.io/diffgi/>
- **论文：** <https://arxiv.org/abs/2607.13365>
- **代码仓（占位）：** <https://github.com/EJShim/diffgi>
- **机构：** CLO Virtual Fashion Inc.
- **入库日期：** 2026-07-27
- **一句话说明：** 官方项目页：薄壳/非流形 DiffGI 管线、TSDF vs occupancy 对比、image-to-3D / VAE / 标签条件结果表与 BibTeX；Code 按钮标注 soon。

## 开源状态（项目页核查，2026-07-27）

| 项 | 状态 |
|----|------|
| Paper | arXiv **2607.13365**（pill 可点） |
| Code | **Code (soon)** — `disabled`，无 GitHub 可点链接 |
| 关联公开仓 | `EJShim/diffgi` 存在，但仅托管本项目页静态资源（`docs/`） |
| 结论 | **宣称将开源 / 待发布** — 入库日无可运行训练/推理代码或权重 |

## 页面结构（策展）

| 区块 | 内容要点 |
|------|----------|
| Hero | 单图/图案 → 薄壳 3D；秒级；可下至 CPU |
| Main Contributions | 薄壳非流形；连续 TSDF + DMS 端到端；\(32\times32\) 潜扩散轻量 |
| Pipeline | Mesh→TSDF GI→VAE→DMS→表面损失反传；DiT 条件生成 |
| Representation Effect | DiffGI vs 二值 GI；子像素边界 |
| Results | Image-to-3D 表；VAE 重建表；标签生成；插值；效率表 |
| BibTeX | `shim2026diffgi`（ECCV 2026） |

## 对 wiki 的映射

- 论文：[`sources/papers/diffgi_arxiv_2607_13365.md`](../papers/diffgi_arxiv_2607_13365.md)
- 代码占位：[`sources/repos/diffgi.md`](../repos/diffgi.md)
- 沉淀 **[`wiki/entities/paper-diffgi.md`](../../wiki/entities/paper-diffgi.md)**
