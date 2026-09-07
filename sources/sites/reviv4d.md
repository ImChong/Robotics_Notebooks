# reviv4d.github.io — ReViV 项目页

> 来源归档

- **标题：** ReViV: Reconstructing the Viewer and the View in 4D from Monocular Egocentric Video
- **类型：** site
- **链接：**
  - <https://reviv4d.github.io/>
  - <https://arxiv.org/abs/2607.17790>
- **代码：** <https://github.com/lvsean/reviv4d>
- **机构：** ETH Zürich · Delft University of Technology · Microsoft
- **会议：** ECCV 2026
- **入库日期：** 2026-09-07
- **一句话说明：** ReViV 官方项目页：单目 egocentric RGB → 统一重建全身/双手/注视/深度/相机；展示 MGET 架构与多 benchmark 定性/定量结果。

## 开源核查（步骤 2.5，2026-09-07）

| 组件 | 状态 |
|------|------|
| 项目页 Header | 链到 arXiv PDF；页内 **Code** 指向 GitHub |
| 论文 | arXiv:2607.17790；摘要写明 "Code and models are fully open-sourced" |
| 代码 | **已开源** — [lvsean/reviv4d](https://github.com/lvsean/reviv4d)，Apache 2.0 |
| 权重 | **已发布** — [polybox](https://polybox.ethz.ch/index.php/s/LHz64M2YnRo3CpL) 两套 checkpoint；[LICENSE_WEIGHTS](https://github.com/lvsean/reviv4d/blob/main/LICENSE_WEIGHTS) **限非商用研究** |
| 数据 | **不随仓库分发**；训练集需从各官方源获取 |
| 依赖 tokenizer | NVIDIA Cosmos（HF gated）；README 提供下载脚本 |

## 对 wiki 的映射

- [ReViV 论文实体页](../../wiki/entities/paper-reviv4d.md)
- [ReViV 论文归档](../papers/reviv4d_arxiv_2607_17790.md)
- [ReViV 官方仓库](../repos/reviv4d.md)
