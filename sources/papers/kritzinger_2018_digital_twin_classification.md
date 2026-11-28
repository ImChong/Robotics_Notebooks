# Kritzinger et al. (2018) — Digital Twin 三级集成度分类文献记录

> 来源归档（ingest · 仅公开元数据核对，原文未取得）

- **标题：** Digital Twin in manufacturing: A categorical literature review and classification
- **作者：** Werner Kritzinger、Matthias Karner、Georg Traar、Jan Henjes、Wilfried Sihn
- **发表：** *IFAC-PapersOnLine*, Vol. 51, Issue 11, pp. 1016–1022, 2018（16th IFAC Symposium on Information Control Problems in Manufacturing, INCOM 2018）
- **DOI：** [10.1016/j.ifacol.2018.08.474](https://doi.org/10.1016/j.ifacol.2018.08.474)
- **入库日期：** 2026-11-28
- **核对范围：** 核对 OpenAlex OA 书目元数据与公开摘要转述；尝试抓取 ScienceDirect 页面返回 HTTP 403、机构仓储链接未返回可下载 PDF，故未取得并通读论文全文。**不列入任何 wiki 页 `original:` 字段**。
- **通读：** 未完成
- **一句话说明：** 制造业数字孪生领域最常用的三分集成度框架：按物理与数字对象之间的数据流自动化程度，区分 **Digital Model（无自动交换）**、**Digital Shadow（物理→数字单向自动流）** 与 **Digital Twin（物理↔数字双向自动集成）**。

## 公开信息核对要点（依据公开书目与摘要转述，待全文复核）

| 分级 | 数据流方向（物理对象 ↔ 数字对象） | 典型工程形态 |
|---|---|---|
| **Digital Model** | 手工/离线更新（无自动数据流） | CAD 模型、离线 URDF/MJCF 仿真、静态工厂布局 |
| **Digital Shadow** | **物理 → 数字** 单向自动流动 | 真实状态遥测回放、日志复现、单向参数辨识更新 |
| **Digital Twin** | **物理 ↔ 数字** 双向自动集成 | 状态实时同步 + 仿真预测/控制指令反作用于物理系统 |

> 边界提示：该三分法精确统计（如文献占比数字）未读原文确认，本归档仅保留其被广泛引用的三类定性定义作为概念分级参考。

## 对 wiki 与路线的映射

- [`wiki/concepts/digital-twin.md`](../../wiki/concepts/digital-twin.md) — 仅列入 `sources:` 作分级坐标，不列入 `original:`
- [`roadmap/depth-robot-modeling-digital-twin.md`](../../roadmap/depth-robot-modeling-digital-twin.md) — Stage 0 与 Stage 6 演进主轴参考
