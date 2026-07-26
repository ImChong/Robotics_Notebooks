# FindAnything

> 来源归档

- **标题：** FindAnything: Open-Vocabulary and Object-Centric Mapping for Robot Exploration in Any Environment
- **类型：** site（项目页）
- **论文：** https://arxiv.org/abs/2504.08603
- **项目页：** https://ethz-mrl.github.io/findanything/
- **机构：** 慕尼黑工业大学（Technical University of Munich）& 苏黎世联邦理工学院（ETH Zurich）
- **入库日期：** 2026-07-26
- **一句话说明：** 将 SAM/eSAM 二维区域与视觉语言特征聚合到对象级三维体素子地图；强调内存可扩展与机载实时；已演示 Jetson Orin NX 级部署与语言引导探索。
- **沉淀到 wiki：** [findanything](../../wiki/entities/findanything.md)（site/项目实体）；[GO2 三维语义建图 Query](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)

---

## 项目页要点（步骤 2.5 核查）

- **表示：** 几何体素子地图 + 对象级开放词汇特征聚合（非逐体素堆满 CLIP）。
- **部署叙事：** 可在 **Nvidia Jetson Orin NX** 等资源受限平台机载运行（项目页 MAV 演示）。
- **下游：** 自然语言查询调制探索（如 Search and Rescue 仿真）。

## 开源状态

- **宣称将开源 / 待并入宿主仓**：项目页写明 *「The code will be hosted in the OKVIS2-X repository, as an additional feature」*。
- **截至 2026-07-26：** 以项目页 + arXiv 为准；勿假设已有独立可 pip 安装的 FindAnything 单仓。跟进时应打开 OKVIS2-X 宿主仓核对是否已合入。

## 对 wiki 的映射

- 实体（项目页占位）：[findanything](../../wiki/entities/findanything.md) — 开源仓落地后再补 `sources/repos/` 与源码分析
- Query：[go2-3d-semantic-mapping-sam-pipeline](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)
