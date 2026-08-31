# ViTaR（视触觉残差适配基础 VLA）

> 来源归档（ingest）

- **标题：** ViTaR: Visuo-Tactile Residual Adaptation for Foundation VLA Manipulation
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.15816>
- **机构：** 北京理工大学（BIT）
- **项目页：** <https://icr-lab.github.io/ViTaR>
- **入库日期：** 2026-08-31
- **一句话说明：** 冻结 OpenVLA-OFT，用 Effect-Guided Modeling + Residual Action Modulation 注入有界视触觉残差；UniVTAC 61.3%（+30.6 pt），真机 +30.0 pt。

## 核心摘录（MVP）

### 1) 触觉作执行调制器

- **摘录要点：** 不把触觉并入动作生成输入，而在冻结 VLA 语义动作上选择并缩放 **有界残差**，避免触觉覆盖视觉先验。
- **对 wiki 的映射：**
  - [ViTaR](../../wiki/entities/paper-vitar.md) — 方法动机。

### 2) 两阶段设计

- **摘录要点：** Effect-Guided Modeling 判断局部修正是否合理；Residual Action Modulation 按实时视触觉连续调节增益。
- **对 wiki 的映射：**
  - [ViTaR](../../wiki/entities/paper-vitar.md) — 方法栈。

### 3) 开源状态（截至 2026-08-31）

- **摘录要点：** **待发布**。项目页标注 **Code Coming soon**；截至入库日无官方可运行仓库链。
- **对 wiki 的映射：**
  - [ViTaR 项目页](../sites/icr-lab-vitar.md)

## 当前提炼状态

- [x] arXiv 摘要与项目页已对齐摘录
- [x] 步骤 2.5：代码未发布
- [x] wiki 映射：`wiki/entities/paper-vitar.md` 新建
