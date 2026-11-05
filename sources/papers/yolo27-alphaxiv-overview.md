# YOLO27: An Overview of Dual-Scale Architecture and Query-Based Real-Time Computer Vision

> 论文来源归档（ingest）

- **作者：** Ranjan Sapkota（康奈尔大学）
- **日期与版本：** 2026-09-17，alphaXiv 页面显示 v2；核对日期 2026-11-05。
- **论文：** <https://www.alphaxiv.org/abs/2609.yolo27-dual-scale-query-based-vision>；[PDF](https://www.alphaxiv.org/abs/2609.yolo27-dual-scale-query-based-vision.pdf)
- **类型：** alphaXiv 综述；页面给出 Figshare DOI 10.6084/m9.figshare.33916828，不应将 alphaXiv 的自定义标识视为 arXiv 编号。
- **官方预览：** <https://docs.ultralytics.com/models/yolo27/>。
- **核对范围：** 已查看 PDF 的摘要、架构、评测、复现和声明等部分；尚未完成全文逐页核对，后续需通读余下章节。
- **开放状态：** 官方说明 YOLO27 权重、配置与实现尚未发布；论文作者未独立验证预发布模型。

## 论文摘录与映射

1. n/s 使用双尺度 CNN，默认 one-to-many 与 NMS，也可选择 one-to-one 的 NMS-free 推理；m/l 采用 query decoder。→ [YOLO27 实体](../../wiki/entities/paper-yolo27-overview.md)。
2. PDF 中 640 输入检测预览为 42.3–60.4 mAP、0.62–2.32 ms，硬件为 RTX PRO 6000 / TensorRT 11 FP16，NMS-free 路径。→ 实体页评测。
3. 论文基于官方预览且没有独立验证；需控制输入、精度、硬件与推理头，不能视为机器人端到端延迟。→ 实体页局限。
4. 官方页面当前预览指标不同（640 输入 42.6–60.5 mAP、0.44–1.86 ms），且将 m/l 都描述为 UltraViT；论文则称 m 使用 CNN 骨干、l 使用 UltraViT。版本之间不应混用。→ 实体页差异说明。
