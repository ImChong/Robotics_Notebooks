# YOLO v1 论文通读（感知栈选型视角）

> 来源归档（原文通读 · 支撑 `wiki/queries/robot-perception-stack-selection-loop.md`）

- **标题：** You Only Look Once: Unified, Real-Time Object Detection
- **作者：** Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
- **类型：** paper / computer-vision / object-detection / real-time
- **原文：** <https://arxiv.org/abs/1506.02640>（2026-11 通过 arXiv PDF v5 获取全文）
- **一句话说明：** 把检测重构为单网络回归（S×S 网格、每格 B 框 + C 类条件概率），以 45 FPS 开启端到端实时检测；感知栈 ② 层「速度换定位精度」取舍的源头。
- **核对范围：** 通读摘要、§1 引言、§2 统一检测（含网络、训练损失、推理、局限）、§3 与 DPM/R-CNN/MultiBox/OverFeat/MultiGrasp 的对比、§4 实验（4.1–4.5）、§5 实时应用、§6 结论；参考文献表仅浏览。论文无附录。SAM / SAM 2 / RF-DETR 等 wiki 页其他来源未在本档通读。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **统一回归（§1–2）：** 单个 CNN 一次前向同时预测所有框与类别，全图训练，可端到端优化；基础模型 45 FPS（Titan X，无批处理），Fast YOLO 155 FPS，延迟 < 25 ms。
- **输出编码（§2）：** 每格预测 B 个框（x, y, w, h, 置信度），置信度定义为 Pr(Object)·IOU；每格一组条件类概率；VOC 取 S=7、B=2、C=20，输出 7×7×30；测试时类别概率与框置信度相乘得类别置信度。
- **网络（§2.1–2.2）：** 24 层卷积 + 2 全连接，1×1 降维 + 3×3 卷积；ImageNet 预训练前 20 层（224×224，top-5 88%），检测时升到 448×448；Fast YOLO 为 9 层卷积。
- **损失（§2.2，式 3）：** 平方和误差，λcoord=5、λnoobj=0.5；预测宽高的平方根以部分缓解大小框误差等权；按最高 IOU 指派「负责」的预测器；约 135 epoch，dropout 0.5 与 HSV/缩放/平移增强。
- **推理（§2.3）：** 每图 98 个框；NMS 对 YOLO 不像对 R-CNN/DPM 那样关键，但仍增加 23% mAP。
- **局限（§2.4）：** 每格仅两框一类，成群小目标（如鸟群）差；对新长宽比泛化弱；特征较粗；小框定位误差代价大；主要误差来自定位。
- **速度–精度（§4.1，表 1）：** Fast YOLO 52.7 mAP/155 FPS；YOLO 63.4 mAP/45 FPS；Faster R-CNN VGG-16 73.2 mAP/7 FPS；Faster R-CNN ZF 62.1/18 FPS；Fast R-CNN 70.0/0.5 FPS；30Hz DPM 26.1 mAP。
- **误差分析（§4.2，图 4）：** YOLO 定位错误 19.0%、背景误报 4.75%；Fast R-CNN 定位 8.6%、背景 13.6%（约为 YOLO 的 3 倍）。
- **组合（§4.3，表 2）：** 用 YOLO 过滤 Fast R-CNN 背景误报，mAP 71.8→75.0（+3.2）；与其他 Fast R-CNN 变体集成仅 +0.3–0.6，说明增益来自误差类型互补。
- **VOC 2012（§4.4）：** YOLO 57.9 mAP，小目标（bottle、sheep、tv）比 R-CNN 低 8–10 点。
- **跨域泛化（§4.5，图 5）：** 在 Picasso / People-Art 艺术画上 YOLO 退化小于 R-CNN（R-CNN 依赖为自然图像调的 Selective Search）；Picasso AP 53.3 vs R-CNN 10.4 vs DPM 37.8。
- **实时部署（§5）：** 接摄像头可保持实时，含取图与显示耗时，行为类似跟踪系统。

## 对 wiki 的映射

- [`wiki/queries/robot-perception-stack-selection-loop.md`](../../wiki/queries/robot-perception-stack-selection-loop.md)（② 层：mAP 高不是万能旗标，速度/定位精度取舍）
- [`wiki/entities/paper-yolo-unified-realtime-detection.md`](../../wiki/entities/paper-yolo-unified-realtime-detection.md)

## 参考来源

- 论文：<https://arxiv.org/abs/1506.02640>
