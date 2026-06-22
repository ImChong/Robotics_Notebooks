# rf-detr

> 来源归档

- **标题：** RF-DETR（Roboflow Detection Transformer）
- **类型：** repo / computer-vision / object-detection / instance-segmentation
- **来源：** Roboflow + CMU（Deva Ramanan、Neehar Peri 等）
- **链接：** <https://github.com/roboflow/rf-detr>
- **Stars / Forks：** ~8k+ / —（2026-06）
- **PyPI：** `pip install rfdetr`（Python ≥3.10）
- **文档：** <https://rfdetr.roboflow.com/latest/>
- **论文：** [arXiv:2511.09554](../papers/rf_detr_arxiv_2511_09554.md)
- **入库日期：** 2026-06-22
- **一句话说明：** Roboflow 开源的 **实时 DETR 族**：DINOv2 骨干 + weight-sharing NAS，支持 **检测 / 实例分割 / 关键点（preview）** 统一 API，面向 **COCO 与自定义数据集微调**，可导出 **ONNX / TensorRT / TFLite**。
- **沉淀到 wiki：** 是 → [`wiki/entities/rf-detr.md`](../../wiki/entities/rf-detr.md)

---

## 核心定位

**RF-DETR** 是 Roboflow 维护的 **closed-vocabulary specialist 检测 Transformer**，在 ICLR 2026 论文中提出。与 YOLO 系不同，它 **无需 NMS**，与 GroundingDINO 等 VLM 检测器不同，它 **无文本编码器、推理更快**，适合 **固定类别、低延迟、需 fine-tune 到垂直域** 的机器人/工业视觉栈。

典型工作流：`pip install rfdetr` → 加载 `RFDETRLarge()` 等预训练权重 → `model.train(dataset_dir=..., epochs=50)`（COCO JSON 或 YOLO 格式）→ `model.export(format="onnx")` → TensorRT / Jetson / Roboflow Inference 部署。

---

## 模型族与许可

| 规模 | 检测类名示例 | COCO AP50:95 | T4 延迟 (ms) | 许可 |
|------|-------------|--------------|--------------|------|
| Nano | `RFDETRNano` | 48.4 | 2.3 | Apache 2.0 |
| Small | `RFDETRSmall` | 53.0 | 3.5 | Apache 2.0 |
| Medium | `RFDETRMedium` | 54.7 | 4.4 | Apache 2.0 |
| Large | `RFDETRLarge` | 56.5 | 6.8 | Apache 2.0 |
| XL / 2XL | `RFDETRXLarge` 等 | 58.6 / **60.1** | 11.5 / 17.2 | PML 1.0（`rfdetr[plus]`） |

分割：`RFDETRSegNano` … `RFDETRSeg2XL`；关键点 preview：`RFDETRKeypointPreview`（COCO 17 keypoints）。

**注意：** 当前 `RFDETRLarge` 默认 checkpoint 为 `rf-detr-large-2026.pth`；旧 `rf-detr-large.pth` 为 legacy。

---

## 仓库能力摘要

1. **训练：** COCO JSON / YOLO 格式；Roboflow 数据集可直接导出；≥8 GB VRAM 推荐（N/S 可 6 GB + 小 batch）。
2. **推理：** PyTorch 原生；CPU 评估支持；TensorRT FP16 为论文基准配置。
3. **导出：** `model.export(format="onnx")` → ONNX Runtime / OpenCV DNN；再转 TensorRT（`trtexec` 等）。
4. **部署教程：** NVIDIA Jetson（Roboflow Inference）、Roboflow 云训练 + 边缘部署文档。
5. **基准工具：** 独立 latency benchmarking 脚本（200 ms buffer 协议）。

---

## 与机器人栈的关系

- **机载实时感知：** Nano/Small 在 T4/Jetson 级 GPU 上 **2–4 ms** 量级，可作为 **球/人/障碍检测** 的 YOLO 替代，且 **无 NMS 后处理**。
- **垂直域 fine-tune：** RF100-VL 表明对 **与 COCO 分布差异大** 的数据集，RF-DETR 比 YOLOv8/v11 更稳；适合 **工厂缺陷、专用部件、仿真–真机 gap** 场景。
- **分割级联：** 同一 backbone 的 Seg 变体可输出 **实例 mask**，减少「检测 + 独立分割模型」双栈维护。
- **与开放词汇组合：** 固定高频类用 RF-DETR specialist，长尾/语言指令类仍可用 GroundingDINO 等 VLM 兜底（见 [目标检测模型选型 Query](../../wiki/queries/object-detection-model-selection.md)）。

---

## 对 wiki 的映射

- 主实体：[`wiki/entities/rf-detr.md`](../../wiki/entities/rf-detr.md)
- 方法页补充：[`wiki/methods/object-detection.md`](../../wiki/methods/object-detection.md)（端到端 DETR / 无 NMS 路线）
- 选型 Query：[`wiki/queries/object-detection-model-selection.md`](../../wiki/queries/object-detection-model-selection.md)
