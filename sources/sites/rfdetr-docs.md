# RF-DETR 官方文档站

> 来源归档

- **标题：** RF-DETR Documentation
- **类型：** site / docs / computer-vision
- **链接：** <https://rfdetr.roboflow.com/latest/>
- **维护方：** Roboflow
- **关联仓库：** [rf-detr](../repos/rf_detr.md)
- **关联论文：** [arXiv:2511.09554](../papers/rf_detr_arxiv_2511_09554.md)
- **入库日期：** 2026-06-22
- **一句话说明：** RF-DETR 的 **安装、Quickstart、微调、ONNX/TensorRT 导出、Jetson 部署与 benchmark 表** 的官方入口；与 GitHub README 互补，教程粒度更细。

---

## 站点结构（2026-06）

| 章节 | 内容 |
|------|------|
| Install | `pip install rfdetr` / `uv add rfdetr`；源码安装与 dev 环境 |
| Quickstart | 检测 / 分割 / 关键点 preview 加载与推理 |
| Train | COCO JSON、YOLO 格式、`model.train(...)` 参数 |
| Export | ONNX、TensorRT 转换路径 |
| Benchmarks | N/S/M/L/XL/2XL 检测、Seg、Keypoint 完整表格 |
| Tutorials | 自定义数据集训练（视频）、Jetson 部署、Roboflow 云工作流 |
| FAQ | 与 YOLO11 对比、VRAM、许可、checkpoint 版本 |

---

## 工程要点摘录

### 安装与依赖

- Python **≥3.10**
- 核心包：`pip install rfdetr`；XL/2XL 需 `pip install "rfdetr[plus]"`（PML 1.0）

### 微调最小示例

```python
from rfdetr import RFDETRLarge
model = RFDETRLarge()
model.train(dataset_dir='./dataset', epochs=50, batch_size=4)
```

数据集：`dataset_file: "coco"` 或 `"yolo"`；检测与分割共用标注格式，由模型类区分任务。

### 导出

```python
model.export(format="onnx")
```

ONNX → TensorRT 需在 CUDA 环境用 `trtexec` 或项目 helper。

### 官方 benchmark 摘要（T4, TensorRT FP16, batch 1）

**检测：** RF-DETR-L **56.5 AP50:95 @ 6.8 ms**；RF-DETR-2XL **60.1 AP50:95 @ 17.2 ms**。

**分割：** RF-DETR-Seg-L **47.1 AP50:95 @ 8.8 ms**。

**关键点（preview）：** **71.8 AP50:95 @ 9.7 ms**（COCO person keypoints）。

### 与 YOLO11 对比（FAQ）

RF-DETR-L：**56.5 AP50:95 @ 6.8 ms** vs YOLO11x **54.7 AP** 且延迟更高；DINOv2 骨干在 **RF100-VL** 等域偏移 benchmark 上更强。

---

## 对 wiki 的映射

- [`wiki/entities/rf-detr.md`](../../wiki/entities/rf-detr.md) — 工程安装、模型族、部署与 benchmark
- [`sources/repos/rf_detr.md`](../repos/rf_detr.md) — 代码仓库与 Stars/结构
- [`sources/papers/rf_detr_arxiv_2511_09554.md`](../papers/rf_detr_arxiv_2511_09554.md) — 方法与 NAS 机制
