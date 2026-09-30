# Ultralytics/YOLO26（Hugging Face）

> 来源归档（ingest）

- **标题：** Ultralytics YOLO26 — pretrained weights collection
- **类型：** model / huggingface
- **链接：** <https://huggingface.co/Ultralytics/YOLO26>
- **代码：** <https://github.com/ultralytics/ultralytics>
- **论文：** <https://arxiv.org/abs/2606.03748>
- **文档：** <https://docs.ultralytics.com/models/yolo26/>
- **入库日期：** 2026-09-30
- **一句话说明：** Ultralytics 官方 HF 组织下的 **YOLO26 全任务权重入口**（detect/seg/pose/obb/cls/depth/sem）；与 PyPI `ultralytics` 包联动，首次使用自动从 release 下载 `.pt`。

## 开源状态

| 项 | 状态 |
|----|------|
| 权重 | **已发布** — 各尺寸 n/s/m/l/x 及 `-seg` / `-pose` / `-obb` 等变体 |
| 许可 | 与主仓一致 **AGPL-3.0**（商用见 Enterprise） |
| 用法 | `pip install ultralytics` → `YOLO("yolo26n.pt")` 或 `yolo predict model=yolo26n.pt` |

## 模型卡要点（Detection · COCO）

| 模型 | mAP50-95 | mAP50-95 (e2e) | T4 TRT10 (ms) | params (M) |
|------|----------|----------------|---------------|------------|
| YOLO26n | 40.9 | 40.1 | 1.7 | 2.4 |
| YOLO26s | 48.6 | 47.8 | 2.5 | 9.5 |
| YOLO26m | 53.1 | 52.5 | 4.7 | 20.4 |
| YOLO26l | 55.0 | 54.4 | 6.2 | 24.8 |
| YOLO26x | 57.5 | 56.9 | 11.8 | 55.7 |

（分割/姿态/OBB/深度/语义/分类表见 HF 页；与 GitHub README 对齐。）

## 对 wiki 的映射

- 论文实体：[`wiki/entities/paper-yolo26-unified-realtime-e2e-vision.md`](../../wiki/entities/paper-yolo26-unified-realtime-e2e-vision.md)
- 工程实体：[`wiki/entities/ultralytics.md`](../../wiki/entities/ultralytics.md)
- 论文归档：[`sources/papers/yolo26_arxiv_2606_03748.md`](../papers/yolo26_arxiv_2606_03748.md)
