# Ultralytics Docs

> 来源归档

- **标题：** Ultralytics YOLO Docs
- **类型：** site / documentation
- **URL：** <https://docs.ultralytics.com/>
- **代码：** <https://github.com/ultralytics/ultralytics>
- **平台：** <https://platform.ultralytics.com>
- **入库日期：** 2026-07-26
- **一句话说明：** Ultralytics YOLO 官方文档枢纽：任务（detect/segment/…）、模式（train/val/predict/export/track）、模型目录、导出与部署、许可说明。

## 开源状态（项目页/文档核查，2026-07-26）

| 项 | 状态 |
|----|------|
| Code | **已开源** — 文档明确指向 GitHub + `pip install ultralytics` |
| Models | YOLO26 / YOLO11 等权重可自动下载；生产推荐 YOLO26 与 YOLO11 |
| License | AGPL-3.0 + Enterprise 双轨（文档 Licensing 节） |
| 额外产品 | Ultralytics Platform（标注/云训练/部署）；Inference（Rust，无 Python 运行时） |

## 文档结构（策展）

- **Task / Mode / Args** — `yolo [TASK] MODE ARGS` 统一语法
- **Models** — YOLO26、YOLO11、SAM、RT-DETR 等可挂到 `model=`
- **Guides / Integrations** — Jetson、导出、超参、W&B/Roboflow 等
- **Licensing** — AGPL vs Enterprise

## 对 wiki 的映射

- 仓库归档：[`sources/repos/ultralytics.md`](../repos/ultralytics.md)
- 沉淀 **[`wiki/entities/ultralytics.md`](../../wiki/entities/ultralytics.md)**
