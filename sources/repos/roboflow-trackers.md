# roboflow/trackers

- **类型：** repo / multi-object-tracking / computer-vision
- **官方仓库：** <https://github.com/roboflow/trackers>
- **官方文档：** <https://trackers.roboflow.com/latest/>
- **许可证：** Apache-2.0（以仓库 LICENSE 为准）
- **核查日期：** 2026-10-19
- **沉淀到 wiki：** [Roboflow Trackers 实体页](../../wiki/entities/roboflow-trackers.md)

## 仓库要点

Roboflow Trackers 是 Python 多目标跟踪库，提供 SORT、ByteTrack、OC-SORT、BoT-SORT、C-BIoU 与 McByte 的模块化实现。它接收检测器产生的边界框，可与 YOLO、DETR、RT-DETR 等任意检测模型组合；Python API 以统一的 `update(detections, frame=None)` 调用切换跟踪算法，也提供 CLI 跟踪视频、摄像头流或图像目录。

官方文档列出 MOT17、SportsMOT、SoccerNet 与 DanceTrack 的默认参数基准，并提供 HOTA、MOTA、IDF1 等评测入口。BoT-SORT 与 McByte 支持相机运动补偿；McByte 需要额外的可选依赖。

## 复现入口

| 用途 | 入口 |
|------|------|
| 安装 | `pip install trackers` |
| Python API | `from trackers import ByteTrackTracker`，再调用 `tracker.update(detections)` |
| CLI | `trackers track --source video.mp4 --tracker bytetrack ...` |
| 算法说明 | [官方算法与基准](https://trackers.roboflow.com/latest/) |
| 参数搜索 | `trackers[tune]` 可选依赖 |

## 开源与运行边界

- **已开源：** 官方 GitHub 仓库包含实现，仓库声明 Apache-2.0。
- **Python 要求：** 官方 README 标明 Python ≥ 3.10。
- **检测器解耦：** 基础库不限定检测推理框架；Python 用法通过 `supervision.Detections` 传入检测结果。
- **可选依赖：** McByte 的 SAM/Cutie 等依赖不随基础安装默认启用；安装和权重要求按该算法文档分别核查。
- **评测边界：** 官方表格报告四个数据集的 HOTA，表中使用默认参数；评测结果不等同于目标摄像头上的端到端速度或精度保证。

## 关键入口

- [GitHub 仓库与 README](https://github.com/roboflow/trackers)
- [官方文档：Quickstart 与算法对比](https://trackers.roboflow.com/latest/)
- [安装指南](https://trackers.roboflow.com/latest/learn/install/)
- [跟踪指南](https://trackers.roboflow.com/latest/learn/track/)
- [Apache-2.0 LICENSE](https://github.com/roboflow/trackers/blob/develop/LICENSE)

## 对 wiki 的映射

- [Roboflow Trackers 实体页](../../wiki/entities/roboflow-trackers.md) — 通用多目标跟踪库与工程选型
- [目标检测方法页](../../wiki/methods/object-detection.md) — tracking-by-detection 的检测前置
