# Ultralytics（YOLO 官方 Python 包）

> 来源归档

- **标题：** Ultralytics YOLO（ultralytics/ultralytics）
- **类型：** repo / computer-vision / object-detection / tooling
- **组织：** Ultralytics
- **链接：** <https://github.com/ultralytics/ultralytics>
- **文档：** <https://docs.ultralytics.com/>
- **主页 / 平台：** <https://www.ultralytics.com/> · <https://platform.ultralytics.com>
- **PyPI：** `pip install ultralytics`（入库日快照约 **8.4.x**；当前主推模型族 **YOLO26** / **YOLO11**）
- **许可：** **AGPL-3.0**（研究/开源友好）；商用另见 [Enterprise License](https://www.ultralytics.com/license)
- **入库日期：** 2026-07-26
- **一句话说明：** YOLO 系列工程主仓：统一 CLI/Python API 覆盖检测、实例/语义分割、分类、姿态、OBB、深度、跟踪与导出（ONNX/TensorRT/OpenVINO 等）。
- **沉淀到 wiki：** [`wiki/entities/ultralytics.md`](../../wiki/entities/ultralytics.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态 |
|----|------|
| 代码 | **已开源** — GitHub `ultralytics/ultralytics`（AGPL-3.0） |
| 文档 | <https://docs.ultralytics.com/>（任务 / 模式 / 模型 / 导出齐全） |
| 权重 | 首次推理自动从 Ultralytics assets release 下载（如 `yolo26n.pt`） |
| 商用 | AGPL 传染性强；闭源产品需 **Enterprise License** |

---

## 核心定位

把「YOLO 家族」做成 **一个包 + 一套语法**：`yolo [TASK] MODE ARGS` 与 `from ultralytics import YOLO`。相对 [YOLO v1 论文实体](../../wiki/entities/paper-yolo-unified-realtime-detection.md) 的历史范式，本仓是 **当前机器人机载检测的默认工程入口**；与 [RF-DETR](../../wiki/entities/rf-detr.md) 对照时：YOLO 生态更大、教程更多，但默认许可是 AGPL，且经典 YOLO 仍依赖 NMS（**YOLO26** 文档宣称端到端 **NMS-free**）。

---

## 仓库入口（README / Docs）

| 组件 | 说明 |
|------|------|
| 安装 | `pip install ultralytics`（Python≥3.8，PyTorch≥1.8）；亦支持 Conda / Docker / 源码 |
| CLI | `yolo predict model=yolo26n.pt source=...`；`train` / `val` / `export` / `track` |
| Python | `YOLO("yolo26n.pt")` → `train` / `val` / `__call__` / `export(format="onnx")` |
| 任务 | detect / segment / semantic / classify / pose / obb / depth + track 模式 |
| 模型族 | YOLO26（当前主推）、YOLO11、YOLOv8 及更早；另集成 RT-DETR、SAM 等（见 Docs Models） |
| 导出 | ONNX、TensorRT、OpenVINO、CoreML、TFLite 等（`model.export`） |
| 集成 | W&B、Comet、Roboflow、OpenVINO 等 |

### YOLO26 Detection（COCO val，README 表）

| 模型 | mAP50-95 | T4 TensorRT10 (ms) | params (M) |
|------|----------|--------------------|------------|
| YOLO26n | 40.9 | 1.7 | 2.4 |
| YOLO26s | 48.6 | 2.5 | 9.5 |
| YOLO26m | 53.1 | 4.7 | 20.4 |
| YOLO26l | 55.0 | 6.2 | 24.8 |
| YOLO26x | 57.5 | 11.8 | 55.7 |

（另有 e2e 列；速度为官方 EC2 P4d / TensorRT 协议，详见 Docs。）

---

## 与机器人栈的关系

- **机载实时检测默认起点：** [Booster RoboCup Demo](../../wiki/entities/booster-robocup-demo.md) 用 YOLOv8；[人形足球](../../wiki/tasks/humanoid-soccer.md) 寻球模块同属 YOLO 谱系。
- **框提示 → 分割：** 检测框可作 [SAM](https://docs.ultralytics.com/models/sam/) / 下游 mask 管线提示（与语义建图「检测器 + 分割」叙事一致）。
- **选型对照：** 垂直域 / 无 NMS / DINOv2 迁移看 [RF-DETR](../../wiki/entities/rf-detr.md)；开放词汇长尾另接 Grounding 类模型（见 [检测选型 Query](../../wiki/queries/object-detection-model-selection.md)）。
- **许可红线：** 闭源机器人产品内嵌 AGPL 权重/代码前必须评估 Enterprise 或换许可更宽松栈。

---

## 对 wiki 的映射

- 主实体页：**`wiki/entities/ultralytics.md`**
- 交叉：[object-detection](../../wiki/methods/object-detection.md)、[paper-yolo-unified-realtime-detection](../../wiki/entities/paper-yolo-unified-realtime-detection.md)、[rf-detr](../../wiki/entities/rf-detr.md)、[booster-robocup-demo](../../wiki/entities/booster-robocup-demo.md)、[object-detection-model-selection](../../wiki/queries/object-detection-model-selection.md)
