# Ultralytics YOLO26 官方边缘部署、量化与蒸馏文档核验

> 官方技术文档与源码入口归档（site / YOLO26 / edge deployment / quantization）。

- **机构：** Ultralytics
- **主要文档：** <https://docs.ultralytics.com/modes/export/>
- **NMS-free 指南：** <https://docs.ultralytics.com/guides/end2end-detection/>
- **TensorRT 指南：** <https://docs.ultralytics.com/integrations/tensorrt/>
- **蒸馏指南：** <https://docs.ultralytics.com/guides/knowledge-distillation/>
- **YOLO26 模型表：** <https://docs.ultralytics.com/models/yolo26/>
- **YOLO27 预览：** <https://docs.ultralytics.com/models/yolo27/>
- **官方代码仓库：** <https://github.com/ultralytics/ultralytics>
- **源码入口：** [`engine/model.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/model.py)、[`engine/exporter.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py)、[`nn/distill_model.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py)
- **核查日期：** 2026-11-22（官方文档滚动更新，API 用法应锁定实际安装版本）
- **核对范围：** 阅读官方 Export 文档的格式/参数/量化兼容表、PTQ 与 QAT 案例、NMS-free 全篇推理头说明、TensorRT 的导出/校准/精度限制、Knowledge Distillation 全篇工作机制与任务限制、YOLO26 检测评测表及 YOLO27 官方预览的发布状态；核对 GitHub 当前 main 的 `Model.export`、`Exporter.__call__`、`Exporter.export_onnx`、`Exporter.export_engine`、`DistillationModel` 初始化和 `forward/loss` 关键源码。并非已执行本地硬件基准。
- **通读：** 已完成

## 文档结论（截至核查日）

1. **YOLO26 DFL-free；NMS-free 是可选推理头。** `predict/val` 默认 one-to-many + NMS；`nms=False` 切 one-to-one，检测导出 one-to-one 输出形状为 `(N,300,6)`，one-to-many 原始输出为 `(N,nc+4,8400)`（640 输入）。部分格式或量化配置会回退到 one-to-many。
2. **导出参数：** `model.export(format="onnx", imgsz=640, nms=False)`；`format="engine", quantize=16` 表示 FP16，`format="engine", quantize=8, data="my_data.yaml"` 表示 PTQ INT8。`half=True`、`int8=True` 仍兼容但已弃用。`tflite` 别名迁移至 `litert`。
3. **格式精度矩阵：** ONNX、TensorRT、OpenVINO 支持 FP32/FP16/INT8（但导出过程不等同于每个后端都有相同速度）；NCNN 支持 FP32/FP16，不支持官方 INT8 导出；LiteRT 的 FP16 和 INT8 行为不能等同于 TensorRT。
4. **PTQ 校准：** 代表性样本应来自目标业务分布。TensorRT 官方建议至少约 500 张代表性图片；`coco8.yaml` 的验证集只有 4 张，不适合当正式校准集。QAT 则可在训练端 `train(quantize=8)` 微调，并使用独立验证确认收益。
5. **官方 TensorRT 实测：** 相比 FP32，INT8 在不同模型上出现 1.7–2.9 mAP 或更大差异（以该官方文档所列版本/条件为准）；部分模型 INT8 引擎并不小于 FP32。性能不从 FP32/INT8 位宽直接线性推算。
6. **蒸馏：** `model.train(data=..., epochs=..., distill_model="yolo26s.pt")`。`DistillationModel` 冻结教师模型，用 hook 捕获来自检测头输入的 Neck 特征，学生特征经过双 1×1 Conv + ReLU 投影，经教师置信度加权的 L2 特征损失与检测任务损失合并；`dis` 控制损失系数。仅导出学生参数。推荐同家族，同任务；分类/语义/深度/RT-DETR 不支持，段分割/姿态/OBB 技术可用但精度收益尚未验证。
7. **YOLO27：** 官方预览尚未给出权重及明确日期，官网例子标记为 release 之后使用，不可直接当作可安装基线。

## 源码运行路径（已核对的入口）

- 导出：`ultralytics.engine.model.Model.export` → `Exporter(overrides=...)(model=self.model)` → `Exporter.__call__` 参数/设备/格式校验 → `export_onnx`（使用 ONNX opset、torch2onnx、必要时 simplify/INT8）→ 对 TensorRT 调用 `export_engine` → `onnx2engine`（INT8 PTQ 时传入校准 dataloader）→ `*.engine`。
- 训练蒸馏：`model.train(distill_model=...)` 构造 `DistillationModel`；`loss` 每 batch 对 frozen teacher 做 no-grad 前向、对 student 做前向，结合检测损失与蒸馏损失；`fuse` 返回学生网络用于部署。
- **代码开放情况：** 训练/推理/导出主代码已开源；底层 TensorRT、NCNN、OpenVINO 另由相应工具链提供，未对所有硬件跑通。

## 对 wiki 的映射

- [YOLO26 边缘部署与模型压缩方法](../../wiki/methods/yolo26-edge-model-optimization.md)
- [Ultralytics 工具链实体](../../wiki/entities/ultralytics.md)
- [Ultralytics 源码仓库归档](../repos/ultralytics.md)