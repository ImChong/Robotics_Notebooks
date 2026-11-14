# Ultralytics Signature Detection Dataset

> 官方数据集来源归档（dataset）

- **名称：** Signature Detection Dataset
- **机构：** Ultralytics
- **官方文档：** <https://docs.ultralytics.com/datasets/detect/signature>
- **平台入口：** <https://platform.ultralytics.com/ultralytics/datasets/signature>
- **入库日期：** 2026-11-14
- **核对范围：** 阅读官方签名数据集文档的规模、划分、类别、训练用途和示例；平台页面返回 403，未核验登录后的 UI 和数据下载。
- **通读：** 已完成

## 数据与用途

- **178 张文档图像**，`train=143`、`val=35`，只标注一类 `signature`，无单独 test 划分。
- 任务是 **签名区域检测/定位**，不是笔迹文本识别，也不能仅凭检测判定签名真伪。
- 官方示例可由 `signature.yaml` 触发数据下载并训练 YOLO 检测器，用于文档表单前处理。
- 部署场景应先定位签名/表格/隐私字段，再将需要识别的子区域送给 OCR 或其他解析器。

## 开放状态

官方文档列出可自动下载的数据集和训练入口；平台页本次不可直接浏览，未独立下载验证。数据集许可和图像来源使用前应另核。

## 对 wiki 的映射

- [YOLO26 OCR 方法及文档分区](../../wiki/methods/yolo26-ocr-document-parsing.md)
- [Ultralytics 主实体](../../wiki/entities/ultralytics.md)
