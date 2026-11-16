# Ultralytics Scene Text OCR using YOLO26

> 官方技术文档归档（site）

- **标题：** Scene Text OCR using Ultralytics YOLO26
- **文档：** <https://docs.ultralytics.com/guides/ocr>
- **源码：** <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/ocr.py>
- **官网演示：** <https://www.ultralytics.com/use-cases/document-processing>
- **开源项目：** <https://github.com/ultralytics/ultralytics>
- **入库日期：** 2026-11-14
- **核对范围：** 阅读官方 OCR Guide 全部章节，核对官方源码 `ultralytics/solutions/ocr.py` 中 `OCR.__init__`、`OCR.read_words`、`OCR.lines`、`OCR.process`、`TextRecognizer.__call__`；没有本地安装、运行或重新评测。
- **通读：** 已完成

## 项目/开放核查（2026-11-14）

| 对象 | 状态及依据 |
|------|-------------|
| 框架 / 识别编排代码 | **已开源**：`ultralytics/ultralytics` 中 `ultralytics/solutions/ocr.py`，仓库主许可证 AGPL-3.0 |
| 检测模型 | 官方文档列出 5 种 YOLO26-Text checkpoint（n/s/m/l/x）；首次 OCR 可自动下载 |
| 识别权重 | PP-OCRv6 medium ONNX 模型由官方 assets 自动下载，文档称权重 Apache-2.0 |
| 运行环境 | `ultralytics>=8.4.211`；默认识别器 Python 3.11+、ONNX Runtime 1.26+；GPU 可选 TensorRT |
| 自定义模型 | 检测器可换、识别器可换；整条 OCR pipeline **不能**只导出检测器即得 |
| 平台和在线演示 | 官方网页提供体验；在线上传会发送页面用于远端推理，平台登录页未能直接读取 |

## 源码入口和 I/O

- `from ultralytics import solutions; ocr = solutions.OCR(); result = ocr(image)`
- `OCR.process()`：调用模型检测，兼容 bbox、OBB、多边形检测结果；调用 `read_words()` 做纠正方向、裁切、识别、排序。
- `TextRecognizer.__call__()`：48 像素高度缩放、按宽度排序、batch=8、ONNX Runtime 或 TensorRT，CTC 解码文字与置信度。
- 返回 `result.texts`：`text`、`confidence`、`detection_confidence`、`line`、`polygon` 等；`plot_im` 返回叠加显示图。
- PDF 示例先用 PyMuPDF 把每页渲染成图片；`solutions.OCR` 本身不是 PDF 文件解析器。

## 来源与映射

- [官方技术文章](../blogs/ultralytics_yolo26_ocr_python_2026_11_12.md)
- [既有 Ultralytics 仓库归档](../repos/ultralytics.md)
- [YOLO26 OCR 方法页](../../wiki/methods/yolo26-ocr-document-parsing.md)
