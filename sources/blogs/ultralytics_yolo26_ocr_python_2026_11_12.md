# OCR in Python with Ultralytics YOLO26: read documents, photos and PDFs at any angle

> 官方博客来源归档（blog / OCR / document analysis）

- **标题：** OCR in Python with Ultralytics YOLO26: read documents, photos and PDFs at any angle
- **作者：** Jing Qiu（Ultralytics）
- **发布日期：** 2026-11-12
- **主文：** <https://www.ultralytics.com/blog/yolo-ocr-python>
- **官方实现文档：** <https://docs.ultralytics.com/guides/ocr>
- **代码：** <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/ocr.py>
- **在线演示：** <https://www.ultralytics.com/use-cases/document-processing>
- **入库日期：** 2026-11-14
- **核对范围：** 通读官方英文博客全文（机制、示例、基准、语言覆盖、适用边界），并核对官方 OCR Guide 与 GitHub `ultralytics/solutions/ocr.py` 的公开实现；未实际执行基准和在线推理。
- **通读：** 已完成

## 原文要点

1. `solutions.OCR()` 默认以 `yolo26x-text.pt` 的 OBB 检测结果为输入，由 `PP-OCRv6_medium_rec.onnx` 逐词识别，再按文本行组织，返回原图坐标多边形、文本、行号与检测/识别两个置信度。
2. 整页旋转：基于长宽比至少 2:1 的文本框估算方向；对两种相反方向各抽取最宽的 8 个词识别，以置信度选择正向，再聚合文本行。
3. **官方自测而非独立评测：** RTX PRO 6000、`imgsz=1280`，80 张 HierText 验证图片词级 F1 **0.722**，6 页 PDF **0.931**；密集场景某样本 251 词的 `read_words` 耗时 ONNX Runtime CUDA **249ms**、缓存 FP16 TensorRT **82ms**，不是端到端模型时延。
4. `ultralytics>=8.4.211`；默认识别器需 Python 3.11+ / ONNX Runtime 1.26+。可选 `recognizer="tensorrt"`，也能传自定义 `(text, confidence)` 识别回调。
5. 默认 PP-OCRv6 支持中、日及多种拉丁文字；韩文、西里尔、阿拉伯等需替换对应识别器；整页以主导方向旋转，混排方向及跨页复杂结构需独立验证。
6. PDF 先逐页渲染为图片，本地路径不上传文档；官网在线演示会把页面上传到远端，含敏感信息时注意脱敏。

## 用户提供的短链归档（未直接解析，不将猜测当成重定向结果）

- 未标注链接 1：<https://bit.ly/4sTyjuJ>（目标未核实）
- 未标注链接 2：<https://bit.ly/4k9X2sw>（目标未核实）
- HierText：<https://bit.ly/4iFK44J> → 已另核官方项目 <https://github.com/google-research-datasets/hiertext>，**未声称短链等价**
- 开源 OCR 模型概览：<https://bit.ly/4zCLnqp> → 官方相关文章 <https://www.ultralytics.com/blog/popular-open-source-ocr-models-and-how-they-work>
- 签名数据集：<https://bit.ly/3URLyQ1> → 官方数据集说明 <https://docs.ultralytics.com/datasets/detect/signature>
- 智能文档分析指南：<https://bit.ly/4iEI1xI> → 官方文章 <https://www.ultralytics.com/blog/using-ultralytics-yolo11-for-smart-document-analysis>
- Ultralytics Platform：<https://bit.ly/48cU27F> → 官方平台 <https://platform.ultralytics.com/>
- 在线 OCR 演示：<https://bit.ly/4znz2pR> → 官方演示 <https://www.ultralytics.com/use-cases/document-processing>

以上箭头表示通过**内容与官方网站**找到的对应参考资料，并非短链跳转验证结果。Platform 页面当前访问受限，平台功能仅据官方博客/文档记载。

## 相关一手资料

- [官方 OCR 文档归档](../sites/ultralytics_yolo26_ocr_guide.md)
- [HierText 官方数据集归档](../datasets/hiertext.md)
- [Signature 官方数据集归档](../datasets/ultralytics-signature.md)
- [Ultralytics 源码仓库归档](../repos/ultralytics.md)

## 对 wiki 的映射

- [YOLO26 + PP-OCRv6 OCR 方法](../../wiki/methods/yolo26-ocr-document-parsing.md)
- [HierText 数据集](../../wiki/entities/hiertext.md)
