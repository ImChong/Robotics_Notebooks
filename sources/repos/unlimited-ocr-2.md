# baidu/Unlimited-OCR

> 来源归档（repo）

- **名称：** Unlimited OCR Works 2
- **类型：** repo / document-parsing / ocr
- **代码：** <https://github.com/baidu/Unlimited-OCR>
- **权重：** <https://huggingface.co/baidu/Unlimited-OCR-2>
- **技术报告：** <https://github.com/baidu/Unlimited-OCR/blob/main/Unlimited-OCR-2.pdf>
- **机构：** 百度（Baidu）
- **仓库许可证：** MIT；模型权重许可请另查模型卡
- **入库日期：** 2026-10-27

## 内容与开放状态

官方 README 记载 2026-10-27 发布第二版；SR-SWA 将长序列的文本–视觉注意力交互压缩到固定数量的工作引用，面向多页长文档解析。仓库提供推理客户端、后处理工具及定制 SGLang wheel；README 给出 Transformers 模型加载与单页、多页推理示例，并链接模型权重。完整训练代码及数据是否发布尚未核实，模型卡本次未能访问。

## 运行入口

- Transformers：加载 `baidu/Unlimited-OCR-2`，多页调用 `infer_multi`，单页调用 `infer`。
- SGLang：README 给出服务器启动参数、PDF 转图与客户端示例；`infer.py` 发送基准文档，`utils/process_unlimitedOCR2.py` 转成评测 Markdown。
- 作者列出的测试环境为 Python 3.10 和 NVIDIA A800 80GB；这不是最低硬件要求。

## 对 wiki 的映射

- [Unlimited OCR Works 2](../../wiki/entities/unlimited-ocr-2.md)
