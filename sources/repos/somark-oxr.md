# SoMarkAI/OXR

> 来源归档（repo）

- **名称：** OXR（Optical Everything Recognition）
- **类型：** repo / document-parsing / OCR
- **代码：** <https://github.com/SoMarkAI/OXR>
- **权重：** <https://huggingface.co/SoMarkAI/OXR-1.0>
- **演示：** <https://free-oxr.somark.cn/>
- **API：** <https://free-oxr.somark.cn/api/v1>
- **官网：** <https://www.somark.cn>
- **机构：** SoMark
- **入库日期：** 2026-10-27

## 内容与开放状态

官方仓库提供文档解析推理与部署代码，模型权重可公开获取；训练代码和数据是否完整公开尚未核实。OXR 采用独立可替换的版面分析模块，识别与阅读顺序由 OXR 模型处理。官方称模型为 1.3B 参数，后训练使用递归自改进（RSI）。技术报告尚待发布。

仓库采用基于 Apache 2.0 并附加条款的 OXR Open Source License，使用前应阅读具体许可；模型权重许可亦应查看模型卡。

## 运行入口

README 提供本地部署、模型下载和服务启动步骤，也提供免费 API。官方建议本地低并发测试使用支持 BF16 的 NVIDIA GPU 和 24 GB 显存；免费服务不保证稳定性，不建议用于生产。

## 对 wiki 的映射

- [OXR](../../wiki/entities/somark-oxr.md)
