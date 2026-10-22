# MOSS-Image 官方项目页

- **类型：** 官方项目页
- **标题：** MOSS-Image — Text-to-Image Generation with Native Layout Control
- **链接：** <https://openmoss.ai/MOSS-Image/>
- **代码：** <https://github.com/OpenMOSS/MOSS-Image>
- **报告：** <https://arxiv.org/abs/2610.23374>
- **核查日期：** 2026-10-22
- **一句话说明：** OpenMOSS 的文生图模型家族，包含 Base 与 SFT 权重，以双语文字渲染、长提示词及原生布局控制为重点。

## 官方页面摘录

- **模型：** 5B DiT backbone；Base 与 SFT 共用架构和推理接口，SFT 由 Base 在策展数据上微调。
- **提示词：** 支持最多 4K tokens 的提示词；项目页演示自然语言和结构化 JSON layout 输入。
- **布局：** 可在 JSON 中描述场景元素、文字和 bounding boxes；编辑文字、元素描述和边界框以改变图像内容及构图。
- **风格：** 官方展示插画、摄影、视觉设计、动漫与写实等示例。
- **开源状态：** 项目页链接到 GitHub、Base / SFT Hugging Face 权重；模型权重注明 Apache-2.0。代码及第三方组件的许可证应分别查看。
- **API：** 项目仓库新闻提到 2026-10-20 起可通过 MOSI API Playground 或 API 使用；模型目录标注 Base 与 SFT API 模型为 Beta。
- **桌面产品关联：** MosMos（<https://mosmos.io/>）是 MOSI 生态中的独立语音输入与会议记录应用；未发现它属于 MOSS-Image 模型或推理链路的证据。

## 相关来源

- [官方仓库归档](../repos/moss-image.md)
- [技术报告摘录](../papers/moss_image_arxiv_2610_23374.md)
