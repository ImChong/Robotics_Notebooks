# OpenMOSS/MOSS-Image

> 来源归档（repo + weights + API）

- **标题：** MOSS-Image
- **类型：** 文生图模型 / 推理代码 / 权重 / API
- **组织：** OpenMOSS / MOSI.AI
- **代码：** <https://github.com/OpenMOSS/MOSS-Image>
- **项目页：** <https://openmoss.ai/MOSS-Image/>
- **技术报告：** [arXiv:2610.23374](https://arxiv.org/abs/2610.23374)
- **Base 权重：** <https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base>
- **SFT 权重：** <https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT>
- **在线 API：** <https://platform.mosi.cn/>
- **许可证：** 仓库与模型卡标注 Apache-2.0；第三方依赖保留各自许可证
- **首次入库：** 2026-10-22
- **一句话说明：** 5B 单流 DiT 文生图模型，重点支持中英文文本渲染、长提示词及结构化 JSON 布局控制。

## 公开资源与开源核查（2026-10-22）

| 项 | 结论 |
|----|------|
| 代码 | 官方 GitHub 含 Python / Diffusers 推理入口、Gradio 界面、ComfyUI 集成与用户指南 |
| Base 权重 | Hugging Face 提供 MOSS-Image-Base |
| SFT 权重 | Hugging Face 提供基于 Base 微调的 MOSS-Image-SFT |
| API | MOSI API 文档列出 `moss-image-base-1.0`、`moss-image-sft-1.0` 两个 Beta 文生图模型，走 `POST /v1/images/generations` |
| 论文 | arXiv:2610.23374；GitHub 仓库内另提供技术报告 PDF |
| 判定 | **已开放推理代码与两套模型权重**；不据模型开放推断训练数据、完整 recipe 或评测框架也已公开 |

## 模型与接口摘要

- **Base：** 基础文生图权重；官方指南说明未对推理参数做后训练或蒸馏，便于社区继续微调，但需要按指南选择合适的采样设置。
- **SFT：** 从 Base 派生的监督微调版本；模型卡展示其在示例中呈现出更清晰的文字分组与布局。
- **提示输入：** 自然语言、tag 序列或 JSON layout；结构化布局可指定 text/object 项及其边界框。
- **本地推理：** Diffusers Python pipeline；官方仓库同时提供 Gradio 与 ComfyUI 入口，README 还列出 vLLM-Omni、SGLang 等生态支持。
- **布局坐标：** 示例使用 `bbox_format: "yxyx"`、`bbox_coord_space: "abs_centralized"`，坐标相对画布中心，顺序为 y-min、x-min、y-max、x-max。

## 对 wiki 的映射

- [MOSS-Image 实体页](../../wiki/entities/moss-image.md)
- [官方项目页归档](../sites/moss-image-project.md)
- [技术报告摘录](../papers/moss_image_arxiv_2610_23374.md)
- 相关产品：[MosMos 桌面语音输入与会议记录工具](../sites/mosmos-io.md)；它是 MOSI 生态中的独立产品，不是 MOSS-Image 的组成模块。
