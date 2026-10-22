# MOSS-Image Technical Report

> 来源归档（paper）

- **标题：** MOSS-Image Technical Report
- **作者：** MOSI.AI
- **机构：** MOSI.AI；技术报告列出复旦大学、上海交通大学与 SII 的作者单位
- **arXiv：** <https://arxiv.org/abs/2610.23374>
- **项目 / 代码：** <https://github.com/OpenMOSS/MOSS-Image>
- **项目页：** <https://openmoss.ai/MOSS-Image/>
- **报告 PDF：** <https://github.com/OpenMOSS/MOSS-Image/blob/main/paper/MOSS_Image_Technical_Report.pdf>
- **提交日期：** 2026-10-16；v1，cs.CV
- **一句话说明：** 介绍一套面向长文本渲染与复杂布局生成的开放文生图系统，由 5B 单流 DiT 和意图感知 prompt enhancement 组成。

## 摘要要点

1. **生成骨干：** 从头预训练 5B 单流 Diffusion Transformer，目标是双语长文本渲染与精确布局控制。
2. **意图增强：** agentic prompt enhancement（PE）把简短或多样的用户请求扩展成较详细的内容、布局规格，再交由生成器渲染。
3. **训练方案：** 报告介绍广覆盖图文数据、多粒度与结构感知文本监督，以及渐进式分辨率训练。
4. **评测基准：** 提出 LongLongTextBench（LLTB）评长篇中英文文本渲染，提出 RealLayoutArena（RLA）评端到端复杂布局图像生成。
5. **报告结果：** 作者报告 LLTB 平均文本覆盖率为 96.3%，并称在开放权重模型中领先；RLA 端到端系统比较排名第二，共用增强提示词的生成器比较排名第一。此处为论文自报结果，不应当作独立复现结论。

## 发布边界

- GitHub 仓库和 Hugging Face 已公开 MOSS-Image-Base、MOSS-Image-SFT 权重及推理相关代码。
- 论文摘要称将发布模型、训练方案与评测框架；本归档只确认模型权重与推理入口已发布，不据此推断训练数据、完整训练 recipe 或评测代码全部公开。
- 项目页面将模型权重标为 Apache-2.0；仓库也提供 Apache-2.0 许可证。第三方组件仍保留各自许可。

## 对 wiki 的映射

- [MOSS-Image 实体页](../../wiki/entities/moss-image.md) — 模型结构、布局输入、版本差异与适用边界
- [MOSS-Image 项目页归档](../sites/moss-image-project.md)
- [MOSS-Image 官方仓库归档](../repos/moss-image.md)

## 原始来源

- [arXiv 摘要](https://arxiv.org/abs/2610.23374)
- [arXiv HTML](https://arxiv.org/html/2610.23374)
