---
type: entity
tags:
  - open-weights
  - image-generation
  - text-to-image
  - diffusion-transformer
  - layout-control
  - mosi
status: complete
updated: 2026-10-22
project_id: moss-image
project: https://openmoss.ai/MOSS-Image/
code: https://github.com/OpenMOSS/MOSS-Image
arxiv: "2610.23374"
related:
  - ../concepts/diffusion-transformer.md
  - ./qwen-image-2-1.md
  - ../methods/generative-data-augmentation.md
sources:
  - ../../sources/repos/moss-image.md
  - ../../sources/sites/moss-image-project.md
  - ../../sources/papers/moss_image_arxiv_2610_23374.md
summary: "OpenMOSS 的 5B 单流 DiT 文生图模型，支持中英文长文本渲染、自然语言或 JSON 布局控制；Base 与 SFT 权重已公开，主要面向信息密集型图像生成。"
---

# MOSS-Image

**MOSS-Image** 是 OpenMOSS 的 5B 单流 Diffusion Transformer 文生图模型家族，目标是生成含有长文本与复杂布局的信息密集型图像。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DiT | Diffusion Transformer | 以 Transformer 为骨干的扩散生成模型 |
| T2I | Text-to-Image | 文本条件图像生成 |
| SFT | Supervised Fine-Tuning | 监督微调；SFT 版本从 Base 派生 |
| PE | Prompt Enhancement | 将简短需求扩展为更详细内容和布局规格的提示增强 |
| LLTB | LongLongTextBench | 评估中英文长文本渲染的基准 |
| RLA | RealLayoutArena | 评估复杂布局图像生成的基准 |

## 为什么重要

开源文生图模型通常更容易生成整体视觉风格，而信息密集图像还要求文字可读、内容不漏、空间关系清晰。MOSS-Image 将文字与布局作为显式生成目标：除自然语言外，还可通过 JSON layout 指定图像元素及其位置。

它可以作为视觉设计、海报和信息图生成工具。对于机器人研究，它与控制策略或视觉语言动作模型无直接关系；若探索生成合成视觉素材，应单独验证图像是否满足真实场景的几何、接触和物理一致性要求。

## 核心原理

### 生成骨干

技术报告介绍从头预训练的 **5B 单流 DiT**。自然语言描述和结构化布局通过同一文本条件路径输入生成器。报告描述的训练配方包括大规模图文数据、多粒度与结构感知文本监督，以及渐进式分辨率训练。

### 从意图到布局

完整系统由意图感知的 prompt enhancement 模块和图像生成器组成：增强模块把用户的简短请求整理为内容与布局规格，DiT 根据这些规格生成图像。不要把该系统级增强器和基础模型本身混为一谈。

### 可控布局输入

官方示例使用 JSON 的 `summary` 与 `items` 描述整体场景、文本和对象。每个元素可包含类别、描述、待绘制文字和边界框。坐标采用 `bbox_format: "yxyx"`、`bbox_coord_space: "abs_centralized"`：以画布中心为原点，顺序为 y-min、x-min、y-max、x-max。

## 模型版本

| 版本 | 定位 | 适用情况 |
|------|------|----------|
| MOSS-Image-Base | 基础文生图模型；便于继续微调，推理设置需按指南选择 | 研究、布局控制试验、定制微调 |
| MOSS-Image-SFT | 在 Base 上监督微调；官方示例中呈现更清晰的文字分组和布局 | 希望开箱获得较规整构图的生成 |

两者使用相同架构与推理接口。该描述是官方版本定位，不代表对所有任务都有普遍质量排序。

## 评测报告

论文提出 LLTB 与 RLA。作者报告 MOSS-Image 在 LLTB 的平均文本覆盖率为 **96.3%**，在开放权重模型中领先；在 RLA 端到端系统比较中排名第二，而在共享增强提示词的生成器比较中排名第一。这些是技术报告中的作者结果，跨基准排序也受提示增强设置影响。

## 工程实践

| 入口 | 用途 | 注意事项 |
|------|------|----------|
| Hugging Face Base / SFT | 加载公开权重 | 模型卡提供 Diffusers 用法；本地推理需要合适的 CUDA GPU 和显存 |
| Diffusers pipeline | Python 推理 | 官方示例推荐从 50 步左右配置开始；具体参数依版本指南 |
| Gradio / ComfyUI | 交互式生成或节点工作流 | 按官方用户指南安装匹配依赖 |
| MOSI API | 云端调用 `moss-image-base-1.0` 或 `moss-image-sft-1.0` | 截至 2026-10-22，平台目录标注 Beta，接口为 `POST /v1/images/generations` |

Base 指南建议以约 1M–2M 总像素、Euler 采样和至少 50 步作为起点；布局输入需让边界框、画布尺寸及提示描述一致。API 费用、可用性和模型快照可能随时间变化，应以平台当前文档为准。

## 局限与风险

- **图像生成器，不是机器人模型：** 不输出关节动作、控制策略或对环境的物理预测。
- **布局不等于物理正确：** JSON bounding boxes 约束的是图像画布位置，不能代替 3D 几何、接触关系或动力学验证。
- **报告结果是作者自报：** 96.3% 与榜单名次应与 LLTB / RLA 的任务定义和提示增强设置一起解读。
- **开源范围需分清：** GitHub 推理代码和 Base/SFT 权重已公开；技术报告所述训练数据、完整训练 recipe 与评测框架不应自动视为都已发布。
- **许可证：** 项目页与模型卡将模型权重标注为 Apache-2.0；仓库代码及第三方组件的细节以各自许可证文件为准。

## 关联页面

- [Diffusion Transformer](../concepts/diffusion-transformer.md) — 生成骨干的架构背景
- [Qwen-Image-2.1](./qwen-image-2-1.md) — 同类开放图像生成模型
- [生成式数据增强](../methods/generative-data-augmentation.md) — 合成视觉素材用于机器人数据的边界与实践

## 参考来源

- [MOSS-Image 官方仓库归档](../../sources/repos/moss-image.md)
- [官方项目页归档](../../sources/sites/moss-image-project.md)
- [技术报告摘录](../../sources/papers/moss_image_arxiv_2610_23374.md)
- [官方项目页](https://openmoss.ai/MOSS-Image/)
- [官方用户指南](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md)
- [MOSI API 模型目录](https://platform.mosi.cn/docs/models/)

## 推荐继续阅读

- [MOSS-Image Technical Report（arXiv:2610.23374）](https://arxiv.org/abs/2610.23374)
