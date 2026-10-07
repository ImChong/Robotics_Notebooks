# PixelUMM（arXiv:2609.38597）

> 来源归档（ingest）

- **标题：** PixelUMM: Encoder-Free Unified Image and Video Understanding and Generation
- **类型：** paper / unified multimodal model / pixel-space image and video modeling
- **arXiv：** <https://arxiv.org/abs/2609.38597>
- **PDF：** <https://arxiv.org/pdf/2609.38597>
- **HTML：** <https://arxiv.org/html/2609.38597>
- **项目页：** <https://nv-tlabs.github.io/PixelUMM/> — [项目页归档](../sites/pixelumm-project.md)
- **官方代码：** <https://github.com/nv-tlabs/PixelUMM> — [代码仓库归档](../repos/nv-tlabs-pixelumm.md)
- **模型权重：** <https://huggingface.co/nvidia/PixelUMM>
- **作者：** Cong Wei、Xuanchi Ren、Bryan Chu、Weiming Ren、Huan Ling、Jiahui Huang、Laura Leal-Taixé、Sanja Fidler、Wenhu Chen、Zian Wang、Jay Zhangjie Wu
- **机构：** NVIDIA；University of Waterloo
- **版本：** v1，2026-09-29 提交；arXiv 预印本
- **入库日期：** 2026-10-07
- **项目实体：** [PixelUMM](../../wiki/entities/paper-pixelumm.md)

## 核心论文摘录

### 1) 用统一像素接口连接图像与视频
图像采用 16×16 RGB patch，视频采用 4×16×16 RGB tubelet。每个 patch/tubelet 经单层线性投影进入同一多模态 Transformer 表征；视觉输入不经过预训练视觉编码器、VAE latent 或离散视觉 tokenizer。
- **对 wiki 的映射：** [PixelUMM 像素接口](../../wiki/entities/paper-pixelumm.md#像素接口与共享主干)。

### 2) MoT 分工、共享自注意力
从 Qwen3 decoder-only Transformer 初始化，使用 token 级路由区分理解与生成专家。两专家各自有归一化、投影和 FFN 参数，但多模态 token 在每个 Transformer block 中通过共享 self-attention 交互。文本使用自回归预测，图像/视频生成在像素空间做 flow matching。
- **对 wiki 的映射：** [PixelUMM 核心机制](../../wiki/entities/paper-pixelumm.md#理解生成分工与共享注意力)。

### 3) 同一序列承载条件与生成目标
干净图像/视频作为理解侧条件，带噪生成目标走生成侧投影；文本、参考视觉条件和噪声视觉 token 在共享注意力中交互，扩展到图像到视频与视频编辑。
- **对 wiki 的映射：** [PixelUMM 流程总览](../../wiki/entities/paper-pixelumm.md#流程总览)。

### 4) 训练与评测的读法
论文报告图像/视频理解与生成上的竞争性表现，并比较 patch/tubelet 压缩率、decoder head、模型规模与算力。项目页指出，对照模型训练数据不同，横向基准不能单独证明架构更优或收敛更快。线性像素输出头存在边界伪影；卷积输出头可缓解，但公开 checkpoint 和基准模型仍使用默认线性头。
- **对 wiki 的映射：** [PixelUMM 评测与结论](../../wiki/entities/paper-pixelumm.md#评测与结果)。

## 资源与许可边界

论文、项目页、源码与模型权重分别发布。代码以 Apache-2.0 为主（保留逐文件第三方声明）；模型权重使用 NVIDIA One-Way Noncommercial License，仅限非商业研究或评估。详见[官方仓库归档](../repos/nv-tlabs-pixelumm.md)。现有资料没有机器人动作策略或实机控制验证。
